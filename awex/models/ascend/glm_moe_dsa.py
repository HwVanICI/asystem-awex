# Licensed to the Awex developers under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

"""Unquantized GLM DSA weight exchange for MindSpeed and vLLM-Ascend.

K/V up-projections use [heads, head_dim, kv_lora_rank] on the wire.
This preserves head-wise TP sharding and permits writes into SFA's runtime
weights after vLLM has disposed of the original kv_b_proj parameter.
"""

from copy import copy

import torch

from awex.converter.vllm_converter import VLLMToHFWeightConverter
from awex.sharding.param_sharding import (
    LinearMLAShardingMixin,
    ShardingStrategy,
    ShardingType,
)


def _config(config):
    if getattr(config, "quantization_config", None):
        raise NotImplementedError(
            "GLM DSA AWEX currently supports unquantized weights only"
        )
    # Adapt GLM field names for inherited converters without mutating the model config.
    result = copy(config)
    result.num_key_value_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )
    result.num_experts = config.n_routed_experts
    return result


def _nope_dim(config) -> int:
    # HF and MindSpeed use different names for the non-RoPE key dimension.
    value = getattr(config, "qk_nope_head_dim", None)
    return int(value if value is not None else config.qk_head_dim)


def _rope_dim(config) -> int:
    # Likewise, accept both config names for the positional part of each head.
    value = getattr(config, "qk_rope_head_dim", None)
    return int(value if value is not None else config.qk_pos_emb_head_dim)


class GLMMoeDSAShardingStrategy(LinearMLAShardingMixin, ShardingStrategy):
    def get_shared_expert_sharding_strategy(self, parameter_name, **kwargs):
        # GLM's shared expert is a TP-sharded MLP, not a routed EP expert.
        return self.get_mlp_sharding_strategy(parameter_name, **kwargs)

    def get_sharding_strategy(self, parameter_name, **kwargs):
        # DSA indexer heads are replicated on both sides, not split by attention TP.
        if ".attention.indexer." in parameter_name:
            return ShardingType.NO_SHARDING, 0, 1
        # EP may be enabled even when attention TP is one.
        if ".mlp.experts." in parameter_name:
            return self.get_expert_sharding_strategy(parameter_name, **kwargs)
        return super().get_sharding_strategy(parameter_name, **kwargs)


def _build_mcore_converter():
    # Keep Megatron/MindSpeed imports out of inference-only processes.
    from awex.converter.mcore_converter import (
        LinearMLAMcoreConverterMixin,
        McoreToHFWeightConverter,
        get_full_tensor,
    )

    class GLMMoeDSAMcoreConverter(
        LinearMLAMcoreConverterMixin, McoreToHFWeightConverter
    ):
        def __init__(self, hf_config, rank_info, infer_conf, tf_config):
            if getattr(tf_config, "mtp_num_layers", 0):
                raise NotImplementedError("GLM DSA AWEX does not exchange MTP weights")
            super().__init__(_config(hf_config), rank_info, infer_conf, tf_config)
            # Send separate A projections; vLLM exposes views into its fused tensor.
            self.fuse_qkv_a_proj = False

        def _fuse_gate_up_proj(self, name: str) -> bool:
            # Separate gate/up tensors can be resharded across different TP sizes.
            return False

        def _convert_gate(self, name: str, parameter: torch.Tensor):
            # GLM uses AscendGateLinear on vLLM-Ascend, which explicitly stores
            # router weights in FP32. The generic AWEX converter instead follows
            # infer_conf.router_dtype (BF16 by default), causing metadata mismatch.
            # Cast only the exported tensor; do not change Megatron's parameter
            # storage or confuse moe_router_dtype (compute) with transfer dtype.
            # Keep this override GLM-specific: Qwen uses a different gate class.
            return "mlp.gate.weight", parameter.to(torch.float32)

        def _convert_expert_bias_param(self, name, parameter, layer_number):
            # Preserve the FP32 correction buffer; the generic converter downcasts it.
            return "mlp.gate.expert_bias", parameter

        def _convert_mla_attention_param(self, name, parameter, layer_number):
            # Map only indexers present in the model; shared-indexer layers add none.
            index_prefix = "self_attention.core_attention.indexer."
            index_names = {
                "linear_wq_b.weight": "wq_b.weight",
                "linear_wk.weight": "wk.weight",
                "linear_weights_proj.weight": "weights_proj.weight",
                "k_norm.weight": "k_norm.weight",
                "k_norm.bias": "k_norm.bias",
            }
            if name.startswith(index_prefix):
                suffix = name[len(index_prefix) :]
                if suffix not in index_names:
                    raise NotImplementedError(f"Unsupported GLM indexer weight: {name}")
                if suffix in {"k_norm.weight", "k_norm.bias"}:
                    # The GLM DSA indexer's LayerNorm parameters are FP32 on the
                    # inspected vLLM side, while Megatron exports them as BF16.
                    # Match destination storage for metadata AND payload transfer.
                    # This is separate from the MoE router; indexer projection
                    # weights retain their original dtype. Leave receiver tensors
                    # and strict dtype validation intact.
                    parameter = parameter.to(torch.float32)
                return [(f"attention.indexer.{index_names[suffix]}", parameter)]
            norms = {
                "self_attention.q_layernorm.weight": "attention.q_a_layernorm.weight",
                "self_attention.kv_layernorm.weight": "attention.kv_a_layernorm.weight",
            }
            if name in norms:
                return [(norms[name], parameter)]
            down = {
                "self_attention.linear_q_down_proj.weight": (
                    "attention.q_a_proj.weight",
                    self.hf_config.q_lora_rank,
                ),
                "self_attention.linear_kv_down_proj.weight": (
                    "attention.kv_a_proj_with_mqa.weight",
                    self.hf_config.kv_lora_rank + _rope_dim(self.hf_config),
                ),
            }
            if name in down:
                target, rows = down[name]
                if (
                    parameter.ndim != 2
                    or parameter.shape[1] != self.hf_config.hidden_size
                ):
                    raise ValueError(
                        f"Unexpected GLM A projection shape: {name}: {parameter.shape}"
                    )
                # A projections may already be replicated. Gather only true TP shards
                # to avoid duplicating rows; rank zero supplies the replicated export.
                if parameter.shape[0] != rows:
                    if parameter.shape[0] * self.rank_info.tp_size != rows:
                        raise ValueError(
                            f"Unexpected GLM A projection rows: {name}: {parameter.shape}"
                        )
                    parameter = get_full_tensor(parameter, dim=0)
                return [(target, parameter)] if self.rank_info.tp_rank == 0 else []
            # Expose heads explicitly so TP resharding keeps each head's K/V rows
            # together, matching the bridge's per-head [K; V] checkpoint layout.
            kv_rank = self.hf_config.kv_lora_rank
            for kind, dim in (
                ("k", _nope_dim(self.hf_config)),
                ("v", self.hf_config.v_head_dim),
            ):
                if name == f"self_attention.linear_{kind}_up_proj.weight":
                    heads = self.hf_config.num_attention_heads // self.rank_info.tp_size
                    if tuple(parameter.shape) != (heads * dim, kv_rank):
                        raise ValueError(
                            f"Unexpected GLM {kind.upper()} projection: {parameter.shape}"
                        )
                    return [
                        (
                            f"attention.{kind}_up_proj.weight",
                            parameter.reshape(heads, dim, kv_rank),
                        )
                    ]
            # Non-absorbed MLA stores K and V together; use the same wire layout.
            if name == "self_attention.linear_kv_up_proj.weight":
                kdim, vdim = _nope_dim(self.hf_config), self.hf_config.v_head_dim
                weight = parameter.reshape(-1, kdim + vdim, kv_rank)
                return [
                    ("attention.k_up_proj.weight", weight[:, :kdim]),
                    ("attention.v_up_proj.weight", weight[:, kdim:]),
                ]
            return super()._convert_mla_attention_param(name, parameter, layer_number)

    return GLMMoeDSAMcoreConverter


class GLMMoeDSAVLLMConverter(VLLMToHFWeightConverter):
    def __init__(self, model_config, infer_engine_config, rank_info):
        super().__init__(_config(model_config), infer_engine_config, rank_info)
        self._runtime_copies = []

    def _normalize_name(self, name: str) -> str:
        # Both backends must describe the same logical tensor with the same name.
        name = name.replace(".self_attn.", ".attention.")
        name = name.replace(".experts.routed_experts.", ".experts.")
        name = name.replace(".attention.o_proj.", ".attention.dense.")
        return name

    def _matrix(self, parameter, rows, cols, name):
        # Receiver tensors must remain views: cloning here would redirect updates
        # into detached storage instead of the weights used by inference.
        if tuple(parameter.shape) == (rows, cols):
            return parameter
        if tuple(parameter.shape) == (cols, rows) and rows != cols:
            return parameter.t()
        raise ValueError(
            f"Unexpected GLM weight shape for {name}: {parameter.shape}; expected {(rows, cols)}"
        )

    def convert_param(self, name, parameter):
        name = self._normalize_name(name)
        if name.endswith(".mlp.gate.e_score_correction_bias"):
            return [(name.replace("e_score_correction_bias", "expert_bias"), parameter)]
        config = self.model_config
        # vLLM packs Q-A first, then KV-A (including RoPE rows). Slice writable
        # views to match the separate tensors exported by the training converter.
        if name.endswith(".attention.fused_qkv_a_proj.weight"):
            qrows = config.q_lora_rank
            kvrows = config.kv_lora_rank + _rope_dim(config)
            weight = self._matrix(parameter, qrows + kvrows, config.hidden_size, name)
            prefix = name.removesuffix("fused_qkv_a_proj.weight")
            return [
                (prefix + "q_a_proj.weight", weight[:qrows]),
                (prefix + "kv_a_proj_with_mqa.weight", weight[qrows:]),
            ]
        if ".attention.indexer." in name:
            # Newer vLLM packs the indexer key and score projections into one GEMM.
            if name.endswith(".wk_weights_proj.weight"):
                dim = config.index_head_dim
                weight = self._matrix(
                    parameter, dim + config.index_n_heads, config.hidden_size, name
                )
                prefix = name.removesuffix("wk_weights_proj.weight")
                return [
                    (prefix + "wk.weight", weight[:dim]),
                    (prefix + "weights_proj.weight", weight[dim:]),
                ]
            suffix = name.split(".attention.indexer.", 1)[1]
            dims = {
                "wq_b.weight": (
                    config.index_n_heads * config.index_head_dim,
                    config.q_lora_rank,
                ),
                "wk.weight": (config.index_head_dim, config.hidden_size),
                "weights_proj.weight": (config.index_n_heads, config.hidden_size),
            }
            if suffix in dims:
                return [(name, self._matrix(parameter, *dims[suffix], name))]
            if suffix in {"k_norm.weight", "k_norm.bias"}:
                return [(name, parameter)]
            raise NotImplementedError(f"Unsupported GLM indexer weight: {name}")
        # Synthetic names from iter_model_parameters already reference SFA's
        # live K/V tensors in canonical [heads, head_dim, latent] order.
        if name.endswith(
            (".attention.k_up_proj.weight", ".attention.v_up_proj.weight")
        ):
            return [(name, parameter)]
        if name.endswith(".attention.kv_b_proj.weight"):
            kdim, vdim = _nope_dim(config), config.v_head_dim
            heads = config.num_attention_heads // self.tp_size
            weight = self._matrix(
                parameter, heads * (kdim + vdim), config.kv_lora_rank, name
            )
            # view, never reshape: the receiver must alias the original storage.
            weight = weight.view(heads, kdim + vdim, config.kv_lora_rank)
            prefix = name.removesuffix("kv_b_proj.weight")
            return [
                (prefix + "k_up_proj.weight", weight[:, :kdim]),
                (prefix + "v_up_proj.weight", weight[:, kdim:]),
            ]
        return super().convert_param(name, parameter)

    def iter_model_parameters(self, model):
        """Include SFA runtime tensors, retaining their captured addresses."""
        self._runtime_copies = []
        replacements = {}
        extras = []
        for prefix, module in model.named_modules():
            if not prefix.endswith(".self_attn"):
                continue
            # SFA's implementation is nested inside the MLA wrapper and owns
            # derived tensors that ordinary named_parameters() does not expose.
            attention = getattr(module, "mla_attn", None)
            attention = getattr(attention, "mla_attn", attention)
            if attention is None:
                attention = getattr(module, "attn", None)
            impl = getattr(attention, "impl", None)
            if impl is None or not hasattr(impl, "W_UK_T"):
                continue
            if getattr(impl, "enable_dsa_cp", False) or hasattr(impl, "wd_qkv"):
                raise NotImplementedError(
                    "GLM AWEX supports unquantized SFA without inference DSA-CP"
                )
            # kv_b_proj may have zero backing storage after SFA initialization.
            replacements[prefix + ".kv_b_proj.weight"] = None
            # K is [heads, Kdim, latent]; V is stored as [heads, latent, Vdim].
            # Transposing V yields a writable view, not a replacement allocation.
            extras.extend(
                [
                    (prefix + ".k_up_proj.weight", impl.W_UK_T),
                    (prefix + ".v_up_proj.weight", impl.W_UV.transpose(1, 2)),
                ]
            )
            if hasattr(impl, "weight_dq"):
                # PROLOG_V3 maintains additional copies of the A/Q projections.
                fused = module.fused_qkv_a_proj.weight
                qweight = module.q_b_proj.weight
                if (
                    fused.untyped_storage().nbytes() == 0
                    or qweight.untyped_storage().nbytes() == 0
                ):
                    raise NotImplementedError(
                        "GLM AWEX does not support disposed A/Q weights on disaggregated decode workers"
                    )
                rows = self.model_config.q_lora_rank
                full = self._matrix(
                    fused,
                    rows
                    + self.model_config.kv_lora_rank
                    + _rope_dim(self.model_config),
                    self.model_config.hidden_size,
                    prefix,
                )
                # Keep live source views: subsequent receives update these bases,
                # and post_update_weights refreshes the separate kernel copies.
                self._runtime_copies.extend(
                    [
                        (impl.weight_dq, full[:rows].t()),
                        (impl.weight_dkv_kr, full[rows:].t()),
                        (impl.weight_uq_qr, qweight.t()),
                    ]
                )
        for name, param in model.named_parameters():
            if name not in replacements:
                yield name, param
        yield from extras

    @torch.no_grad()
    def post_update_weights(self):
        # Copy in place to preserve addresses referenced by captured inference graphs.
        for destination, source in self._runtime_copies:
            if destination.shape != source.shape:
                raise ValueError(
                    f"GLM SFA runtime weight shape mismatch: {destination.shape} != {source.shape}"
                )
            destination.copy_(source)


# The existing Ascend registry discovers this entry; the factory delays Megatron
# imports until a training converter is requested. Other model entries are unchanged.
CONFIG = {
    "model_name": "GlmMoeDsaForCausalLM",
    "sharding_strategy": GLMMoeDSAShardingStrategy,
    "mcore_converter": _build_mcore_converter,
    "vllm_converter": GLMMoeDSAVLLMConverter,
}
