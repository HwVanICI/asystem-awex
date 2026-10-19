# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import re
from typing import Any

import torch

from awex.converter.mcore_converter import McoreToHFWeightConverter
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.weights_converter import append_scale_inv, normalize_scale_inv_name
from awex.models.ascend.qwen3_5 import (
    Qwen3VLShardingStrategy as Qwen3_5ShardingStrategy,
    reshard_visual_attn_qkv,
)
from awex.sharding.param_sharding import ShardingType


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def get_qwen3_vl_text_config(config: Any) -> Any:
    """Return the text sub-config from a composite Qwen3-VL config."""
    return _config_value(config, "text_config", config)


class Qwen3VLShardingStrategy(Qwen3_5ShardingStrategy):
    """TP/EP rules for Qwen3-VL language and vision parameters.

    Extends the Qwen3.5 ascend sharding strategy with deepstack-merger
    support for the vision tower.
    """

    _deepstack_sharding_dims = {
        "linear_fc1.weight": 0,
        "linear_fc1.bias": 0,
        "linear_fc2.weight": 1,
    }

    def _get_visual_sharding_strategy(self, parameter_name: str, **kwargs):
        tp_size = self.rank_info.tp_size
        if tp_size == 1:
            return ShardingType.NO_SHARDING, 0, 1

        visual_prefix = "model.visual."
        if parameter_name.startswith(visual_prefix):
            suffix = parameter_name[len(visual_prefix) :]

            # Deepstack merger params: strip "deepstack_merger_list.{i}." and
            # look up the remaining suffix (e.g. "linear_fc1.weight").
            if suffix.startswith("deepstack_merger_list."):
                parts = suffix.split(".", 2)
                if len(parts) >= 3:
                    merger_suffix = parts[2]
                    dim = self._deepstack_sharding_dims.get(merger_suffix)
                    if dim is not None:
                        return ShardingType.TP_SHARDING, dim, tp_size
                return ShardingType.NO_SHARDING, 0, 1

        return super()._get_visual_sharding_strategy(parameter_name, **kwargs)


class VLLMToHFWeightConverterQwen3VL(SGlangToHFWeightConverter):
    """Normalize current vLLM Qwen3-VL target parameters.

    vLLM exposes the vision tower under ``visual.*`` and the language model
    under ``language_model.*`` (or top-level for non-VLM checkpoints).  The
    converter restores a stable checkpoint-like namespace and remaps text
    parameters to ``model.language_model.*``.
    """

    def __init__(self, model_config, infer_engine_config, rank_info):
        self.vl_model_config = model_config
        super().__init__(
            get_qwen3_vl_text_config(model_config), infer_engine_config, rank_info
        )

    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_visual_param(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        return [(f"model.{name}", parameter)]

    def _normalize_name(self, name: str) -> str:
        name, has_scale_inv = normalize_scale_inv_name(name)
        replacements = [
            (".self_attn.attn.qkv", ".attention.query_key_value_proj"),
            (".self_attn.attn.qkv_proj", ".attention.query_key_value_proj"),
            (".self_attn.qkv", ".attention.query_key_value_proj"),
            (".self_attn.qkv_proj", ".attention.query_key_value_proj"),
            (".self_attn.attn.o_proj", ".attention.dense"),
            (".self_attn.o_proj", ".attention.dense"),
            (".self_attn.proj", ".attention.dense"),
            (".self_attn.q_norm", ".attention.query_layernorm"),
            (".self_attn.k_norm", ".attention.key_layernorm"),
        ]
        for old, new in replacements:
            if old in name:
                name = name.replace(old, new)
        # Guard against double normalization.
        name = name.replace("query_key_value_proj_proj", "query_key_value_proj")
        return append_scale_inv(name, has_scale_inv)

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if name.startswith("visual."):
            converted_params = self._convert_visual_param(name, parameter)
            return converted_params

        language_prefix = "language_model."
        name = name.replace(language_prefix, "").replace(
            "shared_expert.", "shared_experts."
        )

        converted_params = super().convert_param(
            self._normalize_name(name), parameter
        )

        if len(converted_params) == 1 and converted_params[0][0] == "lm_head.weight":
            return converted_params
        converted_params = [
            (s.replace("model.", "model.language_model.", 1), t)
            for s, t in converted_params
        ]

        return converted_params


_VISION_LAYER_PATTERN = re.compile(r"vision_model\.decoder\.layers\.(\d+)\.(.+)")
_VISION_DEEPSTACK_PATTERN = re.compile(
    r"vision_model\.decoder\.deepstack_merger_list\.(\d+)\.(.+)"
)
_VISION_DIRECT_NAMES = {
    "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
    "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
    "vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
    "vision_model.merger.patch_norm.weight": "model.visual.merger.norm.weight",
    "vision_model.merger.patch_norm.bias": "model.visual.merger.norm.bias",
    "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
    "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
    "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
    "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
}


class McoreToHFWeightConverterQwen3VL(McoreToHFWeightConverter):
    """Extend the base converter with vision-tower parameter mappings.

    The text decoder uses standard GQA attention, so attention conversion
    is handled entirely by the base ``_convert_attention_param``.  Vision-
    tower parameters—including deepstack mergers—are mapped to the
    ``model.visual.*`` namespace expected by the inference side.
    """

    _vision_layer_mapping = {
        "self_attention.linear_qkv.weight": "attn.qkv.weight",
        "self_attention.linear_qkv.bias": "attn.qkv.bias",
        "self_attention.linear_proj.weight": "attn.proj.weight",
        "self_attention.linear_proj.bias": "attn.proj.bias",
        "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
        "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
        "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
        "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
        "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
        "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
    }

    _deepstack_mapping = {
        "patch_norm.weight": "norm.weight",
        "patch_norm.bias": "norm.bias",
        "linear_fc1.weight": "linear_fc1.weight",
        "linear_fc1.bias": "linear_fc1.bias",
        "linear_fc2.weight": "linear_fc2.weight",
        "linear_fc2.bias": "linear_fc2.bias",
    }

    def __init__(self, hf_config, rank_info, infer_conf, tf_config):
        self.vl_hf_config = hf_config
        super().__init__(
            get_qwen3_vl_text_config(hf_config),
            rank_info,
            infer_conf,
            tf_config=tf_config,
        )
        self.vision_config = _config_value(hf_config, "vision_config")

    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_vision_qkv(
        self, layer_number: str, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Reshard vision-tower QKV from train TP to infer TP layout."""
        if self.vision_config is None:
            raise ValueError(
                "vision_config is required for Qwen3-VL visual QKV"
            )
        packed = reshard_visual_attn_qkv(
            parameter,
            self.infer_atten_tp_size,
            self.vision_config,
            int(self.rank_info.attn_tp_rank),
            max(1, int(self.rank_info.attn_tp_size)),
        )
        kind = "weight" if name.endswith("weight") else "bias"
        return [(f"model.visual.blocks.{layer_number}.attn.qkv.{kind}", packed)]

    def _convert_vision_param(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if name in _VISION_DIRECT_NAMES:
            return [(_VISION_DIRECT_NAMES[name], parameter)]

        match = _VISION_LAYER_PATTERN.fullmatch(name)
        if match is not None:
            layer_number, remaining_name = match.groups()
            base = f"model.visual.blocks.{layer_number}"
            if remaining_name in (
                "self_attention.linear_qkv.weight",
                "self_attention.linear_qkv.bias",
            ):
                return self._convert_vision_qkv(
                    layer_number, remaining_name, parameter
                )
            target_suffix = self._vision_layer_mapping.get(remaining_name)
            if target_suffix is not None:
                return [(f"{base}.{target_suffix}", parameter)]

        match = _VISION_DEEPSTACK_PATTERN.fullmatch(name)
        if match is not None:
            merger_number, remaining_name = match.groups()
            target_suffix = self._deepstack_mapping.get(remaining_name)
            if target_suffix is not None:
                base = f"model.visual.deepstack_merger_list.{merger_number}"
                return [(f"{base}.{target_suffix}", parameter)]

        raise ValueError(f"Unknown Qwen3-VL vision parameter: {name}")

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int = None
    ) -> list[tuple[str, torch.Tensor]]:
        name = name.replace("module.", "")
        if name.startswith("vision_model."):
            return self._convert_vision_param(name, parameter)
        if not name.startswith("language_model."):
            raise ValueError(f"Unknown Qwen3-VL parameter: {name}")

        language_name = name[len("language_model.") :]
        converted = []
        for hf_name, hf_param in super().convert_param(
            language_name, parameter, vp_stage=vp_stage
        ):
            if hf_name.startswith("model."):
                hf_name = hf_name.replace("model.", "model.language_model.", 1)
            converted.append((hf_name, hf_param))

        return converted


CONFIG = [
    {
        "model_name": "Qwen3VLForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterQwen3VL,
        "vllm_converter": VLLMToHFWeightConverterQwen3VL,
    },
    {
        "model_name": "Qwen3VLMoeForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterQwen3VL,
        "vllm_converter": VLLMToHFWeightConverterQwen3VL,
    },
]