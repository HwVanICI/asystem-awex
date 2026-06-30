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

"""Converter for vLLM parameter names to HF/Megatron-friendly names.

We normalize vLLM self-attention names to the HF-style projection naming
used by Awex converters (e.g., qkv/qkv_proj -> query_key_value_proj).
"""

from typing import List, Tuple

import torch

from awex.converter.sglang_converter import (
    LinearMLASGlangConverterMixin,
    SGlangToHFWeightConverter,
)
from awex.converter.weights_converter import append_scale_inv, normalize_scale_inv_name


class VLLMToHFWeightConverter(
    LinearMLASGlangConverterMixin,
    SGlangToHFWeightConverter,
):
    def _use_fsdp_train_names(self) -> bool:
        train_engine_name = self._cfg_value(
            self.infer_engine_config, "train_engine_name", None
        )
        return (
            isinstance(train_engine_name, str) and train_engine_name.lower() == "fsdp"
        )

    def _normalize_name(self, name: str) -> str:
        name, has_scale_inv = normalize_scale_inv_name(name)
        if self._use_fsdp_train_names():
            replacements = [
                (".self_attn.attn.qkv_proj", ".self_attn.qkv_proj"),
                (".self_attn.attn.qkv", ".self_attn.qkv"),
                (".self_attn.attn.o_proj", ".self_attn.o_proj"),
                (".self_attn.proj", ".self_attn.o_proj"),
            ]
            for old, new in replacements:
                if old in name:
                    name = name.replace(old, new)
            return append_scale_inv(name, has_scale_inv)

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

    def _qkv_split_sizes(self, parameter: torch.Tensor) -> Tuple[int, int, int]:
        head_dim = getattr(self.model_config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(self.model_config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Cannot split qkv without head_dim or hidden_size.")
            head_dim = hidden_size // self.total_num_heads

        q_size = self.total_num_heads * head_dim
        kv_size = self.total_kv_heads * head_dim
        total_size = q_size + 2 * kv_size
        local_size = parameter.shape[0]
        if local_size == total_size:
            return q_size, kv_size, kv_size

        attn_tp_size = (
            getattr(self.rank_info, "attn_tp_size", None) or self.tp_size or 1
        )
        if (
            attn_tp_size > 1
            and total_size % attn_tp_size == 0
            and local_size == total_size // attn_tp_size
            and q_size % attn_tp_size == 0
            and kv_size % attn_tp_size == 0
        ):
            return (
                q_size // attn_tp_size,
                kv_size // attn_tp_size,
                kv_size // attn_tp_size,
            )

        head_units = self.total_num_heads + 2 * self.total_kv_heads
        if head_units > 0 and local_size % head_units == 0:
            local_head_dim = local_size // head_units
            return (
                self.total_num_heads * local_head_dim,
                self.total_kv_heads * local_head_dim,
                self.total_kv_heads * local_head_dim,
            )

        raise ValueError(
            "Cannot split qkv parameter with shape "
            f"{tuple(parameter.shape)} for num_attention_heads={self.total_num_heads}, "
            f"num_key_value_heads={self.total_kv_heads}, head_dim={head_dim}."
        )

    def _split_qkv_to_fsdp_names(
        self, name: str, parameter: torch.Tensor, has_scale_inv: bool
    ) -> List[Tuple[str, torch.Tensor]]:
        q_size, k_size, v_size = self._qkv_split_sizes(parameter)
        if name.startswith("attention."):
            q_name = "self_attn.q_proj.weight"
            k_name = "self_attn.k_proj.weight"
            v_name = "self_attn.v_proj.weight"
            if name.endswith(".bias"):
                q_name = q_name.replace(".weight", ".bias")
                k_name = k_name.replace(".weight", ".bias")
                v_name = v_name.replace(".weight", ".bias")
        else:
            q_name = name.replace("qkv_proj", "q_proj").replace("qkv", "q_proj")
            k_name = name.replace("qkv_proj", "k_proj").replace("qkv", "k_proj")
            v_name = name.replace("qkv_proj", "v_proj").replace("qkv", "v_proj")
        return [
            (append_scale_inv(q_name, has_scale_inv), parameter.narrow(0, 0, q_size)),
            (
                append_scale_inv(k_name, has_scale_inv),
                parameter.narrow(0, q_size, k_size),
            ),
            (
                append_scale_inv(v_name, has_scale_inv),
                parameter.narrow(0, q_size + k_size, v_size),
            ),
        ]

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        if not self._use_fsdp_train_names():
            return super()._convert_attention_param(name, parameter, layer_number)

        base_name, has_scale_inv = normalize_scale_inv_name(name)
        if base_name in {
            "self_attn.qkv_proj.weight",
            "self_attn.qkv_proj.bias",
            "self_attn.qkv.weight",
            "self_attn.qkv.bias",
            "attention.query_key_value_proj.weight",
            "attention.query_key_value_proj.bias",
            "attention.query_key_value.weight",
            "attention.query_key_value.bias",
        }:
            return self._split_qkv_to_fsdp_names(base_name, parameter, has_scale_inv)

        if base_name in {
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.bias",
        }:
            return [(append_scale_inv(base_name, has_scale_inv), parameter)]

        return super()._convert_attention_param(name, parameter, layer_number)

    def _convert_layer_norm_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        if self._use_fsdp_train_names():
            base_name, has_scale_inv = normalize_scale_inv_name(name)
            if base_name in {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}:
                return [(append_scale_inv(base_name, has_scale_inv), parameter)]
        return super()._convert_layer_norm_param(name, parameter, layer_number)

    def convert_param(self, name, parameter):
        return super().convert_param(self._normalize_name(name), parameter)
