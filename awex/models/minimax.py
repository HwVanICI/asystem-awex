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

from typing import Dict

from transformers import PretrainedConfig

from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.vllm_converter import VLLMToHFWeightConverter
from awex.sharding.param_sharding import ShardingStrategy
from awex.sharding.rank_info import RankInfo


def _normalize_minimax_name(name: str) -> str:
    # MiniMax HF/vLLM names use block_sparse_moe where Awex canonical names
    # reuse the Qwen-style mlp namespace.
    return name.replace(".block_sparse_moe.", ".mlp.")


class MiniMaxShardingStrategy(ShardingStrategy):
    def get_sharding_strategy(self, parameter_name, **kwargs):
        if (
            "attention.query_layernorm.weight" in parameter_name
            or "attention.key_layernorm.weight" in parameter_name
        ):
            return self.get_attention_sharding_strategy(parameter_name, **kwargs)
        return super().get_sharding_strategy(parameter_name, **kwargs)


def _build_mcore_converter_minimax():
    from awex.converter.mcore_converter import McoreToHFWeightConverter

    class McoreToHFWeightConverterMiniMax(McoreToHFWeightConverter):
        def __init__(
            self,
            hf_config: PretrainedConfig,
            rank_info: RankInfo,
            infer_conf: Dict,
            tf_config,
        ):
            super().__init__(hf_config, rank_info, infer_conf, tf_config=tf_config)

        def convert_param(self, name, parameter, vp_stage=None):
            return super().convert_param(
                _normalize_minimax_name(name), parameter, vp_stage=vp_stage
            )

    return McoreToHFWeightConverterMiniMax


class MiniMaxSGlangToHFWeightConverter(SGlangToHFWeightConverter):
    def convert_param(self, name, parameter):
        return super().convert_param(_normalize_minimax_name(name), parameter)


class MiniMaxVLLMToHFWeightConverter(VLLMToHFWeightConverter):
    def convert_param(self, name, parameter):
        return super().convert_param(_normalize_minimax_name(name), parameter)


CONFIG = {
    "model_name": "MiniMaxM2ForCausalLM",
    "sharding_strategy": MiniMaxShardingStrategy,
    "mcore_converter": _build_mcore_converter_minimax,
    "sglang_converter": MiniMaxSGlangToHFWeightConverter,
    "vllm_converter": MiniMaxVLLMToHFWeightConverter,
}
