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

from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from awex.config import InferenceConfig
from awex.models.registry import (
    get_infer_weights_converter,
    get_sharding_strategy,
    get_train_weights_converter,
)
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo


class _MiniMaxConfig(PretrainedConfig):
    model_type = "minimax_m2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.architectures = ["MiniMaxM2ForCausalLM"]
        self.num_hidden_layers = 1
        self.num_attention_heads = 48
        self.num_key_value_heads = 8
        self.num_local_experts = 256


def _make_rank_info(
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    attn_tp_size: int | None = None,
) -> RankInfo:
    if attn_tp_size is None:
        attn_tp_size = tp_size
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        dp_size=1,
        dp_rank=0,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=tp_rank,
        attn_tp_size=attn_tp_size,
        attn_dp_rank=0,
        world_size=max(tp_size, 1) * max(ep_size, 1),
        global_rank=tp_rank,
        local_rank=tp_rank,
        engine_rank=0,
        is_infer=False,
    )


def test_minimax_vllm_converter_normalizes_block_sparse_moe_names():
    cfg = _MiniMaxConfig()
    infer_config = InferenceConfig(tp_size=1, ep_size=1)
    rank_info = _make_rank_info()

    converter = get_infer_weights_converter(
        "vllm",
        "MiniMaxM2ForCausalLM",
        cfg,
        rank_info,
        infer_config,
    )

    converted = converter.convert_param(
        "model.layers.0.block_sparse_moe.experts.w2_weight",
        torch.randn(1, 4, 2),
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.0.down_proj.weight"
    ]


def test_minimax_vllm_converter_maps_e_score_correction_bias_to_expert_bias():
    cfg = _MiniMaxConfig()
    infer_config = InferenceConfig(tp_size=1, ep_size=1)
    rank_info = _make_rank_info()

    converter = get_infer_weights_converter(
        "vllm",
        "MiniMaxM2ForCausalLM",
        cfg,
        rank_info,
        infer_config,
    )

    converted = converter.convert_param(
        "model.layers.0.block_sparse_moe.e_score_correction_bias",
        torch.randn(256),
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.gate.expert_bias"
    ]


def test_minimax_train_converter_uses_num_local_experts_fallback():
    cfg = _MiniMaxConfig()
    rank_info = _make_rank_info(tp_size=1, ep_size=2, ep_rank=1)
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace(num_local_experts=256)

    converter = get_train_weights_converter(
        "mcore",
        "MiniMaxM2ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )

    converted = converter.convert_param(
        "decoder.layers.0.block_sparse_moe.experts.local_experts.0.linear_fc1.weight",
        torch.randn(4, 2),
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.128.gate_proj.weight",
        "model.layers.0.mlp.experts.128.up_proj.weight",
    ]


def test_minimax_train_converter_normalizes_router_expert_bias_name():
    cfg = _MiniMaxConfig()
    rank_info = _make_rank_info(tp_size=1, ep_size=1)
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace(num_local_experts=256)

    converter = get_train_weights_converter(
        "mcore",
        "MiniMaxM2ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )

    converted = converter.convert_param(
        "decoder.layers.0.block_sparse_moe.router.expert_bias",
        torch.randn(256),
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.gate.expert_bias"
    ]


def test_minimax_query_layernorm_uses_tp_sharding():
    strategy_cls = get_sharding_strategy("MiniMaxM2ForCausalLM")
    rank_info = _make_rank_info(tp_size=4)
    strategy = strategy_cls(
        engine_name="vllm",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=None,
        tp_size=4,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    q_norm = strategy.get_sharding_strategy(
        "model.layers.0.attention.query_layernorm.weight"
    )
    k_norm = strategy.get_sharding_strategy(
        "model.layers.0.attention.key_layernorm.weight"
    )
    input_norm = strategy.get_sharding_strategy(
        "model.layers.0.input_layernorm.weight"
    )

    assert q_norm == (ShardingType.TP_SHARDING, 0, 4)
    assert k_norm == (ShardingType.TP_SHARDING, 0, 4)
    assert input_norm == (ShardingType.NO_SHARDING, 0, 1)


def test_minimax_expert_bias_is_not_tp_sharded():
    strategy_cls = get_sharding_strategy("MiniMaxM2ForCausalLM")
    rank_info = _make_rank_info(tp_size=4)
    strategy = strategy_cls(
        engine_name="vllm",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=None,
        tp_size=4,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    expert_bias = strategy.get_sharding_strategy(
        "model.layers.0.mlp.gate.expert_bias"
    )

    assert expert_bias == (ShardingType.NO_SHARDING, 0, 1)
