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

import pytest

from awex.models.ascend.glm_moe_dsa import GLMMoeDSAShardingStrategy
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo


def _strategy(engine, tp, dp, cp, ep):
    rank = RankInfo(
        tp_rank=0,
        tp_size=tp,
        pp_rank=0,
        pp_size=1,
        dp_size=dp,
        dp_rank=0,
        ep_rank=0,
        ep_size=ep,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=tp,
        attn_dp_rank=0,
        world_size=tp * dp * cp,
        global_rank=0,
        local_rank=0,
        engine_rank=0,
        is_infer=engine == "vllm",
        cp_size=cp,
    )
    return GLMMoeDSAShardingStrategy(
        engine_name=engine,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=tp,
        tp_size=tp,
        ep_size=ep,
        ep_tp_size=1,
        rank_info=rank,
    )


@pytest.mark.parametrize(
    "engine,tp,dp,cp,ep",
    [("vllm", 4, 32, 1, 128), ("mcore", 4, 1, 8, 32), ("vllm", 1, 128, 1, 128)],
)
@pytest.mark.parametrize(
    "projection,dim", [("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)]
)
def test_glm_shared_experts_follow_tp_not_ep(engine, tp, dp, cp, ep, projection, dim):
    """Shared MLP weights must not be multiplied by the routed-expert group size."""
    strategy = _strategy(engine, tp, dp, cp, ep)
    actual = strategy.get_sharding_strategy(
        f"model.layers.68.mlp.shared_experts.{projection}.weight"
    )

    kind = ShardingType.TP_SHARDING if tp > 1 else ShardingType.NO_SHARDING
    assert actual == (kind, dim, tp)


@pytest.mark.parametrize(
    "engine,tp,dp,cp,ep",
    [("vllm", 4, 32, 1, 128), ("mcore", 4, 1, 8, 32), ("vllm", 1, 128, 1, 128)],
)
def test_glm_routed_experts_keep_ep_sharding(engine, tp, dp, cp, ep):
    """Fixing the shared expert must not change routed experts, including TP1."""
    strategy = _strategy(engine, tp, dp, cp, ep)
    actual = strategy.get_sharding_strategy(
        "model.layers.68.mlp.experts.0.up_proj.weight"
    )

    assert actual == (ShardingType.EP_SHARDING, 0, ep)


def test_glm_shared_expert_global_numel_matches_training():
    """The observed inference shard expands to 2048*6144, not 128*512*6144."""
    strategy = _strategy("vllm", 4, 32, 1, 128)
    _, _, shards = strategy.get_sharding_strategy(
        "model.layers.68.mlp.shared_experts.up_proj.weight"
    )

    assert 512 * 6144 * shards == 2048 * 6144
