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

import pytest
import torch

from awex.meta.meta_resolver import get_num_hidden_layers
from awex.models.ascend.qwen3_vl import (
    McoreToHFWeightConverterQwen3VL,
    Qwen3VLShardingStrategy,
    VLLMToHFWeightConverterQwen3VL,
)
from awex.converter.mcore_converter import McoreToHFWeightConverter
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.sharding.param_sharding import ShardingType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def text_config():
    return SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_experts=4,
        tie_word_embeddings=False,
    )


@pytest.fixture
def vl_config(text_config):
    return SimpleNamespace(
        architectures=["Qwen3VLForConditionalGeneration"],
        text_config=text_config,
        vision_config=SimpleNamespace(num_heads=2, hidden_size=4),
    )


@pytest.fixture
def tf_config():
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        num_layers=2,
    )


@pytest.fixture
def rank_info():
    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_size=1,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_rank=0,
        world_size=1,
        global_rank=0,
        local_rank=0,
        engine_rank=0,
        is_infer=False,
        cp_rank=0,
        cp_size=1,
        cp_mode="none",
    )


@pytest.fixture
def infer_engine_config():
    return SimpleNamespace(tp_size=1, ep_size=1, device_backend="cpu")


@pytest.fixture(autouse=True)
def _mock_get_full_tensor(monkeypatch):
    """Avoid Megatron TP-group initialization in CPU unit tests."""
    try:
        monkeypatch.setattr(
            "awex.converter.mcore_converter.get_full_tensor",
            lambda parameter, dim=0: parameter,
        )
    except (AttributeError, ImportError):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_converter(vl_config, rank_info, tf_config, **infer_conf):
    config = {"infer_atten_tp_size": 1, **infer_conf}
    return McoreToHFWeightConverterQwen3VL(vl_config, rank_info, config, tf_config)


def _make_infer_converter(vl_config, rank_info, infer_engine_config):
    return VLLMToHFWeightConverterQwen3VL(vl_config, infer_engine_config, rank_info)


# ---------------------------------------------------------------------------
# Tests – config resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(num_hidden_layers=12), 12),
        (SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=28)), 28),
        ({"text_config": {"num_hidden_layers": 48}}, 48),
    ],
)
def test_num_hidden_layers_supports_flat_and_composite_configs(config, expected):
    assert get_num_hidden_layers(config) == expected


def test_num_hidden_layers_rejects_missing_decoder_depth():
    with pytest.raises(AttributeError, match="num_hidden_layers"):
        get_num_hidden_layers(SimpleNamespace(vision_config=SimpleNamespace()))


# ---------------------------------------------------------------------------
# Tests – converter wiring
# ---------------------------------------------------------------------------


def test_mcore_converter_stores_vl_config(vl_config, rank_info, tf_config):
    converter = _make_train_converter(vl_config, rank_info, tf_config)
    assert converter.vl_hf_config is vl_config
    assert converter.hf_config is vl_config.text_config
    assert converter.vision_config is vl_config.vision_config


def test_vllm_converter_stores_vl_config(vl_config, rank_info, infer_engine_config):
    converter = _make_infer_converter(vl_config, rank_info, infer_engine_config)
    assert converter.vl_model_config is vl_config


# ---------------------------------------------------------------------------
# Tests – train/infer name and value parity
# ---------------------------------------------------------------------------


def test_language_qkv_has_train_infer_name_and_value_parity(
    vl_config, tf_config, rank_info, infer_engine_config
):
    """Both converters must emit the same canonical names for fused QKV.

    MCore: ``linear_qkv`` → ``self_attn.qkv_proj`` → ``_normalize_attn_name``
    → ``attention.query_key_value_proj`` → prefix →
    ``model.language_model.layers.0.attention.query_key_value_proj.weight``

    vLLM: ``self_attn.qkv_proj`` → ``_normalize_name`` →
    ``attention.query_key_value_proj`` → prefix → same.
    """
    train_converter = _make_train_converter(
        vl_config, rank_info, tf_config, router_dtype="fp32"
    )
    infer_converter = _make_infer_converter(vl_config, rank_info, infer_engine_config)
    mcore_qkv = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)

    train_params = train_converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "self_attention.linear_qkv.weight",
        mcore_qkv,
    )
    fused_qkv = torch.cat([parameter for _, parameter in train_params], dim=0)
    infer_params = infer_converter.convert_param(
        "language_model.model.layers.0.self_attn.qkv_proj.weight", fused_qkv
    )

    expected_names = [
        "model.language_model.layers.0.attention.query_key_value_proj.weight",
    ]
    assert [name for name, _ in train_params] == expected_names
    assert [name for name, _ in infer_params] == expected_names
    for (_, train_param), (_, infer_param) in zip(
        train_params, infer_params, strict=False
    ):
        torch.testing.assert_close(train_param, infer_param, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["weight", "bias"])
def test_vision_qkv_tp2_matches_vllm_rank_layout(
    kind, vl_config, tf_config, rank_info
):
    """Vision-tower QKV must be resharded from train-TP to infer-TP layout."""
    converter = _make_train_converter(
        vl_config, rank_info, tf_config, infer_atten_tp_size=2
    )
    tail_shape = (4,) if kind == "weight" else ()
    q = [torch.full((2, *tail_shape), 10 + head) for head in range(2)]
    k = [torch.full((2, *tail_shape), 20 + head) for head in range(2)]
    v = [torch.full((2, *tail_shape), 30 + head) for head in range(2)]
    mcore_qkv = torch.cat(
        [torch.cat((q[head], k[head], v[head]), dim=0) for head in range(2)],
        dim=0,
    )

    [(name, packed_qkv)] = converter.convert_param(
        f"module.module.vision_model.decoder.layers.1.self_attention.linear_qkv.{kind}",
        mcore_qkv,
    )

    assert name == f"model.visual.blocks.1.attn.qkv.{kind}"
    for rank, rank_shard in enumerate(torch.chunk(packed_qkv, 2, dim=0)):
        expected = torch.cat((q[rank], k[rank], v[rank]), dim=0)
        torch.testing.assert_close(rank_shard, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mcore_name", "vllm_name", "canonical_name"),
    [
        (
            "module.module.vision_model.patch_embed.proj.weight",
            "visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.weight",
        ),
        (
            "module.module.vision_model.decoder.layers.1.mlp.linear_fc1.weight",
            "visual.blocks.1.mlp.linear_fc1.weight",
            "model.visual.blocks.1.mlp.linear_fc1.weight",
        ),
        (
            "module.module.vision_model.merger.linear_fc2.weight",
            "visual.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.weight",
        ),
        (
            "module.module.vision_model.decoder.deepstack_merger_list.2."
            "patch_norm.weight",
            "visual.deepstack_merger_list.2.norm.weight",
            "model.visual.deepstack_merger_list.2.norm.weight",
        ),
    ],
)
def test_vision_parameters_have_train_infer_parity(
    mcore_name,
    vllm_name,
    canonical_name,
    vl_config,
    tf_config,
    rank_info,
    infer_engine_config,
):
    train_converter = _make_train_converter(vl_config, rank_info, tf_config)
    infer_converter = _make_infer_converter(vl_config, rank_info, infer_engine_config)
    parameter = torch.arange(8, dtype=torch.float32)

    train_params = train_converter.convert_param(mcore_name, parameter)
    infer_params = infer_converter.convert_param(vllm_name, parameter)

    assert [name for name, _ in train_params] == [canonical_name]
    assert [name for name, _ in infer_params] == [canonical_name]
    torch.testing.assert_close(train_params[0][1], infer_params[0][1], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Tests – MoE expert numbering
# ---------------------------------------------------------------------------


def test_moe_expert_reuses_qwen3_global_ep_numbering(vl_config, tf_config, rank_info):
    rank_info.ep_size = 2
    rank_info.ep_rank = 1
    converter = _make_train_converter(vl_config, rank_info, tf_config)

    converted = converter.convert_param(
        "module.module.language_model.decoder.layers.0.mlp.experts."
        "linear_fc2.weight0",
        torch.ones(8, 6),
    )

    assert converted[0][0] == (
        "model.language_model.layers.0.mlp.experts.2.down_proj.weight"
    )


# ---------------------------------------------------------------------------
# Tests – tied embeddings (no duplicate lm_head)
# ---------------------------------------------------------------------------


def test_tied_embeddings_produces_single_lm_head_on_mcore(
    vl_config, tf_config, rank_info
):
    """With tie_word_embeddings=True, MCore must not duplicate lm_head.weight.

    MindSpeed exposes ``output_layer.weight`` even when tied, so the base
    converter already produces ``lm_head.weight``.  The converter-level tie
    logic is disabled to avoid a duplicate.
    """
    vl_config.text_config.tie_word_embeddings = True
    converter = _make_train_converter(vl_config, rank_info, tf_config)
    embedding = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    # output_layer.weight → lm_head.weight (via base converter)
    output_params = converter.convert_param(
        "module.module.language_model.output_layer.weight", embedding
    )
    assert [name for name, _ in output_params] == ["lm_head.weight"]

    # embedding.word_embeddings.weight → model.language_model.embed_tokens.weight
    # (tie logic is commented out, so no extra lm_head.weight)
    embed_params = converter.convert_param(
        "module.module.language_model.embedding.word_embeddings.weight", embedding
    )
    names = [name for name, _ in embed_params]
    assert "lm_head.weight" not in names
    assert any("embed_tokens.weight" in n for n in names)


def test_tied_embeddings_produces_single_lm_head_on_vllm(
    vl_config, rank_info, infer_engine_config
):
    """With tie_word_embeddings=True, vLLM must not duplicate lm_head.weight.

    vLLM exposes ``lm_head.weight`` directly (even when tied), so the
    converter's direct mapping already produces it.  The converter-level tie
    logic is disabled to avoid a duplicate.
    """
    vl_config.text_config.tie_word_embeddings = True
    converter = _make_infer_converter(vl_config, rank_info, infer_engine_config)
    embedding = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    # vLLM exposes lm_head.weight directly (even when tied).
    lm_head_params = converter.convert_param("lm_head.weight", embedding)
    assert [name for name, _ in lm_head_params] == ["lm_head.weight"]

    # embed_tokens must not produce an extra lm_head.weight.
    embed_params = converter.convert_param(
        "language_model.model.embed_tokens.weight", embedding
    )
    names = [name for name, _ in embed_params]
    assert "lm_head.weight" not in names
    assert any("embed_tokens.weight" in n for n in names)


# ---------------------------------------------------------------------------
# Tests – sharding strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_type", "expected_dim"),
    [
        ("model.visual.patch_embed.proj.weight", ShardingType.NO_SHARDING, 0),
        ("model.visual.blocks.0.attn.qkv.weight", ShardingType.TP_SHARDING, 0),
        ("model.visual.blocks.0.attn.proj.weight", ShardingType.TP_SHARDING, 1),
        (
            "model.visual.deepstack_merger_list.0.linear_fc1.weight",
            ShardingType.TP_SHARDING,
            0,
        ),
        (
            "model.visual.deepstack_merger_list.0.linear_fc2.weight",
            ShardingType.TP_SHARDING,
            1,
        ),
        (
            "model.visual.deepstack_merger_list.0.linear_fc2.bias",
            ShardingType.NO_SHARDING,
            0,
        ),
    ],
)
def test_vision_sharding_matches_vllm_layout(
    name, expected_type, expected_dim, rank_info
):
    rank_info.tp_size = 2
    rank_info.attn_tp_size = 2
    strategy = Qwen3VLShardingStrategy(
        engine_name="vllm",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
        device_backend="cpu",
    )

    sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(name)

    assert sharding_type is expected_type
    assert sharding_dim == expected_dim
    assert num_shards == (2 if expected_type is ShardingType.TP_SHARDING else 1)


def test_vision_pos_embedding_is_replicated_on_mcore_and_vllm(rank_info):
    """pos_embed is replicated on both mcore and vLLM (bridge layout)."""
    rank_info.tp_size = 2
    common = {
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "moe_dense_tp_size": 2,
        "tp_size": 2,
        "ep_size": 1,
        "ep_tp_size": 1,
        "rank_info": rank_info,
        "device_backend": "cpu",
    }

    mcore = Qwen3VLShardingStrategy(engine_name="mcore", **common)
    vllm = Qwen3VLShardingStrategy(engine_name="vllm", **common)
    name = "model.visual.pos_embed.weight"
    assert mcore.get_sharding_strategy(name) == (ShardingType.NO_SHARDING, 0, 1)
    assert vllm.get_sharding_strategy(name) == (ShardingType.NO_SHARDING, 0, 1)


# ---------------------------------------------------------------------------
# Tests – pipeline-parallel layer remapping
# ---------------------------------------------------------------------------


def test_pipeline_mapping_only_rewrites_language_layers(
    vl_config, tf_config, rank_info
):
    """PP layer-id remapping applies to language layers but not vision layers."""
    rank_info.pp_rank = 1
    rank_info.pp_size = 2
    converter = _make_train_converter(vl_config, rank_info, tf_config)
    converter._pp_stage_layer_id_map = {(1, 0): {0: 1}}
    language_weight = torch.ones(8, 8)
    vision_weight = torch.ones(4, 4)

    [(language_name, _)] = converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "self_attention.linear_proj.weight",
        language_weight,
    )
    [(vision_name, _)] = converter.convert_param(
        "module.module.vision_model.decoder.layers.0.self_attention.linear_proj.weight",
        vision_weight,
    )

    # Language layer 0 → remapped to layer 1 (PP stage 1).
    # linear_proj → attention.dense (via _normalize_attn_name).
    assert language_name == (
        "model.language_model.layers.1.attention.dense.weight"
    )
    # Vision layer index is NOT remapped.
    assert vision_name == "model.visual.blocks.0.attn.proj.weight"


# ---------------------------------------------------------------------------
# Tests – error paths
# ---------------------------------------------------------------------------


def test_unknown_vision_parameter_fails_fast(vl_config, tf_config, rank_info):
    converter = _make_train_converter(vl_config, rank_info, tf_config)

    with pytest.raises(ValueError, match="Unknown Qwen3-VL vision parameter"):
        converter.convert_param(
            "module.module.vision_model.unknown.weight", torch.ones(2)
        )


def test_non_language_model_parameter_fails_fast(vl_config, tf_config, rank_info):
    """Parameters without 'language_model.' or 'vision_model.' prefix raise ValueError."""
    converter = _make_train_converter(vl_config, rank_info, tf_config)

    with pytest.raises(ValueError, match="Unknown Qwen3-VL parameter"):
        converter.convert_param(
            "module.module.unknown_module.weight", torch.ones(2)
        )