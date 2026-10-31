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

"""Refresh Ascend kernel copies after exchanging canonical Qwen3.5 weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from awex.config import InferenceConfig
from awex.models.ascend.qwen3_5 import VLLMToHFWeightConverterQwen3VL


@pytest.fixture
def converter():
    text_config = PretrainedConfig(num_attention_heads=8, num_key_value_heads=8)
    return VLLMToHFWeightConverterQwen3VL(
        SimpleNamespace(text_config=text_config),
        InferenceConfig(),
        SimpleNamespace(tp_rank=0, ep_rank=0),
    )


def make_model(cache_kind=None):
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.linear_attn = nn.Module()
    conv = model.layer.linear_attn.conv1d = nn.Conv1d(6, 6, 4, groups=6, bias=False)
    model.layer.mlp = nn.Module()
    gate = model.layer.mlp.gate = nn.Linear(6, 4, bias=False, dtype=torch.bfloat16)
    if cache_kind:
        for module, name, shape in (
            (conv, "ascend_conv1d_weight", (4, 6)),
            (gate, "weight_fp32", (4, 6)),
        ):
            tensor = torch.zeros(shape)
            if cache_kind == "parameter":
                module.register_parameter(
                    name, nn.Parameter(tensor, requires_grad=False)
                )
            elif cache_kind == "buffer":
                module.register_buffer(name, tensor)
            else:
                setattr(module, name, tensor)
    return model, conv, gate


@pytest.mark.parametrize("cache_kind", ["parameter", "buffer", "attribute"])
def test_refresh_preserves_storage_across_updates(converter, cache_kind):
    model, conv, gate = make_model(cache_kind)
    destinations = (conv.ascend_conv1d_weight, gate.weight_fp32)
    pointers = [tensor.data_ptr() for tensor in destinations]
    parameters = dict(converter.iter_model_parameters(model))
    assert set(parameters) == {
        "layer.linear_attn.conv1d.weight",
        "layer.mlp.gate.weight",
    }
    assert parameters["layer.linear_attn.conv1d.weight"] is conv.weight
    assert parameters["layer.mlp.gate.weight"] is gate.weight
    for offset in (2.0, 5.0):
        with torch.no_grad():
            conv.weight.copy_(torch.arange(24).reshape(6, 1, 4) + offset)
            gate.weight.copy_(torch.arange(24).reshape(4, 6) + offset)
        converter.post_update_weights()
        assert conv.ascend_conv1d_weight is destinations[0]
        assert gate.weight_fp32 is destinations[1]
        assert [tensor.data_ptr() for tensor in destinations] == pointers
        torch.testing.assert_close(
            conv.ascend_conv1d_weight, conv.weight.squeeze(1).t()
        )
        torch.testing.assert_close(gate.weight_fp32, gate.weight.float())


def test_legacy_layout_and_reenumeration_clear_cached_sources(converter):
    model, conv, _ = make_model("parameter")
    list(converter.iter_model_parameters(model))
    legacy_model, _, _ = make_model()
    assert dict(converter.iter_model_parameters(legacy_model)) == dict(
        legacy_model.named_parameters()
    )
    with torch.no_grad():
        conv.weight.fill_(7)
    converter.post_update_weights()
    assert torch.count_nonzero(conv.ascend_conv1d_weight) == 0


def test_post_update_before_enumeration_is_noop(converter):
    converter.post_update_weights()


@pytest.mark.parametrize("layout", ["source_rank", "source_channels", "packed_shape"])
def test_invalid_conv_layout_is_rejected(converter, layout):
    model, conv, _ = make_model("parameter")
    if layout == "source_rank":
        conv.weight = nn.Parameter(torch.zeros(6, 4))
    elif layout == "source_channels":
        conv.weight = nn.Parameter(torch.zeros(6, 2, 4))
    else:
        conv.ascend_conv1d_weight = nn.Parameter(torch.zeros(6, 4))
    with pytest.raises(ValueError, match="Unsupported Ascend GDN convolution layout"):
        list(converter.iter_model_parameters(model))


@pytest.mark.parametrize("layout", ["shape", "dtype"])
def test_invalid_router_layout_is_rejected(converter, layout):
    model, _, gate = make_model("parameter")
    gate.weight_fp32 = nn.Parameter(
        torch.zeros(6, 4)
        if layout == "shape"
        else torch.zeros(4, 6, dtype=torch.bfloat16)
    )
    with pytest.raises(ValueError, match="Unsupported Ascend router layout"):
        list(converter.iter_model_parameters(model))


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("stage", ["single", "first", "last"])
def test_tied_head_metadata_and_reader_agree(converter, monkeypatch, tied, stage):
    from awex.meta import infer_meta_resolver
    from awex.reader.weights_reader import WorkerWeightsReader

    model = nn.Module()
    model.language_model = nn.Module()
    language_model = model.language_model
    language_model.model = nn.Module()
    # Tied vLLM models retain embeddings on the last pipeline stage too.
    has_embedding = stage != "last" or tied
    has_head = stage != "first"
    if has_embedding:
        language_model.model.embed_tokens = nn.Embedding(4, 2)
    if has_head:
        language_model.lm_head = nn.Linear(2, 4, bias=False)
        if tied:
            language_model.lm_head.weight = language_model.model.embed_tokens.weight
    model.config = SimpleNamespace(tie_word_embeddings=tied)
    rank_info = SimpleNamespace(
        pp_rank=int(stage == "last"),
        pp_size=1 if stage == "single" else 2,
        global_rank=0,
    )
    monkeypatch.setattr(
        infer_meta_resolver,
        "get_rank_info_extractor",
        lambda _: lambda *args: rank_info,
    )
    monkeypatch.setattr(
        infer_meta_resolver,
        "get_infer_weights_converter",
        lambda *args, **kwargs: converter,
    )
    meta = infer_meta_resolver.InferParamMetaResolver._get_model_param_info(
        "vllm",
        InferenceConfig(),
        convert_params=True,
        model=model,
        model_context={"pp_rank": rank_info.pp_rank, "pp_size": rank_info.pp_size},
    )
    reader = SimpleNamespace(
        model=model,
        weight_converter=converter,
        hf_config=model.config,
        rank_info=rank_info,
    )
    WorkerWeightsReader.initialize(reader)
    expected = set()
    if has_embedding:
        expected.add("model.language_model.embed_tokens.weight")
    if has_head:
        expected.add("lm_head.weight")
    names = [param["name"] for param in meta["params_meta"]]
    assert set(names) == expected
    assert len(names) == len(expected)
    assert set(reader.parameters) == expected
    if has_head:
        head = reader.parameters["lm_head.weight"]
        assert head is language_model.lm_head.weight
        if has_embedding:
            embedding = reader.parameters["model.language_model.embed_tokens.weight"]
            assert (head is embedding) == tied
        pointer = head.data_ptr()
        with torch.no_grad():
            for value in (2.0, 5.0):
                head.copy_(torch.full_like(head, value))
                converter.post_update_weights()
                assert head.data_ptr() == pointer
                torch.testing.assert_close(
                    language_model.lm_head.weight,
                    torch.full_like(head, value),
                    rtol=0,
                    atol=0,
                )
