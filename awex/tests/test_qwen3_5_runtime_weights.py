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
