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

from awex.transfer import nccl_stream_batch as transport_module


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_colocate_fanout_reuses_snapshot_without_aliasing_source(
    monkeypatch, noncontiguous
):
    """Identical sends share storage, but stay isolated from source mutations."""
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    if noncontiguous:
        source = source.t()
    slices = (slice(None), slice(None))
    partial = (slice(0, 1), slice(None))

    def operation(peer, name="weight", train_slices=slices):
        return SimpleNamespace(
            recv_rank=peer,
            send_shard_meta=SimpleNamespace(name=name),
            train_slices=train_slices,
        )

    send_plan = SimpleNamespace(
        operations={
            1: [operation(1)],
            2: [operation(2), operation(2, train_slices=partial)],
            3: [operation(3, name="other")],
        }
    )
    captured = {}

    def inspect_transfer(rank, size, sends, *args):
        captured.update(sends)
        source.fill_(-1)

    monkeypatch.setattr(
        transport_module.dist,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(tensor=tensor),
    )
    monkeypatch.setattr(transport_module.hang_detector, "submit", lambda *a, **k: None)
    monkeypatch.setattr(transport_module.device_util, "synchronize", lambda: None)
    transport = object.__new__(transport_module.NcclColocateStreamBatchTransport)
    monkeypatch.setattr(
        transport, "execute_recursive_partition_stream_transfer", inspect_transfer
    )
    expected = source.clone()
    mapping = {i: i for i in range(4)}
    transport.update_weights_in_colocate_mode(
        mapping,
        mapping,
        0,
        "test",
        4,
        send_plan,
        SimpleNamespace(operations={}),
        None,
        {"weight": source, "other": source.clone()},
        {},
    )

    first = captured[1][0][1].tensor
    repeated = captured[2][0][1].tensor
    assert first is repeated
    assert first.data_ptr() != source.data_ptr()
    assert first is not captured[2][1][1].tensor
    assert first is not captured[3][0][1].tensor
    assert first.is_contiguous()
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(captured[2][1][1].tensor, expected[:1], rtol=0, atol=0)
