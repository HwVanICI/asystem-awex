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

import weakref
from contextlib import nullcontext
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


@pytest.mark.parametrize("npu_batch", [False, True])
def test_colocate_noncontiguous_receives_copy_back_with_bounded_buffers(
    monkeypatch, npu_batch
):
    """Wait before copying back, and release staging tensors between rows."""

    class NpuTensor(torch.Tensor):
        @property
        def device(self):
            # Keep CPU storage while selecting the NPU batching branch.
            return SimpleNamespace(type="npu")

    targets = [torch.zeros(3, 4).t() for _ in range(6)]
    if npu_batch:
        targets = [target.as_subclass(NpuTensor) for target in targets]
    live_buffers = []
    completed = []
    batches = []
    peak = 0

    def receive(tensor, peer, group):
        nonlocal peak
        assert tensor.is_contiguous()
        live_buffers.append(weakref.ref(tensor))
        peak = max(peak, sum(ref() is not None for ref in live_buffers))

        def wait():
            tensor.fill_(peer + 1)
            completed.append(peer)

        return SimpleNamespace(tensor=tensor, wait=wait)

    def batch(ops):
        batches.append([op.peer for op in ops])
        return [op.op(op.tensor, op.peer, group=op.group) for op in ops]

    monkeypatch.setenv("AWEX_USE_BATCH_SEND_RECV", "1")
    monkeypatch.setattr(transport_module.dist, "irecv", receive)
    monkeypatch.setattr(transport_module.dist, "batch_isend_irecv", batch)
    monkeypatch.setattr(
        transport_module.dist,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    )
    monkeypatch.setattr(transport_module.device_util, "stream", lambda _: nullcontext())
    monkeypatch.setattr(transport_module.device_util, "synchronize", lambda: None)
    transport = object.__new__(transport_module.NcclColocateStreamBatchTransport)
    transport._stream_pool = [None, None]
    ops = {
        peer: [
            (None, SimpleNamespace(op=receive, tensor=t, peer=peer, group=None))
            for t in targets[peer * 3 : (peer + 1) * 3]
        ]
        for peer in range(2)
    }

    assert transport._execute_ops_concurrent(ops, range(2)) == 6

    assert sorted(completed) == [0, 0, 0, 1, 1, 1]
    assert batches == ([[0, 1]] * 3 if npu_batch else [])
    assert peak == 2
    assert all(ref() is None for ref in live_buffers)
    for peer in range(2):
        for target in targets[peer * 3 : (peer + 1) * 3]:
            target = target.as_subclass(torch.Tensor)
            torch.testing.assert_close(
                target, torch.full_like(target, peer + 1), rtol=0, atol=0
            )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("direction", ["send", "recv"])
def test_colocate_npu_respects_batch_send_recv_flag(monkeypatch, enabled, direction):
    """Send/receive rows keep uneven peer counts and wait for completion."""
    monkeypatch.setenv("AWEX_USE_BATCH_SEND_RECV", "1" if enabled else "0")
    direct_calls, batches, waited = [], [], []
    expected_group = object()

    def transfer(tensor, peer, group):
        assert group is expected_group
        direct_calls.append(peer)
        return SimpleNamespace(wait=lambda: waited.append([peer]))

    def batch(ops):
        assert all(op.op is transfer and op.group is expected_group for op in ops)
        batches.append([op.peer for op in ops])
        return [
            SimpleNamespace(
                tensors=[op.tensor for op in ops],
                wait=lambda: waited.append([op.peer for op in ops]),
            )
        ]

    def operation(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    monkeypatch.setattr(
        transport_module.dist, "isend" if direction == "send" else "irecv", transfer
    )
    monkeypatch.setattr(transport_module.dist, "batch_isend_irecv", batch)
    monkeypatch.setattr(transport_module.dist, "P2POp", operation)
    monkeypatch.setattr(transport_module.device_util, "stream", lambda _: nullcontext())
    monkeypatch.setattr(transport_module.device_util, "synchronize", lambda: None)
    transport = object.__new__(transport_module.NcclColocateStreamBatchTransport)
    transport._stream_pool = [None, None]
    tensor = SimpleNamespace(
        device=SimpleNamespace(type="npu"), is_contiguous=lambda: True
    )
    ops = {
        peer: [
            (None, operation(transfer, tensor, peer, expected_group))
            for _ in range(count)
        ]
        for peer, count in ((1, 2), (2, 1))
    }
    ops[0] = []

    assert transport._execute_ops_concurrent(ops, range(3)) == 3
    assert batches == ([[1, 2], [1]] if enabled else [])
    assert direct_calls == ([] if enabled else [1, 2, 1])
    assert sorted(waited) == sorted(batches if enabled else [[1], [2], [1]])


def test_colocate_empty_peer_operations_returns_zero():
    transport = object.__new__(transport_module.NcclColocateStreamBatchTransport)
    assert transport._execute_ops_concurrent({0: []}, [0]) == 0
