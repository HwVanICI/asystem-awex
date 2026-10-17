from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from awex.reader.nccl_reader import NCCLWorkerWeightsReader
from awex.util import process_group
from awex.writer import nccl_writer


@pytest.mark.parametrize("rank,world_size", [(0, 1), (0, 4), (3, 4), (1, 3)])
@pytest.mark.parametrize("use_batch", [False, True])
def test_p2p_warmup_respects_batch_selection(monkeypatch, rank, world_size, use_batch):
    """Verify payloads and completion for single, paired and ring warmups."""
    group = object()
    events = []

    def isend(tensor, peer, group, tag=0):
        assert tensor.tolist() == [rank] * 4
        events.append(("send", peer))
        return SimpleNamespace(wait=lambda: events.append("wait"))

    def irecv(tensor, peer, group, tag=0):
        tensor.fill_(peer)
        events.append(("recv", peer))
        return SimpleNamespace(wait=lambda: events.append("wait"))

    def batch(ops):
        assert use_batch, "Non-batch warmup must never call batch_isend_irecv"
        return [op.op(op.tensor, op.peer, op.group) for op in ops]

    monkeypatch.setattr(process_group.device_util, "current_device", lambda: 0)
    monkeypatch.setattr(process_group.device_util, "get_torch_device", lambda _: "cpu")
    monkeypatch.setattr(process_group.device_util, "synchronize", Mock())
    monkeypatch.setattr(process_group.dist, "barrier", Mock())
    monkeypatch.setattr(process_group.dist, "isend", isend)
    monkeypatch.setattr(process_group.dist, "irecv", irecv)
    monkeypatch.setattr(
        process_group.dist,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group, tag=0
        ),
    )
    batch_mock = Mock(side_effect=batch)
    monkeypatch.setattr(process_group.dist, "batch_isend_irecv", batch_mock)

    process_group.setup_batch_isend_irecv(
        group, rank, world_size, tensor_size=4, use_batch=use_batch
    )

    expected = {
        (0, 1): [],
        (0, 4): [("recv", 2), "wait"],
        (3, 4): [("send", 1), "wait"],
        (1, 3): [("recv", 0), ("send", 2), "wait", "wait"],
    }
    assert events == expected[(rank, world_size)]
    assert batch_mock.call_count == int(use_batch and world_size > 1)


@pytest.mark.parametrize(
    "cls", [nccl_writer.NCCLWeightsWriter, NCCLWorkerWeightsReader]
)
def test_nonbatch_handshake_forwards_transport_selection(monkeypatch, cls):
    """Reader and writer must both honor AWEX_USE_BATCH_SEND_RECV=0 at setup."""
    worker = object.__new__(cls)
    worker.weights_update_group = object()
    worker.transfer_rank = 1
    worker.transfer_world_size = 8
    worker.world_size = 8
    worker.enable_colocate_mode = False
    worker.use_batch_send_recv = False
    warmup = Mock()
    monkeypatch.setattr(process_group, "setup_batch_isend_irecv", warmup)
    monkeypatch.setattr(nccl_writer, "setup_batch_isend_irecv", warmup)

    if cls is NCCLWorkerWeightsReader:
        worker._shake_hands_with_writer()
    else:
        worker._shake_hands_with_reader()

    warmup.assert_called_once_with(worker.weights_update_group, 1, 8, use_batch=False)
