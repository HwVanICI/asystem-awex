from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from awex.reader.nccl_reader import NCCLWorkerWeightsReader
from awex.writer.nccl_writer import NCCLWeightsWriter


@pytest.mark.parametrize("cls", [NCCLWeightsWriter, NCCLWorkerWeightsReader])
def test_hccl_group_destroy_releases_reference_and_is_idempotent(monkeypatch, cls):
    """Release HCCL resources, not only the distributed registry entry."""
    worker = object.__new__(cls)
    group = object()
    worker.destroy_pg_after_update = True
    worker.comm_backend = "hccl"
    if cls is NCCLWorkerWeightsReader:
        worker.backend = "hccl"
    worker.already_initialized = True
    worker.weights_update_group = group
    events = []
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: events.append("synchronize"),
            empty_cache=lambda: events.append("empty_cache"),
        ),
        raising=False,
    )
    monkeypatch.setattr(dist, "destroy_process_group", lambda pg: events.append(pg))

    worker._destroy_weights_exchange_process_group()
    worker._destroy_weights_exchange_process_group()

    assert events == ["synchronize", group, "empty_cache"]
    assert worker.weights_update_group is None
    assert worker.already_initialized is False


@pytest.mark.parametrize("cls", [NCCLWeightsWriter, NCCLWorkerWeightsReader])
@pytest.mark.parametrize(
    "enabled, backend, initialized",
    [(False, "hccl", True), (True, "nccl", True), (True, "hccl", False)],
)
def test_group_destroy_preserves_inactive_paths(
    monkeypatch, cls, enabled, backend, initialized
):
    """Do not destroy an inactive group or the default process group."""
    worker = object.__new__(cls)
    worker.destroy_pg_after_update = enabled
    worker.comm_backend = backend
    if cls is NCCLWorkerWeightsReader:
        worker.backend = backend
    worker.already_initialized = initialized
    worker.weights_update_group = object() if initialized else None
    group = worker.weights_update_group
    destroy = Mock()
    monkeypatch.setattr(dist, "destroy_process_group", destroy)

    worker._destroy_weights_exchange_process_group()

    destroy.assert_not_called()
    assert worker.weights_update_group is group
    assert worker.already_initialized is initialized


@pytest.mark.parametrize("cls", [NCCLWeightsWriter, NCCLWorkerWeightsReader])
def test_hccl_group_reinitializes_for_next_update(monkeypatch, cls):
    """Each update can recreate the released communicator with the same topology."""
    from awex.util import device as device_util
    from awex.util import process_group
    from awex.writer import nccl_writer

    worker = object.__new__(cls)
    worker.destroy_pg_after_update = True
    worker.comm_backend = "hccl"
    worker.already_initialized = False
    worker.weights_update_group = None
    worker.master_address = "127.0.0.1"
    worker.master_port = 12345
    worker.transfer_rank = 0
    worker.transfer_world_size = 256
    worker.world_size = 256
    worker.backend = "hccl"
    worker.engine_rank = 0
    worker.barrier_device = 0
    groups = [object(), object()]
    events = []
    group_iter = iter(groups)

    def create_group(**kwargs):
        events.append("create")
        return next(group_iter)

    initialize = Mock(side_effect=create_group)
    destroy = Mock(side_effect=lambda group: events.append("destroy"))
    monkeypatch.setattr(process_group, "init_weights_update_group", initialize)
    monkeypatch.setattr(nccl_writer, "init_weights_update_group", initialize)
    monkeypatch.setattr(dist, "barrier", Mock())
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    monkeypatch.setattr(device_util, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: events.append("synchronize"),
            empty_cache=lambda: events.append("empty_cache"),
        ),
        raising=False,
    )

    for group in groups:
        worker._init_weights_exchange_process_group()
        worker._init_weights_exchange_process_group()
        assert worker.weights_update_group is group
        assert worker.already_initialized is True
        worker._destroy_weights_exchange_process_group()

    assert initialize.call_count == 2
    assert destroy.call_count == 2
    assert worker.weights_update_group is None
    assert worker.already_initialized is False
    assert (
        events
        == [
            "synchronize",
            "empty_cache",
            "create",
            "synchronize",
            "destroy",
            "empty_cache",
        ]
        * 2
    )
