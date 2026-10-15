"""Cross-DP identities must survive vLLM's dense EngineCore config reset."""

from types import SimpleNamespace

import pytest

from awex import vllm_plugin


def _capture_context(**kwargs):
    return kwargs["model_context"]


@pytest.mark.parametrize(
    ("moe", "dp_size", "dp_index", "engine_rank"),
    [(False, 8, 2, 1), (False, 1, 0, 3), (True, 8, 2, 1)],
    ids=["dense-internal-dp", "dense-external-dp", "moe-internal-dp"],
)
def test_dp_identity_survives_configless_reader_rpc(
    monkeypatch, moe, dp_size, dp_index, engine_rank
):
    """Metadata and later reader RPCs must share the same instance-local ranks."""
    from vllm.distributed import parallel_state
    from vllm.v1.worker import worker_base

    worker_dp_size = dp_size if moe else 1
    ep_size = dp_size * 4 if moe else 1
    ep_rank = dp_index * 4 + 3 if moe else 0
    for name, rank, size in (
        ("get_tp_group", 3, 4),
        ("get_pp_group", 0, 1),
        ("get_ep_group", ep_rank, ep_size),
    ):
        monkeypatch.setattr(
            parallel_state,
            name,
            lambda rank=rank, size=size: SimpleNamespace(
                rank_in_group=rank, world_size=size
            ),
        )
    worker_class = type("TestWorker", (), {})
    monkeypatch.setattr(worker_base, "WorkerBase", worker_class)
    vllm_plugin._patch_awex_worker()
    worker = worker_class()
    worker.local_rank = 3
    worker.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                rank=3,
                world_size=4,
                world_size_across_dp=worker_dp_size * 4,
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                data_parallel_size=worker_dp_size,
                data_parallel_rank=dp_index if moe else 0,
                data_parallel_index=dp_index,
                enable_expert_parallel=moe,
            )
        ),
        model=None,
    )

    for kwargs in (
        {
            "infer_engine_config": {
                "dp_size": None if moe else dp_size,
                "engine_rank": engine_rank,
                "num_engines": 4,
            }
        },
        {},
    ):
        context = worker.awex_execute(__name__, "_capture_context", kwargs)
        assert context["dp_rank"] == dp_index
        assert context["dp_size"] == dp_size
        assert context["global_rank"] == dp_index * 4 + 3
        assert context["world_size"] == dp_size * 4
        assert context["engine_rank"] == engine_rank
        assert (context["ep_rank"], context["ep_size"]) == (ep_rank, ep_size)
