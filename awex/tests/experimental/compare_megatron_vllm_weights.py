#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _apply_device_backend(device_backend: str) -> str:
    from awex.util import device as device_util

    if device_backend and device_backend != "auto":
        os.environ["AWEX_DEVICE_TYPE"] = device_backend
    return device_util.get_device_type()


def _dist_backend_for(device_type: str) -> str:
    if device_type == "npu":
        return "hccl"
    if device_type == "cuda":
        return "nccl"
    return "gloo"


def _maybe_init_vllm_ascend_runtime(device_type: str, vllm_config) -> None:
    if device_type != "npu":
        return

    try:
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.distributed.parallel_state import (
            init_ascend_model_parallel,
        )
        from vllm_ascend.utils import (
            check_ascend_device_type,
            register_ascend_customop,
        )
    except Exception as exc:
        raise RuntimeError(
            "device_backend=npu requires vllm_ascend runtime initialization"
        ) from exc

    register_ascend_customop(vllm_config)
    init_ascend_config(vllm_config)
    check_ascend_device_type()
    init_ascend_model_parallel(vllm_config.parallel_config)


def _resolve_requested_infer_device_backend(
    args: argparse.Namespace, runtime_device_backend: str
) -> str:
    return args.infer_device_backend or runtime_device_backend


def _resolve_requested_router_dtype(args: argparse.Namespace, hf_config) -> str:
    return args.infer_router_dtype or getattr(hf_config, "router_dtype", "bf16")


def _resolve_requested_expert_bias_dtype(args: argparse.Namespace, hf_config) -> str:
    return args.infer_expert_bias_dtype or _resolve_requested_router_dtype(
        args, hf_config
    )


def _build_requested_infer_conf(
    args: argparse.Namespace,
    hf_config,
    default_infer_atten_tp_size: int,
    runtime_device_backend: str,
) -> dict:
    infer_device_backend = _resolve_requested_infer_device_backend(
        args, runtime_device_backend
    )
    return {
        "engine_name": "vllm",
        "infer_atten_tp_size": (
            args.infer_atten_tp_size
            if args.infer_atten_tp_size is not None
            else default_infer_atten_tp_size
        ),
        "router_dtype": _resolve_requested_router_dtype(args, hf_config),
        "expert_bias_dtype": _resolve_requested_expert_bias_dtype(args, hf_config),
        "num_query_groups": getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
        "device_backend": infer_device_backend,
        "infer_engine_config": {
            "device_backend": infer_device_backend,
            "comm_backend": _dist_backend_for(infer_device_backend),
        },
    }


def _observe_vllm_infer_conf(model, rank_info, runtime_device_backend: str) -> dict:
    router_dtype = None
    expert_bias_dtype = None
    for name, param in model.named_parameters():
        if name.endswith(".gate.weight") or ".gate.weight" in name:
            router_dtype = str(param.dtype).removeprefix("torch.")
        elif name.endswith(".expert_bias") or ".expert_bias" in name:
            expert_bias_dtype = str(param.dtype).removeprefix("torch.")
        if router_dtype is not None and expert_bias_dtype is not None:
            break
    return {
        "engine_name": "vllm",
        "infer_atten_tp_size": rank_info.attn_tp_size,
        "router_dtype": router_dtype,
        "expert_bias_dtype": expert_bias_dtype,
        "device_backend": runtime_device_backend,
    }


def _compare_infer_conf(requested: dict, observed: dict) -> dict[str, dict[str, str]]:
    mismatches = {}
    for key, requested_value in requested.items():
        observed_value = observed.get(key)
        if observed_value is None or str(requested_value) != str(observed_value):
            mismatches[key] = {
                "requested": str(requested_value),
                "observed": str(observed_value),
            }
    return mismatches



def _hash_name(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


_LAYER_ID_RE = re.compile(r"(?:^|\\.)layers\\.(\\d+)\\.")


def _layer_id_from_name(name: str) -> int | None:
    match = _LAYER_ID_RE.search(name)
    if not match:
        return None
    return int(match.group(1))


def _should_include_name(
    name: str, max_layers: int | None, include_non_layer: bool
) -> bool:
    if max_layers is None:
        return True
    layer_id = _layer_id_from_name(name)
    if layer_id is None:
        return include_non_layer
    return layer_id < max_layers


def _save_manifest(manifest_path: Path, entries: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, sort_keys=True)


def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["name"]: entry for entry in entries}


def _maybe_get_tf_config(model):
    for attr in ("transformer_config", "config"):
        cfg = getattr(model, attr, None)
        if cfg is not None:
            return cfg
    return None


def _get_model_arch_name(hf_config) -> str:
    architectures = getattr(hf_config, "architectures", None)
    if architectures:
        return architectures[0]
    raise ValueError("HF config must define architectures[0] for converter lookup.")


def _dump_megatron_hf_weights(args: argparse.Namespace) -> None:
    import torch

    from awex.converter.mcore_converter import get_mcore_model_parameters
    from awex.models.registry import get_train_weights_converter
    from awex.sharding.rank_info import RankInfo
    from awex.tests.test_utils import megatron_model_from_hf
    from awex.util import device as device_util

    out_dir = Path(args.out_dir).resolve()
    weights_dir = out_dir / "megatron_hf_tensors"
    weights_dir.mkdir(parents=True, exist_ok=True)

    models, hf_config = megatron_model_from_hf(
        model_path=args.model_path,
        use_mbridge=not args.no_mbridge,
    )

    infer_conf = _build_requested_infer_conf(
        args,
        hf_config,
        default_infer_atten_tp_size=1,
        runtime_device_backend=device_util.get_device_type(),
    )

    rank_info = RankInfo(
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
        is_infer=True,
    )

    entries: list[dict] = []
    name_to_tensor: dict[str, torch.Tensor] = {}
    duplicate_conflicts: list[str] = []
    model_arch_name = _get_model_arch_name(hf_config)

    with torch.no_grad():
        for model in models:
            tf_config = _maybe_get_tf_config(model)
            converter = get_train_weights_converter(
                "mcore",
                model_arch_name,
                hf_config,
                rank_info,
                infer_conf,
                tf_config=tf_config,
            )
            for name, param in get_mcore_model_parameters(model).items():
                converted = converter.convert_param(name, param.detach())
                for hf_name, hf_tensor in converted:
                    if not _should_include_name(
                        hf_name, args.max_layers, args.include_non_layer
                    ):
                        continue
                    existing = name_to_tensor.get(hf_name)
                    if existing is not None:
                        if existing.shape != hf_tensor.shape or not torch.allclose(
                            existing, hf_tensor
                        ):
                            duplicate_conflicts.append(hf_name)
                        continue
                    name_to_tensor[hf_name] = hf_tensor.detach().cpu()

    for hf_name, tensor in name_to_tensor.items():
        file_name = f"{_hash_name(hf_name)}.pt"
        file_path = weights_dir / file_name
        torch.save(tensor, file_path)
        entries.append(
            {
                "name": hf_name,
                "file": str(file_path),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
        )

    _save_manifest(out_dir / "megatron_hf_manifest.json", entries)

    if duplicate_conflicts:
        conflict_path = out_dir / "megatron_duplicate_conflicts.json"
        with conflict_path.open("w", encoding="utf-8") as f:
            json.dump(sorted(set(duplicate_conflicts)), f, indent=2)

    if device_util.get_device_type() == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_util.get_device_type() == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is not None and hasattr(npu_mod, "empty_cache"):
            npu_mod.empty_cache()
    try:
        from megatron.core import parallel_state as mpu

        if mpu.model_parallel_is_initialized():
            mpu.destroy_model_parallel()
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _compare_with_vllm(args: argparse.Namespace) -> None:
    import tempfile

    import torch
    from transformers import AutoConfig
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    from vllm.model_executor.model_loader import get_model

    from awex.config import InferenceConfig
    from awex.models.registry import get_infer_weights_converter
    from awex.sharding.rank_info import RankInfo
    from awex.util import device as device_util

    out_dir = Path(args.out_dir).resolve()
    manifest_path = out_dir / "megatron_hf_manifest.json"
    manifest = _load_manifest(manifest_path)

    hf_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model_arch_name = _get_model_arch_name(hf_config)
    infer_config = InferenceConfig(tp_size=1, ep_size=1)
    rank_info = RankInfo(
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
        is_infer=True,
    )
    converter = get_infer_weights_converter(
        "vllm",
        model_arch_name,
        hf_config,
        rank_info,
        infer_config,
    )

    if not model_parallel_is_initialized():
        temp_file = tempfile.mkstemp()[1]
        backend = _dist_backend_for(device_util.get_device_type())
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend=backend,
        )
        initialize_model_parallel(1, 1)

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            enforce_eager=True,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        load_config=LoadConfig(
            load_format=args.vllm_load_format,
            download_dir=args.download_dir,
        ),
    )

    _maybe_init_vllm_ascend_runtime(device_util.get_device_type(), vllm_config)
    model = get_model(vllm_config=vllm_config)
    requested_infer_conf = _build_requested_infer_conf(
        args,
        hf_config,
        default_infer_atten_tp_size=1,
        runtime_device_backend=device_util.get_device_type(),
    )
    observed_infer_conf = _observe_vllm_infer_conf(
        model, rank_info, device_util.get_device_type()
    )
    infer_conf_mismatch = _compare_infer_conf(
        {
            "engine_name": requested_infer_conf["engine_name"],
            "infer_atten_tp_size": requested_infer_conf["infer_atten_tp_size"],
            "router_dtype": requested_infer_conf["router_dtype"],
            "expert_bias_dtype": requested_infer_conf["expert_bias_dtype"],
            "device_backend": requested_infer_conf["device_backend"],
        },
        observed_infer_conf,
    )
    if infer_conf_mismatch:
        print(f"Infer config mismatch detected: {infer_conf_mismatch}")
        print(
            "Hint: set --infer-router-dtype / --infer-expert-bias-dtype / "
            "--infer-device-backend / --infer-atten-tp-size to match the real vLLM runtime before "
            "interpreting dtype or shape mismatches."
        )

    vllm_hf_weights: dict[str, torch.Tensor] = {}
    vllm_conflicts: list[str] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            for hf_name, hf_tensor in converter.convert_param(name, param.detach()):
                if not _should_include_name(
                    hf_name, args.max_layers, args.include_non_layer
                ):
                    continue
                existing = vllm_hf_weights.get(hf_name)
                if existing is not None:
                    if existing.shape != hf_tensor.shape or not torch.allclose(
                        existing, hf_tensor
                    ):
                        vllm_conflicts.append(hf_name)
                    continue
                vllm_hf_weights[hf_name] = hf_tensor.detach().cpu()

    missing_in_megatron: list[str] = []
    missing_in_vllm: list[str] = []
    shape_mismatch: list[dict] = []
    dtype_mismatch: list[dict] = []
    value_mismatch: list[dict] = []

    for name, vllm_tensor in vllm_hf_weights.items():
        entry = manifest.get(name)
        if entry is None:
            missing_in_megatron.append(name)
            continue
        megatron_tensor = torch.load(entry["file"], map_location="cpu")
        if list(megatron_tensor.shape) != list(vllm_tensor.shape):
            shape_mismatch.append(
                {
                    "name": name,
                    "megatron_shape": list(megatron_tensor.shape),
                    "vllm_shape": list(vllm_tensor.shape),
                }
            )
            continue
        if str(megatron_tensor.dtype) != str(vllm_tensor.dtype):
            dtype_mismatch.append(
                {
                    "name": name,
                    "megatron_dtype": str(megatron_tensor.dtype),
                    "vllm_dtype": str(vllm_tensor.dtype),
                }
            )
            if args.compare_values_strict:
                vllm_tensor = vllm_tensor.to(megatron_tensor.dtype)
            else:
                continue
        if not torch.allclose(
            megatron_tensor,
            vllm_tensor,
            rtol=args.rtol,
            atol=args.atol,
        ):
            diff = (megatron_tensor - vllm_tensor).abs()
            value_mismatch.append(
                {
                    "name": name,
                    "max_abs_diff": diff.max().item(),
                    "mean_abs_diff": diff.mean().item(),
                }
            )

    for name in manifest.keys():
        if name not in vllm_hf_weights:
            missing_in_vllm.append(name)

    report = {
        "infer_conf": {
            "requested": requested_infer_conf,
            "observed": observed_infer_conf,
            "mismatch": infer_conf_mismatch,
        },
        "missing_in_megatron": missing_in_megatron,
        "missing_in_vllm": missing_in_vllm,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "value_mismatch": value_mismatch,
        "vllm_duplicate_conflicts": sorted(set(vllm_conflicts)),
    }
    report_path = out_dir / "megatron_vllm_compare_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("Comparison complete.")
    print(f"Missing in Megatron: {len(missing_in_megatron)}")
    print(f"Missing in vLLM: {len(missing_in_vllm)}")
    print(f"Shape mismatch: {len(shape_mismatch)}")
    print(f"Dtype mismatch: {len(dtype_mismatch)}")
    print(f"Value mismatch: {len(value_mismatch)}")
    print(f"Report: {report_path}")

    del model
    if device_util.get_device_type() == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_util.get_device_type() == "npu":
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is not None and hasattr(npu_mod, "empty_cache"):
            npu_mod.empty_cache()
    try:
        cleanup_dist_env_and_memory()
    except Exception:
        pass


def _run_subprocess(stage: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--model-path",
        args.model_path,
        "--out-dir",
        args.out_dir,
        "--dtype",
        args.dtype,
        "--vllm-load-format",
        args.vllm_load_format,
        "--device-backend",
        args.device_backend,
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
    ]
    if args.infer_router_dtype:
        cmd.extend(["--infer-router-dtype", args.infer_router_dtype])
    if args.infer_expert_bias_dtype:
        cmd.extend(["--infer-expert-bias-dtype", args.infer_expert_bias_dtype])
    if args.infer_device_backend:
        cmd.extend(["--infer-device-backend", args.infer_device_backend])
    if args.infer_atten_tp_size is not None:
        cmd.extend(["--infer-atten-tp-size", str(args.infer_atten_tp_size)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.no_mbridge:
        cmd.append("--no-mbridge")
    if args.compare_values_strict:
        cmd.append("--compare-values-strict")
    if args.max_layers is not None:
        cmd.extend(["--max-layers", str(args.max_layers)])
    if args.include_non_layer:
        cmd.append("--include-non-layer")
    if args.download_dir:
        cmd.extend(["--download-dir", args.download_dir])

    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Megatron->HF converted weights against vLLM->HF converted weights."
        )
    )
    parser.add_argument("--model-path", required=True, help="HF model path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--dtype", default="auto", help="Weight dtype")
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Only compare layers [0, max_layers).",
    )
    parser.add_argument(
        "--include-non-layer",
        action="store_true",
        help="Include non-layer weights when max-layers is set.",
    )
    parser.add_argument(
        "--vllm-load-format",
        default="auto",
        help="vLLM load format (auto, safetensors, pt, etc)",
    )
    parser.add_argument(
        "--device-backend",
        choices=["auto", "cuda", "npu", "cpu"],
        default="auto",
        help="Device backend to use (auto/cuda/npu/cpu).",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional HF download cache dir",
    )
    parser.add_argument(
        "--infer-router-dtype",
        choices=["bf16", "fp16", "fp32"],
        default=None,
        help="Override infer_conf.router_dtype used by Megatron-side conversion.",
    )
    parser.add_argument(
        "--infer-expert-bias-dtype",
        choices=["bf16", "fp16", "fp32"],
        default=None,
        help="Override infer_conf.expert_bias_dtype used by Megatron-side conversion.",
    )
    parser.add_argument(
        "--infer-device-backend",
        choices=["cuda", "npu", "cpu"],
        default=None,
        help="Override infer_conf.device_backend used by Megatron-side conversion.",
    )
    parser.add_argument(
        "--infer-atten-tp-size",
        type=int,
        default=None,
        help="Override infer_conf.infer_atten_tp_size used by Megatron-side conversion.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading HF models",
    )
    parser.add_argument(
        "--no-mbridge",
        action="store_true",
        help="Use Megatron DCP conversion instead of mbridge loading",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--compare-values-strict",
        action="store_true",
        help="Compare values even if dtype mismatches",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "megatron_dump", "vllm_compare"],
        default="all",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    _apply_device_backend(args.device_backend)

    if args.stage == "megatron_dump":
        _dump_megatron_hf_weights(args)
        return
    if args.stage == "vllm_compare":
        _compare_with_vllm(args)
        return

    _run_subprocess("megatron_dump", args)
    _run_subprocess("vllm_compare", args)


if __name__ == "__main__":
    main()
