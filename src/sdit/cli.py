"""One entrypoint for generation and synchronized end-to-end measurement."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys

from . import __version__
from .config import InferenceConfig, MODEL_IDS


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_prompts(path, prompt, seed):
    if path:
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    else:
        records = [{"id": "sample", "prompt": prompt, "seed": seed}]
    if not records:
        raise ValueError("Prompt manifest is empty")
    ids = set()
    for r in records:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", r["id"]):
            raise ValueError("Each sample needs a safe id of 1–96 letters/digits/underscores/hyphens")
        if r["id"] in ids:
            raise ValueError(f"Duplicate sample id: {r['id']}")
        ids.add(r["id"])
        if not isinstance(r.get("prompt"), str) or not r["prompt"].strip():
            raise ValueError("Each sample needs a nonempty prompt")
        if type(r.get("seed")) is not int or not 0 <= r["seed"] < 2**63:
            raise ValueError("Each sample needs an explicit integer seed in [0, 2**63)")
        if not isinstance(r.get("negative_prompt", ""), str):
            raise ValueError("negative_prompt must be a string")
    return records


def environment():
    packages = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "numpy", "scikit-image", "scikit-learn", "beautifulsoup4", "ftfy", "Pillow", "scipy", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch
    return {"python": platform.python_version(), "packages": packages,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads(), "release_version": __version__}


def source_hashes():
    root = Path(__file__).resolve().parent.parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ("sdit", "ras") for p in sorted((root / name).rglob("*.py"))}


def parser():
    p = argparse.ArgumentParser(description="SDiT image inference and benchmarking")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print environment and CUDA availability")
    for name in ("generate", "benchmark"):
        q = sub.add_parser(name)
        q.add_argument("--config", type=Path)
        for key, typ in [("model", str), ("method", str), ("steps", int), ("rho", float), ("anchors", int), ("hops", int), ("backend", str), ("height", int), ("width", int), ("guidance-scale", float), ("warmup-steps", int)]:
            q.add_argument("--" + key, type=typ)
        q.add_argument("--analysis", action="store_true", default=None)
        prompts = q.add_mutually_exclusive_group(required=True)
        prompts.add_argument("--prompt")
        prompts.add_argument("--prompts", type=Path, help="JSONL with id, prompt and seed")
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--model-path", help="Local Diffusers snapshot or Hub identifier")
        q.add_argument("--local-files-only", action="store_true")
        q.add_argument("--device", default="cuda:0")
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--threads", type=int, default=4)
        q.add_argument("--dry-run", action="store_true", help="Validate inputs without loading a model")
        if name == "benchmark":
            q.add_argument("--repeats", type=int, default=2)
            q.add_argument("--warmup-runs", type=int, default=1)
    return p


def main():
    p = parser()
    a = p.parse_args()
    if a.command == "doctor":
        print(json.dumps(environment(), indent=2))
        return
    keys = ("model", "method", "steps", "rho", "anchors", "hops", "backend", "height", "width", "guidance_scale", "warmup_steps", "analysis")
    try:
        c = InferenceConfig.load(a.config, **{k: getattr(a, k) for k in keys})
        prompts = read_prompts(a.prompts, a.prompt, a.seed)
        if a.threads < 1:
            raise ValueError("threads must be positive")
        if a.command == "benchmark" and (a.repeats < 1 or a.warmup_runs < 1):
            raise ValueError("Benchmark requires at least one warm-up and one measured repeat")
        if a.command == "benchmark" and c.analysis:
            raise ValueError("Analysis instrumentation changes latency; use generate --analysis separately")
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError("Output directory must be new or empty to protect existing results")
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if a.dry_run:
        print(json.dumps({"config": c.to_dict(), "samples": prompts}, indent=2))
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(a.threads))
    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(a.threads)
    from .engine import Runner
    a.output.mkdir(parents=True, exist_ok=True)
    # Save verbose inherited engine output for debugging, away from the CLI summary.
    with (a.output / "engine.log").open("w") as log, redirect_stdout(log):
        runner = Runner(c, a.model_path, a.device, a.local_files_only)
    meta = {"config": c.to_dict(), "environment": environment(),
            "model": runner.model_id, "source_sha256": source_hashes(),
            "prompts": prompts, "command": a.command,
            "timing_scope": "CUDA-synchronized pipeline call through PIL conversion; excludes loading, request reset and image writing"}
    write_json(a.output / "run.json", meta)
    records = []
    repeats = a.repeats if a.command == "benchmark" else 1
    warmups = a.warmup_runs if a.command == "benchmark" else 0
    for sample in prompts:
        for repeat in range(-warmups, repeats):
            with (a.output / "engine.log").open("a") as log, redirect_stdout(log):
                image, result = runner.generate(sample["prompt"], sample["seed"], sample.get("negative_prompt", ""))
            if repeat < 0:
                print(f"{sample['id']}: excluded warm-up complete", flush=True)
                continue
            filename = sample["id"] + (f"-run{repeat + 1}" if a.command == "benchmark" else "") + ".png"
            image.save(a.output / filename)
            result.update(id=sample["id"], seed=sample["seed"], repeat=repeat + 1,
                          image=filename, pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            records.append(result)
            write_json(a.output / "results.json", records)
            print(f"{filename}: {result['wall_seconds']:.3f} s", flush=True)
    summary = {"samples": len(prompts), "measured_runs": len(records),
               "excluded_warmup_runs_per_sample": warmups,
               "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in records),
               "min_wall_seconds": min(r["wall_seconds"] for r in records),
               "max_wall_seconds": max(r["wall_seconds"] for r in records),
               "repeat_outputs_identical": all(len({r["pixel_sha256"] for r in records if r["id"] == s["id"]}) == 1 for s in prompts)}
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
