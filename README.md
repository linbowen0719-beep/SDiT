# SDiT: Region-Adaptive Sparse Inference for Diffusion Transformers

SDiT uses a shared content-aware partition to coordinate selective denoising,
image KV reuse, local attention and remote-region summaries. This repository
provides the image-generation implementation for **Lumina-Next** and
**HunyuanDiT**, with reproducible configuration files and a command-line runner.

## Installation

Use Linux, Python 3.10 or 3.11, and an NVIDIA CUDA GPU. Validation uses an
RTX 6000 Ada with 48 GB VRAM. The default path does not need a custom CUDA
extension. Use a fresh environment to avoid conflicts with another `ras` install.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[dev]'
sdit doctor
```

Diffusers 0.35.1 and Transformers 4.56.0 are pinned because this implementation
patches model internals. Download the model through its original provider and
follow its model license. The runner accepts a Hub identifier or a local
Diffusers-format snapshot; model weights are not included here.

## Generate an image

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 sdit generate \
  --config configs/lumina_sdit_r20.json \
  --prompt 'Peacock in a sunlit garden' --seed 42 \
  --output outputs/peacock
```

For a local model, append `--model-path /path/to/model --local-files-only`.
HunyuanDiT is available through `configs/hunyuan_sdit_r20.json`, but its
current presets are experimental: the validation prompt shows visible texture
artifacts at low active ratios. See [validation](docs/validation.md).
The defaults are 30 steps,
20% dense warm-up, one-hop attention, one remote anchor per region and
active-region ratio 0.2. Lumina uses BF16 and guidance 4; Hunyuan uses FP16 and
guidance 5. Batch size is one, including the model's classifier-free guidance.

To generate the same inputs with dense inference:

```bash
sdit generate --config configs/lumina_sdit_r20.json --method dense \
  --prompts examples/prompts.jsonl --output outputs/dense
sdit generate --config configs/lumina_sdit_r20.json \
  --prompts examples/prompts.jsonl --output outputs/sdit
```

Every JSONL record specifies `id`, `prompt`, and `seed`; `negative_prompt` is
optional. Outputs include PNG images, resolved settings, package versions,
source hashes, timing records and engine logs. Existing nonempty output
folders are rejected. `--dry-run` validates inputs without loading weights.

## Benchmark

```bash
sdit benchmark --config configs/lumina_sdit_r20.json \
  --prompt 'Peacock in a sunlit garden' --seed 42 \
  --warmup-runs 1 --repeats 2 --output outputs/benchmark
```

The benchmark excludes one full warm-up generation and averages two measured
GPU runs. The timed pipeline call includes text encoding, denoising, VAE
and PIL conversion, with CUDA synchronization at both boundaries. Loading,
per-request state reset and file writing are outside the interval. Each repeat
starts from the same seed and empty inference caches. Run competing methods on
the same idle GPU with matching prompts, steps, resolution and guidance.

`--method temporal` retains full image KV context while skipping inactive
queries. `--method ras` enables the Lumina token-selection baseline;
`--method teacache` uses the included TeaCache adaptation. These are explicit
configurations, not promises to reproduce every historical baseline row.

For diagnostic density and module timing, use `sdit generate --analysis`.
Instrumentation is disabled by default and rejected by the clean benchmark.
The release uses grouped PyTorch SDPA with Triton cache/projection helpers;
see [execution details](docs/backends.md).

## Quality evaluation

```bash
sdit-evaluate --generated outputs/sdit --dense outputs/dense \
  --output outputs/quality.json
```

This checks matching prompt/seed/model/step settings, then computes paired
PSNR and SSIM without downloading a metric model. Install `.[metrics]` to add
`--lpips` and `--fid`. With `--fid --real /path/to/real-images`, the output also
includes FID against the explicitly supplied real-image set. Full metric
protocols and examples are in [evaluation](docs/evaluation.md).

## Implementation and paper correspondence

The default policy follows Sections 4.1–4.3 of the
[manuscript](https://github.com/linbowen0719-beep/ICLR-2027-SDiT):

- Quickshift partitions the predicted clean latent at the warm-up boundary.
- Priority combines equally weighted absolute Laplacian and residual magnitude,
  multiplied by `exp(0.1 * tau)` before region averaging.
- Inactive tokens retain their latest native predictions and layer KV states.
- Local regions retain token-level KV; remote regions use centroid-nearest keys,
  mean values and a log-count bias in the same softmax.
- Region-contiguous layouts and grouped SDPA execute the sparse workload.

[Implementation details](docs/implementation.md) specify boundary handling,
subgroup construction and scheduling order. The release aligns these rules with
the current manuscript; it is not a recovered snapshot of every historical
experiment. The paper's reported timings and aggregate metrics have not been
remeasured with this release. [Validation](docs/validation.md) lists the checks
actually completed.

## Repository layout

- `src/sdit/`: validated public configuration, runner, CLI and evaluation.
- `src/ras/`: scheduling, attention, cache and model adapters.
- `configs/`: explicit image-model configurations.
- `tests/`: cache-refresh, graph-layout, scheduling, diagnostics and CLI checks.
- `docs/`: implementation, evaluation, backend and validation records.

```bash
python -m pytest -q
python -m build
```

The inherited manager holds process-global state. Use one worker process per
GPU and do not call two runners concurrently in the same process. Video,
training, CPU inference and arbitrary Diffusers versions are not supported by
this release interface.

## Attribution and license

The implementation builds on [Microsoft RAS](https://github.com/microsoft/RAS)
and [Hugging Face Diffusers](https://github.com/huggingface/diffusers).
See [NOTICE](NOTICE), [LICENSE](LICENSE), and the retained third-party licenses.
Weights and datasets remain subject to their original licenses.
