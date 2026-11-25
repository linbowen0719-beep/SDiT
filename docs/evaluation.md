# Evaluation protocol

## Paired images

Prepare a JSONL manifest with one selected caption and explicit seed per image:

```json
{"id":"000000000001","prompt":"A cat by a window.","seed":42}
```

Run `sdit generate` twice using the same manifest, model checkpoint, step count,
resolution, guidance and negative prompts, once with `--method dense` and once
with the chosen accelerated method. Keep output folders separate. The evaluator
rejects mismatched inputs and incomplete records. A small example manifest is
provided for smoke tests; it is not the paper's 5,000-image evaluation set.
Dataset downloading, licensing and caption/image selection belong to the
caller's dataset preparation step.

## Metric definitions

- **FID_d** compares the generated image distribution with the paired dense
  generation distribution.
- **FID_r** compares generated images with an explicitly supplied real-image
  distribution, such as the selected COCO images.
- **PSNR, SSIM and LPIPS** compare each accelerated output to the dense output
  for the same prompt and seed.

The evaluator's SSIM uses scikit-image's channel-wise implementation, data range
1 and default window. LPIPS uses the AlexNet variant. FID uses torch-fidelity
0.3.0's Inception features. These named implementations make new evaluations
reproducible; differences in historical feature extractors, resizing or sample
sets can change numbers. Do not mix the two FID reference sets.

```bash
python -m pip install -e '.[metrics]'
sdit evaluate --generated outputs/sdit --dense outputs/dense \
  --lpips --fid --real /path/to/selected-real-images \
  --output outputs/quality-full.json
```

Metric models may download their pretrained weights on first use. Use `--cpu`
if a GPU is unavailable for metric computation. Real-image FID uses the entire
explicitly supplied directory: prepare the exact intended reference set there.
PSNR is infinite for identical pairs; JSON uses `null` with `psnr_infinite=true`
rather than a nonstandard Infinity literal.

## Timing

The benchmark defaults to one excluded full-generation warm-up and two measured
repeats per prompt. Both repeats reset all inference caches and reuse the same
seed. `summary.json` reports mean/min/max and whether repeat output hashes agree.
The timer includes the pipeline through PIL conversion. It excludes model
loading, request configuration/reset and disk I/O. Use the same timing scope
for every method and report the GPU, dtype, software versions and configuration.
Two repeats are a quick measurement, not a latency confidence interval.

The benchmark stores peak allocated CUDA bytes, including resident model
weights, rather than attributing memory to individual caches. Analysis runs
with synchronized internal events have a different timing perturbation and
are kept outside the clean benchmark command.
