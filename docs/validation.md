# Release validation — 2026-11-24

The current release was checked on NVIDIA RTX 6000 Ada GPUs with Python
3.10.20, PyTorch 2.6.0+cu124, Diffusers 0.35.1 and Transformers 4.56.0.
Resolved configurations, environment versions, output hashes and sparse backend
reports are recorded in `validation.json`.

## Numerical and functional checks

The full suite passed 96 tests with CUDA available. With CUDA hidden,
79 passed and 17 GPU tests were skipped.

Tests cover the detail/residual formula, starvation-based region selection,
geometric representatives, disjoint subgroup coverage, unscaled prediction
reuse and Hunyuan clean-latent conversion. A direct numerical comparison checks
grouped sparse attention against the explicit local-plus-remote shared-softmax
formula, including mean values and log-count bias. Existing tests check full
and partial anchor refresh, unchanged caches, region layouts, host scheduling,
request resets and opt-in density analysis.

Both Lumina-Next and HunyuanDiT completed 30-step 1024×1024 generation with
`rho=0.2`, six dense warm-up steps, one-hop local context and one remote anchor.
Each model used one excluded warm-up generation and two measured repeats of
“Peacock in a sunlit garden”, seed 42. Both repeat pairs were pixel-identical.
The sparse backend reports contain grouped SDPA calls. These runs were
functional checks on separate GPUs, so their runtimes are not benchmark claims.

## Quality and scope

The Hunyuan `rho=0.2` peacock sample still has visible texture artifacts relative
to dense inference. Hunyuan presets are therefore marked experimental.
Increasing the active ratio is an available control, not a verified guarantee
of a particular quality improvement. This release's policy corrections change
outputs relative to the earlier development snapshot.

These checks establish the tested numerical rules and executable inference
paths. They do not reproduce the paper's 5,000-image aggregate metrics or
historical timings. Optional LPIPS/FID measurement requires metric weights and
the evaluation image sets; a full dataset run was not part of release checks.
