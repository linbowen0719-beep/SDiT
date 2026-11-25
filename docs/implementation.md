# Method and implementation correspondence

The public configurations implement the reference policy in Sections 4.1–4.3
and Appendix D of the current manuscript. The table maps that policy to code.

| Paper component | Implementation |
|---|---|
| Predicted clean latent | `ClusterAdaptiveScheduler.step` and `ClusterAdaptiveDDPMScheduler._predicted_clean_sample` |
| Quickshift partition | `ClusterBasedRASScheduler.spatial_aware_clustering` |
| Eq. 3 detail and residual | `sdit.policy.detail_residual_score` |
| Region priority and starvation | `ClusterAdaptiveScheduler.cluster_adaptive_selection` |
| Geometric remote representatives | `sdit.policy.spatial_anchor_groups` |
| Local KV, mean remote V, shared softmax | `RASLuminaAttnProcessor2_0._cluster_concat_attention` |
| Hunyuan adapter | `RASHunyuanAttnProcessor2_0` |
| Request/cache reset | `sdit.engine.Runner.reset` |

## Partition and priority

Warm-up lasts `ceil(0.2 * steps)` iterations: 6/3/2 for 30/15/10 steps.
Quickshift consumes predicted-clean-latent features averaged within each 2×2
latent patch and min–max normalized per channel. It uses kernel size 2.3,
maximum distance 6, ratio 0.5 and `convert2lab=False`, without PCA. The partition
and adjacency graph remain fixed for the rest of the request.

The clean estimate is `x - sigma * native_output` for Lumina. Hunyuan converts
its native epsilon/sample/velocity prediction using the DDPM alpha schedule.
Both use this clean estimate for partitioning and scoring, rather than the
next noisy sample.

The per-location score is the channel mean of
`0.5 * abs(Laplacian(clean)) + 0.5 * abs(current - clean)`.
The Laplacian uses a four-neighbor stencil with replicated image boundaries.
Scores are averaged over each 2×2 patch, multiplied by
`exp(starvation_scale * consecutive_skips)`, then averaged over its region.
Whole-region updates keep the patch counters within a region equal, so this
matches the region counter in Eq. 3. The default starvation scale is 0.1.

The scheduler ranks `max(1, floor(rho * number_of_regions))` regions. Token
fractions depend on region sizes. Selection uses the latest available clean
estimate: the scheduler prepares the next iteration's active set after the
current denoising evaluation. Warm-up's final evaluation prepares the first
sparse iteration. No additional dense denoiser pass is used for scoring.

## Caches and anchors

The assembled native prediction preserves inactive entries exactly; Lumina
sigma-ratio extrapolation is disabled. Active KV entries refresh at each image
attention layer, while inactive entries retain their most recent KV states.
Text conditioning and positional encodings follow the model adapters.

At `m=1`, the token nearest the spatial centroid supplies the representative
key, and all values in that remote region are averaged. At `m>1`, deterministic
spatial subgroups use farthest-point initialization and up to eight centroid/
medoid refinement rounds. Every token belongs to exactly one nonempty subgroup;
each subgroup supplies a representative key and a mean value. Regions with
fewer than `m` tokens use one group per token. Group metadata is fixed along
with the partition.

Local token logits and remote-anchor logits share one softmax. Each anchor
receives additive bias `log(group_size)` (`gamma=1`). Updated groups recompute
and overwrite their value means; unchanged groups reuse their mean cache.
Partial-group helpers retain a tested fallback, although public configurations
refresh complete regions.

## Execution and comparisons

Tokens are placed in region-contiguous spans. Compatible sparse jobs use
PyTorch SDPA with explicit masks for padding and anchor biases. Triton helpers
handle cache writes and partial projections. See `backends.md`.

`dense` uses the native pipeline. `temporal` uses the same region priority and
prediction/KV reuse, with full image KV context for each active query. `ras`
is the included Lumina token-selection implementation, using standard-deviation
priority. `teacache` selects the included adaptation with an explicit threshold.

## Release provenance

The retained engine originates from the 2026-11-16 research snapshot. The public
release changes partition/scoring input, score terms, prediction extrapolation
and anchor geometry to follow the current manuscript. It also removes the
experimental external attention backends and unsupported model adapters.
`source_manifest.json` records original and release hashes for retained files.

These policy corrections change outputs relative to the earlier development
snapshot. Historical paper timings and dataset metrics are not relabeled as
measurements of this release. The validation record covers numerical tests and
repeatable inference; full dataset reproduction remains separate work.
