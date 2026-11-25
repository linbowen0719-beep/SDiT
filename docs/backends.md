# Sparse execution

This release exposes one backend, `sdpa`. Active image queries see their local
region neighborhood plus remote representative keys and mean values, with
additive log-count bias in a common softmax. Compatible region jobs are grouped
to reduce dispatch overhead. Masked padding does not contribute to attention.

PyTorch selects the native CUDA SDPA implementation supported by each shape,
dtype and mask. The repository does not include the external FlashAttention
package, varlen/indexed paths, or their build instructions. No custom C++
extension is required. Triton helpers for partial projection and cache updates
are included and use the Triton installation provided with PyTorch.

Runtime records include the effective sparse backend calls. These counts cover
instrumented sparse image attention, rather than every operation in the model.
For comparable timings use the same GPU, software stack, prompt, seed,
resolution, sampling steps and guidance for all methods. Do not combine this
release's runtimes with historical baseline runtimes from another environment.
