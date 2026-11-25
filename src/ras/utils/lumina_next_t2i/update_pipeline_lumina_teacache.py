from ...modules.lumina_next_t2i.teacache_forward import (
    LUMINA_NEXT_TEACACHE_COEFFICIENTS,
    get_teacache_stats,
    reset_teacache_state,
    teacache_forward,
)


def update_lumina_pipeline_teacache(
    pipeline,
    rel_l1_thresh=0.3,
    num_inference_steps=30,
    enable_teacache=True,
):
    pipeline.transformer.forward = teacache_forward.__get__(
        pipeline.transformer,
        pipeline.transformer.__class__,
    )
    pipeline.transformer._teacache_enabled = bool(enable_teacache)
    pipeline.transformer._teacache_rel_l1_thresh = float(rel_l1_thresh)
    pipeline.transformer._teacache_num_steps = int(num_inference_steps)
    pipeline.transformer._teacache_coefficients = LUMINA_NEXT_TEACACHE_COEFFICIENTS
    reset_teacache_state(pipeline.transformer)
    return pipeline


def reset_lumina_teacache(pipeline, num_inference_steps=None, rel_l1_thresh=None, enable_teacache=None):
    if num_inference_steps is not None:
        pipeline.transformer._teacache_num_steps = int(num_inference_steps)
    if rel_l1_thresh is not None:
        pipeline.transformer._teacache_rel_l1_thresh = float(rel_l1_thresh)
    if enable_teacache is not None:
        pipeline.transformer._teacache_enabled = bool(enable_teacache)
    reset_teacache_state(pipeline.transformer)


def get_lumina_teacache_stats(pipeline):
    return get_teacache_stats(pipeline.transformer)
