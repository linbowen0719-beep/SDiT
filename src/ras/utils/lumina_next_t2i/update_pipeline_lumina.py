from ...modules.lumina_next_t2i.transformer_forward import ras_forward
from ...modules.lumina_next_t2i.block_forward import ras_block_forward
from ...modules.attention_processor import RASLuminaAttnProcessor2_0
from ...schedulers import RASFlowMatchEulerDiscreteScheduler, AdaptiveComplexityScheduler


def update_lumina_pipeline(pipeline, use_adaptive=False, syn_timesteps=None, custom_timesteps=None):
    """
    Update Lumina pipeline with RAS or Adaptive Complexity scheduler.

    Args:
        pipeline: Diffusers LuminaPipeline instance
        use_adaptive: If True, use AdaptiveComplexityScheduler instead of RASFlowMatchEulerDiscreteScheduler
        syn_timesteps: List of synchronization points for adaptive mode
        custom_timesteps: Custom timestep schedule for adaptive mode

    Returns:
        Updated pipeline
    """
    # Choose scheduler based on mode
    if use_adaptive:
        scheduler = AdaptiveComplexityScheduler.from_config(
            pipeline.scheduler.config,
            syn_timesteps=syn_timesteps,
            custom_timesteps=custom_timesteps
        )
    else:
        scheduler = RASFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    pipeline.scheduler = scheduler

    # Apply RAS transformer forward and attention processors
    pipeline.transformer.forward = ras_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.layers:
        block.forward = ras_block_forward.__get__(block, block.__class__)
        block.attn1.set_processor(RASLuminaAttnProcessor2_0())
        block.attn2.set_processor(RASLuminaAttnProcessor2_0())

    return pipeline
