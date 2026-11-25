from ...modules.attention_processor import RASHunyuanAttnProcessor2_0
from ...modules.hunyuan_dit.transformer_forward import ras_forward


def update_hunyuan_pipeline(pipeline):
    pipeline.transformer.forward = ras_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.blocks:
        block.attn1.set_processor(RASHunyuanAttnProcessor2_0())
        block.attn2.set_processor(RASHunyuanAttnProcessor2_0())
    return pipeline
