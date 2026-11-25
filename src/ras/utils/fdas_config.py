"""
FDAS (Frequency-Decoupled Adaptive Sampling) 配置文件
"""

FDAS_CONFIG = {
    # ============ 频率分解参数 ============
    "frequency_cutoff": 0.25,  # 低频截断比例（保留 25% 的低频）

    # ============ 采样策略参数 ============
    "high_freq_sample_ratio": 0.3,  # 高频部分的 RAS 采样比率（只计算 30% 的高频 tokens）

    # ============ 低频处理策略 ============
    "enable_low_freq_bypass": True,  # 是否启用低频 bypass（从 LR 继承）
    "low_freq_from_lr_threshold": 0.3,  # 低于此归一化 timestep 时，低频从 LR 继承

    # ============ RAS 集成参数 ============
    "enable_high_freq_ras": True,  # 是否对高频部分使用 RAS
    "scheduler_start_step": 4,  # RAS 开始步骤
    "scheduler_end_step": 24,  # RAS 结束步骤

    # ============ 调试和可视化 ============
    "debug_mode": False,  # 是否启用调试模式（打印频率信息）
    "save_frequency_visualizations": False,  # 是否保存频率可视化
}


def get_fdas_config(custom_config: dict = None) -> dict:
    """
    获取 FDAS 配置

    Args:
        custom_config: 自定义配置，会覆盖默认配置

    Returns:
        合并后的配置字典
    """
    config = FDAS_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config
