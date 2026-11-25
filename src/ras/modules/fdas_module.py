"""
Frequency-Decoupled Adaptive Sampling (FDAS) Module

核心思想：
- 低频信息从 LR 图像继承（bypass 扩散过程）
- 高频信息用 RAS 加速生成
- 避免 KV Cache 时间不同步问题
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class FrequencyDecomposer:
    """
    频率分解器

    使用 2D FFT 将 latent 分解为低频和高频成分
    """

    def __init__(self, cutoff_ratio: float = 0.25):
        """
        Args:
            cutoff_ratio: 低频截断比例（保留前 cutoff_ratio 的频率）
                         例如 0.25 表示保留最低的 25% 频率作为低频
        """
        self.cutoff_ratio = cutoff_ratio
        self._low_pass_mask_cache = {}  # 缓存 mask 避免重复计算

    def _create_low_pass_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        创建低通滤波器 mask

        Args:
            height, width: latent 的空间尺寸
            device, dtype: tensor 的设备和数据类型

        Returns:
            mask: [H, W//2+1] 低通滤波器（用于 rfft2）
        """
        cache_key = (height, width, device, dtype)
        if cache_key in self._low_pass_mask_cache:
            return self._low_pass_mask_cache[cache_key]

        # 计算截断频率
        cutoff_h = int(height * self.cutoff_ratio)
        cutoff_w = int(width * self.cutoff_ratio)

        # 创建圆形低通滤波器（更平滑）
        # rfft2 的输出维度是 [H, W//2+1]
        freq_h = torch.fft.fftfreq(height, device=device)[:, None]  # [H, 1]
        freq_w = torch.fft.rfftfreq(width, device=device)[None, :]  # [1, W//2+1]

        # 归一化频率（0-1范围）
        freq_h_norm = freq_h.abs() / 0.5  # fftfreq 范围是 [-0.5, 0.5]
        freq_w_norm = freq_w.abs() / 0.5

        # 计算到原点的距离（频率空间中）
        freq_radius = torch.sqrt(freq_h_norm**2 + freq_w_norm**2)

        # 创建 mask：距离小于 cutoff_ratio 的为低频
        mask = (freq_radius <= self.cutoff_ratio).to(dtype)

        self._low_pass_mask_cache[cache_key] = mask
        return mask

    def decompose(
        self,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 latent 分解为低频和高频成分

        Args:
            latent: [B, C, H, W] 输入 latent

        Returns:
            low_freq: [B, C, H, W] 低频成分
            high_freq: [B, C, H, W] 高频成分
        """
        B, C, H, W = latent.shape
        device = latent.device
        original_dtype = latent.dtype

        # FFT 不支持 bfloat16，需要转换为 float32
        if original_dtype == torch.bfloat16:
            latent = latent.float()

        # 2D FFT (使用 rfft2 节省计算和内存)
        latent_fft = torch.fft.rfft2(latent, norm='ortho')  # [B, C, H, W//2+1]

        # 创建低通滤波器 (使用 float32 以兼容 FFT)
        low_pass_mask = self._create_low_pass_mask(H, W, device, torch.float32)
        high_pass_mask = 1.0 - low_pass_mask

        # 应用滤波器
        low_freq_fft = latent_fft * low_pass_mask[None, None, :, :]
        high_freq_fft = latent_fft * high_pass_mask[None, None, :, :]

        # 逆 FFT
        low_freq = torch.fft.irfft2(low_freq_fft, s=(H, W), norm='ortho')
        high_freq = torch.fft.irfft2(high_freq_fft, s=(H, W), norm='ortho')

        # 转回原来的 dtype
        if original_dtype == torch.bfloat16:
            low_freq = low_freq.bfloat16()
            high_freq = high_freq.bfloat16()

        return low_freq, high_freq

    def compose(
        self,
        low_freq: torch.Tensor,
        high_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        合并低频和高频成分

        Args:
            low_freq: [B, C, H, W] 低频成分
            high_freq: [B, C, H, W] 高频成分

        Returns:
            latent: [B, C, H, W] 合并后的 latent
        """
        return low_freq + high_freq

    def get_frequency_energy(
        self,
        latent: torch.Tensor
    ) -> Tuple[float, float]:
        """
        计算低频和高频的能量占比

        Args:
            latent: [B, C, H, W]

        Returns:
            low_energy_ratio: 低频能量占比
            high_energy_ratio: 高频能量占比
        """
        low_freq, high_freq = self.decompose(latent)

        total_energy = (latent ** 2).sum()
        low_energy = (low_freq ** 2).sum()
        high_energy = (high_freq ** 2).sum()

        low_ratio = (low_energy / total_energy).item()
        high_ratio = (high_energy / total_energy).item()

        return low_ratio, high_ratio


class FDASManager:
    """
    FDAS 管理器

    管理频率解耦的采样策略
    """

    def __init__(
        self,
        cutoff_ratio: float = 0.25,
        high_freq_sample_ratio: float = 0.3,
        low_freq_from_lr_threshold: float = 0.3,
        enable_low_freq_bypass: bool = True
    ):
        """
        Args:
            cutoff_ratio: 频率截断比例
            high_freq_sample_ratio: 高频部分的 RAS 采样比率
            low_freq_from_lr_threshold: 低于此 timestep 时，低频从 LR 继承
            enable_low_freq_bypass: 是否启用低频 bypass
        """
        self.decomposer = FrequencyDecomposer(cutoff_ratio)
        self.high_freq_sample_ratio = high_freq_sample_ratio
        self.low_freq_from_lr_threshold = low_freq_from_lr_threshold
        self.enable_low_freq_bypass = enable_low_freq_bypass

        # 缓存 LR 的低频 latent（避免重复编码）
        self.lr_low_freq_latent = None

    def set_lr_low_freq_latent(self, lr_latent: torch.Tensor):
        """
        设置 LR 图像的低频 latent（在早期步骤中使用）

        Args:
            lr_latent: [B, C, H, W] LR 图像的 VAE latent
        """
        low_freq, _ = self.decomposer.decompose(lr_latent)
        self.lr_low_freq_latent = low_freq

    def should_use_lr_low_freq(self, normalized_timestep: float) -> bool:
        """
        判断当前 timestep 是否应该使用 LR 的低频

        Args:
            normalized_timestep: 归一化的 timestep (0-1)
                0.0 = 最接近纯噪声（去噪开始）
                1.0 = 最接近干净图像（去噪结束）

        Returns:
            bool: True 表示应该使用 LR 的低频
        """
        if not self.enable_low_freq_bypass:
            return False

        # 后期步骤（接近图像）使用 LR 的低频来保持一致性
        # 注意：应该是 > threshold，不是 <
        return normalized_timestep > self.low_freq_from_lr_threshold

    def process_latent(
        self,
        latent: torch.Tensor,
        normalized_timestep: float,
        use_ras_for_high_freq: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        处理 latent：分解、决定策略

        Args:
            latent: [B, C, H, W] 当前 latent
            normalized_timestep: 归一化的 timestep
            use_ras_for_high_freq: 是否对高频使用 RAS

        Returns:
            low_freq: [B, C, H, W] 低频成分
            high_freq: [B, C, H, W] 高频成分
            low_freq_from_lr: bool，True 表示低频来自 LR
        """
        # 分解
        low_freq, high_freq = self.decomposer.decompose(latent)

        # 判断是否使用 LR 的低频
        low_freq_from_lr = False
        if self.should_use_lr_low_freq(normalized_timestep):
            if self.lr_low_freq_latent is not None:
                low_freq = self.lr_low_freq_latent
                low_freq_from_lr = True

        return low_freq, high_freq, low_freq_from_lr

    def reset(self):
        """重置 FDAS 管理器（新的推理任务开始时调用）"""
        self.lr_low_freq_latent = None
        self.decomposer._low_pass_mask_cache.clear()


# 全局 FDAS 管理器实例
FDAS_MANAGER = FDASManager()
