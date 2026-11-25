"""
频率分析工具

用于可视化和分析 FDAS 的频率分解结果
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def compute_frequency_spectrum(latent: torch.Tensor) -> torch.Tensor:
    """
    计算 latent 的频率谱

    Args:
        latent: [B, C, H, W]

    Returns:
        spectrum: [B, C, H, W//2+1] 频率谱的幅度
    """
    # 2D FFT
    latent_fft = torch.fft.rfft2(latent, norm='ortho')

    # 计算幅度
    spectrum = torch.abs(latent_fft)

    return spectrum


def visualize_frequency_decomposition(
    latent: torch.Tensor,
    low_freq: torch.Tensor,
    high_freq: torch.Tensor,
    save_path: str,
    channel_idx: int = 0
):
    """
    可视化频率分解结果

    Args:
        latent: [B, C, H, W] 原始 latent
        low_freq: [B, C, H, W] 低频成分
        high_freq: [B, C, H, W] 高频成分
        save_path: 保存路径
        channel_idx: 要可视化的通道索引
    """
    # 提取单个 batch 和通道
    latent_ch = latent[0, channel_idx].cpu().numpy()
    low_ch = low_freq[0, channel_idx].cpu().numpy()
    high_ch = high_freq[0, channel_idx].cpu().numpy()

    # 计算频谱
    latent_spectrum = compute_frequency_spectrum(latent[0:1, channel_idx:channel_idx+1])
    low_spectrum = compute_frequency_spectrum(low_freq[0:1, channel_idx:channel_idx+1])
    high_spectrum = compute_frequency_spectrum(high_freq[0:1, channel_idx:channel_idx+1])

    latent_spectrum_np = latent_spectrum[0, 0].cpu().numpy()
    low_spectrum_np = low_spectrum[0, 0].cpu().numpy()
    high_spectrum_np = high_spectrum[0, 0].cpu().numpy()

    # 创建 2x3 子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：空间域
    axes[0, 0].imshow(latent_ch, cmap='viridis')
    axes[0, 0].set_title(f'Original Latent (Ch {channel_idx})')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(low_ch, cmap='viridis')
    axes[0, 1].set_title('Low Frequency')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(high_ch, cmap='viridis')
    axes[0, 2].set_title('High Frequency')
    axes[0, 2].axis('off')

    # 第二行：频域（对数尺度）
    axes[1, 0].imshow(np.log(latent_spectrum_np + 1e-10), cmap='hot')
    axes[1, 0].set_title('Original Spectrum (log)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.log(low_spectrum_np + 1e-10), cmap='hot')
    axes[1, 1].set_title('Low Freq Spectrum (log)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.log(high_spectrum_np + 1e-10), cmap='hot')
    axes[1, 2].set_title('High Freq Spectrum (log)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 频率分解可视化已保存到: {save_path}")


def analyze_frequency_energy(
    latent: torch.Tensor,
    low_freq: torch.Tensor,
    high_freq: torch.Tensor,
    verbose: bool = True
) -> dict:
    """
    分析频率能量分布

    Args:
        latent: [B, C, H, W] 原始 latent
        low_freq: [B, C, H, W] 低频成分
        high_freq: [B, C, H, W] 高频成分
        verbose: 是否打印详细信息

    Returns:
        stats: 统计信息字典
    """
    # 计算能量（L2 norm）
    total_energy = (latent ** 2).sum().item()
    low_energy = (low_freq ** 2).sum().item()
    high_energy = (high_freq ** 2).sum().item()

    # 计算占比
    low_ratio = low_energy / total_energy
    high_ratio = high_energy / total_energy

    # 重建误差（验证分解是否完美）
    reconstructed = low_freq + high_freq
    reconstruction_error = ((latent - reconstructed) ** 2).sum().item()
    relative_error = reconstruction_error / total_energy

    stats = {
        "total_energy": total_energy,
        "low_energy": low_energy,
        "high_energy": high_energy,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "reconstruction_error": reconstruction_error,
        "relative_error": relative_error,
    }

    if verbose:
        print("\n" + "="*60)
        print("📊 频率能量分析")
        print("="*60)
        print(f"总能量: {total_energy:.4f}")
        print(f"低频能量: {low_energy:.4f} ({low_ratio*100:.2f}%)")
        print(f"高频能量: {high_energy:.4f} ({high_ratio*100:.2f}%)")
        print(f"重建误差: {reconstruction_error:.2e} (相对误差: {relative_error:.2e})")
        print("="*60 + "\n")

    return stats


def plot_frequency_energy_distribution(
    latent: torch.Tensor,
    cutoff_ratio: float,
    save_path: str
):
    """
    绘制频率能量分布曲线

    Args:
        latent: [B, C, H, W]
        cutoff_ratio: 低频截断比例
        save_path: 保存路径
    """
    # 计算频谱
    latent_fft = torch.fft.rfft2(latent, norm='ortho')
    spectrum = torch.abs(latent_fft)

    # 对所有 batch 和通道取平均
    spectrum_mean = spectrum.mean(dim=(0, 1)).cpu().numpy()  # [H, W//2+1]

    # 计算径向频率分布
    H, W = spectrum_mean.shape
    center_h, center_w = 0, 0  # DC 分量在左上角

    # 创建频率坐标
    freq_h = np.fft.fftfreq(H * 2, d=1.0)[:H]
    freq_w = np.fft.rfftfreq(W * 2, d=1.0)

    freq_h_grid, freq_w_grid = np.meshgrid(freq_h, freq_w, indexing='ij')
    freq_radius = np.sqrt(freq_h_grid**2 + freq_w_grid**2)

    # 按半径分桶
    max_radius = freq_radius.max()
    num_bins = 100
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    radial_energy = []

    for i in range(num_bins):
        mask = (freq_radius >= bin_edges[i]) & (freq_radius < bin_edges[i+1])
        if mask.sum() > 0:
            radial_energy.append(spectrum_mean[mask].mean())
        else:
            radial_energy.append(0)

    # 归一化半径
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    normalized_radius = bin_centers / max_radius

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_radius, radial_energy, linewidth=2, label='Radial Energy')
    plt.axvline(x=cutoff_ratio, color='r', linestyle='--', linewidth=2,
                label=f'Cutoff ({cutoff_ratio:.2f})')

    plt.xlabel('Normalized Frequency (0=DC, 1=Nyquist)', fontsize=12)
    plt.ylabel('Average Spectral Magnitude', fontsize=12)
    plt.title('Frequency Energy Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 频率能量分布图已保存到: {save_path}")
