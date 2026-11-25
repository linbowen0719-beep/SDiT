"""
Cluster Adaptive Scheduler with 8-Neighbor Expansion

核心改进:
1. 基于复杂度的聚类选择（选择前50%聚类，可配置）
2. 8-邻域扩展：每个选中的patch包含其8个邻居
3. 保持RAS的防饥饿机制
4. 每步重新聚类（基于当前复杂度）

与ClusterBasedRASScheduler的区别:
- ClusterBasedRASScheduler: 固定选择top-k簇，簇内所有patches
- ClusterAdaptiveScheduler: 选择top-X%簇，每个patch扩展8-邻域
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, Tuple, Union
from pathlib import Path

from .cluster_based_ras_scheduler import ClusterBasedRASScheduler
from .ras_scheduling_flow_match_euler_discrete import RASFlowMatchEulerDiscreteSchedulerOutput
from ras.utils import ras_manager
from ras.utils.patch_utils import (
    expand_patches_8_neighbors,
    expand_patches_4_neighbors,
    expand_patches_no_expansion,
    expand_patches_by_radius
)


class ClusterAdaptiveScheduler(ClusterBasedRASScheduler):
    """
    聚类自适应调度器，增加8-邻域扩展

    改进:
    - 基于复杂度选择前X%聚类（默认50%）
    - 每个选中patch扩展8-邻域
    - 每步重新聚类
    - 保持原RAS的防饥饿机制
    """

    # Internal A/B switch inherited by DDPM/DPM wrappers without forwarding a
    # RAS-only constructor argument into their diffusers parent constructors.
    enable_lazy_anchor_metadata = True

    def __init__(
        self,
        *args,
        num_clusters: int = 16,
        cluster_selection_ratio: float = 0.5,
        recluster_every_step: bool = True,
        recluster_every_n_steps: int = 0,
        final_sync_steps: int = 3,
        segmentation_algorithm: str = "quickshift",
        use_momentum_prediction: bool = True,
        save_x_pred: bool = False,
        x_pred_save_steps: list = None,
        x_pred_output_dir: str = None,
        vae_model: any = None,
        quickshift_kernel_size: float = 2.3,
        quickshift_max_dist: float = 6.0,
        quickshift_ratio: float = 0.5,
        enable_partition_metadata_cache: bool = True,
        enable_cpu_graph_metadata: bool = True,
        enable_host_selection_schedule: bool = True,
        defer_dense_cluster_mask: bool = False,
        **kwargs
    ):
        """
        Args:
            num_clusters: 聚类数量（仅用于kmeans，其他算法作为选择比例参考）
            cluster_selection_ratio: 每步选择多少比例的聚类（0.5 = 50%）
            recluster_every_step: 是否每步重新聚类
            recluster_every_n_steps: 若 >0，则每隔N个RAS步重新聚类；recluster_every_step优先
            final_sync_steps: 最后N步禁用adaptive，改为全部同步更新
            segmentation_algorithm: 分割算法 ("kmeans", "slic", "felzenszwalb", "watershed", "quickshift")
            use_momentum_prediction: 是否启用sigma缩放（True: ε̂ = ε × Δσ_ratio, False: 直接复用ε）
        """
        # 初始化父类（top_k_clusters在父类中不会被使用，因为我们重写了选择逻辑）
        super().__init__(
            *args,
            num_clusters=num_clusters,
            top_k_clusters=max(1, int(num_clusters * cluster_selection_ratio)),
            segmentation_algorithm=segmentation_algorithm,
            quickshift_kernel_size=quickshift_kernel_size,
            quickshift_max_dist=quickshift_max_dist,
            quickshift_ratio=quickshift_ratio,
            **kwargs
        )

        self.cluster_selection_ratio = cluster_selection_ratio
        self.recluster_every_step = recluster_every_step
        self.recluster_every_n_steps = int(recluster_every_n_steps or 0)
        self.last_recluster_step = None
        self.final_sync_steps = final_sync_steps
        self.use_momentum_prediction = use_momentum_prediction
        self.enable_partition_metadata_cache = bool(enable_partition_metadata_cache)
        self.enable_cpu_graph_metadata = bool(enable_cpu_graph_metadata)
        self.enable_host_selection_schedule = bool(enable_host_selection_schedule)
        self._active_selection_cpu = None
        self._cached_selection_cpu = None
        self.defer_dense_cluster_mask = bool(defer_dense_cluster_mask)
        self._partition_metadata_cache = None
        self._cluster_mask_density_cache = None
        self._cluster_far_ids_cpu_cache = None

        # ============ x_pred 保存配置 ============
        self.save_x_pred = save_x_pred
        self.x_pred_save_steps = x_pred_save_steps if x_pred_save_steps is not None else []
        self.x_pred_output_dir = x_pred_output_dir if x_pred_output_dir is not None else "outputs/debug/predictions"
        self.vae_model = vae_model

        # ============ 累计统计字段 ============
        self.total_patches = 0  # 总patch数
        self.cumulative_updated_patches = 0  # 累计更新的patch数
        self.step_update_history = []  # 每步的更新patch数历史
        self.cluster_update_counts: Optional[np.ndarray] = None  # 每个cluster被选中的累计次数

        # ============ 时间测量字段 ============
        self.clustering_times = []  # 每步的聚类时间
        self.metric_compute_times = []  # 每步计算metric的时间
        self.kmeans_times = []  # 每步K-means的时间
        self.selection_times = []  # 每步选择和扩展的时间
        self.total_clustering_time = 0.0  # 总聚类时间

        # ============ 注意力稀疏度统计 ============
        self.attn_mask_density_history = []    # cluster graph mask 密度 (越低越稀疏)
        self.window_area_ratio_history = []    # 滑动窗口面积占比 (bounding box 近似)

        # ============ Sigma-scaled noise reuse ============
        self.cached_noise = None  # 最近一步被跳过区域的噪声预测
        self.cached_index_signature = None  # 记录噪声对应的索引签名
        self.cached_region_weights = None  # 可选的区域权重
        self.prev_delta_sigma = None  # 上一步的 sigma 变化量 (用于缩放)

        print(f"✅ ClusterAdaptiveScheduler初始化完成:")
        print(f"   聚类数量: {self.num_clusters}")
        print(f"   选择比例: {self.cluster_selection_ratio:.0%}")
        print(f"   邻域扩展模式: {self._get_expansion_mode_desc()}")
        print(f"   每步重新聚类: {self.recluster_every_step}")
        if not self.recluster_every_step and self.recluster_every_n_steps > 0:
            print(f"   间隔重新聚类: 每 {self.recluster_every_n_steps} 个RAS步")
        print(f"   最后{self.final_sync_steps}步: 禁用adaptive，全部同步更新")
        print(f"   噪声复用策略: {'启用 Sigma 缩放 (ε̂ = ε × Δσ_ratio)' if self.use_momentum_prediction else '直接复用 (ε̂ = ε)'}")

    def _get_partition_metadata(self, device):
        """Cache immutable membership while retaining raster order within each cluster.

        Scores and their reductions are deliberately not cached: using the same
        ordered gather followed by ``mean`` preserves the selection arithmetic.
        The CPU snapshot also detects an in-place edit of the numpy labels.
        """
        labels = np.asarray(self.cluster_labels)
        device = torch.device(device)
        cached = self._partition_metadata_cache
        if (cached is not None and cached["device"] == device
                and cached["labels_cpu"].dtype == labels.dtype
                and np.array_equal(cached["labels_cpu"], labels)):
            return cached
        labels_cpu = labels.copy()
        members_cpu = {
            int(cid): tuple(np.flatnonzero(labels_cpu == cid).tolist())
            for cid in np.unique(labels_cpu)
        }
        cached = {
            "device": device,
            "labels_cpu": labels_cpu,
            "labels_tensor": torch.from_numpy(labels_cpu).to(device),
            "members_cpu": members_cpu,
            "members_device": {
                cid: torch.tensor(indices, device=device, dtype=torch.long)
                for cid, indices in members_cpu.items()
            },
        }
        self._partition_metadata_cache = cached
        return cached

    def _attention_mask_density(self, mask):
        """Read a partition's diagnostic density once, not once per denoising step."""
        if not self.enable_partition_metadata_cache:
            return mask.float().mean().item()
        try:
            version = mask._version
        except RuntimeError:
            # Inference tensors have no version counter. The scheduler owns this
            # mask and only replaces it when the partition changes.
            version = None
        cached = self._cluster_mask_density_cache
        if cached is None or cached[0] is not mask or cached[1] != version:
            cached = (mask, version, mask.float().mean().item())
            self._cluster_mask_density_cache = cached
        return cached[2]

    def _publish_cluster_pack_metadata_cpu(self):
        """Publish versioned, partition-level host indices for attention pack plans.

        Consumers must validate both metadata_version and every reference in
        metadata before using these tuples. Query ranges still change each step.
        A single transfer replaces per-layer/per-step transfers of static lists.
        """
        manager = ras_manager.MANAGER
        manager.cluster_pack_metadata_cpu = None
        if (not self.enable_partition_metadata_cache
                or not getattr(manager, "enable_cluster_fused_pack", False)
                or manager.cluster_key_indices is None
                or manager.cluster_remote_anchor_all_indices is None
                or manager.cluster_remote_anchor_packed_positions is None
                or manager.cluster_remote_anchor_all_logit_bias is None):
            return
        metadata = (
            manager.cluster_key_indices, manager.cluster_local_key_spans,
            manager.cluster_remote_anchor_all_indices,
            manager.cluster_remote_anchor_packed_positions,
            manager.cluster_remote_anchor_all_logit_bias,
        )
        entries = [("all_anchor_keys", None, metadata[2])]
        entries.extend(("local_keys", cid, indices) for cid, indices in metadata[0].items())
        entries.extend(("packed_positions", cid, indices) for cid, indices in metadata[3].items())
        flattened = torch.cat([indices.reshape(-1) for _, _, indices in entries]).detach().cpu().tolist()
        result = {"metadata_version": int(manager.cluster_remote_anchor_metadata_version),
                  "metadata": metadata, "local_keys": {}, "packed_positions": {}}
        cursor = 0
        for field, cid, indices in entries:
            length = int(indices.numel())
            values = tuple(flattened[cursor:cursor + length])
            if cid is None:
                result[field] = values
            else:
                result[field][cid] = values
            cursor += length
        manager.cluster_pack_metadata_cpu = result

    def _get_expansion_mode_desc(self):
        """
        获取邻域扩展模式的描述
        """
        mode = ras_manager.MANAGER.neighbor_expansion_mode
        if mode == "0":
            return "无扩展（仅选中的patches）"
        elif mode == "4":
            return "4-邻域（上下左右）"
        else:  # mode == "8"
            return "8-邻域（包含对角线）"

    def _save_x_pred_vae_decoded(self, sample: torch.FloatTensor, model_output: torch.FloatTensor, sigma: float, current_step: int):
        """
        保存 x_pred (预测的干净图像) 的 VAE 解码结果

        对于 Flow Matching: x_pred = sample - sigma * model_output

        Args:
            sample: 当前 latent x_t (B, C, H, W)
            model_output: 模型输出 (velocity/噪声预测) (B, C, H, W)
            sigma: 当前时刻的 sigma 值
            current_step: 当前时间步
        """
        if not self.save_x_pred or current_step not in self.x_pred_save_steps:
            return

        if self.vae_model is None:
            print(f"   ⚠️  警告: VAE model 未设置，无法保存 x_pred")
            return

        try:
            from PIL import Image
            from pathlib import Path

            # 创建输出目录
            output_path = Path(self.x_pred_output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"   📸 保存 step {current_step} 的 x_pred VAE 解码结果...")

            # 计算 x_pred (预测的干净图像)
            # 对于 Flow Matching: x_pred = x_t - sigma * v
            with torch.no_grad():
                # 确保 sigma 和 model_output 的数据类型一致
                sigma_tensor = torch.tensor(sigma, dtype=model_output.dtype, device=model_output.device)
                x_pred = sample - sigma_tensor * model_output

                # 缩放 latents
                latents_scaled = x_pred / self.vae_model.config.scaling_factor
                # 解码
                image_tensor = self.vae_model.decode(latents_scaled, return_dict=False)[0]

                # 转换为 PIL 图像 (假设使用标准的后处理)
                image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                image_np = (image_np * 255).round().astype("uint8")

                # 保存第一张图像
                pil_image = Image.fromarray(image_np[0])
                save_path = output_path / f"x_pred_step_{current_step:03d}.png"
                pil_image.save(save_path)
                print(f"   ✅ 已保存: {save_path}")

        except Exception as e:
            print(f"   ⚠️  保存 x_pred 失败: {e}")

    def _save_sample_tensor(self, sample: torch.FloatTensor, current_step: int, output_dir: str = "outputs/debug/samples"):
        """
        保存sample张量为numpy文件和可视化图像

        Args:
            sample: 要保存的latent张量 (B, C, H, W)
            current_step: 当前时间步
            output_dir: 输出目录
        """
        if not bool(getattr(ras_manager.MANAGER, "save_scheduler_samples", False)):
            return
        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 保存为numpy文件
            sample_np = sample.cpu().detach().numpy()
            np_filename = output_path / f"timestep_{current_step:03d}.npy"
            np.save(str(np_filename), sample_np)

            # 尝试保存可视化图像（如果有PIL库）
            try:
                from PIL import Image
                # 将sample张量标准化到0-255范围用于可视化
                sample_vis = sample.cpu().detach().squeeze(0)  # 移除batch维度

                # 如果是多通道，取第一个通道或平均
                if sample_vis.dim() == 3:
                    sample_vis = sample_vis[0] if sample_vis.shape[0] > 0 else sample_vis

                # 标准化到0-255
                sample_min = sample_vis.min()
                sample_max = sample_vis.max()
                if sample_max > sample_min:
                    sample_vis = ((sample_vis - sample_min) / (sample_max - sample_min) * 255).to(torch.uint8)
                else:
                    sample_vis = torch.zeros_like(sample_vis, dtype=torch.uint8)

                # 保存为PNG
                img = Image.fromarray(sample_vis.numpy(), mode='L')
                img_filename = output_path / f"timestep_{current_step:03d}.png"
                img.save(str(img_filename))

                # print(f"   💾 保存sample: {np_filename.name}, {img_filename.name}")
            except ImportError:
                pass  # PIL不可用，只保存numpy文件

        except Exception as e:
            print(f"   ⚠️  保存sample失败: {e}")

    def update_patch_statistics(self, num_updated_patches, total_patches):
        """
        更新累计的patch统计信息

        Args:
            num_updated_patches: 当前步更新的patch数
            total_patches: 总patch数
        """
        if self.total_patches == 0:
            self.total_patches = total_patches

        self.cumulative_updated_patches += num_updated_patches
        self.step_update_history.append(num_updated_patches)

    def get_cumulative_update_ratio(self):
        """
        获取累计更新的patch比例

        Returns:
            float: 累计更新比例 (0-1)
        """
        if self.total_patches == 0:
            return 0.0
        return self.cumulative_updated_patches / self.total_patches

    def print_clustering_timing_summary(self, total_inference_time):
        """
        打印聚类时间统计摘要

        Args:
            total_inference_time: 总推理时间（秒）
        """
        if not self.clustering_times:
            return

        total_clustering_time = sum(self.clustering_times)
        num_steps = len(self.clustering_times)

        clustering_ratio = total_clustering_time / max(total_inference_time, 0.001) * 100

        print("\n" + "="*80)
        print("⏱️  聚类时间分析")
        print("="*80)

        print(f"\n【总体聚类时间】")
        print(f"  总聚类时间: {total_clustering_time:.3f}s")
        print(f"  总推理时间: {total_inference_time:.3f}s")
        print(f"  聚类占比: {clustering_ratio:.1f}%")

        density_report = ras_manager.MANAGER.get_cluster_attention_density_report()
        if density_report["overall_density"] is not None:
            overall_density = density_report["overall_density"]
            n_recorded = len(density_report["steps"])
            print(f"\n【注意力稀疏度统计 (累计 {n_recorded} 步)】")
            print(f"  Cluster graph mask 累计密度: {overall_density:.1%}  → 理论最优加速比 {1/overall_density:.2f}x")
        elif self.attn_mask_density_history:
            avg_density = sum(self.attn_mask_density_history) / len(self.attn_mask_density_history)
            n_recorded = len(self.attn_mask_density_history)
            print(f"\n【注意力稀疏度统计 (累计 {n_recorded} 步)】")
            print(f"  Cluster graph mask 平均密度: {avg_density:.1%}  → 理论最优加速比 {1/avg_density:.2f}x")

        if num_steps > 0:
            avg_clustering_time = total_clustering_time / num_steps
            max_clustering_time = max(self.clustering_times)
            min_clustering_time = min(self.clustering_times)

            print(f"\n【每步聚类时间】")
            print(f"  平均: {avg_clustering_time*1000:.2f}ms")
            print(f"  最大: {max_clustering_time*1000:.2f}ms")
            print(f"  最小: {min_clustering_time*1000:.2f}ms")

        # 详细时间分解
        if self.metric_compute_times and self.selection_times:
            total_metric_time = sum(self.metric_compute_times)
            total_selection_time = sum(self.selection_times)

            metric_ratio = total_metric_time / max(total_clustering_time, 0.001) * 100
            selection_ratio = total_selection_time / max(total_clustering_time, 0.001) * 100

            print(f"\n【聚类时间分解】")
            print(f"  Metric计算: {total_metric_time:.3f}s ({metric_ratio:.1f}%)")
            print(f"  选择和扩展: {total_selection_time:.3f}s ({selection_ratio:.1f}%)")

        print("="*80 + "\n")

    def _clear_cached_history(self):
        """
        清理噪声缓存相关的状态。
        """
        self.cached_noise = None
        self.cached_index_signature = None
        self.cached_region_weights = None
        self.prev_delta_sigma = None

    def _update_cached_noise(
        self,
        latent_cached_indices: Optional[torch.Tensor],
        new_noise: Optional[torch.Tensor],
        delta_sigma: float,
        region_weights: Optional[torch.Tensor] = None,
    ):
        """
        更新噪声缓存（仅保存最近一步，用于sigma缩放）。

        Args:
            latent_cached_indices: 当前被跳过区域的latent索引。
            new_noise: 当前步对这些区域的噪声预测ε_t。
            delta_sigma: 当前步的sigma变化量 |σ_next - σ_current|
            region_weights: 可选的区域重要性权重。
        """
        if (
            latent_cached_indices is None
            or new_noise is None
            or latent_cached_indices.numel() == 0
        ):
            self._clear_cached_history()
            ras_manager.MANAGER.cached_index = None
            ras_manager.MANAGER.cached_scaled_noise = None
            return

        self.cached_noise = new_noise.detach().clone()
        host = getattr(self, "_cached_selection_cpu", None)
        self.cached_index_signature = (tuple(host[1].tolist())
            if host is not None and host[0] is latent_cached_indices
            else tuple(latent_cached_indices.detach().cpu().tolist()))
        self.prev_delta_sigma = delta_sigma

        ras_manager.MANAGER.cached_index = latent_cached_indices
        ras_manager.MANAGER.cached_scaled_noise = self.cached_noise

        if region_weights is not None:
            self.cached_region_weights = region_weights.detach().clone()
        else:
            self.cached_region_weights = None

    def _compute_scaled_noise(
        self,
        current_delta_sigma: float,
        region_weights: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        基于sigma缩放因子来调整复用的噪声（不使用动量）。

        公式：
        ε̂_{t+1} = ε_t × (Δσ_{t+1} / Δσ_t)

        其中：
        - ε_t 是上一步缓存的噪声
        - Δσ_t 是上一步的sigma变化量
        - Δσ_{t+1} 是当前步的sigma变化量

        Args:
            current_delta_sigma: 当前步的sigma变化量 |σ_{t+1} - σ_t|
            region_weights: 可选的区域权重

        Returns:
            torch.Tensor: 缩放后的噪声；若无缓存则返回None
        """
        if self.cached_noise is None:
            return None

        # 如果启用动量预测（实际上这里是sigma缩放）
        if self.use_momentum_prediction and self.prev_delta_sigma is not None:
            # 避免除零
            if self.prev_delta_sigma > 1e-8:
                scale_factor = current_delta_sigma / self.prev_delta_sigma
                scaled_noise = self.cached_noise * scale_factor
            else:
                scaled_noise = self.cached_noise.clone()
        else:
            # 直接复用，不缩放
            scaled_noise = self.cached_noise.clone()

        # 应用区域权重（如果有）
        effective_weights = region_weights if region_weights is not None else self.cached_region_weights
        if effective_weights is not None:
            scaled_noise = scaled_noise * effective_weights.to(scaled_noise.device, scaled_noise.dtype)

        return scaled_noise

    def print_statistics_summary(self):
        """
        打印累计统计摘要
        """
        if self.total_patches == 0 or len(self.step_update_history) == 0:
            return

        num_steps = len(self.step_update_history)

        # 计算每步的百分比
        step_percentages = []
        for num_updated in self.step_update_history:
            if self.total_patches > 0:
                percentage = (num_updated / self.total_patches) * 100
                step_percentages.append(percentage)

        # 计算平均百分比，排除100%的全量推理
        adaptive_percentages = [p for p in step_percentages if p < 100]
        avg_percentage = sum(adaptive_percentages) / len(adaptive_percentages) if adaptive_percentages else 0

        print("\n" + "="*80)
        print("📊 推理统计汇总")
        print("="*80)
        print(f"总patches数: {self.total_patches}")
        print(f"推理步数: {num_steps}")
        print(f"累计更新patches: {self.cumulative_updated_patches}")
        print(f"平均每步更新百分比（排除全量推理）: {avg_percentage:.1f}%")
        print(f"平均每步更新patches: {self.cumulative_updated_patches / max(1, num_steps):.0f}")
        print(f"最大每步更新百分比: {max(step_percentages):.1f}%")
        print(f"最小每步更新百分比: {min(step_percentages):.1f}%")
        print("="*80 + "\n")

    def _compute_spatial_aware_metric(self, x_pred_squeezed):
        """
        计算空间感知复杂度指标

        结合了：
        1. 像素方差（衡量局部复杂度）
        2. 梯度幅度（衡量边缘/纹理）
        3. 拉普拉斯算子（衡量细节）

        Args:
            x_pred_squeezed: (H, W, C) tensor，通道维度已排列到最后

        Returns:
            metric: (H, W) tensor，空间感知复杂度
        """
        # 计算通道维度的方差作为基础复杂度
        variance = torch.var(x_pred_squeezed, dim=-1)  # (H, W)

        # 计算梯度幅度（用第一个通道或所有通道的平均）
        if x_pred_squeezed.shape[-1] > 1:
            # 多通道：计算每个通道梯度后求平均
            grad_x_list = []
            grad_y_list = []
            for c in range(x_pred_squeezed.shape[-1]):
                channel = x_pred_squeezed[..., c]
                # 计算梯度
                grad_x = torch.gradient(channel, dim=1)[0]
                grad_y = torch.gradient(channel, dim=0)[0]
                grad_x_list.append(grad_x)
                grad_y_list.append(grad_y)
            grad_x = torch.stack(grad_x_list, dim=-1).mean(dim=-1)
            grad_y = torch.stack(grad_y_list, dim=-1).mean(dim=-1)
        else:
            # 单通道
            channel = x_pred_squeezed[..., 0]
            grad_x = torch.gradient(channel, dim=1)[0]
            grad_y = torch.gradient(channel, dim=0)[0]

        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)  # (H, W)

        # 计算拉普拉斯算子（细节检测）
        laplacian = self._compute_laplacian(x_pred_squeezed)  # (H, W)

        # 综合复杂度分数（权重可根据需要调整）
        # 权重设置：方差40%、梯度30%、拉普拉斯30%
        metric = 0.4 * variance + 0.3 * grad_magnitude + 0.3 * laplacian

        return metric

    def _compute_spatial_aware_new_metric(self, x_pred_squeezed, x_current_squeezed=None, alpha=1.0, gamma=1.0, beta=1.0):
        """
        计算新的空间感知复杂度指标（基于公式5）

        公式: C_i^t = (1/C) * Σ_c [ α√(∂_x² + ∂_y²) + γ|Δx_pred| + β|x - x_pred| ]

        结合了：
        1. edge/texture: α × √[(∂_x x_pred)² + (∂_y x_pred)²] - 梯度幅度
        2. fine details: γ × |Δx_pred| - 拉普拉斯算子（细节）
        3. residual noise: β × |x^t - x_pred^t| - 残差噪声

        Args:
            x_pred_squeezed: (H, W, C) tensor，预测的x_pred，通道维度已排列到最后
            x_current_squeezed: (H, W, C) tensor，当前的x^t，可选，用于计算残差项
            alpha: edge/texture项的权重
            gamma: fine details项的权重
            beta: residual noise项的权重

        Returns:
            metric: (H, W) tensor，空间感知复杂度
        """
        H, W, C = x_pred_squeezed.shape

        # 初始化每个通道的复杂度
        per_channel_complexity = []

        for c in range(C):
            channel = x_pred_squeezed[..., c]  # (H, W)

            # 1. Edge/Texture 项: α × √[(∂_x x_pred)² + (∂_y x_pred)²]
            grad_x = torch.gradient(channel, dim=1)[0]  # (H, W)
            grad_y = torch.gradient(channel, dim=0)[0]  # (H, W)
            edge_texture = alpha * torch.sqrt(grad_x**2 + grad_y**2)  # (H, W)

            # 2. Fine Details 项: γ × |Δx_pred| (拉普拉斯算子)
            laplacian = self._compute_laplacian_single_channel(channel)  # (H, W)
            fine_details = gamma * torch.abs(laplacian)  # (H, W)

            # 3. Residual Noise 项: β × |x^t - x_pred^t|
            if x_current_squeezed is not None:
                x_current_channel = x_current_squeezed[..., c]  # (H, W)
                residual_noise = beta * torch.abs(x_current_channel - channel)  # (H, W)
            else:
                # 如果没有提供x_current，residual项为0
                residual_noise = torch.zeros_like(channel)

            # 合并三项
            channel_complexity = edge_texture + fine_details + residual_noise  # (H, W)
            per_channel_complexity.append(channel_complexity)

        # 对所有通道求平均: (1/C) * Σ_c [...]
        metric = torch.stack(per_channel_complexity, dim=-1).mean(dim=-1)  # (H, W)

        return metric

    def _compute_laplacian_single_channel(self, channel):
        """
        计算单通道的拉普拉斯算子响应

        Args:
            channel: (H, W) tensor，单通道数据

        Returns:
            laplacian: (H, W) tensor，拉普拉斯算子响应
        """
        H, W = channel.shape

        # 边界处理：使用反射填充
        padded = torch.nn.functional.pad(channel.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')

        # 提取3x3邻域
        patches = torch.nn.functional.unfold(padded, kernel_size=3, padding=0)
        patches = patches.view(1, 9, H, W)

        # 拉普拉斯卷积核
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=channel.dtype, device=channel.device).view(-1)

        # 应用卷积
        laplacian = (patches.squeeze(0) * laplacian_kernel.unsqueeze(-1).unsqueeze(-1)).sum(0)

        return laplacian

    def _compute_laplacian(self, x_pred_squeezed):
        """
        计算拉普拉斯算子响应（多通道版本）

        Args:
            x_pred_squeezed: (H, W, C) tensor

        Returns:
            laplacian_score: (H, W) tensor
        """
        H, W, C = x_pred_squeezed.shape

        # 对每个通道计算拉普拉斯，然后取平均
        laplacian_responses = []

        for c in range(C):
            channel = x_pred_squeezed[..., c]  # (H, W)
            response = self._compute_laplacian_single_channel(channel)
            laplacian_responses.append(response)

        # 取绝对值并求平均
        laplacian_score = torch.stack(laplacian_responses, dim=0).abs().mean(0)

        return laplacian_score

    def _compute_frequency_metric(self, x_pred_squeezed, alpha=1.0, gamma=1.0):
        """
        计算基于频率的复杂度指标（仅包含edge/texture和fine details项）

        公式: C_i^t = (1/C) * Σ_c [ α√(∂_x² + ∂_y²) + γ|Δx_pred| ]

        结合了：
        1. edge/texture: α × √[(∂_x x_pred)² + (∂_y x_pred)²] - 梯度幅度
        2. fine details: γ × |Δx_pred| - 拉普拉斯算子（细节）

        Args:
            x_pred_squeezed: (H, W, C) tensor，预测的x_pred，通道维度已排列到最后
            alpha: edge/texture项的权重，默认1.0
            gamma: fine details项的权重，默认1.0

        Returns:
            metric: (H, W) tensor，频率域复杂度
        """
        H, W, C = x_pred_squeezed.shape

        # 初始化每个通道的复杂度
        per_channel_complexity = []

        for c in range(C):
            channel = x_pred_squeezed[..., c]  # (H, W)

            # 1. Edge/Texture 项: α × √[(∂_x x_pred)² + (∂_y x_pred)²]
            grad_x = torch.gradient(channel, dim=1)[0]  # (H, W)
            grad_y = torch.gradient(channel, dim=0)[0]  # (H, W)
            edge_texture = alpha * torch.sqrt(grad_x**2 + grad_y**2)  # (H, W)

            # 2. Fine Details 项: γ × |Δx_pred| (拉普拉斯算子)
            laplacian = self._compute_laplacian_single_channel(channel)  # (H, W)
            fine_details = gamma * torch.abs(laplacian)  # (H, W)

            # 合并两项（无残差项）
            channel_complexity = edge_texture + fine_details  # (H, W)
            per_channel_complexity.append(channel_complexity)

        # 对所有通道求平均: (1/C) * Σ_c [...]
        metric = torch.stack(per_channel_complexity, dim=-1).mean(dim=-1)  # (H, W)

        return metric

    def _compute_noise_metric(self, x_pred_squeezed, x_current_squeezed, beta=1.0):
        """
        计算基于噪声残差的复杂度指标（仅包含residual noise项）

        公式: C_i^t = (1/C) * Σ_c [ β|x - x_pred| ]

        仅使用：
        - residual noise: β × |x^t - x_pred^t| - 残差噪声

        Args:
            x_pred_squeezed: (H, W, C) tensor，预测的x_pred，通道维度已排列到最后
            x_current_squeezed: (H, W, C) tensor，当前的x^t，必需
            beta: residual noise项的权重，默认1.0

        Returns:
            metric: (H, W) tensor，噪声残差复杂度
        """
        H, W, C = x_pred_squeezed.shape

        # 初始化每个通道的复杂度
        per_channel_complexity = []

        for c in range(C):
            x_pred_channel = x_pred_squeezed[..., c]  # (H, W)
            x_current_channel = x_current_squeezed[..., c]  # (H, W)

            # Residual Noise 项: β × |x^t - x_pred^t|
            residual_noise = beta * torch.abs(x_current_channel - x_pred_channel)  # (H, W)

            per_channel_complexity.append(residual_noise)

        # 对所有通道求平均: (1/C) * Σ_c [...]
        metric = torch.stack(per_channel_complexity, dim=-1).mean(dim=-1)  # (H, W)

        return metric

    def _compute_patch_complexity_from_diff(self, diff, height, width, sample=None):
        """
        基于diff（残差）计算每个patch的复杂度分数

        Args:
            diff: 残差 (B, C, H, W)
            height, width: latent空间的高宽
            sample: 当前sample (B, C, H, W)，用于spatial_aware_new的残差项计算

        Returns:
            patch_scores: (num_patches,) tensor
        """
        patch_size = ras_manager.MANAGER.patch_size
        num_h = height // patch_size
        num_w = width // patch_size

        # 计算diff在通道维度上的统计特性
        diff_squeezed = diff.squeeze(0).permute(1, 2, 0)  # (H, W, C)

        if ras_manager.MANAGER.metric == "std":
            metric = torch.std(diff_squeezed, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "l2norm":
            metric = torch.norm(diff_squeezed, p=2, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "spatial_aware":
            metric = self._compute_spatial_aware_metric(diff_squeezed)  # (H, W)
        elif ras_manager.MANAGER.metric == "spatial_aware_new":
            # 对于spatial_aware_new，diff本身作为x_pred（因为我们在分析diff的复杂度）
            # 如果有sample，可以用于计算残差项
            x_current_squeezed = sample.squeeze(0).permute(1, 2, 0) if sample is not None else None
            metric = self._compute_spatial_aware_new_metric(
                diff_squeezed,
                x_current_squeezed=x_current_squeezed,
                alpha=1.0, gamma=1.0, beta=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "frequency":
            # 频率域复杂度：仅edge/texture + fine details（无残差项）
            metric = self._compute_frequency_metric(
                diff_squeezed,
                alpha=1.0, gamma=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "noise":
            # 噪声残差复杂度：仅residual noise项
            if sample is None:
                raise ValueError("noise metric requires sample (x_current) to be provided")
            x_current_squeezed = sample.squeeze(0).permute(1, 2, 0)
            metric = self._compute_noise_metric(
                diff_squeezed,
                x_current_squeezed=x_current_squeezed,
                beta=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "random":
            # 纯随机复杂度：返回均匀分布的随机值
            H, W = diff_squeezed.shape[0], diff_squeezed.shape[1]
            metric = torch.rand(H, W, device=diff_squeezed.device, dtype=diff_squeezed.dtype)  # (H, W)
        else:
            raise ValueError(f"Unknown metric: {ras_manager.MANAGER.metric}")

        # 聚合到patch级别
        metric_patches = metric.view(
            num_h, patch_size,
            num_w, patch_size
        ).transpose(-2, -3).mean(-1).mean(-1).view(-1)  # (num_patches,)

        return metric_patches

    def _compute_patch_complexity_from_x_pred(self, x_pred, height, width, sample=None):
        """
        基于x_pred（预测样本）计算每个patch的复杂度分数

        Args:
            x_pred: 预测的下一步样本 (B, C, H, W)
            height, width: latent空间的高宽
            sample: 当前sample (B, C, H, W)，用于spatial_aware_new的残差项计算

        Returns:
            patch_scores: (num_patches,) tensor
        """
        patch_size = ras_manager.MANAGER.patch_size
        num_h = height // patch_size
        num_w = width // patch_size

        # 计算x_pred在通道维度上的统计特性
        x_pred_squeezed = x_pred.squeeze(0).permute(1, 2, 0)  # (H, W, C)

        if ras_manager.MANAGER.metric == "paper":
            from sdit.policy import detail_residual_score
            metric = detail_residual_score(x_pred, sample).squeeze(0)
        elif ras_manager.MANAGER.metric == "std":
            metric = torch.std(x_pred_squeezed, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "l2norm":
            metric = torch.norm(x_pred_squeezed, p=2, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "spatial_aware":
            metric = self._compute_spatial_aware_metric(x_pred_squeezed)  # (H, W)
        elif ras_manager.MANAGER.metric == "spatial_aware_new":
            # 对于spatial_aware_new，需要x_current来计算残差项
            x_current_squeezed = sample.squeeze(0).permute(1, 2, 0) if sample is not None else None
            metric = self._compute_spatial_aware_new_metric(
                x_pred_squeezed,
                x_current_squeezed=x_current_squeezed,
                alpha=1.0, gamma=1.0, beta=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "frequency":
            # 频率域复杂度：仅edge/texture + fine details（无残差项）
            metric = self._compute_frequency_metric(
                x_pred_squeezed,
                alpha=1.0, gamma=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "noise":
            # 噪声残差复杂度：仅residual noise项
            if sample is None:
                raise ValueError("noise metric requires sample (x_current) to be provided")
            x_current_squeezed = sample.squeeze(0).permute(1, 2, 0)
            metric = self._compute_noise_metric(
                x_pred_squeezed,
                x_current_squeezed=x_current_squeezed,
                beta=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "random":
            # 纯随机复杂度：返回均匀分布的随机值
            H, W = x_pred_squeezed.shape[0], x_pred_squeezed.shape[1]
            metric = torch.rand(H, W, device=x_pred_squeezed.device, dtype=x_pred_squeezed.dtype)  # (H, W)
        else:
            raise ValueError(f"Unknown metric: {ras_manager.MANAGER.metric}")

        # 聚合到patch级别
        metric_patches = metric.view(
            num_h, patch_size,
            num_w, patch_size
        ).transpose(-2, -3).mean(-1).mean(-1).view(-1)  # (num_patches,)

        return metric_patches

    def compute_complexity_from_diff(self, diff, cluster_labels, height, width, sample=None):
        """
        基于diff（残差）计算复杂度分数

        Args:
            diff: 残差 (B, C, H, W)
            cluster_labels: (num_patches,) patch到簇的映射
            height, width: latent空间的高宽
            sample: 当前sample (B, C, H, W)，用于spatial_aware_new的残差项计算

        Returns:
            cluster_scores: (num_clusters,) tensor
        """
        patch_size = ras_manager.MANAGER.patch_size
        num_h = height // patch_size
        num_w = width // patch_size

        # 计算diff在通道维度上的统计特性
        diff_squeezed = diff.squeeze(0).permute(1, 2, 0)  # (H, W, C)

        if ras_manager.MANAGER.metric == "std":
            # 计算每个位置跨通道的标准差
            metric = torch.std(diff_squeezed, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "l2norm":
            # 计算每个位置的L2范数
            metric = torch.norm(diff_squeezed, p=2, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "spatial_aware":
            # 空间感知复杂度：结合方差、梯度和细节
            metric = self._compute_spatial_aware_metric(diff_squeezed)  # (H, W)
        elif ras_manager.MANAGER.metric == "spatial_aware_new":
            # 新的空间感知复杂度公式
            x_current_squeezed = sample.squeeze(0).permute(1, 2, 0) if sample is not None else None
            metric = self._compute_spatial_aware_new_metric(
                diff_squeezed,
                x_current_squeezed=x_current_squeezed,
                alpha=1.0, gamma=1.0, beta=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "frequency":
            # 频率域复杂度：仅edge/texture + fine details（无残差项）
            metric = self._compute_frequency_metric(
                diff_squeezed,
                alpha=1.0, gamma=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "noise":
            # 噪声残差复杂度：仅residual noise项
            if sample is None:
                raise ValueError("noise metric requires sample (x_current) to be provided")
            x_current_squeezed = sample.squeeze(0).permute(1, 2, 0)
            metric = self._compute_noise_metric(
                diff_squeezed,
                x_current_squeezed=x_current_squeezed,
                beta=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "random":
            # 纯随机复杂度：返回均匀分布的随机值
            H, W = diff_squeezed.shape[0], diff_squeezed.shape[1]
            metric = torch.rand(H, W, device=diff_squeezed.device, dtype=diff_squeezed.dtype)  # (H, W)
        else:
            raise ValueError(f"Unknown metric: {ras_manager.MANAGER.metric}")

        # 聚合到patch级别
        metric_patches = metric.view(
            num_h, patch_size,
            num_w, patch_size
        ).transpose(-2, -3).mean(-1).mean(-1).view(-1)  # (num_patches,)

        actual_clusters = self.actual_num_clusters if self.actual_num_clusters is not None else self.num_clusters

        # 聚合到簇级别
        cluster_scores = torch.zeros(
            actual_clusters,
            device=diff.device
        )

        cluster_labels_tensor = torch.from_numpy(cluster_labels).to(diff.device)

        for k in range(actual_clusters):
            mask = (cluster_labels_tensor == k)
            if mask.sum() > 0:
                # 使用top-50%均值（更鲁棒）
                cluster_patch_scores = metric_patches[mask]
                top_k = max(1, len(cluster_patch_scores) // 2)
                cluster_scores[k] = torch.topk(cluster_patch_scores, top_k).values.mean()

        return cluster_scores

    def compute_complexity_from_x_pred(self, x_pred, cluster_labels, height, width, sample=None):
        """
        基于x_pred（预测样本）计算复杂度分数

        x_pred的复杂度通过其空间梯度或变异性来衡量

        Args:
            x_pred: 预测的下一步样本 (B, C, H, W)
            cluster_labels: (num_patches,) patch到簇的映射
            height, width: latent空间的高宽
            sample: 当前sample (B, C, H, W)，用于spatial_aware_new的残差项计算

        Returns:
            cluster_scores: (num_clusters,) tensor
        """
        patch_size = ras_manager.MANAGER.patch_size
        num_h = height // patch_size
        num_w = width // patch_size

        # 计算x_pred在通道维度上的统计特性
        x_pred_squeezed = x_pred.squeeze(0).permute(1, 2, 0)  # (H, W, C)

        if ras_manager.MANAGER.metric == "std":
            # 计算每个位置跨通道的标准差
            metric = torch.std(x_pred_squeezed, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "l2norm":
            # 计算每个位置的L2范数
            metric = torch.norm(x_pred_squeezed, p=2, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "spatial_aware":
            # 空间感知复杂度：结合方差、梯度和细节
            metric = self._compute_spatial_aware_metric(x_pred_squeezed)  # (H, W)
        elif ras_manager.MANAGER.metric == "spatial_aware_new":
            # 新的空间感知复杂度公式
            x_current_squeezed = sample.squeeze(0).permute(1, 2, 0) if sample is not None else None
            metric = self._compute_spatial_aware_new_metric(
                x_pred_squeezed,
                x_current_squeezed=x_current_squeezed,
                alpha=1.0, gamma=1.0, beta=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "frequency":
            # 频率域复杂度：仅edge/texture + fine details（无残差项）
            metric = self._compute_frequency_metric(
                x_pred_squeezed,
                alpha=1.0, gamma=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "noise":
            # 噪声残差复杂度：仅residual noise项
            if sample is None:
                raise ValueError("noise metric requires sample (x_current) to be provided")
            x_current_squeezed = sample.squeeze(0).permute(1, 2, 0)
            metric = self._compute_noise_metric(
                x_pred_squeezed,
                x_current_squeezed=x_current_squeezed,
                beta=1.0
            )  # (H, W)
        elif ras_manager.MANAGER.metric == "random":
            # 纯随机复杂度：返回均匀分布的随机值
            H, W = x_pred_squeezed.shape[0], x_pred_squeezed.shape[1]
            metric = torch.rand(H, W, device=x_pred_squeezed.device, dtype=x_pred_squeezed.dtype)  # (H, W)
        else:
            raise ValueError(f"Unknown metric: {ras_manager.MANAGER.metric}")

        # 聚合到patch级别
        metric_patches = metric.view(
            num_h, patch_size,
            num_w, patch_size
        ).transpose(-2, -3).mean(-1).mean(-1).view(-1)  # (num_patches,)

        actual_clusters = self.actual_num_clusters if self.actual_num_clusters is not None else self.num_clusters

        # 聚合到簇级别
        cluster_scores = torch.zeros(
            actual_clusters,
            device=x_pred.device
        )

        cluster_labels_tensor = torch.from_numpy(cluster_labels).to(x_pred.device)

        for k in range(actual_clusters):
            mask = (cluster_labels_tensor == k)
            if mask.sum() > 0:
                # 使用top-50%均值（更鲁棒）
                cluster_patch_scores = metric_patches[mask]
                top_k = max(1, len(cluster_patch_scores) // 2)
                cluster_scores[k] = torch.topk(cluster_patch_scores, top_k).values.mean()

        return cluster_scores

    def _build_cluster_graph_metadata_legacy(self, labels_t, nhop_neighborhoods, num_patches, device):
        # Per-cluster key index lists — used for full-step per-cluster SDPA
        cluster_key_indices = {}
        cluster_far_cluster_ids = {}
        cluster_far_cluster_ids_cpu = {}
        cluster_local_block_indices = []
        cluster_local_key_spans = {}
        sorted_cluster_ids = sorted(nhop_neighborhoods.keys())
        all_cluster_ids = torch.tensor(
            sorted_cluster_ids,
            device=device,
            dtype=torch.long,
        )
        cluster_local_key_span_starts = torch.full(
            (all_cluster_ids.numel(),),
            -1,
            device=device,
            dtype=torch.long,
        )
        cluster_local_key_span_lengths = torch.zeros(
            all_cluster_ids.numel(),
            device=device,
            dtype=torch.long,
        )
        cluster_local_cluster_mask = torch.zeros(
            all_cluster_ids.numel(),
            all_cluster_ids.numel(),
            device=device,
            dtype=torch.bool,
        )
        for row, c_id in enumerate(sorted_cluster_ids):
            neighbor_clusters = nhop_neighborhoods[c_id]
            cluster_far_cluster_ids_cpu[c_id] = tuple(
                cid for cid in sorted_cluster_ids if cid not in neighbor_clusters
            )
            k_mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
            for nc in neighbor_clusters:
                k_mask |= (labels_t == nc)
            key_indices = torch.where(k_mask)[0]
            cluster_key_indices[c_id] = torch.sort(
                ras_manager.MANAGER.map_patch_indices_to_cluster_order(key_indices)
            ).values
            neighbor_ids = torch.tensor(
                sorted(neighbor_clusters),
                device=device,
                dtype=torch.long,
            )
            neighbor_compact = torch.searchsorted(all_cluster_ids, neighbor_ids)
            cluster_local_cluster_mask[row, neighbor_compact] = True
            cluster_local_block_indices.append(neighbor_compact)
            if neighbor_compact.numel() > 0:
                first_block = int(neighbor_compact[0].item())
                last_block = int(neighbor_compact[-1].item())
                block_count = int(neighbor_compact.numel())
                if last_block - first_block + 1 == block_count:
                    span_start = int(ras_manager.MANAGER.cluster_block_starts[first_block].item())
                    span_end = int(ras_manager.MANAGER.cluster_block_ends[last_block].item())
                    cluster_local_key_spans[c_id] = (span_start, span_end)
                    cluster_local_key_span_starts[row] = span_start
                    cluster_local_key_span_lengths[row] = span_end - span_start
                else:
                    cluster_local_key_spans[c_id] = None
            else:
                cluster_local_key_spans[c_id] = None
            far_mask = ~torch.isin(all_cluster_ids, neighbor_ids)
            cluster_far_cluster_ids[c_id] = all_cluster_ids[far_mask]
        max_local_blocks = max((int(indices.numel()) for indices in cluster_local_block_indices), default=0)
        if max_local_blocks > 0:
            cluster_local_block_indices_padded = torch.zeros(
                all_cluster_ids.numel(),
                max_local_blocks,
                device=device,
                dtype=torch.long,
            )
            cluster_local_block_counts = torch.zeros(
                all_cluster_ids.numel(),
                device=device,
                dtype=torch.long,
            )
            for row, block_indices in enumerate(cluster_local_block_indices):
                block_count = int(block_indices.numel())
                cluster_local_block_counts[row] = block_count
                if block_count > 0:
                    cluster_local_block_indices_padded[row, :block_count] = block_indices
        else:
            cluster_local_block_indices_padded = torch.empty(
                all_cluster_ids.numel(),
                0,
                device=device,
                dtype=torch.long,
            )
            cluster_local_block_counts = torch.zeros(
                all_cluster_ids.numel(),
                device=device,
                dtype=torch.long,
            )
        ras_manager.MANAGER.cluster_key_indices = cluster_key_indices
        ras_manager.MANAGER.cluster_far_cluster_ids = cluster_far_cluster_ids
        self._cluster_far_ids_cpu_cache = (cluster_far_cluster_ids, cluster_far_cluster_ids_cpu)
        ras_manager.MANAGER.cluster_local_cluster_mask = cluster_local_cluster_mask
        ras_manager.MANAGER.cluster_local_block_indices_padded = cluster_local_block_indices_padded
        ras_manager.MANAGER.cluster_local_block_counts = cluster_local_block_counts
        ras_manager.MANAGER.cluster_local_block_max_count = max_local_blocks
        ras_manager.MANAGER.cluster_local_key_spans = cluster_local_key_spans
        ras_manager.MANAGER.cluster_local_key_span_starts = cluster_local_key_span_starts
        ras_manager.MANAGER.cluster_local_key_span_lengths = cluster_local_key_span_lengths


    def _build_cluster_graph_metadata_cpu(self, nhop_neighborhoods, device):
        """Build the fixed graph in cluster order with two batched uploads.

        Block ranges already contain CPU integers from the stable permutation.
        Concatenating those sorted ranges is exactly the legacy GPU mask->where
        ->inverse permutation->sort result, without one synchronization per row.
        """
        manager = ras_manager.MANAGER
        cluster_ids = sorted(nhop_neighborhoods)
        if set(cluster_ids) != set(manager.cluster_block_ranges):
            raise ValueError("Graph and cluster permutation must contain the same labels")
        compact = {cid: row for row, cid in enumerate(cluster_ids)}
        local_keys, far_ids, blocks, spans = {}, {}, {}, {}
        mask = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=np.bool_)
        starts = np.full(len(cluster_ids), -1, dtype=np.int64)
        lengths = np.zeros(len(cluster_ids), dtype=np.int64)
        counts = np.zeros(len(cluster_ids), dtype=np.int64)
        for row, cid in enumerate(cluster_ids):
            neighbors = sorted(nhop_neighborhoods[cid])
            if any(neighbor not in compact for neighbor in neighbors):
                raise ValueError("Graph contains a neighbor outside the partition")
            block_ids = np.asarray([compact[neighbor] for neighbor in neighbors], dtype=np.int64)
            blocks[cid] = block_ids
            mask[row, block_ids] = True
            counts[row] = len(block_ids)
            local_keys[cid] = np.concatenate([
                np.arange(*manager.cluster_block_ranges[neighbor], dtype=np.int64)
                for neighbor in neighbors
            ]) if neighbors else np.empty(0, dtype=np.int64)
            far_ids[cid] = tuple(other for other in cluster_ids if other not in nhop_neighborhoods[cid])
            if len(block_ids) and int(block_ids[-1] - block_ids[0] + 1) == len(block_ids):
                start = manager.cluster_block_ranges[neighbors[0]][0]
                end = manager.cluster_block_ranges[neighbors[-1]][1]
                spans[cid] = (start, end)
                starts[row], lengths[row] = start, end - start
            else:
                spans[cid] = None
        max_blocks = max((len(indices) for indices in blocks.values()), default=0)
        padded = np.zeros((len(cluster_ids), max_blocks), dtype=np.int64)
        for row, cid in enumerate(cluster_ids):
            padded[row, :len(blocks[cid])] = blocks[cid]

        entries = [("keys", cid, local_keys[cid]) for cid in cluster_ids]
        entries += [("far", cid, np.asarray(far_ids[cid], dtype=np.int64)) for cid in cluster_ids]
        entries += [("starts", None, starts), ("lengths", None, lengths),
                    ("padded", None, padded), ("counts", None, counts)]
        storage = torch.from_numpy(np.concatenate([array.reshape(-1) for _, _, array in entries])).to(device)
        views = {"keys": {}, "far": {}}
        offset = 0
        for field, cid, array in entries:
            view = storage[offset:offset + array.size].view(array.shape)
            if cid is None:
                views[field] = view
            else:
                views[field][cid] = view
            offset += array.size
        manager.cluster_key_indices = views["keys"]
        manager.cluster_far_cluster_ids = views["far"]
        self._cluster_far_ids_cpu_cache = (views["far"], far_ids)
        manager.cluster_local_cluster_mask = torch.from_numpy(mask).to(device)
        manager.cluster_local_block_indices_padded = views["padded"]
        manager.cluster_local_block_counts = views["counts"]
        manager.cluster_local_block_max_count = max_blocks
        manager.cluster_local_key_spans = spans
        manager.cluster_local_key_span_starts = views["starts"]
        manager.cluster_local_key_span_lengths = views["lengths"]

    def _prepare_cluster_attention_mask(self, nhop_neighborhoods, num_patches, device):
        """Keep an optional dense fallback available without allocating it eagerly."""
        manager = ras_manager.MANAGER
        manager.cluster_attention_mask_factory = None
        manager.cluster_attention_mask_density = None
        labels = self.cluster_labels.copy()
        neighborhoods = {int(cid): frozenset(neighbors) for cid, neighbors in nhop_neighborhoods.items()}
        permutation = manager.cluster_perm

        def materialize():
            mask = self._build_patch_attention_mask(labels, neighborhoods, num_patches, device=device)
            return mask.index_select(0, permutation).index_select(1, permutation)

        # A missing getter means this caller cannot guarantee dense fallback;
        # retain the eager implementation in that case, regardless of the flag.
        if self.defer_dense_cluster_mask and callable(getattr(manager, "get_cluster_attention_mask", None)):
            ranges = manager.cluster_block_ranges
            true_count = sum(
                (ranges[cid][1] - ranges[cid][0]) * sum(
                    ranges[neighbor][1] - ranges[neighbor][0] for neighbor in neighbors
                ) for cid, neighbors in neighborhoods.items()
            )
            manager.cluster_attention_mask_density = float(np.float32(true_count / (num_patches * num_patches)))
            manager.cluster_attention_mask = None
            manager.cluster_attention_mask_factory = materialize
        else:
            manager.cluster_attention_mask = materialize()

    def _cluster_graph_density(self):
        manager = ras_manager.MANAGER
        density = getattr(manager, "cluster_attention_mask_density", None)
        return density if density is not None else self._attention_mask_density(manager.cluster_attention_mask)

    @staticmethod
    def _build_cluster_adjacency_graph_cpu(cluster_labels, num_h, num_w):
        """Vectorize four edge directions of the undirected 8-neighbor graph."""
        labels = np.asarray(cluster_labels).reshape(num_h, num_w)
        ids, compact_flat = np.unique(labels, return_inverse=True)
        compact = compact_flat.reshape(num_h, num_w)
        pairs = (
            (compact[:, :-1], compact[:, 1:]),
            (compact[:-1, :], compact[1:, :]),
            (compact[:-1, :-1], compact[1:, 1:]),
            (compact[:-1, 1:], compact[1:, :-1]),
        )
        encoded_edges = []
        for left, right in pairs:
            different = left != right
            first, second = left[different], right[different]
            encoded_edges.append(np.minimum(first, second) * len(ids) + np.maximum(first, second))
        graph = {int(cid): set() for cid in ids}
        for edge in np.unique(np.concatenate(encoded_edges)):
            first, second = divmod(int(edge), len(ids))
            a, b = int(ids[first]), int(ids[second])
            graph[a].add(b)
            graph[b].add(a)
        return graph

    def _build_cluster_adjacency_graph(self, cluster_labels, num_h, num_w):
        """
        Build a spatial adjacency graph between clusters.

        Two clusters are connected by an edge if any patch from one cluster
        is an 8-neighbor of any patch from the other cluster.

        Args:
            cluster_labels: flat numpy array of shape [num_h * num_w]
            num_h, num_w: patch grid dimensions

        Returns:
            adj: dict {cluster_id: set of neighbor cluster_ids}
        """
        if self.enable_cpu_graph_metadata:
            return self._build_cluster_adjacency_graph_cpu(cluster_labels, num_h, num_w)
        adj = {}
        for idx in range(num_h * num_w):
            i, j = divmod(idx, num_w)
            c = int(cluster_labels[idx])
            if c not in adj:
                adj[c] = set()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < num_h and 0 <= nj < num_w:
                        nc = int(cluster_labels[ni * num_w + nj])
                        if nc != c:
                            adj[c].add(nc)
                            if nc not in adj:
                                adj[nc] = set()
                            adj[nc].add(c)
        return adj

    def _compute_nhop_neighborhoods(self, adj_graph, n_hops=2):
        """
        For each cluster, compute the set of clusters reachable within n_hops
        via BFS, including the cluster itself.

        Args:
            adj_graph: dict {cluster_id: set of neighbor cluster_ids}
            n_hops: number of hops (default 2)

        Returns:
            neighborhoods: dict {cluster_id: frozenset of reachable cluster_ids}
        """
        neighborhoods = {}
        for start in adj_graph:
            visited = {start}
            frontier = {start}
            for _ in range(n_hops):
                next_frontier = set()
                for node in frontier:
                    for neighbor in adj_graph.get(node, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier
                if not frontier:
                    break
            neighborhoods[start] = frozenset(visited)
        return neighborhoods

    def _build_patch_attention_mask(self, cluster_labels, nhop_neighborhoods, num_patches, device):
        """
        Build a boolean attention mask of shape [num_patches, num_patches].

        For query patch q in cluster c, mask[q, k] = True iff patch k belongs
        to any cluster within the n-hop neighborhood of c.

        All patches in the same cluster share the same mask row, so we
        process cluster by cluster for efficiency.

        Args:
            cluster_labels: flat numpy array [num_patches]
            nhop_neighborhoods: dict {cluster_id: frozenset of cluster_ids}
            num_patches: total number of patches
            device: torch device

        Returns:
            mask: bool tensor [num_patches, num_patches] on device
        """
        labels = torch.from_numpy(cluster_labels).long()  # [num_patches], on CPU
        mask = torch.zeros(num_patches, num_patches, dtype=torch.bool)  # CPU first

        unique_clusters = labels.unique().tolist()

        # Pre-compute per-cluster membership masks (CPU, reused below)
        cluster_member_mask = {c: (labels == c) for c in unique_clusters}

        for c_id in unique_clusters:
            q_mask = cluster_member_mask[c_id]  # [num_patches] bool

            # Union of all key patches in the n-hop neighborhood
            neighbor_clusters = nhop_neighborhoods.get(c_id, frozenset([c_id]))
            k_mask = torch.zeros(num_patches, dtype=torch.bool)
            for nc in neighbor_clusters:
                if nc in cluster_member_mask:
                    k_mask |= cluster_member_mask[nc]

            # All query patches in cluster c_id get the same key mask
            mask[q_mask] = k_mask

        return mask.to(device)

    def _compute_cluster_window_size(self, cluster_labels, nhop_neighborhoods, num_h, num_w):
        """
        Compute the 1D window half-size from the maximum
        bounding rectangle of any cluster's n-hop neighborhood union.

        For a 2D patch grid stored in row-major order, two patches at
        positions (r1,c1) and (r2,c2) have 1D distance |(r2-r1)*num_w + (c2-c1)|.
        To guarantee all patches within a bounding box of H_box×W_box are
        reachable, the required 1D window half-size is:
            (H_box - 1) * num_w + (W_box - 1)

        Returns:
            window_half (int): 1D window half-size
            max_h_box (int): maximum bounding box height across all clusters
            max_w_box (int): maximum bounding box width across all clusters
        """
        max_h_box, max_w_box = 0, 0

        # Pre-compute patch row/col arrays per cluster for efficiency
        cluster_rows = {}
        cluster_cols = {}
        for c_id in np.unique(cluster_labels):
            patch_indices = np.where(cluster_labels == c_id)[0]
            cluster_rows[c_id] = patch_indices // num_w
            cluster_cols[c_id] = patch_indices % num_w

        for c_id, neighbor_clusters in nhop_neighborhoods.items():
            all_rows = np.concatenate([cluster_rows[nc] for nc in neighbor_clusters if nc in cluster_rows])
            all_cols = np.concatenate([cluster_cols[nc] for nc in neighbor_clusters if nc in cluster_cols])
            if len(all_rows) == 0:
                continue
            h_box = int(all_rows.max() - all_rows.min() + 1)
            w_box = int(all_cols.max() - all_cols.min() + 1)
            max_h_box = max(max_h_box, h_box)
            max_w_box = max(max_w_box, w_box)

        window_half = (max_h_box - 1) * num_w + (max_w_box - 1)
        return window_half, max_h_box, max_w_box

    def _select_cluster_anchor_patches(self, patch_indices, patch_scores, num_w):
        """
        选择每个 cluster 的固定远端 anchor，全部都是真实 patch token:
        1. center anchor: 最接近 cluster 几何中心的真实 patch
        2. salient anchor: cluster 内复杂度最高的真实 patch
        3. 额外 anchor: 做轻量 farthest-point spatial coverage
        """
        if patch_indices.numel() == 0:
            return patch_indices

        requested_anchor_count = int(ras_manager.MANAGER.cluster_remote_anchor_count)
        if requested_anchor_count <= 0:
            return patch_indices[:0]

        anchor_count = min(requested_anchor_count, int(patch_indices.numel()))

        rows = torch.div(patch_indices, num_w, rounding_mode="floor").to(dtype=torch.float32)
        cols = torch.remainder(patch_indices, num_w).to(dtype=torch.float32)
        center_row = rows.mean()
        center_col = cols.mean()
        center_dist = (rows - center_row).square() + (cols - center_col).square()
        center_patch = patch_indices[center_dist.argmin()]

        if patch_indices.numel() == 1 or anchor_count == 1:
            return center_patch.unsqueeze(0)

        cluster_scores = patch_scores.index_select(0, patch_indices)
        selected = [center_patch]

        masked_scores = cluster_scores.masked_fill(patch_indices == center_patch, float("-inf"))
        selected.append(patch_indices[masked_scores.argmax()])

        spatial_points = torch.stack([rows, cols], dim=-1)
        while len(selected) < anchor_count:
            selected_tensor = torch.stack(selected).to(device=patch_indices.device)
            selected_mask = (patch_indices[:, None] == selected_tensor[None, :]).any(dim=1)
            selected_positions = torch.stack(
                [
                    torch.div(selected_tensor, num_w, rounding_mode="floor").to(dtype=torch.float32),
                    torch.remainder(selected_tensor, num_w).to(dtype=torch.float32),
                ],
                dim=-1,
            )
            min_dist = torch.cdist(spatial_points[None], selected_positions[None]).squeeze(0).min(dim=1).values
            coverage_score = min_dist + 0.01 * cluster_scores.to(dtype=torch.float32)
            coverage_score = coverage_score.masked_fill(selected_mask, float("-inf"))
            selected.append(patch_indices[coverage_score.argmax()])

        anchors = torch.stack(selected)
        return torch.unique(anchors, sorted=True)

    def _compute_remote_anchor_patch_features(self, anchor_source, patch_scores, height, width):
        from sdit.policy import token_coordinates
        return token_coordinates(height // ras_manager.MANAGER.patch_size,
                                 width // ras_manager.MANAGER.patch_size,
                                 patch_scores.device)

    def _assign_anchor_groups(self, patch_indices, anchor_patch_indices, patch_features):
        if anchor_patch_indices.numel() == 0:
            empty_indices = torch.empty(0, 0, device=patch_indices.device, dtype=torch.long)
            empty_mask = torch.empty(0, 0, device=patch_indices.device, dtype=torch.bool)
            empty_weights = torch.empty(0, device=patch_indices.device, dtype=torch.float32)
            return empty_indices, empty_mask, empty_weights

        if anchor_patch_indices.numel() >= patch_indices.numel():
            group_indices = patch_indices.view(-1, 1)
            group_mask = torch.ones_like(group_indices, dtype=torch.bool)
            group_weights = torch.ones(anchor_patch_indices.numel(), device=patch_indices.device, dtype=torch.float32)
            return group_indices, group_mask, group_weights

        cluster_features = patch_features.index_select(0, patch_indices)
        anchor_features = patch_features.index_select(0, anchor_patch_indices)
        distances = torch.cdist(cluster_features[None], anchor_features[None]).squeeze(0)
        assignments = distances.argmin(dim=1)
        group_weights = torch.bincount(assignments, minlength=anchor_patch_indices.numel()).to(dtype=torch.float32)
        max_group_size = max(1, int(group_weights.max().item()))
        group_indices = torch.zeros(
            anchor_patch_indices.numel(),
            max_group_size,
            device=patch_indices.device,
            dtype=torch.long,
        )
        group_mask = torch.zeros_like(group_indices, dtype=torch.bool)

        for anchor_idx in range(anchor_patch_indices.numel()):
            members = patch_indices[assignments == anchor_idx]
            if members.numel() == 0:
                members = anchor_patch_indices[anchor_idx].view(1)
                group_weights[anchor_idx] = 1.0
            member_count = min(int(members.numel()), max_group_size)
            group_indices[anchor_idx, :member_count] = members[:member_count]
            group_mask[anchor_idx, :member_count] = True

        return group_indices, group_mask, group_weights

    def _select_cluster_anchor_patches_coreset(self, patch_indices, patch_scores, patch_features):
        from sdit.policy import spatial_anchor_groups
        return spatial_anchor_groups(patch_indices, patch_features,
                                     ras_manager.MANAGER.cluster_remote_anchor_count)

    def _pad_and_concat_anchor_groups(self, group_blocks, mask_blocks, device):
        from ras.utils.lazy_anchor_metadata import pad_and_concat_anchor_groups
        return pad_and_concat_anchor_groups(group_blocks, mask_blocks, device)

    def _build_remote_anchor_metadata(self, cluster_labels_tensor, patch_scores, num_w, anchor_source=None, height=None, width=None):
        """
        为每个 query cluster 预拼接远端 anchor 索引。

        近端 1-hop exact kept 仍然完全由 cluster_key_indices 决定。
        这里只为远端 cluster 提供极少量固定的真实 token，避免在线打分开销。
        """
        manager = ras_manager.MANAGER
        manager.cluster_pack_metadata_cpu = None
        requested_anchor_count = int(getattr(manager, "cluster_remote_anchor_count", 0))
        use_value_coreset = bool(getattr(manager, "enable_cluster_remote_anchor_value_coreset", False))
        lazy_fallback_metadata = (
            bool(getattr(self, "enable_lazy_anchor_metadata", True))
            and use_value_coreset and requested_anchor_count == 1
        )
        relevance_routing_effective = (
            bool(getattr(manager, "enable_cluster_relevance_routing", True))
            and float(getattr(manager, "cluster_relevance_topk_ratio", 0.0)) > 0.0
        )

        def _stats(values, prefix):
            if not values:
                return {
                    f"{prefix}_min": 0,
                    f"{prefix}_avg": 0.0,
                    f"{prefix}_max": 0,
                    f"{prefix}_sum": 0,
                }
            return {
                f"{prefix}_min": min(values),
                f"{prefix}_avg": float(sum(values)) / len(values),
                f"{prefix}_max": max(values),
                f"{prefix}_sum": sum(values),
            }

        def _record_empty_anchor_summary(reason):
            if not getattr(manager, "enable_cluster_anchor_diagnostics", False):
                return
            summary = {
                "reason": reason,
                "requested_anchor_count": requested_anchor_count,
                "use_value_coreset": use_value_coreset,
                "relevance_routing_effective": relevance_routing_effective,
                "all_anchor_count": 0,
            }
            summary.update(_stats([], "source_anchor_count"))
            summary.update(_stats([], "query_remote_anchor_count"))
            summary.update(_stats([], "query_source_cluster_count"))
            summary.update(_stats([], "anchor_group_size"))
            manager.set_cluster_anchor_metadata_summary(summary)

        if (
            manager.cluster_far_cluster_ids is None
            or not manager.has_cluster_permutation()
            or manager.cluster_block_ranges is None
        ):
            manager.cluster_remote_anchor_indices = None
            manager.cluster_remote_anchor_weights = None
            manager.cluster_remote_anchor_group_indices = None
            manager.cluster_remote_anchor_group_mask = None
            manager.cluster_remote_anchor_source_cluster_ids = None
            manager.cluster_remote_anchor_source_indices = None
            manager.cluster_remote_anchor_source_weights = None
            manager.cluster_remote_anchor_source_group_indices = None
            manager.cluster_remote_anchor_source_group_mask = None
            manager.cluster_remote_anchor_all_indices = None
            manager.cluster_remote_anchor_all_weights = None
            manager.cluster_remote_anchor_all_group_indices = None
            manager.cluster_remote_anchor_all_group_mask = None
            manager.cluster_remote_anchor_packed_positions = None
            manager.cluster_remote_anchor_merged_key_indices = None
            manager.cluster_remote_anchor_token_to_packed_anchor = None
            manager.cluster_remote_anchor_packed_counts = None
            manager.cluster_remote_anchor_group_token_order = None
            manager.cluster_remote_anchor_group_lengths = None
            manager.cluster_remote_anchor_group_token_order_is_cluster_order = False
            manager.cluster_remote_anchor_all_logit_bias = None
            manager.cluster_remote_anchor_metadata_version += 1
            _record_empty_anchor_summary("metadata_unavailable")
            return

        if requested_anchor_count <= 0:
            manager.cluster_remote_anchor_indices = None
            manager.cluster_remote_anchor_weights = None
            manager.cluster_remote_anchor_group_indices = None
            manager.cluster_remote_anchor_group_mask = None
            manager.cluster_remote_anchor_source_cluster_ids = None
            manager.cluster_remote_anchor_source_indices = None
            manager.cluster_remote_anchor_source_weights = None
            manager.cluster_remote_anchor_source_group_indices = None
            manager.cluster_remote_anchor_source_group_mask = None
            manager.cluster_remote_anchor_all_indices = None
            manager.cluster_remote_anchor_all_weights = None
            manager.cluster_remote_anchor_all_group_indices = None
            manager.cluster_remote_anchor_all_group_mask = None
            manager.cluster_remote_anchor_packed_positions = None
            manager.cluster_remote_anchor_merged_key_indices = None
            manager.cluster_remote_anchor_token_to_packed_anchor = None
            manager.cluster_remote_anchor_packed_counts = None
            manager.cluster_remote_anchor_group_token_order = None
            manager.cluster_remote_anchor_group_lengths = None
            manager.cluster_remote_anchor_group_token_order_is_cluster_order = False
            manager.cluster_remote_anchor_all_logit_bias = None
            manager.cluster_remote_anchor_metadata_version += 1
            _record_empty_anchor_summary("anchor_count_zero")
            return

        patch_features = None
        if use_value_coreset:
            patch_features = self._compute_remote_anchor_patch_features(anchor_source, patch_scores, height, width)

        per_cluster_anchor_indices = {}
        per_cluster_anchor_weights = {}
        per_cluster_anchor_group_indices = {}
        per_cluster_anchor_group_mask = {}
        empty = torch.empty(0, device=patch_scores.device, dtype=torch.long)
        empty_weights = torch.empty(0, device=patch_scores.device, dtype=patch_scores.dtype)
        empty_group_indices = torch.empty(0, 0, device=patch_scores.device, dtype=torch.long)
        empty_group_mask = torch.empty(0, 0, device=patch_scores.device, dtype=torch.bool)

        # Reuse the partition's raster-ordered members without scalar GPU reads.
        # Direct external callers retain the original tensor-based path.
        partition = self._partition_metadata_cache if self.enable_partition_metadata_cache else None
        if partition is not None and partition["labels_tensor"] is not cluster_labels_tensor:
            partition = None
        _all_cluster_ids_sorted = sorted(manager.cluster_block_ranges.keys())
        if partition is None:
            _sorted_patch_order = torch.argsort(cluster_labels_tensor, stable=True)
            _sorted_patch_labels = cluster_labels_tensor[_sorted_patch_order]
            _cid_tensor = torch.tensor(
                _all_cluster_ids_sorted, device=cluster_labels_tensor.device, dtype=cluster_labels_tensor.dtype
            )
            _bounds_lo = torch.searchsorted(_sorted_patch_labels, _cid_tensor, right=False)
            _bounds_hi = torch.searchsorted(_sorted_patch_labels, _cid_tensor, right=True)

        for _ci, cluster_id in enumerate(_all_cluster_ids_sorted):
            if partition is None:
                _lo, _hi = int(_bounds_lo[_ci]), int(_bounds_hi[_ci])
                patch_indices = _sorted_patch_order[_lo:_hi]
            else:
                patch_indices = partition["members_device"].get(cluster_id, empty)
            if patch_indices.numel() == 0:
                per_cluster_anchor_indices[int(cluster_id)] = empty
                per_cluster_anchor_weights[int(cluster_id)] = empty_weights
                per_cluster_anchor_group_indices[int(cluster_id)] = empty_group_indices
                per_cluster_anchor_group_mask[int(cluster_id)] = empty_group_mask
                continue

            if use_value_coreset:
                anchor_patch_indices, group_indices, group_mask, group_weights = self._select_cluster_anchor_patches_coreset(
                    patch_indices,
                    patch_scores,
                    patch_features,
                )
                per_cluster_anchor_indices[int(cluster_id)] = manager.map_patch_indices_to_cluster_order(anchor_patch_indices)
                per_cluster_anchor_weights[int(cluster_id)] = group_weights.to(device=patch_scores.device, dtype=patch_scores.dtype)
                per_cluster_anchor_group_indices[int(cluster_id)] = manager.map_patch_indices_to_cluster_order(group_indices)
                per_cluster_anchor_group_mask[int(cluster_id)] = group_mask
            else:
                anchor_patch_indices = self._select_cluster_anchor_patches(
                    patch_indices,
                    patch_scores,
                    num_w,
                )
                anchor_count = max(1, int(anchor_patch_indices.numel()))
                represented_token_count = patch_indices.numel() / anchor_count
                per_cluster_anchor_indices[int(cluster_id)] = torch.sort(
                    manager.map_patch_indices_to_cluster_order(anchor_patch_indices)
                ).values
                per_cluster_anchor_weights[int(cluster_id)] = torch.full(
                    (anchor_count,),
                    float(represented_token_count),
                    device=patch_scores.device,
                    dtype=patch_scores.dtype,
                )
                per_cluster_anchor_group_indices[int(cluster_id)] = empty_group_indices
                per_cluster_anchor_group_mask[int(cluster_id)] = empty_group_mask

        cluster_remote_anchor_indices = {}
        cluster_remote_anchor_weights = {}
        cluster_remote_anchor_group_indices = {}
        cluster_remote_anchor_group_mask = {}
        cluster_remote_anchor_source_cluster_ids = {}
        cluster_remote_anchor_packed_positions = {}
        cluster_remote_anchor_merged_key_indices = {}
        build_merged_key_indices = bool(getattr(manager, "enable_cluster_remote_anchor_merged_key_fast_path", False))
        source_anchor_positions = {}
        if use_value_coreset:
            all_anchor_blocks = []
            all_weight_blocks = []
            all_group_blocks = []
            all_group_mask_blocks = []
            cursor = 0
            for source_cluster_id in sorted(per_cluster_anchor_indices.keys()):
                anchor_idx = per_cluster_anchor_indices[int(source_cluster_id)]
                if anchor_idx is None or anchor_idx.numel() == 0:
                    source_anchor_positions[int(source_cluster_id)] = empty
                    continue
                anchor_count = int(anchor_idx.numel())
                positions = torch.arange(
                    cursor,
                    cursor + anchor_count,
                    device=patch_scores.device,
                    dtype=torch.long,
                )
                source_anchor_positions[int(source_cluster_id)] = positions
                cursor += anchor_count
                all_anchor_blocks.append(anchor_idx)
                all_weight_blocks.append(per_cluster_anchor_weights[int(source_cluster_id)])
                all_group_blocks.append(per_cluster_anchor_group_indices[int(source_cluster_id)])
                all_group_mask_blocks.append(per_cluster_anchor_group_mask[int(source_cluster_id)])

            if all_anchor_blocks:
                all_anchor_indices = torch.cat(all_anchor_blocks, dim=0)
                all_anchor_weights = torch.cat(all_weight_blocks, dim=0)
                all_group_indices, all_group_mask = self._pad_and_concat_anchor_groups(
                    all_group_blocks,
                    all_group_mask_blocks,
                    patch_scores.device,
                )
                token_to_packed_anchor = torch.full(
                    (cluster_labels_tensor.numel(),),
                    -1,
                    device=patch_scores.device,
                    dtype=torch.long,
                )
                packed_group_ids = torch.arange(
                    all_group_indices.shape[0],
                    device=patch_scores.device,
                    dtype=torch.long,
                )[:, None].expand_as(all_group_indices)
                token_to_packed_anchor.index_copy_(
                    0,
                    all_group_indices[all_group_mask],
                    packed_group_ids[all_group_mask],
                )
                group_lengths = all_group_mask.sum(dim=1).to(dtype=torch.long)
                group_token_order = all_group_indices[all_group_mask]
                expected_group_token_order = torch.arange(
                    cluster_labels_tensor.numel(),
                    device=patch_scores.device,
                    dtype=torch.long,
                )
                group_token_order_is_cluster_order = (
                    group_token_order.numel() == expected_group_token_order.numel()
                    and torch.equal(group_token_order, expected_group_token_order)
                    and manager.cluster_block_lengths is not None
                    and group_lengths.numel() == manager.cluster_block_lengths.numel()
                    and torch.equal(
                        group_lengths.to(device=manager.cluster_block_lengths.device),
                        manager.cluster_block_lengths,
                    )
                )
                packed_anchor_counts = group_lengths.to(dtype=patch_scores.dtype)
            else:
                all_anchor_indices = empty
                all_anchor_weights = empty_weights
                all_group_indices = empty_group_indices
                all_group_mask = empty_group_mask
                token_to_packed_anchor = torch.empty(0, device=patch_scores.device, dtype=torch.long)
                group_token_order = empty
                group_lengths = torch.empty(0, device=patch_scores.device, dtype=torch.long)
                group_token_order_is_cluster_order = False
                packed_anchor_counts = empty_weights
        else:
            all_anchor_indices = None
            all_anchor_weights = None
            all_group_indices = None
            all_group_mask = None
            token_to_packed_anchor = None
            group_token_order = None
            group_lengths = None
            group_token_order_is_cluster_order = False
            packed_anchor_counts = None

        for cluster_id, far_cluster_ids in manager.cluster_far_cluster_ids.items():
            anchor_blocks = []
            weight_blocks = []
            group_blocks = []
            group_mask_blocks = []
            source_cluster_blocks = []
            packed_position_blocks = []
            far_ids_cache = self._cluster_far_ids_cpu_cache if self.enable_partition_metadata_cache else None
            if far_ids_cache is not None and far_ids_cache[0] is manager.cluster_far_cluster_ids:
                far_cluster_ids_list = far_ids_cache[1][cluster_id]
            else:
                far_cluster_ids_list = far_cluster_ids.detach().cpu().tolist()
            for far_cluster_id in far_cluster_ids_list:
                anchor_idx = per_cluster_anchor_indices.get(int(far_cluster_id))
                if anchor_idx is not None and anchor_idx.numel() > 0:
                    anchor_blocks.append(anchor_idx)
                    weight_blocks.append(per_cluster_anchor_weights[int(far_cluster_id)])
                    source_cluster_blocks.append(int(far_cluster_id))
                    if use_value_coreset:
                        if not lazy_fallback_metadata:
                            group_blocks.append(per_cluster_anchor_group_indices[int(far_cluster_id)])
                            group_mask_blocks.append(per_cluster_anchor_group_mask[int(far_cluster_id)])
                        packed_position_blocks.append(source_anchor_positions[int(far_cluster_id)])

            if anchor_blocks:
                cluster_remote_anchor_indices[int(cluster_id)] = torch.cat(anchor_blocks, dim=0)
                cluster_remote_anchor_weights[int(cluster_id)] = torch.cat(weight_blocks, dim=0)
                if use_value_coreset:
                    if not lazy_fallback_metadata:
                        group_indices, group_mask = self._pad_and_concat_anchor_groups(
                            group_blocks,
                            group_mask_blocks,
                            patch_scores.device,
                        )
                        cluster_remote_anchor_group_indices[int(cluster_id)] = group_indices
                        cluster_remote_anchor_group_mask[int(cluster_id)] = group_mask
                    cluster_remote_anchor_source_cluster_ids[int(cluster_id)] = tuple(source_cluster_blocks)
                    cluster_remote_anchor_packed_positions[int(cluster_id)] = torch.cat(packed_position_blocks, dim=0)
                    local_key_indices = manager.cluster_key_indices.get(int(cluster_id)) if build_merged_key_indices else None
                    if build_merged_key_indices and local_key_indices is not None and not lazy_fallback_metadata:
                        cluster_remote_anchor_merged_key_indices[int(cluster_id)] = torch.cat(
                            [local_key_indices, cluster_remote_anchor_indices[int(cluster_id)]],
                            dim=0,
                        )
            else:
                cluster_remote_anchor_indices[int(cluster_id)] = empty
                cluster_remote_anchor_weights[int(cluster_id)] = empty_weights
                if use_value_coreset:
                    cluster_remote_anchor_group_indices[int(cluster_id)] = empty_group_indices
                    cluster_remote_anchor_group_mask[int(cluster_id)] = empty_group_mask
                    cluster_remote_anchor_source_cluster_ids[int(cluster_id)] = ()
                    cluster_remote_anchor_packed_positions[int(cluster_id)] = empty
                    local_key_indices = manager.cluster_key_indices.get(int(cluster_id)) if build_merged_key_indices else None
                    if build_merged_key_indices and local_key_indices is not None:
                        cluster_remote_anchor_merged_key_indices[int(cluster_id)] = local_key_indices

        if lazy_fallback_metadata:
            from ras.utils.lazy_anchor_metadata import LazyRemoteAnchorGroups, LazyMergedAnchorKeys
            fallback_groups = LazyRemoteAnchorGroups(
                cluster_remote_anchor_source_cluster_ids,
                per_cluster_anchor_group_indices, per_cluster_anchor_group_mask,
                patch_scores.device,
            )
            cluster_remote_anchor_group_indices = fallback_groups.indices
            cluster_remote_anchor_group_mask = fallback_groups.mask
            if build_merged_key_indices:
                cluster_remote_anchor_merged_key_indices = LazyMergedAnchorKeys(
                    manager.cluster_key_indices, cluster_remote_anchor_indices)

        manager.cluster_remote_anchor_indices = cluster_remote_anchor_indices
        manager.cluster_remote_anchor_weights = cluster_remote_anchor_weights
        manager.cluster_remote_anchor_group_indices = cluster_remote_anchor_group_indices if use_value_coreset else None
        manager.cluster_remote_anchor_group_mask = cluster_remote_anchor_group_mask if use_value_coreset else None
        manager.cluster_remote_anchor_source_cluster_ids = cluster_remote_anchor_source_cluster_ids if use_value_coreset else None
        manager.cluster_remote_anchor_source_indices = per_cluster_anchor_indices if use_value_coreset else None
        manager.cluster_remote_anchor_source_weights = per_cluster_anchor_weights if use_value_coreset else None
        manager.cluster_remote_anchor_source_group_indices = per_cluster_anchor_group_indices if use_value_coreset else None
        manager.cluster_remote_anchor_source_group_mask = per_cluster_anchor_group_mask if use_value_coreset else None
        manager.cluster_remote_anchor_all_indices = all_anchor_indices if use_value_coreset else None
        manager.cluster_remote_anchor_all_weights = all_anchor_weights if use_value_coreset else None
        manager.cluster_remote_anchor_all_group_indices = all_group_indices if use_value_coreset else None
        manager.cluster_remote_anchor_all_group_mask = all_group_mask if use_value_coreset else None
        manager.cluster_remote_anchor_packed_positions = cluster_remote_anchor_packed_positions if use_value_coreset else None
        manager.cluster_remote_anchor_merged_key_indices = (
            cluster_remote_anchor_merged_key_indices if use_value_coreset and build_merged_key_indices else None
        )
        manager.cluster_remote_anchor_token_to_packed_anchor = token_to_packed_anchor if use_value_coreset else None
        manager.cluster_remote_anchor_packed_counts = packed_anchor_counts if use_value_coreset else None
        manager.cluster_remote_anchor_group_token_order = group_token_order if use_value_coreset else None
        manager.cluster_remote_anchor_group_lengths = group_lengths if use_value_coreset else None
        manager.cluster_remote_anchor_group_token_order_is_cluster_order = (
            group_token_order_is_cluster_order if use_value_coreset else False
        )

        # Precompute logn bias for all packed anchors once at scheduling time,
        # so each attention block forward can skip the log() computation.
        _precomputed_logit_bias = None
        if (
            use_value_coreset
            and all_anchor_weights is not None
            and all_anchor_weights.numel() > 0
        ):
            _weight_mode = getattr(manager, "cluster_remote_anchor_weight_mode", "none").strip().lower()
            _alpha = float(getattr(manager, "cluster_remote_anchor_logit_bias_alpha", 1.0))
            if _weight_mode == "logn" and _alpha != 0.0:
                _precomputed_logit_bias = (
                    torch.log(all_anchor_weights.to(dtype=torch.float32).clamp_min(1.0)) * _alpha
                )
        manager.cluster_remote_anchor_all_logit_bias = _precomputed_logit_bias

        if getattr(manager, "enable_cluster_anchor_diagnostics", False):
            source_counts = [int(v.numel()) for v in per_cluster_anchor_indices.values()]
            query_counts = [int(v.numel()) for v in cluster_remote_anchor_indices.values()]
            if use_value_coreset:
                source_cluster_counts = [
                    len(v) for v in cluster_remote_anchor_source_cluster_ids.values()
                ]
                if all_anchor_weights is not None and all_anchor_weights.numel() > 0:
                    group_sizes = [
                        int(round(float(v)))
                        for v in all_anchor_weights.detach().cpu().reshape(-1).tolist()
                    ]
                else:
                    group_sizes = []
            else:
                source_cluster_counts = [
                    int(v.numel()) for v in manager.cluster_far_cluster_ids.values()
                ]
                weight_values = []
                for weight_block in per_cluster_anchor_weights.values():
                    if weight_block is not None and weight_block.numel() > 0:
                        weight_values.extend(weight_block.detach().cpu().reshape(-1).tolist())
                group_sizes = [int(round(float(v))) for v in weight_values]

            summary = {
                "reason": "ok",
                "requested_anchor_count": requested_anchor_count,
                "use_value_coreset": use_value_coreset,
                "relevance_routing_effective": relevance_routing_effective,
                "all_anchor_count": (
                    int(all_anchor_indices.numel())
                    if all_anchor_indices is not None
                    else int(sum(source_counts))
                ),
            }
            summary.update(_stats(source_counts, "source_anchor_count"))
            summary.update(_stats(query_counts, "query_remote_anchor_count"))
            summary.update(_stats(source_cluster_counts, "query_source_cluster_count"))
            summary.update(_stats(group_sizes, "anchor_group_size"))
            manager.set_cluster_anchor_metadata_summary(summary)

        manager.cluster_remote_anchor_metadata_version += 1
        self._publish_cluster_pack_metadata_cpu()

    def cluster_adaptive_selection(self, sample, x_pred, height, width, diff=None):
        """
        基于聚类的自适应patch选择（含8-邻域扩展）+ Patch级防饥饿机制

        流程:
        1. 空间感知聚类（每步或首次）
        2. 计算每个patch的复杂度分数
        3. 应用防饥饿机制（patch级别）
        4. 选择前X%簇（基于簇内patch平均分数）
        5. 对簇内patches进行8-邻域扩展
        6. 返回最终选中的patches

        Args:
            sample: 当前latent (B, C, H, W)
            x_pred: prediction tensor (B, C, H, W), used for clustering and scoring
            height, width: latent空间的高宽
            diff: 残差 (B, C, H, W)，可选，用于complexity_source="diff"

        Returns:
            latent_cached_indices: 被缓存（跳过）的latent索引
            other_patchified_indices: 被选中（更新）的patch索引
        """
        # ===== 总聚类时间测量开始 =====
        step_clustering_start = time.time()

        patch_size = ras_manager.MANAGER.patch_size
        num_h = height // patch_size
        num_w = width // patch_size
        num_patches = num_h * num_w
        cluster_layout_updated = False

        current_step = ras_manager.MANAGER.current_step
        should_recluster = (
            not self.clustered
            or self.recluster_every_step
            or (
                self.recluster_every_n_steps > 0
                and self.last_recluster_step is not None
                and current_step - self.last_recluster_step >= self.recluster_every_n_steps
            )
        )

        # Step 1: 聚类（每步、按间隔或仅首次）
        if should_recluster:
            # New partitions/images must never reuse old mask or host metadata.
            self._partition_metadata_cache = None
            self._cluster_mask_density_cache = None
            self._cluster_far_ids_cpu_cache = None
            ras_manager.MANAGER.cluster_pack_metadata_cpu = None
            prediction_source = getattr(ras_manager.MANAGER, "cluster_prediction_source", "legacy")
            prediction_source = prediction_source.strip().lower()
            clustering_input = sample if prediction_source == "legacy" else x_pred
            self.cluster_labels = self.spatial_aware_clustering(clustering_input, height, width)
            cluster_layout_updated = True
            is_first_cluster = not self.clustered
            self.last_recluster_step = current_step
            if not self.clustered:
                self.clustered = True
                actual_clusters = self.actual_num_clusters if self.actual_num_clusters is not None else self.num_clusters
                print(f"   🔸 完成空间感知聚类: {num_patches} patches → {actual_clusters} clusters")

                # 初始化 patch 级 drop_cnt（第一次聚类时）
                self.patch_drop_cnt = torch.zeros(num_patches, device=x_pred.device)
            else:
                actual_clusters = self.actual_num_clusters if self.actual_num_clusters is not None else self.num_clusters
                print(f"   🔸 重新聚类（sync后）: {num_patches} patches → {actual_clusters} clusters")
                # 重置 drop_cnt，避免旧计数影响新 cluster 布局
                self.patch_drop_cnt = torch.zeros(num_patches, device=x_pred.device)

            # Build cluster graph attention mask whenever clustering is (re)done
            if ras_manager.MANAGER.spatial_attention_enabled():
                labels_t = torch.from_numpy(self.cluster_labels).long().to(x_pred.device)
                # 每次重新聚类都强制重置 permutation，保证 cluster_key_indices 与 permutation 一致
                ras_manager.MANAGER.initialize_cluster_permutation(labels_t)

                n_hops = ras_manager.MANAGER.cluster_graph_attention_hops
                adj_graph = self._build_cluster_adjacency_graph(self.cluster_labels, num_h, num_w)
                nhop_neighborhoods = self._compute_nhop_neighborhoods(adj_graph, n_hops=n_hops)

                self._prepare_cluster_attention_mask(nhop_neighborhoods, num_patches, x_pred.device)

                if self.enable_cpu_graph_metadata:
                    self._build_cluster_graph_metadata_cpu(nhop_neighborhoods, x_pred.device)
                else:
                    self._build_cluster_graph_metadata_legacy(
                        labels_t, nhop_neighborhoods, num_patches, x_pred.device
                    )

                density = self._cluster_graph_density()
                print(f"   🔸 聚类图注意力 ({n_hops}-hop): mask密度={density:.1%} → 理论最优加速比 {1/density:.2f}x")

        # 每个 adaptive step 都记录一次密度（mask不变时仍记录，保证累计统计跨越所有步）
        if ras_manager.MANAGER.spatial_attention_enabled() and (
                ras_manager.MANAGER.cluster_attention_mask is not None
                or getattr(ras_manager.MANAGER, "cluster_attention_mask_factory", None) is not None):
            density = self._cluster_graph_density()
            self.attn_mask_density_history.append(density)
            self.window_area_ratio_history.append(density)

        # 确保 patch_drop_cnt 大小正确
        if not hasattr(self, 'patch_drop_cnt') or len(self.patch_drop_cnt) != num_patches:
            self.patch_drop_cnt = torch.zeros(num_patches, device=x_pred.device)

        # Step 2: 计算每个 patch 的复杂度分数（基于complexity_source选择）
        metric_compute_start = time.time()

        if ras_manager.MANAGER.complexity_source == "diff":
            if diff is None:
                raise ValueError("diff is required when complexity_source='diff'")
            patch_scores = self._compute_patch_complexity_from_diff(diff, height, width, sample=sample)
        elif ras_manager.MANAGER.complexity_source == "x_pred":
            patch_scores = self._compute_patch_complexity_from_x_pred(x_pred, height, width, sample=sample)
        else:
            raise ValueError(f"Unknown complexity_source: {ras_manager.MANAGER.complexity_source}")

        metric_compute_time = time.time() - metric_compute_start
        self.metric_compute_times.append(metric_compute_time)

        # Step 3: 应用防饥饿机制（patch级别）
        patch_scores_with_starvation = patch_scores * torch.exp(
            ras_manager.MANAGER.starvation_scale * self.patch_drop_cnt
        )
        partition = self._get_partition_metadata(x_pred.device) if self.enable_partition_metadata_cache else None
        cluster_labels_tensor = (partition["labels_tensor"] if partition is not None
                                 else torch.from_numpy(self.cluster_labels).to(x_pred.device))

        if (
            cluster_layout_updated
            and ras_manager.MANAGER.spatial_attention_enabled()
            and ras_manager.MANAGER.use_cluster_remote_anchors
        ):
            self._build_remote_anchor_metadata(
                cluster_labels_tensor,
                patch_scores,
                num_w,
                anchor_source=x_pred,
                height=height,
                width=width,
            )
        elif cluster_layout_updated:
            ras_manager.MANAGER.cluster_pack_metadata_cpu = None
            ras_manager.MANAGER.cluster_remote_anchor_indices = None
            ras_manager.MANAGER.cluster_remote_anchor_weights = None
            ras_manager.MANAGER.cluster_remote_anchor_group_indices = None
            ras_manager.MANAGER.cluster_remote_anchor_group_mask = None
            ras_manager.MANAGER.cluster_remote_anchor_source_cluster_ids = None
            ras_manager.MANAGER.cluster_remote_anchor_source_indices = None
            ras_manager.MANAGER.cluster_remote_anchor_source_weights = None
            ras_manager.MANAGER.cluster_remote_anchor_source_group_indices = None
            ras_manager.MANAGER.cluster_remote_anchor_source_group_mask = None
            ras_manager.MANAGER.cluster_remote_anchor_all_indices = None
            ras_manager.MANAGER.cluster_remote_anchor_all_weights = None
            ras_manager.MANAGER.cluster_remote_anchor_all_group_indices = None
            ras_manager.MANAGER.cluster_remote_anchor_all_group_mask = None
            ras_manager.MANAGER.cluster_remote_anchor_packed_positions = None
            ras_manager.MANAGER.cluster_remote_anchor_merged_key_indices = None
            ras_manager.MANAGER.cluster_remote_anchor_token_to_packed_anchor = None
            ras_manager.MANAGER.cluster_remote_anchor_packed_counts = None
            ras_manager.MANAGER.cluster_remote_anchor_group_token_order = None
            ras_manager.MANAGER.cluster_remote_anchor_group_lengths = None
            ras_manager.MANAGER.cluster_remote_anchor_group_token_order_is_cluster_order = False
            ras_manager.MANAGER.cluster_remote_anchor_metadata_version += 1

        # Step 4: 聚合到簇级别（取簇内patch的平均分数）
        actual_clusters = self.actual_num_clusters if self.actual_num_clusters is not None else self.num_clusters
        cluster_scores = torch.zeros(actual_clusters, device=x_pred.device)

        for k in range(actual_clusters):
            if partition is not None:
                indices = partition["members_device"].get(k)
                if indices is not None and indices.numel() > 0:
                    cluster_scores[k] = patch_scores_with_starvation.index_select(0, indices).mean()
            else:
                mask = (cluster_labels_tensor == k)
                if mask.sum() > 0:
                    cluster_scores[k] = patch_scores_with_starvation[mask].mean()

        # Step 5: 选择前X%簇（基于实际簇数计算）
        num_selected_clusters = max(1, int(actual_clusters * self.cluster_selection_ratio))
        selected_clusters = torch.topk(
            cluster_scores,
            num_selected_clusters
        ).indices.cpu().numpy()

        # 累计记录每个cluster被选中的次数
        if self.cluster_update_counts is None or len(self.cluster_update_counts) != actual_clusters:
            self.cluster_update_counts = np.zeros(actual_clusters, dtype=np.int32)
        self.cluster_update_counts[selected_clusters] += 1

        # Step 6: 收集被选中簇中的所有patches + 选择和扩展时间测量
        selection_start = time.time()

        selected_patches_before_expansion = []

        for cluster_id in selected_clusters:
            if partition is not None:
                cluster_patch_indices = partition["members_cpu"].get(int(cluster_id), ())
            else:
                cluster_mask = (cluster_labels_tensor == cluster_id)
                cluster_patch_indices = torch.where(cluster_mask)[0].tolist()
            selected_patches_before_expansion.extend(cluster_patch_indices)

        # Host membership already determines the radius-zero index set. Build
        # its complement and latent offsets here instead of GPU nonzero/sorts.
        self._active_selection_cpu = None
        self._cached_selection_cpu = None
        if (partition is not None and ras_manager.MANAGER.neighbor_expansion_mode == "0"
                and getattr(self, "enable_host_selection_schedule", True)):
            selected_cpu = np.zeros(num_patches, dtype=np.bool_)
            selected_cpu[selected_patches_before_expansion] = True
            other_cpu = np.flatnonzero(selected_cpu)
            cached_cpu = np.flatnonzero(~selected_cpu)
            patch_size = ras_manager.MANAGER.patch_size
            bases = (cached_cpu // num_w) * patch_size * width + (cached_cpu % num_w) * patch_size
            offsets = (np.arange(patch_size)[:, None] * width + np.arange(patch_size)[None, :]).reshape(-1)
            latent_cpu = (bases[:, None] + offsets[None, :]).reshape(-1)
            packed_cpu = np.concatenate((other_cpu, latent_cpu, selected_cpu.astype(np.int64)))
            packed = torch.from_numpy(packed_cpu).to(x_pred.device)
            other_patchified_indices = packed[:other_cpu.size]
            latent_cached_indices = packed[other_cpu.size:other_cpu.size+latent_cpu.size]
            selected_mask = packed[other_cpu.size+latent_cpu.size:] != 0
            self.patch_drop_cnt = torch.where(selected_mask, 0.0, self.patch_drop_cnt + 1)
            expanded_patches = other_patchified_indices
            expansion_mode_name = "radius=0 (no expansion)"
            self._active_selection_cpu = (other_patchified_indices, other_cpu)
            self._cached_selection_cpu = (latent_cached_indices, latent_cpu)
        else:
            # Step 6: 根据neighbor_expansion_mode选择邻域扩展方式
            expansion_mode = ras_manager.MANAGER.neighbor_expansion_mode

            # 支持两种模式：
            # 1. 旧模式：字符串 "0", "4", "8"
            # 2. 新模式：整数半径 radius (0, 1, 2, 3, ...)
            try:
                # 尝试作为整数半径解析
                radius = int(expansion_mode)
                expanded_patches = expand_patches_by_radius(
                    patch_indices=selected_patches_before_expansion,
                    grid_h=num_h,
                    grid_w=num_w,
                    radius=radius,
                    device=x_pred.device
                )
                if radius == 0:
                    expansion_mode_name = "无扩展 (radius=0)"
                elif radius == 1:
                    expansion_mode_name = "8-邻域 (radius=1, 3x3)"
                elif radius == 2:
                    expansion_mode_name = "24-邻域 (radius=2, 5x5)"
                else:
                    expansion_mode_name = f"半径={radius} ({2*radius+1}x{2*radius+1})"
            except ValueError:
                # 回退到旧的字符串模式（向后兼容）
                if expansion_mode == "0":
                    expanded_patches = expand_patches_no_expansion(
                        patch_indices=selected_patches_before_expansion,
                        grid_h=num_h,
                        grid_w=num_w,
                        device=x_pred.device
                    )
                    expansion_mode_name = "无扩展"
                elif expansion_mode == "4":
                    expanded_patches = expand_patches_4_neighbors(
                        patch_indices=selected_patches_before_expansion,
                        grid_h=num_h,
                        grid_w=num_w,
                        device=x_pred.device
                    )
                    expansion_mode_name = "4-邻域"
                else:  # expansion_mode == "8"
                    expanded_patches = expand_patches_8_neighbors(
                        patch_indices=selected_patches_before_expansion,
                        grid_h=num_h,
                        grid_w=num_w,
                        device=x_pred.device
                    )
                    expansion_mode_name = "8-邻域"

            # Step 7: 确定最终选中和缓存的patches
            all_patches = torch.arange(num_patches, device=x_pred.device)
            selected_mask = torch.zeros(num_patches, dtype=torch.bool, device=x_pred.device)
            selected_mask[expanded_patches] = True

            cached_mask = ~selected_mask
            cached_patchified_indices = all_patches[cached_mask]
            other_patchified_indices = all_patches[selected_mask]

            # Step 8: 更新 patch 级 drop 计数（防饥饿）
            # 被选中的patch重置为0，未被选中的+1
            self.patch_drop_cnt[selected_mask] = 0
            self.patch_drop_cnt[cached_mask] += 1

            # Step 9: 转换为latent索引
            latent_cached_indices = self.extract_latents_index_from_patched_latents_index(
                cached_patchified_indices, height, width
            )

        # 记录选择和扩展的时间
        selection_time = time.time() - selection_start
        self.selection_times.append(selection_time)

        # 计算总聚类时间
        step_total_clustering_time = time.time() - step_clustering_start
        self.clustering_times.append(step_total_clustering_time)
        self.total_clustering_time += step_total_clustering_time

        # 日志
        num_selected_before = len(selected_patches_before_expansion)
        num_selected_after = len(expanded_patches)
        actual_ratio = num_selected_after / num_patches
        expansion_ratio = num_selected_after / max(1, num_selected_before)

        print(f"   🔸 选中 {num_selected_clusters}/{actual_clusters} 簇 "
              f"→ {num_selected_before} patches "
              f"→ ({expansion_mode_name}扩展) → {num_selected_after}/{num_patches} patches "
              f"({actual_ratio:.1%}, 扩展 {expansion_ratio:.2f}x) "
              f"[聚类: {step_total_clustering_time*1000:.1f}ms]")

        return latent_cached_indices, other_patchified_indices

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[RASFlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        重写step方法，使用cluster_adaptive_selection替代父类的cluster_based_selection
        """
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        if self.drop_cnt is None or self._step_index == 0:
            self._init_ras_config(sample)

        if self._step_index == 0:
            ras_manager.MANAGER.reset_cache()
            ras_manager.MANAGER.reset_cluster_backend_stats()
            self.clustered = False  # 重置聚类状态
            self.last_recluster_step = None
            self._partition_metadata_cache = None
            self._cluster_mask_density_cache = None
            self._cluster_far_ids_cpu_cache = None
            ras_manager.MANAGER.cluster_pack_metadata_cpu = None
            self._clear_cached_history()
            self.cluster_update_counts = None  # 新图片重置计数

        latent_dim, height, width = sample.shape[-3:]

        # 计算当前步的 sigma 变化量（需要在复用噪声之前）
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        delta_sigma = abs(sigma_next - sigma)

        assert ras_manager.MANAGER.sample_ratio > 0.0 and ras_manager.MANAGER.sample_ratio <= 1.0
        if ras_manager.MANAGER.temporal_selection_enabled() and ras_manager.MANAGER.is_RAS_step:
            flat_output = model_output.squeeze(0).view(latent_dim, -1)
            cached_indices = ras_manager.MANAGER.cached_index
            cached_noise = ras_manager.MANAGER.cached_scaled_noise
            if cached_indices is not None and cached_noise is not None:
                if isinstance(cached_indices, torch.Tensor):
                    indices_empty = cached_indices.numel() == 0
                else:
                    indices_empty = len(cached_indices) == 0
                if not indices_empty:
                    # 使用sigma缩放来复用噪声
                    scaled_noise = self._compute_scaled_noise(delta_sigma.item())
                    if scaled_noise is None:
                        scaled_noise = cached_noise
                    flat_output[:, cached_indices] = scaled_noise
            model_output = flat_output.view(latent_dim, height, width).unsqueeze(0)

        # 保存 x_pred 的 VAE 解码结果（在转换数据类型之前）
        # x_pred = sample - sigma * model_output (预测的干净图像)
        current_step = ras_manager.MANAGER.current_step
        self._save_x_pred_vae_decoded(sample, model_output, sigma.item(), current_step)

        # Upcast to avoid precision issues when computing the clean prediction
        # and prev_sample. Use x0_pred for semantic clustering/scoring, while
        # prev_sample remains the actual scheduler update.
        sample = sample.to(torch.float32)
        model_output_float = model_output.to(dtype=sample.dtype)
        sigma_for_x0 = sigma.to(device=sample.device, dtype=sample.dtype)

        x_pred_clean = sample - sigma_for_x0 * model_output_float
        diff = (sigma_next - sigma).to(device=sample.device, dtype=sample.dtype) * model_output_float

        prev_sample = sample + diff
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)
        prediction_source = getattr(ras_manager.MANAGER, "cluster_prediction_source", "x0_pred")
        prediction_source = prediction_source.strip().lower()
        if prediction_source == "legacy":
            cluster_prediction = prev_sample.to(dtype=sample.dtype)
        elif prediction_source in ("x0_pred", "x0", "clean"):
            cluster_prediction = x_pred_clean
        elif prediction_source in ("prev_sample", "x_next", "next"):
            cluster_prediction = prev_sample.to(dtype=sample.dtype)
        elif prediction_source in ("sample", "current", "xt"):
            cluster_prediction = sample
        else:
            raise ValueError(
                f"Unsupported cluster_prediction_source={prediction_source!r}; "
                "expected 'x0_pred', 'prev_sample', or 'sample'"
            )

        analysis_callback = ras_manager.MANAGER.scheduler_analysis_callback
        if analysis_callback is not None:
            analysis_callback({
                "model": "lumina",
                "step": int(current_step),
                "timestep": timestep,
                "sample": sample.detach(),
                "clean_prediction": x_pred_clean.detach(),
                "model_output": model_output_float.detach(),
                "prev_sample": prev_sample.detach(),
            })

        # print(sample.shape)
        # 保存当前step的sample张量
        self._save_sample_tensor(sample, current_step)

        # 关键修改：使用cluster_adaptive_selection（含8-邻域扩展）
        # 但在最后N步禁用adaptive，改为全部同步更新
        remaining_steps = ras_manager.MANAGER.num_steps - ras_manager.MANAGER.current_step

        temporal_enabled = ras_manager.MANAGER.temporal_selection_enabled()
        spatial_enabled = ras_manager.MANAGER.spatial_attention_enabled()
        if temporal_enabled and ras_manager.MANAGER.is_next_RAS_step:
            if remaining_steps <= self.final_sync_steps:
                # 最后N步：禁用adaptive，临时禁用RAS采样
                print(f"   ⚠️  最后{remaining_steps}步：禁用adaptive，全部同步更新")
                # 临时禁用sample_ratio，让所有patch都被更新
                original_sample_ratio = ras_manager.MANAGER.sample_ratio
                ras_manager.MANAGER.sample_ratio = 1.0
                self._clear_cached_history()
                ras_manager.MANAGER.cached_index = None
                ras_manager.MANAGER.cached_scaled_noise = None
            else:
                # 正常的cluster adaptive选择
                latent_cached_indices, other_patchified_indices = self.cluster_adaptive_selection(
                    sample, cluster_prediction, height, width, diff=diff
                )
                flattened_output = model_output.squeeze(0).view(latent_dim, -1)
                new_cached_noise = None
                if latent_cached_indices is not None:
                    if isinstance(latent_cached_indices, torch.Tensor) and latent_cached_indices.numel() > 0:
                        new_cached_noise = flattened_output[:, latent_cached_indices]
                    elif hasattr(latent_cached_indices, "__len__") and len(latent_cached_indices) > 0:
                        new_cached_noise = flattened_output[:, latent_cached_indices]

                # 更新噪声缓存（包含当前步的delta_sigma）
                self._update_cached_noise(latent_cached_indices, new_cached_noise, delta_sigma.item())

                host_selection = getattr(self, "_active_selection_cpu", None)
                indices_cpu = (host_selection[1] if host_selection is not None
                               and host_selection[0] is other_patchified_indices else None)
                ras_manager.MANAGER.set_active_patch_indices(
                    other_patchified_indices, indices_cpu=indices_cpu)

                # 更新patch统计 (adaptive采样时，更新的patches = other_patchified_indices的数量)
                num_updated = len(other_patchified_indices) if other_patchified_indices is not None else 0
                total_patches = (height // ras_manager.MANAGER.patch_size) * (width // ras_manager.MANAGER.patch_size)
                self.update_patch_statistics(num_updated, total_patches)
        elif spatial_enabled and ras_manager.MANAGER.is_next_RAS_step:
            # Spatial-only: create/reuse the semantic layout, but never skip a
            # query and never populate the temporal prediction cache.
            should_update_layout = (
                not self.clustered
                or self.recluster_every_step
                or (
                    self.recluster_every_n_steps > 0
                    and self.last_recluster_step is not None
                    and current_step - self.last_recluster_step >= self.recluster_every_n_steps
                )
            )
            if should_update_layout:
                self.cluster_adaptive_selection(sample, cluster_prediction, height, width, diff=diff)
            num_patches = (height // ras_manager.MANAGER.patch_size) * (
                width // ras_manager.MANAGER.patch_size
            )
            ras_manager.MANAGER.cached_scaled_noise = None
            ras_manager.MANAGER.set_all_patches_active(num_patches, sample.device)
            self._clear_cached_history()
            self.update_patch_statistics(num_patches, num_patches)
        else:
            # Warm-up阶段或最后同步阶段，所有patches都被更新
            total_patches = (height // ras_manager.MANAGER.patch_size) * (width // ras_manager.MANAGER.patch_size)
            self.update_patch_statistics(total_patches, total_patches)

        # 如果当前步是 sync step，下一步强制重新聚类
        if ras_manager.MANAGER.current_step in ras_manager.MANAGER.error_reset_steps:
            self.clustered = False
            self.last_recluster_step = None
            print(f"   🔄 Sync step {ras_manager.MANAGER.current_step} 完成，下一步将重新聚类")

        # upon completion increase step index by one
        self._step_index += 1
        ras_manager.MANAGER.increase_step()
        if ras_manager.MANAGER.current_step >= ras_manager.MANAGER.num_steps:
            ras_manager.MANAGER.reset_cache()
            self.clustered = False
            self.last_recluster_step = None
            self._clear_cached_history()
            # 推理完成时打印统计摘要
            self.print_statistics_summary()

        if not return_dict:
            return (prev_sample,)

        return RASFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
