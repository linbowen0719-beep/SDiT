"""
Cluster-Based RAS Scheduler

核心思想：
1. 将latent划分为patches并进行空间感知聚类（16簇）
2. 计算每个簇的总complexity score
3. 每步选择top-8簇进行同步更新
4. 保持RAS的防饥饿和同步机制
"""

import torch
import numpy as np
from .ras_scheduling_flow_match_euler_discrete import (
    RASFlowMatchEulerDiscreteScheduler,
    RASFlowMatchEulerDiscreteSchedulerOutput
)
from ras.utils import ras_manager
from typing import Optional, Tuple, Union
from sklearn.cluster import KMeans

# 添加图像分割算法导入
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature


class ClusterBasedRASScheduler(RASFlowMatchEulerDiscreteScheduler):
    """
    基于聚类的RAS调度器

    改进：
    - 空间感知聚类（16簇）
    - 簇级complexity score
    - 簇内同步更新（top-8簇）
    - 保持原RAS的防饥饿和error_reset机制
    """

    def __init__(self, *args, num_clusters=16, top_k_clusters=8, segmentation_algorithm="slic",
                 quickshift_kernel_size=2.3, quickshift_max_dist=6.0, quickshift_ratio=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        # 聚类/分割参数
        self.segmentation_algorithm = segmentation_algorithm
        self.num_clusters = num_clusters  # 仅用于 kmeans
        self.top_k_clusters = top_k_clusters

        # QuickShift参数
        self.quickshift_kernel_size = quickshift_kernel_size
        self.quickshift_max_dist = quickshift_max_dist
        self.quickshift_ratio = quickshift_ratio

        # 聚类状态
        self.cluster_labels = None  # (num_patches,) patch到簇的映射
        self.cluster_drop_cnt = None  # (num_clusters,) 每个簇被跳过的次数
        self.clustered = False
        self.actual_num_clusters = None  # 实际分割得到的簇数

        print(f"✅ ClusterBasedRASScheduler初始化完成:")
        print(f"   分割算法: {self.segmentation_algorithm}")
        if self.segmentation_algorithm == "kmeans":
            print(f"   聚类数量: {self.num_clusters}")
            print(f"   每步选择簇数: {self.top_k_clusters}")
        else:
            print(f"   每步选择比例: {self.top_k_clusters}/{self.num_clusters} (将根据实际簇数调整)")

    def _init_ras_config(self, latents):
        """初始化RAS配置（覆盖父类方法）"""
        super()._init_ras_config(latents)

        # 初始化簇级drop计数（实际数量将在第一次聚类后确定）
        self.cluster_drop_cnt = torch.zeros(
            self.num_clusters,
            device=latents.device
        )
        self.clustered = False

    def _normalize_channels(self, image, eps=1e-8):
        """按通道进行 min-max 归一化到 [0, 1]"""
        channel_min = image.min(axis=(0, 1), keepdims=True)
        channel_max = image.max(axis=(0, 1), keepdims=True)
        scale = np.maximum(channel_max - channel_min, eps)
        normalized = (image - channel_min) / scale
        return normalized

    def _downsample_latent(self, latent, block_size=2):
        """
        使用非重叠块的均值对latent进行空间降采样，与VAE_Samples的离线脚本保持一致。
        """
        if latent.ndim != 4:
            raise ValueError(f"Expected latent with 4 dims (B, C, H, W); got {latent.shape}")

        batch, channels, height, width = latent.shape
        if height % block_size != 0 or width % block_size != 0:
            raise ValueError(
                f"Height and width must be divisible by block_size={block_size}; got {(height, width)}"
            )

        new_h = height // block_size
        new_w = width // block_size
        latent = latent.reshape(batch, channels, new_h, block_size, new_w, block_size)
        downsampled = latent.mean(axis=(3, 5))
        return downsampled

    def _prepare_image_for_segmentation(self, sample):
        """
        将 latent tensor 准备为分割算法的输入格式

        Args:
            sample: (B, C, H, W) tensor

        Returns:
            image: (H, W, C) numpy array, 归一化到 [0, 1]
        """
        # (B, C, H, W) -> (C, H, W)
        sample_np = sample.detach().cpu().float().numpy()
        if sample_np.ndim == 3:
            sample_np = np.expand_dims(sample_np, axis=0)
        if sample_np.shape[0] != 1:
            raise ValueError(f"Expected batch dimension == 1, got {sample_np.shape}")

        block_size = ras_manager.MANAGER.patch_size
        downsampled = self._downsample_latent(sample_np, block_size=block_size)[0]
        # (C, H, W) -> (H, W, C)
        image = np.moveaxis(downsampled, 0, -1)
        # 归一化
        image = self._normalize_channels(image)
        return image.astype(np.float32)

    def _run_slic(self, image, n_segments=40, compactness=1.0):
        """SLIC 超像素分割"""
        labels = segmentation.slic(
            image,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=-1,
        )
        return labels.astype(np.int32)

    def _run_felzenszwalb(self, image, scale=100.0, sigma=0.5, min_size=20):
        """Felzenszwalb 图分割"""
        labels = segmentation.felzenszwalb(
            image,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
            channel_axis=-1,
        )
        return labels.astype(np.int32)

    def _compute_watershed_markers(self, gray_image, min_distance=4, threshold_rel=0.02):
        """为 Watershed 计算标记点"""
        smoothed = filters.gaussian(gray_image, sigma=1.0)
        coords = feature.peak_local_max(
            -smoothed,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
        )

        markers = np.zeros_like(gray_image, dtype=np.int32)
        for idx, (row, col) in enumerate(coords, start=1):
            markers[row, col] = idx

        if markers.max() < 2:
            suppressed = morphology.h_minima(smoothed, h=threshold_rel)
            markers = ndi.label(suppressed)[0]

        if markers.max() < 2:
            try:
                threshold = filters.threshold_otsu(smoothed)
            except ValueError:
                threshold = float(smoothed.mean())
            markers = ndi.label(smoothed > threshold)[0]

        if markers.max() < 2:
            # 回退：在网格上放置标记点
            markers = np.zeros_like(gray_image, dtype=np.int32)
            step = max(8, gray_image.shape[0] // 6)
            marker_id = 1
            for row in range(step // 2, gray_image.shape[0], step):
                for col in range(step // 2, gray_image.shape[1], step):
                    markers[row, col] = marker_id
                    marker_id += 1
            markers = ndi.grey_dilation(markers, size=(3, 3))

        return markers.astype(np.int32)

    def _run_watershed(self, image, compactness=0.001):
        """Watershed 分水岭分割"""
        gray = np.linalg.norm(image, axis=-1)
        gray = filters.gaussian(gray, sigma=0.8)
        gradient = filters.sobel(gray)
        markers = self._compute_watershed_markers(gray)
        labels = segmentation.watershed(
            gradient,
            markers=markers,
            compactness=compactness,
        )
        return labels.astype(np.int32)

    def _run_quickshift(self, image):
        """QuickShift 快速移位分割"""
        labels = segmentation.quickshift(
            image,
            kernel_size=self.quickshift_kernel_size,
            max_dist=self.quickshift_max_dist,
            ratio=self.quickshift_ratio,
            convert2lab=False,
            channel_axis=-1,
        )
        return labels.astype(np.int32)

    def _convert_pixel_labels_to_patch_labels(self, pixel_labels, patch_size):
        """
        将像素级分割标签转换为 patch 级标签

        Args:
            pixel_labels: (H, W) 像素级标签
            patch_size: patch 大小

        Returns:
            patch_labels: (num_patches,) patch 级标签，每个 patch 取众数
        """
        H, W = pixel_labels.shape
        num_h = H // patch_size
        num_w = W // patch_size

        patch_labels = []
        for i in range(num_h):
            for j in range(num_w):
                patch_region = pixel_labels[
                    i*patch_size:(i+1)*patch_size,
                    j*patch_size:(j+1)*patch_size
                ]
                # 取众数作为该 patch 的标签
                unique, counts = np.unique(patch_region, return_counts=True)
                majority_label = unique[np.argmax(counts)]
                patch_labels.append(majority_label)

        return np.array(patch_labels, dtype=np.int32)

    def _run_uniform(self, num_h, num_w, uniform_size=16):
        """
        Uniform 均匀分割 - 将patches均匀分组成固定大小的区域

        例如：64x64 patches，uniform_size=16，将分成 4x4=16 个区域
        现在宽度×2，所以实际是 8x8 区域（宽度翻倍）

        Args:
            num_h: patch grid 的高度
            num_w: patch grid 的宽度
            uniform_size: 每个区域包含多少个 patches（例如16表示4x4区域）

        Returns:
            cluster_labels: (num_patches,) patch级别的标签
        """
        import math

        # 计算每个区域的边长（假设正方形区域），然后宽度×2
        region_side = int(math.sqrt(uniform_size)) * 2

        # 计算区域grid的尺寸
        region_h = math.ceil(num_h / region_side)
        region_w = math.ceil(num_w / region_side)

        cluster_labels = []
        for i in range(num_h):
            for j in range(num_w):
                # 计算该patch属于哪个区域
                region_i = i // region_side
                region_j = j // region_side
                cluster_id = region_i * region_w + region_j
                cluster_labels.append(cluster_id)

        return np.array(cluster_labels, dtype=np.int32)

    def spatial_aware_clustering(self, sample, height, width):
        """
        空间感知聚类/分割

        根据 self.segmentation_algorithm 选择不同的方法：
        - "kmeans": 原始的 K-means 聚类（使用 num_clusters）
        - "slic": SLIC 超像素分割
        - "felzenszwalb": Felzenszwalb 图分割
        - "watershed": Watershed 分水岭分割
        - "quickshift": QuickShift 快速移位分割
        - "uniform": 均匀分割（将patches分成固定大小的正方形区域）
        - "direct": 直接选择（不聚类，每个patch独立，基于复杂度直接选择）

        Args:
            sample: latent tensor (B, C, H, W)
            height, width: latent空间的高宽

        Returns:
            cluster_labels: (num_patches,) numpy array
        """
        patch_size = ras_manager.MANAGER.patch_size
        num_h = height // patch_size
        num_w = width // patch_size
        num_patches = num_h * num_w

        if self.segmentation_algorithm == "direct":
            # ========== Direct 直接选择（不聚类） ==========
            # 每个patch作为独立的"簇"，后续直接基于复杂度分数选择
            # 这样就跳过了聚类步骤，直接选择top-k% patches
            cluster_labels = np.arange(num_patches, dtype=np.int32)
            self.actual_num_clusters = num_patches

        elif self.segmentation_algorithm == "uniform":
            # ========== Uniform 均匀分割 ==========
            # 默认每个区域16个patches (4x4)
            uniform_size = getattr(self, 'uniform_region_size', 16)
            cluster_labels = self._run_uniform(num_h, num_w, uniform_size=uniform_size)
            unique_labels = np.unique(cluster_labels)
            self.actual_num_clusters = len(unique_labels)

        elif self.segmentation_algorithm == "kmeans":
            # ========== 原始 K-means 聚类方法 ==========
            # 提取patch特征
            sample_np = sample.squeeze(0).cpu().float().numpy()  # (C, H, W)

            features = []
            positions = []

            for i in range(num_h):
                for j in range(num_w):
                    # 提取patch
                    patch = sample_np[
                        :,
                        i*patch_size:(i+1)*patch_size,
                        j*patch_size:(j+1)*patch_size
                    ]

                    # 简单特征：通道均值
                    patch_mean = patch.mean(axis=(1, 2))  # (C,)
                    features.append(patch_mean)

                    # 空间坐标（归一化）
                    positions.append([i / (num_h - 1) if num_h > 1 else 0,
                                     j / (num_w - 1) if num_w > 1 else 0])

            features = np.array(features)  # (num_patches, C)
            positions = np.array(positions)  # (num_patches, 2)

            # 组合特征：[语义特征, α×空间坐标]
            alpha = 0.5  # 空间权重
            combined_features = np.concatenate([
                features / (np.abs(features).max() + 1e-8),  # 归一化语义特征
                alpha * positions
            ], axis=1)

            # K-means聚类
            n_clusters = min(self.num_clusters, num_patches)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
            cluster_labels = kmeans.fit_predict(combined_features)
            unique_labels = np.unique(cluster_labels)
            self.actual_num_clusters = len(unique_labels)

        else:
            # ========== 图像分割算法 ==========
            # 准备图像数据
            image = self._prepare_image_for_segmentation(sample)  # (H, W, C)

            # 根据算法选择执行分割
            if self.segmentation_algorithm == "slic":
                pixel_labels = self._run_slic(image, n_segments=40, compactness=1.0)
            elif self.segmentation_algorithm == "felzenszwalb":
                pixel_labels = self._run_felzenszwalb(image, scale=100.0, sigma=0.5, min_size=20)
            elif self.segmentation_algorithm == "watershed":
                pixel_labels = self._run_watershed(image, compactness=0.001)
            elif self.segmentation_algorithm == "quickshift":
                pixel_labels = self._run_quickshift(image)
            else:
                raise ValueError(f"Unknown segmentation algorithm: {self.segmentation_algorithm}")

            # 转换像素级标签到 patch 级标签
            expected_shape = (num_h, num_w)
            if pixel_labels.shape == expected_shape:
                cluster_labels = pixel_labels.reshape(-1)
            else:
                cluster_labels = self._convert_pixel_labels_to_patch_labels(pixel_labels, patch_size)

            # 重新映射标签确保连续（0, 1, 2, ...）
            unique_labels = np.unique(cluster_labels)
            label_mapping = {old: new for new, old in enumerate(unique_labels)}
            cluster_labels = np.array([label_mapping[label] for label in cluster_labels], dtype=np.int32)

            # 更新实际簇数
            self.actual_num_clusters = len(unique_labels)

        # 确保 cluster_labels 为 numpy array 并映射到连续标签
        cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
        unique_labels = np.unique(cluster_labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        cluster_labels = np.array([label_mapping[label] for label in cluster_labels], dtype=np.int32)
        self.actual_num_clusters = len(unique_labels)

        # 调整 cluster_drop_cnt 大小（如果需要）
        actual_clusters = self.actual_num_clusters
        if self.cluster_drop_cnt is None or len(self.cluster_drop_cnt) != actual_clusters:
            device = sample.device if self.cluster_drop_cnt is None else self.cluster_drop_cnt.device
            self.cluster_drop_cnt = torch.zeros(actual_clusters, device=device)

        return cluster_labels

    def compute_cluster_complexity_scores(self, diff, cluster_labels, height, width):
        """
        计算每个簇的complexity score

        Args:
            diff: 去噪增量 (B, C, H, W)
            cluster_labels: (num_patches,) patch到簇的映射
            height, width: latent空间的高宽

        Returns:
            cluster_scores: (num_clusters,) tensor
        """
        patch_size = ras_manager.MANAGER.patch_size
        num_h = height // patch_size
        num_w = width // patch_size

        # 计算patch级metric（与RAS相同）
        diff_squeezed = diff.squeeze(0).permute(1, 2, 0)  # (H, W, C)

        if ras_manager.MANAGER.metric == "std":
            metric = torch.std(diff_squeezed, dim=-1)  # (H, W)
        elif ras_manager.MANAGER.metric == "l2norm":
            metric = torch.norm(diff_squeezed, p=2, dim=-1)  # (H, W)
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

    def cluster_based_selection(self, sample, diff, height, width):
        """
        基于聚类的patch选择

        Args:
            sample: 当前latent (B, C, H, W)
            diff: 去噪增量 (B, C, H, W)
            height, width: latent空间的高宽

        Returns:
            latent_cached_indices: 被缓存（跳过）的latent索引
            other_patchified_indices: 被选中（更新）的patch索引
        """
        patch_size = ras_manager.MANAGER.patch_size
        num_h = height // patch_size
        num_w = width // patch_size
        num_patches = num_h * num_w

        # 第一步：进行聚类
        if not self.clustered:
            self.cluster_labels = self.spatial_aware_clustering(sample, height, width)
            self.clustered = True
            actual_clusters = self.actual_num_clusters if self.actual_num_clusters is not None else self.num_clusters
            print(f"   🔸 完成空间感知聚类: {num_patches} patches → {actual_clusters} clusters")

        # 第二步：计算簇级complexity scores
        cluster_scores = self.compute_cluster_complexity_scores(
            diff, self.cluster_labels, height, width
        )

        # 第三步：应用防饥饿机制（簇级别）
        cluster_scores *= torch.exp(ras_manager.MANAGER.starvation_scale * self.cluster_drop_cnt)

        # 第四步：选择top-k簇（按固定比例向上取整）
        actual_clusters = self.actual_num_clusters if self.actual_num_clusters is not None else self.num_clusters
        if self.segmentation_algorithm == "kmeans":
            # kmeans: 使用 top_k_clusters 作为绝对数量
            top_k = min(self.top_k_clusters, actual_clusters)
        else:
            # 图像分割算法: 按比例计算（向上取整）
            selection_ratio = self.top_k_clusters / self.num_clusters
            top_k = max(1, int(np.ceil(actual_clusters * selection_ratio)))

        selected_clusters = torch.topk(cluster_scores, top_k).indices.cpu().numpy()

        # 第五步：确定被更新的patches（selected clusters内的所有patches）
        cluster_labels_tensor = torch.from_numpy(self.cluster_labels)
        selected_mask = torch.zeros(num_patches, dtype=torch.bool)

        for cluster_id in selected_clusters:
            selected_mask |= (cluster_labels_tensor == cluster_id)

        # 被缓存（跳过）的patches
        cached_mask = ~selected_mask
        cached_patchified_indices = torch.where(cached_mask)[0].to(diff.device)
        other_patchified_indices = torch.where(selected_mask)[0].to(diff.device)

        # 更新簇级drop计数
        for k in range(actual_clusters):
            if k in selected_clusters:
                self.cluster_drop_cnt[k] = 0  # 重置
            else:
                self.cluster_drop_cnt[k] += 1  # 增加

        # 转换为latent索引
        latent_cached_indices = self.extract_latents_index_from_patched_latents_index(
            cached_patchified_indices, height, width
        )

        # 日志
        num_selected_patches = selected_mask.sum().item()
        actual_ratio = num_selected_patches / num_patches
        print(f"   🔸 选中 {top_k}/{self.num_clusters} 簇 → {num_selected_patches}/{num_patches} patches ({actual_ratio:.1%})")

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
        覆盖父类的step方法，使用聚类选择策略
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
            self.clustered = False  # 重置聚类状态

        latent_dim, height, width = sample.shape[-3:]

        assert ras_manager.MANAGER.sample_ratio > 0.0 and ras_manager.MANAGER.sample_ratio <= 1.0
        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
            model_output.squeeze(0).view(latent_dim, -1)[:, ras_manager.MANAGER.cached_index] = ras_manager.MANAGER.cached_scaled_noise
            model_output = model_output.transpose(0, 1).view(latent_dim, height, width).unsqueeze(0)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        diff = (sigma_next - sigma) * model_output

        prev_sample = sample + diff
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # 关键修改：使用聚类选择策略替代原RAS的patch选择
        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_next_RAS_step:
            latent_cached_indices, other_patchified_indices = self.cluster_based_selection(
                sample, diff, height, width
            )
            ras_manager.MANAGER.cached_scaled_noise = model_output.squeeze(0).view(latent_dim, -1)[:, latent_cached_indices]
            ras_manager.MANAGER.cached_index = latent_cached_indices
            ras_manager.MANAGER.other_patchified_index = other_patchified_indices

        # upon completion increase step index by one
        self._step_index += 1
        ras_manager.MANAGER.increase_step()
        if ras_manager.MANAGER.current_step >= ras_manager.MANAGER.num_steps:
            ras_manager.MANAGER.reset_cache()
            self.clustered = False

        if not return_dict:
            return (prev_sample,)

        return RASFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
