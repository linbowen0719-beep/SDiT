"""
Visualization Scheduler - 用于可视化中间推理结果

功能:
1. 基于 ClusterAdaptiveScheduler
2. 保存每一步的 latent 中间结果
3. 可选：保存每一步解码后的图像
4. 提供可视化工具函数
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union
from PIL import Image

from .cluster_adaptive_scheduler import ClusterAdaptiveScheduler
from .ras_scheduling_flow_match_euler_discrete import RASFlowMatchEulerDiscreteSchedulerOutput


class VisualizationScheduler(ClusterAdaptiveScheduler):
    """
    可视化调度器 - 保存推理过程中每一步的中间结果

    新增功能:
    1. 保存每步的 latent 张量
    2. 可选：保存每步解码后的图像预览
    3. 统计信息保存
    """

    def __init__(
        self,
        *args,
        save_latents: bool = True,
        save_decoded_images: bool = False,
        output_dir: str = "outputs/debug/visualizations",
        save_interval: int = 1,  # 每隔几步保存一次
        **kwargs
    ):
        """
        Args:
            save_latents: 是否保存 latent 张量
            save_decoded_images: 是否保存解码后的图像（需要 VAE，较慢）
            output_dir: 输出目录
            save_interval: 保存间隔（1=每步保存，2=每2步保存一次）
        """
        super().__init__(*args, **kwargs)

        self.save_latents = save_latents
        self.save_decoded_images = save_decoded_images
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval

        # 存储中间结果
        self.latent_history = []  # List of latents
        self.timestep_history = []  # List of timesteps
        self.step_indices = []  # List of step indices

        # 用于解码的 VAE（稍后设置）
        self.vae = None

        print(f"✅ VisualizationScheduler 初始化完成:")
        print(f"   保存 latents: {self.save_latents}")
        print(f"   保存解码图像: {self.save_decoded_images}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   保存间隔: 每 {self.save_interval} 步")

    def set_vae(self, vae):
        """设置 VAE 用于解码图像"""
        self.vae = vae
        print(f"✅ VAE 已设置用于图像解码")

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[RASFlowMatchEulerDiscreteSchedulerOutput, tuple]:
        """
        重写 step 方法，在每步推理后保存中间结果
        """
        # 调用父类的 step 方法
        result = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            generator=generator,
            return_dict=return_dict,
        )

        # 获取当前步数
        if isinstance(timestep, torch.Tensor):
            timestep_value = timestep.item()
        else:
            timestep_value = timestep

        current_step = len(self.latent_history)

        # 根据 save_interval 决定是否保存
        if current_step % self.save_interval == 0:
            # 保存当前步的 latent
            if self.save_latents:
                if return_dict:
                    current_latent = result.prev_sample
                else:
                    current_latent = result[0]

                self._save_latent(current_latent, current_step, timestep_value)

            # 保存解码后的图像（可选）
            if self.save_decoded_images and self.vae is not None:
                if return_dict:
                    current_latent = result.prev_sample
                else:
                    current_latent = result[0]

                self._save_decoded_image(current_latent, current_step, timestep_value)

        return result

    def _save_latent(self, latent: torch.Tensor, step: int, timestep: float):
        """
        保存 latent 张量

        Args:
            latent: [B, C, H, W]
            step: 当前步数
            timestep: 当前时间步
        """
        # 保存到内存
        self.latent_history.append(latent.detach().cpu())
        self.timestep_history.append(timestep)
        self.step_indices.append(step)

        # 保存为文件
        latent_path = self.output_dir / f"latent_step{step:03d}_t{timestep:.4f}.pt"
        torch.save({
            "latent": latent.detach().cpu(),
            "step": step,
            "timestep": timestep,
        }, latent_path)

        print(f"   💾 保存 latent: step {step}, timestep {timestep:.4f}")

    def _save_decoded_image(self, latent: torch.Tensor, step: int, timestep: float):
        """
        解码 latent 并保存为图像

        Args:
            latent: [B, C, H, W]
            step: 当前步数
            timestep: 当前时间步
        """
        try:
            with torch.no_grad():
                # 解码
                latent_scaled = latent / self.vae.config.scaling_factor
                image_tensor = self.vae.decode(latent_scaled.to(self.vae.dtype)).sample

                # 转换为 PIL Image
                image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                image_pil = Image.fromarray(image_np)

                # 保存
                image_path = self.output_dir / f"image_step{step:03d}_t{timestep:.4f}.png"
                image_pil.save(image_path)

                print(f"   🖼️  保存图像: step {step}, timestep {timestep:.4f}")

        except Exception as e:
            print(f"   ⚠️  解码图像失败 (step {step}): {e}")

    def save_all_latents(self):
        """
        将所有 latent 保存为单个文件
        """
        if len(self.latent_history) == 0:
            print("⚠️  没有收集到 latent 数据")
            return

        save_path = self.output_dir / "all_latents.pt"

        try:
            # 尝试 stack（如果所有 latent shape 一致）
            latents_stacked = torch.stack(self.latent_history)
            data = {
                "latents": latents_stacked,
                "timesteps": self.timestep_history,
                "step_indices": self.step_indices,
            }
        except:
            # 如果无法 stack，保存为 list
            data = {
                "latents": self.latent_history,
                "timesteps": self.timestep_history,
                "step_indices": self.step_indices,
            }

        torch.save(data, save_path)

        print(f"\n{'='*80}")
        print(f"✅ 保存所有 latent 数据: {save_path}")
        print(f"   总步数: {len(self.latent_history)}")
        print(f"   Latent shape: {self.latent_history[0].shape}")
        print(f"{'='*80}\n")

    def create_video(self, fps: int = 2, video_name: str = "denoising_process.mp4"):
        """
        从保存的图像创建视频（需要已保存解码图像）

        Args:
            fps: 帧率
            video_name: 视频文件名
        """
        try:
            import cv2

            # 获取所有图像文件
            image_files = sorted(self.output_dir.glob("image_step*.png"))

            if len(image_files) == 0:
                print("⚠️  没有找到图像文件，无法创建视频")
                return

            # 读取第一张图像获取尺寸
            first_image = cv2.imread(str(image_files[0]))
            height, width, _ = first_image.shape

            # 创建视频写入器
            video_path = self.output_dir / video_name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            # 写入所有图像
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                video_writer.write(img)

            video_writer.release()

            print(f"\n{'='*80}")
            print(f"✅ 创建视频: {video_path}")
            print(f"   帧数: {len(image_files)}")
            print(f"   FPS: {fps}")
            print(f"   分辨率: {width}x{height}")
            print(f"{'='*80}\n")

        except ImportError:
            print("⚠️  需要安装 opencv-python: pip install opencv-python")
        except Exception as e:
            print(f"⚠️  创建视频失败: {e}")

    def clear_data(self):
        """清空所有保存的数据"""
        self.latent_history = []
        self.timestep_history = []
        self.step_indices = []
        print("✅ 已清空所有可视化数据")
