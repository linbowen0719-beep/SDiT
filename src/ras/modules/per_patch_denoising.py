"""
Per-Patch Denoising with Different Timesteps

This module implements the core logic for applying different denoising strategies
to different patches by leveraging the batch dimension.

Strategy:
- Group A patches: Multi-step denoising (use current timestep)
- Group B patches: Single-step skip (use sync point timestep until final step)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from einops import rearrange


class PerPatchDenoisingManager:
    """
    Manages per-patch denoising with different timestep strategies.

    Uses batch dimension trick: expand patches to batch dimension to allow
    different timesteps per patch.
    """

    def __init__(self, patch_size: int, latent_height: int, latent_width: int):
        """
        Args:
            patch_size: Size of patches in latent space
            latent_height: Height of latent (e.g., 128 for 1024x1024 image)
            latent_width: Width of latent
        """
        self.patch_size = patch_size
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.num_patches_h = latent_height // patch_size
        self.num_patches_w = latent_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Track current timestep for each patch
        self.patch_timesteps = torch.full((self.num_patches,), 1000, dtype=torch.float32)
        self.patch_strategies = {}  # patch_id -> 'a' or 'b'

    def latent_to_patches(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Convert latent to patch representation.

        Args:
            latent: [B, C, H, W]

        Returns:
            patches: [num_patches, B, C, ph, pw]
        """
        B, C, H, W = latent.shape

        patches = rearrange(
            latent,
            'b c (nh ph) (nw pw) -> (nh nw) b c ph pw',
            ph=self.patch_size,
            pw=self.patch_size
        )

        return patches

    def patches_to_latent(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to latent.

        Args:
            patches: [num_patches, B, C, ph, pw]

        Returns:
            latent: [B, C, H, W]
        """
        latent = rearrange(
            patches,
            '(nh nw) b c ph pw -> b c (nh ph) (nw pw)',
            nh=self.num_patches_h,
            nw=self.num_patches_w
        )

        return latent

    def prepare_strategywise_batches(
        self,
        latent: torch.Tensor,
        group_a_indices: torch.Tensor,
        group_b_indices: torch.Tensor,
        timestep_a: float,
        timestep_b: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare separate batches for Group A and Group B with different timesteps.

        Args:
            latent: [B, C, H, W] - original latent (B should be 1 for single image)
            group_a_indices: Indices of complex patches
            group_b_indices: Indices of simple patches
            timestep_a: Timestep for Group A
            timestep_b: Timestep for Group B

        Returns:
            latent_a: [len(group_a), C, ph, pw] - Group A patches
            latent_b: [len(group_b), C, ph, pw] - Group B patches
            timesteps: [len(group_a) + len(group_b)] - Concatenated timesteps
        """
        # Convert to patches
        patches = self.latent_to_patches(latent)  # [num_patches, B, C, ph, pw]

        # Assume B=1, squeeze batch dimension
        patches = patches.squeeze(1)  # [num_patches, C, ph, pw]

        # Extract patches for each group
        patches_a = patches[group_a_indices]  # [len_a, C, ph, pw]
        patches_b = patches[group_b_indices]  # [len_b, C, ph, pw]

        # Create timestep tensors
        timesteps_a = torch.full((len(group_a_indices),), timestep_a, device=latent.device)
        timesteps_b = torch.full((len(group_b_indices),), timestep_b, device=latent.device)

        return patches_a, patches_b, timesteps_a, timesteps_b

    def merge_strategywise_results(
        self,
        patches_a: torch.Tensor,
        patches_b: torch.Tensor,
        group_a_indices: torch.Tensor,
        group_b_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge results from Group A and Group B back into full latent.

        Args:
            patches_a: [len_a, C, ph, pw]
            patches_b: [len_b, C, ph, pw]
            group_a_indices: Indices for Group A
            group_b_indices: Indices for Group B

        Returns:
            latent: [1, C, H, W]
        """
        C, ph, pw = patches_a.shape[1:]

        # Create full patches tensor
        patches_full = torch.zeros(
            self.num_patches, C, ph, pw,
            device=patches_a.device,
            dtype=patches_a.dtype
        )

        # Fill in Group A and Group B
        patches_full[group_a_indices] = patches_a
        patches_full[group_b_indices] = patches_b

        # Add batch dimension
        patches_full = patches_full.unsqueeze(1)  # [num_patches, 1, C, ph, pw]

        # Convert back to latent
        latent = self.patches_to_latent(patches_full)

        return latent

    def denoise_step_strategywise(
        self,
        pipeline,
        latent: torch.Tensor,
        group_a_indices: torch.Tensor,
        group_b_indices: torch.Tensor,
        timestep_a: float,
        timestep_b: float,
        target_timestep_a: float,
        target_timestep_b: float,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_attention_mask: torch.Tensor,
        guidance_scale: float,
        image_rotary_emb: torch.Tensor,
        cross_attention_kwargs: dict,
    ) -> torch.Tensor:
        """
        Perform one denoising step with different timesteps for different groups.

        This is the core function that enables per-patch adaptive sampling.
        """
        # 1. Prepare strategywise batches
        patches_a, patches_b, timesteps_a, timesteps_b = self.prepare_strategywise_batches(
            latent, group_a_indices, group_b_indices, timestep_a, timestep_b
        )

        # 2. Concatenate into single batch
        patches_combined = torch.cat([patches_a, patches_b], dim=0)  # [N_total, C, ph, pw]
        timesteps_combined = torch.cat([timesteps_a, timesteps_b], dim=0)  # [N_total]

        # 3. Expand prompt embeddings to match batch size
        N_total = len(patches_combined)
        prompt_embeds_expanded = prompt_embeds.expand(N_total, -1, -1)
        prompt_attention_mask_expanded = prompt_attention_mask.expand(N_total, -1)
        negative_prompt_embeds_expanded = negative_prompt_embeds.expand(N_total, -1, -1)
        negative_prompt_attention_mask_expanded = negative_prompt_attention_mask.expand(N_total, -1)

        # 4. Prepare for CFG
        if guidance_scale > 1.0:
            patches_model_input = torch.cat([patches_combined] * 2)  # [2*N_total, C, ph, pw]
            timestep_input = torch.cat([timesteps_combined] * 2)  # [2*N_total]
            prompt_embeds_input = torch.cat([negative_prompt_embeds_expanded, prompt_embeds_expanded])
            attention_mask_input = torch.cat([negative_prompt_attention_mask_expanded, prompt_attention_mask_expanded])
        else:
            patches_model_input = patches_combined
            timestep_input = timesteps_combined
            prompt_embeds_input = prompt_embeds_expanded
            attention_mask_input = prompt_attention_mask_expanded

        # 5. Transformer forward
        with torch.no_grad():
            noise_pred = pipeline.transformer(
                hidden_states=patches_model_input,
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds_input,
                encoder_mask=attention_mask_input,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Handle Lumina's output format
            noise_pred = noise_pred.chunk(2, dim=1)[0]

        # 6. Apply CFG
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 7. Split back to Group A and Group B
        noise_pred_a = noise_pred[:len(group_a_indices)]
        noise_pred_b = noise_pred[len(group_a_indices):]

        patches_a_current = patches_a
        patches_b_current = patches_b

        # 8. Apply scheduler step with different dt for each group
        num_train_timesteps = pipeline.scheduler.config.num_train_timesteps

        # Group A step
        sigma_a_current = timestep_a / num_train_timesteps
        sigma_a_next = target_timestep_a / num_train_timesteps
        dt_a = sigma_a_next - sigma_a_current
        patches_a_next = patches_a_current + dt_a * noise_pred_a

        # Group B step
        sigma_b_current = timestep_b / num_train_timesteps
        sigma_b_next = target_timestep_b / num_train_timesteps
        dt_b = sigma_b_next - sigma_b_current
        patches_b_next = patches_b_current + dt_b * noise_pred_b

        # 9. Merge results
        latent_next = self.merge_strategywise_results(
            patches_a_next,
            patches_b_next,
            group_a_indices,
            group_b_indices
        )

        return latent_next

    def update_strategies(self, group_a_indices: torch.Tensor, group_b_indices: torch.Tensor):
        """Update strategy assignments for patches."""
        self.patch_strategies = {}
        for idx in group_a_indices:
            self.patch_strategies[idx.item()] = 'a'
        for idx in group_b_indices:
            self.patch_strategies[idx.item()] = 'b'

    def get_patch_info(self) -> Dict:
        """Get current patch state information for debugging/visualization."""
        return {
            'num_patches': self.num_patches,
            'grid_size': (self.num_patches_h, self.num_patches_w),
            'patch_size': self.patch_size,
            'strategies': self.patch_strategies,
            'timesteps': self.patch_timesteps.tolist()
        }
