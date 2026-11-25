"""
Complexity Calculator for Adaptive Sampling

This module computes complexity scores for image patches based on
downsampling-upsampling reconstruction error (L2 norm).
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange


def compute_patch_complexity(
    latent: torch.Tensor,
    patch_size: int,
    vae_scale: int = 8,
) -> torch.Tensor:
    """
    Compute complexity score for each patch using downsampling-upsampling L2 error.

    The complexity is measured as:
    1. Divide latent into patches
    2. For each patch: downsample 2x -> upsample 2x
    3. Compute L2 norm between original and reconstructed patch

    Args:
        latent: Latent tensor of shape [B, C, H, W]
        patch_size: Size of patches in latent space (e.g., 2)
        vae_scale: VAE downsampling factor (default 8)

    Returns:
        complexity_scores: Tensor of shape [num_patches] containing L2 norm for each patch
    """
    batch_size, channels, height, width = latent.shape

    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w

    # Rearrange into patches: [B, C, H, W] -> [num_patches, B, C, patch_size, patch_size]
    patches = rearrange(
        latent,
        'b c (h p1) (w p2) -> (h w) b c p1 p2',
        p1=patch_size,
        p2=patch_size
    )

    # Reshape for batch processing: [num_patches * B, C, patch_size, patch_size]
    patches_flat = rearrange(patches, 'n b c p1 p2 -> (n b) c p1 p2')

    # Downsample by 2x
    downsampled = F.interpolate(
        patches_flat,
        scale_factor=0.5,
        mode='bilinear',
        align_corners=False
    )

    # Upsample back to original size
    upsampled = F.interpolate(
        downsampled,
        size=(patch_size, patch_size),
        mode='bilinear',
        align_corners=False
    )

    # Compute L2 difference
    diff = patches_flat - upsampled
    diff_flat = diff.reshape(num_patches, batch_size, -1)  # [num_patches, B, C*p1*p2]

    # Compute L2 norm per patch (average over batch)
    complexity = torch.norm(diff_flat, p=2, dim=2).mean(dim=1)  # [num_patches]

    return complexity


def select_strategy_groups(
    complexity_scores: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split patches into two groups based on complexity scores.

    Args:
        complexity_scores: Tensor of shape [num_patches]
        threshold: Ratio for top-k selection (0.5 = top 50%)

    Returns:
        group_a_indices: Indices of complex patches (top threshold%)
        group_b_indices: Indices of simple patches (bottom (1-threshold)%)
    """
    num_patches = len(complexity_scores)

    # Sort by complexity (descending)
    sorted_indices = torch.argsort(complexity_scores, descending=True)

    # Split point
    split_point = int(num_patches * threshold)

    # Group A: high complexity (use multi-step)
    group_a_indices = sorted_indices[:split_point]

    # Group B: low complexity (use single-step)
    group_b_indices = sorted_indices[split_point:]

    return group_a_indices, group_b_indices


def compute_and_split_patches(
    latent: torch.Tensor,
    patch_size: int,
    threshold: float = 0.5,
    vae_scale: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function to compute complexity and split in one call.

    Args:
        latent: Latent tensor [B, C, H, W]
        patch_size: Patch size in latent space
        threshold: Complexity threshold for grouping
        vae_scale: VAE downsampling factor

    Returns:
        complexity_scores: Complexity score per patch [num_patches]
        group_a_indices: Indices for complex patches
        group_b_indices: Indices for simple patches
    """
    complexity = compute_patch_complexity(latent, patch_size, vae_scale)
    group_a, group_b = select_strategy_groups(complexity, threshold)

    return complexity, group_a, group_b


def visualize_complexity_map(
    complexity_scores: torch.Tensor,
    height: int,
    width: int,
    patch_size: int
) -> torch.Tensor:
    """
    Convert complexity scores to a 2D heatmap.

    Args:
        complexity_scores: Tensor of shape [num_patches]
        height: Latent height
        width: Latent width
        patch_size: Patch size

    Returns:
        heatmap: Tensor of shape [H//patch_size, W//patch_size]
    """
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    heatmap = complexity_scores.reshape(num_patches_h, num_patches_w)
    return heatmap


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    print("Testing complexity calculator...")

    # Create dummy latent (1024x1024 image -> 128x128 latent)
    batch_size = 1
    channels = 4
    latent_h, latent_w = 128, 128
    patch_size = 2

    dummy_latent = torch.randn(batch_size, channels, latent_h, latent_w)

    # Compute complexity
    complexity = compute_patch_complexity(dummy_latent, patch_size)
    print(f"Complexity scores shape: {complexity.shape}")
    print(f"Complexity range: [{complexity.min():.4f}, {complexity.max():.4f}]")

    # Split into groups
    group_a, group_b = select_strategy_groups(complexity, threshold=0.5)
    print(f"Group A (complex): {len(group_a)} patches")
    print(f"Group B (simple): {len(group_b)} patches")

    # Visualize
    heatmap = visualize_complexity_map(complexity, latent_h, latent_w, patch_size)
    print(f"Heatmap shape: {heatmap.shape}")

    print("✓ All tests passed!")
