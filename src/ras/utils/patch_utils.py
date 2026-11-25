"""
Utility functions for patch operations

This module provides utilities for patch-based operations,
including 8-neighbor expansion for spatial clustering.
"""

import torch
import numpy as np
from typing import Union, Set


def expand_patches_no_expansion(
    patch_indices: Union[torch.Tensor, np.ndarray, Set[int]],
    grid_h: int,
    grid_w: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    No expansion - return selected patches as-is.

    Args:
        patch_indices: Indices of patches to keep (1D, in raster scan order)
        grid_h: Number of patches in height dimension
        grid_w: Number of patches in width dimension
        device: Device for output tensor

    Returns:
        expanded_indices: Tensor of patch indices (1D, unique, sorted)
    """
    # Convert to set and back to sorted tensor
    if isinstance(patch_indices, torch.Tensor):
        patch_set = set(patch_indices.cpu().numpy().tolist())
    elif isinstance(patch_indices, np.ndarray):
        patch_set = set(patch_indices.tolist())
    else:
        patch_set = set(patch_indices)

    expanded_indices = torch.tensor(
        sorted(list(patch_set)),
        dtype=torch.long,
        device=device
    )

    return expanded_indices


def expand_patches_4_neighbors(
    patch_indices: Union[torch.Tensor, np.ndarray, Set[int]],
    grid_h: int,
    grid_w: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Expand patch indices to include 4-neighbor surface patches.

    For each patch in the input, includes:
    - The patch itself
    - 4 surrounding neighbors (up, down, left, right only, no diagonals)

    Handles boundary cases automatically (edge patches have fewer neighbors).

    Args:
        patch_indices: Indices of patches to expand (1D, in raster scan order)
        grid_h: Number of patches in height dimension
        grid_w: Number of patches in width dimension
        device: Device for output tensor

    Returns:
        expanded_indices: Tensor of expanded patch indices (1D, unique, sorted)

    Example:
        >>> # 4x4 grid, select center patch (index 5)
        >>> expand_patches_4_neighbors([5], 4, 4)
        tensor([1, 4, 5, 6, 9])  # includes 4 neighbors only (no diagonals)
    """
    # Convert to set for efficient operations
    if isinstance(patch_indices, torch.Tensor):
        patch_set = set(patch_indices.cpu().numpy().tolist())
    elif isinstance(patch_indices, np.ndarray):
        patch_set = set(patch_indices.tolist())
    else:
        patch_set = set(patch_indices)

    # 4-neighbor offsets (row, col) - only cardinal directions
    # Order: top, left, right, bottom
    neighbor_offsets = [
        (-1, 0),           # top
        (0, -1), (0, 1),   # left, right
        (1, 0)             # bottom
    ]

    expanded_set = set(patch_set)  # Start with original patches

    for patch_idx in patch_set:
        # Convert 1D index to 2D coordinates
        row = patch_idx // grid_w
        col = patch_idx % grid_w

        # Add all valid neighbors
        for dr, dc in neighbor_offsets:
            neighbor_row = row + dr
            neighbor_col = col + dc

            # Check bounds
            if 0 <= neighbor_row < grid_h and 0 <= neighbor_col < grid_w:
                neighbor_idx = neighbor_row * grid_w + neighbor_col
                expanded_set.add(neighbor_idx)

    # Convert to sorted tensor
    expanded_indices = torch.tensor(
        sorted(list(expanded_set)),
        dtype=torch.long,
        device=device
    )

    return expanded_indices


def expand_patches_8_neighbors(
    patch_indices: Union[torch.Tensor, np.ndarray, Set[int]],
    grid_h: int,
    grid_w: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Expand patch indices to include 8-neighbor surface patches.

    For each patch in the input, includes:
    - The patch itself
    - 8 surrounding neighbors (up, down, left, right, and 4 diagonals)

    Handles boundary cases automatically (edge patches have fewer neighbors).

    Args:
        patch_indices: Indices of patches to expand (1D, in raster scan order)
        grid_h: Number of patches in height dimension
        grid_w: Number of patches in width dimension
        device: Device for output tensor

    Returns:
        expanded_indices: Tensor of expanded patch indices (1D, unique, sorted)

    Example:
        >>> # 4x4 grid, select center patch (index 5)
        >>> expand_patches_8_neighbors([5], 4, 4)
        tensor([0, 1, 2, 4, 5, 6, 8, 9, 10])  # includes 8 neighbors
    """
    # Convert to set for efficient operations
    if isinstance(patch_indices, torch.Tensor):
        patch_set = set(patch_indices.cpu().numpy().tolist())
    elif isinstance(patch_indices, np.ndarray):
        patch_set = set(patch_indices.tolist())
    else:
        patch_set = set(patch_indices)

    # 8-neighbor offsets (row, col)
    # Order: top-left, top, top-right, left, right, bottom-left, bottom, bottom-right
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),  # top row
        (0, -1),           (0, 1),   # middle row (left, right)
        (1, -1),  (1, 0),  (1, 1)    # bottom row
    ]

    expanded_set = set(patch_set)  # Start with original patches

    for patch_idx in patch_set:
        # Convert 1D index to 2D coordinates
        row = patch_idx // grid_w
        col = patch_idx % grid_w

        # Add all valid neighbors
        for dr, dc in neighbor_offsets:
            neighbor_row = row + dr
            neighbor_col = col + dc

            # Check bounds
            if 0 <= neighbor_row < grid_h and 0 <= neighbor_col < grid_w:
                neighbor_idx = neighbor_row * grid_w + neighbor_col
                expanded_set.add(neighbor_idx)

    # Convert to sorted tensor
    expanded_indices = torch.tensor(
        sorted(list(expanded_set)),
        dtype=torch.long,
        device=device
    )

    return expanded_indices


def expand_patches_by_radius(
    patch_indices: Union[torch.Tensor, np.ndarray, Set[int]],
    grid_h: int,
    grid_w: int,
    radius: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Expand patch indices by a given radius using Chebyshev distance (L∞ norm).

    For each patch in the input, includes all patches within the specified radius:
    - radius=0: No expansion (only the patch itself)
    - radius=1: 8-neighbor expansion (3x3 square)
    - radius=2: 24-neighbor expansion (5x5 square)
    - radius=3: 48-neighbor expansion (7x7 square)

    Handles boundary cases automatically (edge patches have fewer neighbors).

    Args:
        patch_indices: Indices of patches to expand (1D, in raster scan order)
        grid_h: Number of patches in height dimension
        grid_w: Number of patches in width dimension
        radius: Expansion radius (0, 1, 2, 3, ...)
        device: Device for output tensor

    Returns:
        expanded_indices: Tensor of expanded patch indices (1D, unique, sorted)

    Example:
        >>> # 5x5 grid, select center patch (index 12)
        >>> expand_patches_by_radius([12], 5, 5, radius=0)
        tensor([12])  # only itself
        >>> expand_patches_by_radius([12], 5, 5, radius=1)
        tensor([6, 7, 8, 11, 12, 13, 16, 17, 18])  # 3x3 = 9 patches
        >>> expand_patches_by_radius([12], 5, 5, radius=2)
        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])  # 5x5 = 25 patches
    """
    # Convert to set for efficient operations
    if isinstance(patch_indices, torch.Tensor):
        patch_set = set(patch_indices.cpu().numpy().tolist())
    elif isinstance(patch_indices, np.ndarray):
        patch_set = set(patch_indices.tolist())
    else:
        patch_set = set(patch_indices)

    expanded_set = set(patch_set)  # Start with original patches

    # Generate all offsets within the radius using Chebyshev distance
    for patch_idx in patch_set:
        # Convert 1D index to 2D coordinates
        row = patch_idx // grid_w
        col = patch_idx % grid_w

        # Add all patches within radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                neighbor_row = row + dr
                neighbor_col = col + dc

                # Check bounds
                if 0 <= neighbor_row < grid_h and 0 <= neighbor_col < grid_w:
                    neighbor_idx = neighbor_row * grid_w + neighbor_col
                    expanded_set.add(neighbor_idx)

    # Convert to sorted tensor
    expanded_indices = torch.tensor(
        sorted(list(expanded_set)),
        dtype=torch.long,
        device=device
    )

    return expanded_indices


def visualize_patch_selection(
    selected_indices: torch.Tensor,
    grid_h: int,
    grid_w: int
) -> np.ndarray:
    """
    Visualize selected patches as a 2D binary mask.

    Args:
        selected_indices: 1D tensor of selected patch indices
        grid_h: Number of patches in height
        grid_w: Number of patches in width

    Returns:
        mask: 2D numpy array (grid_h, grid_w) with 1 for selected, 0 for not selected
    """
    mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

    if isinstance(selected_indices, torch.Tensor):
        indices = selected_indices.cpu().numpy()
    else:
        indices = selected_indices

    for idx in indices:
        row = idx // grid_w
        col = idx % grid_w
        mask[row, col] = 1

    return mask


# Unit tests
if __name__ == "__main__":
    print("Testing patch_utils...")

    # Test 1: Single patch in center of 4x4 grid
    print("\n=== Test 1: Single center patch ===")
    expanded = expand_patches_8_neighbors([5], 4, 4, device="cpu")
    print(f"Input: [5] (center of 4x4)")
    print(f"Output: {expanded.tolist()}")
    expected = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    assert expanded.tolist() == expected, f"Expected {expected}, got {expanded.tolist()}"
    print("✓ Passed")

    # Test 2: Corner patch (top-left)
    print("\n=== Test 2: Top-left corner patch ===")
    expanded = expand_patches_8_neighbors([0], 4, 4, device="cpu")
    print(f"Input: [0] (top-left corner)")
    print(f"Output: {expanded.tolist()}")
    expected = [0, 1, 4, 5]  # only 3 neighbors + itself
    assert expanded.tolist() == expected, f"Expected {expected}, got {expanded.tolist()}"
    print("✓ Passed")

    # Test 3: Multiple patches
    print("\n=== Test 3: Multiple patches ===")
    expanded = expand_patches_8_neighbors([5, 10], 4, 4, device="cpu")
    print(f"Input: [5, 10] (two patches)")
    print(f"Output: {expanded.tolist()}")
    # Should include neighborhoods of both 5 and 10, with overlap removed
    assert len(expanded) > 0 and expanded[0] >= 0
    print("✓ Passed")

    # Test 4: Edge patch
    print("\n=== Test 4: Edge patch (not corner) ===")
    expanded = expand_patches_8_neighbors([1], 4, 4, device="cpu")
    print(f"Input: [1] (top edge)")
    print(f"Output: {expanded.tolist()}")
    expected = [0, 1, 2, 4, 5, 6]  # 5 neighbors + itself
    assert expanded.tolist() == expected, f"Expected {expected}, got {expanded.tolist()}"
    print("✓ Passed")

    # Test 5: Visualization
    print("\n=== Test 5: Visualization ===")
    mask = visualize_patch_selection(torch.tensor([5]), 4, 4)
    print("Mask for patch [5]:")
    print(mask)
    assert mask[1, 1] == 1  # patch 5 is at (1,1)
    print("✓ Passed")

    print("\n✅ All tests passed!")
