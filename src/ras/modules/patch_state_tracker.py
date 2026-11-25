"""
Patch State Tracker

Tracks the denoising progress of each patch throughout the adaptive sampling process.
"""

import torch
from typing import Dict, List, Tuple
import numpy as np


class PatchStateTracker:
    """
    Tracks the current timestep and strategy for each patch during denoising.
    """

    def __init__(self, num_patches: int, initial_timestep: float = 1000.0, device: str = "cuda"):
        """
        Args:
            num_patches: Total number of patches
            initial_timestep: Starting timestep (default 1000)
            device: Device to store tensors on
        """
        self.num_patches = num_patches
        self.device = device

        # Current timestep for each patch
        self.current_timesteps = torch.full(
            (num_patches,), initial_timestep, dtype=torch.float32, device=device
        )

        # Strategy assignment: 'a' or 'b'
        self.strategies = {}  # Dict[int, str]

        # History for visualization
        self.history = []

    def update_strategies(self, group_a_indices: torch.Tensor, group_b_indices: torch.Tensor):
        """
        Update strategy assignments at a synchronization point.

        Args:
            group_a_indices: Indices of complex patches (strategy A)
            group_b_indices: Indices of simple patches (strategy B)
        """
        self.strategies = {}

        for idx in group_a_indices:
            self.strategies[idx.item()] = 'a'

        for idx in group_b_indices:
            self.strategies[idx.item()] = 'b'

    def update_timesteps(self, group_a_indices: torch.Tensor, group_b_indices: torch.Tensor,
                         timestep_a: float, timestep_b: float):
        """
        Update the current timestep for each group.

        Args:
            group_a_indices: Indices for Group A
            group_b_indices: Indices for Group B
            timestep_a: New timestep for Group A
            timestep_b: New timestep for Group B
        """
        self.current_timesteps[group_a_indices] = timestep_a
        self.current_timesteps[group_b_indices] = timestep_b

    def snapshot(self, step_name: str = ""):
        """
        Save current state to history for visualization.

        Args:
            step_name: Optional name for this step
        """
        self.history.append({
            'step_name': step_name,
            'timesteps': self.current_timesteps.cpu().clone(),
            'strategies': self.strategies.copy()
        })

    def get_strategy_indices(self) -> Tuple[List[int], List[int]]:
        """
        Get patch indices for each strategy.

        Returns:
            group_a_indices: List of indices with strategy 'a'
            group_b_indices: List of indices with strategy 'b'
        """
        group_a = [idx for idx, strat in self.strategies.items() if strat == 'a']
        group_b = [idx for idx, strat in self.strategies.items() if strat == 'b']

        return group_a, group_b

    def get_timestep_grid(self, grid_h: int, grid_w: int) -> np.ndarray:
        """
        Get current timesteps as a 2D grid for visualization.

        Args:
            grid_h: Number of patches in height
            grid_w: Number of patches in width

        Returns:
            grid: [grid_h, grid_w] array of timesteps
        """
        timesteps_np = self.current_timesteps.cpu().numpy()
        grid = timesteps_np.reshape(grid_h, grid_w)
        return grid

    def get_strategy_grid(self, grid_h: int, grid_w: int) -> np.ndarray:
        """
        Get strategy assignments as a 2D grid.

        Args:
            grid_h: Number of patches in height
            grid_w: Number of patches in width

        Returns:
            grid: [grid_h, grid_w] array (0 for 'a', 1 for 'b', -1 for unassigned)
        """
        strategy_array = np.full(self.num_patches, -1, dtype=np.int32)

        for idx, strat in self.strategies.items():
            strategy_array[idx] = 0 if strat == 'a' else 1

        grid = strategy_array.reshape(grid_h, grid_w)
        return grid

    def print_status(self, grid_h: int, grid_w: int):
        """Print current denoising status."""
        print(f"\n{'='*60}")
        print("Patch Denoising Status")
        print(f"{'='*60}")

        timestep_grid = self.get_timestep_grid(grid_h, grid_w)
        strategy_grid = self.get_strategy_grid(grid_h, grid_w)

        group_a_count = sum(1 for s in self.strategies.values() if s == 'a')
        group_b_count = sum(1 for s in self.strategies.values() if s == 'b')

        print(f"Group A (complex): {group_a_count} patches")
        print(f"Group B (simple):  {group_b_count} patches")
        print(f"\nTimestep Range: [{timestep_grid.min():.1f}, {timestep_grid.max():.1f}]")

        if len(self.strategies) > 0:
            group_a_timesteps = [self.current_timesteps[i].item()
                                for i, s in self.strategies.items() if s == 'a']
            group_b_timesteps = [self.current_timesteps[i].item()
                                for i, s in self.strategies.items() if s == 'b']

            if len(group_a_timesteps) > 0:
                print(f"Group A avg timestep: {np.mean(group_a_timesteps):.1f}")
            if len(group_b_timesteps) > 0:
                print(f"Group B avg timestep: {np.mean(group_b_timesteps):.1f}")

        print(f"{'='*60}\n")

    def get_history_summary(self) -> List[Dict]:
        """Get a summary of the denoising history."""
        summary = []

        for i, snapshot in enumerate(self.history):
            timesteps = snapshot['timesteps'].numpy()
            strategies = snapshot['strategies']

            summary.append({
                'step': i,
                'step_name': snapshot['step_name'],
                'min_timestep': float(timesteps.min()),
                'max_timestep': float(timesteps.max()),
                'mean_timestep': float(timesteps.mean()),
                'num_group_a': sum(1 for s in strategies.values() if s == 'a'),
                'num_group_b': sum(1 for s in strategies.values() if s == 'b'),
            })

        return summary

    def reset(self, initial_timestep: float = 1000.0):
        """Reset tracker to initial state."""
        self.current_timesteps.fill_(initial_timestep)
        self.strategies = {}
        self.history = []


# Utility functions for visualization
def visualize_patch_timesteps(tracker: PatchStateTracker, grid_h: int, grid_w: int):
    """
    Create a simple text-based visualization of patch timesteps.

    Args:
        tracker: PatchStateTracker instance
        grid_h: Grid height
        grid_w: Grid width
    """
    import matplotlib.pyplot as plt

    timestep_grid = tracker.get_timestep_grid(grid_h, grid_w)
    strategy_grid = tracker.get_strategy_grid(grid_h, grid_w)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Timestep heatmap
    im1 = ax1.imshow(timestep_grid, cmap='viridis', interpolation='nearest')
    ax1.set_title('Current Timesteps')
    ax1.set_xlabel('Patch X')
    ax1.set_ylabel('Patch Y')
    plt.colorbar(im1, ax=ax1, label='Timestep')

    # Strategy map
    im2 = ax2.imshow(strategy_grid, cmap='RdYlBu', interpolation='nearest', vmin=-1, vmax=1)
    ax2.set_title('Strategy Assignment')
    ax2.set_xlabel('Patch X')
    ax2.set_ylabel('Patch Y')

    # Add colorbar with labels
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[-1, 0, 1])
    cbar2.set_label('Strategy')
    cbar2.ax.set_yticklabels(['Unassigned', 'A (Multi-step)', 'B (Single-step)'])

    plt.tight_layout()
    return fig
