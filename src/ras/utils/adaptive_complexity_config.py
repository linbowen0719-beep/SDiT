"""
Adaptive Complexity Sampling Configuration

This module defines the configuration for adaptive complexity-based sampling strategy.
The strategy uses synchronization points to determine when to compute patch complexity
and apply different denoising strategies based on complexity scores.
"""

from typing import List, Tuple
import torch


class AdaptiveComplexityConfig:
    """Configuration for adaptive complexity sampling strategy."""

    # Default timesteps for 30-step inference
    DEFAULT_TIMESTEPS = [
        1000, 967, 934, 900, 867, 834, 800, 767, 734, 700,
        667, 634, 600, 567, 534, 500, 467, 434, 400, 367,
        334, 300, 267, 234, 200, 167, 134, 100, 67, 34, 0
    ]

    # Synchronization points where complexity is computed
    DEFAULT_SYN_TIMESTEPS = [
        1000, 933, 867, 800, 733, 667, 600, 533, 467, 400,
        333, 267, 200, 133, 67, 0
    ]

    def __init__(
        self,
        timesteps: List[int] = None,
        syn_timesteps: List[int] = None,
        complexity_threshold: float = 0.5,
    ):
        """
        Args:
            timesteps: Full timestep schedule for denoising
            syn_timesteps: Synchronization points for complexity computation
            complexity_threshold: Threshold for splitting patches (0.5 = top 50% use strategy A)
        """
        self.timesteps = timesteps or self.DEFAULT_TIMESTEPS
        self.syn_timesteps = syn_timesteps or self.DEFAULT_SYN_TIMESTEPS
        self.complexity_threshold = complexity_threshold

        # Validate configuration
        self._validate()

        # Build mapping from timesteps to syn_timesteps
        self.timestep_to_syn_mapping = self._build_timestep_mapping()

    def _validate(self):
        """Validate configuration parameters."""
        assert len(self.timesteps) > 0, "timesteps cannot be empty"
        assert len(self.syn_timesteps) > 0, "syn_timesteps cannot be empty"
        assert 0 <= self.complexity_threshold <= 1, "complexity_threshold must be in [0, 1]"

        # Check that syn_timesteps are in descending order
        assert all(self.syn_timesteps[i] >= self.syn_timesteps[i+1]
                   for i in range(len(self.syn_timesteps)-1)), \
               "syn_timesteps must be in descending order"

        # Check that timesteps are in descending order
        assert all(self.timesteps[i] >= self.timesteps[i+1]
                   for i in range(len(self.timesteps)-1)), \
               "timesteps must be in descending order"

    def _build_timestep_mapping(self):
        """Build mapping from each timestep to nearest syn_timestep."""
        mapping = {}
        for t in self.timesteps:
            # Find nearest syn_timestep
            nearest_syn = min(self.syn_timesteps, key=lambda s: abs(s - t))
            mapping[t] = nearest_syn
        return mapping

    def is_sync_point(self, timestep: int) -> bool:
        """Check if given timestep is a synchronization point."""
        # Directly check if timestep is in syn_timesteps (with tolerance)
        for sync_t in self.syn_timesteps:
            if abs(timestep - sync_t) <= 5:  # Tolerance of 5
                return True
        return False

    def is_last_sync_point(self, timestep: int) -> bool:
        """Check if given timestep is the last synchronization point."""
        if not self.is_sync_point(timestep):
            return False
        # Find the nearest sync point
        nearest_syn = min(self.syn_timesteps, key=lambda s: abs(s - timestep))
        return nearest_syn == self.syn_timesteps[-1]

    def get_sync_index(self, timestep: int) -> int:
        """Get index of current sync point in syn_timesteps."""
        # Find nearest sync point
        nearest_syn = min(self.syn_timesteps, key=lambda s: abs(s - timestep))
        if abs(timestep - nearest_syn) <= 5:  # Within tolerance
            return self.syn_timesteps.index(nearest_syn)
        return -1

    def get_next_sync_timestep(self, current_timestep: int) -> int:
        """Get the next synchronization timestep."""
        # Find nearest sync point
        nearest_syn = min(self.syn_timesteps, key=lambda s: abs(s - current_timestep))
        if abs(current_timestep - nearest_syn) > 5:  # Not at a sync point
            return None

        current_idx = self.syn_timesteps.index(nearest_syn)
        if current_idx >= len(self.syn_timesteps) - 1:
            return None

        return self.syn_timesteps[current_idx + 1]

    def get_intermediate_timesteps(self, t_start: int, t_end: int) -> List[int]:
        """
        Get all timesteps between two sync points (inclusive of start, exclusive of end).

        Args:
            t_start: Starting timestep (current sync point)
            t_end: Ending timestep (next sync point)

        Returns:
            List of timesteps in the range [t_end, t_start]
        """
        # Filter timesteps that fall in the range
        intermediate = [t for t in self.timesteps if t_end < t <= t_start]
        return sorted(intermediate, reverse=True)  # Descending order

    def get_all_intermediate_including_end(self, t_start: int, t_end: int) -> List[int]:
        """
        Get all timesteps between two sync points including both endpoints.

        Args:
            t_start: Starting timestep (current sync point)
            t_end: Ending timestep (next sync point)

        Returns:
            List of timesteps in the range [t_end, t_start] (both inclusive)
        """
        intermediate = [t for t in self.timesteps if t_end <= t <= t_start]
        return sorted(intermediate, reverse=True)


# Global config instance (can be overridden)
_global_config = None


def get_config() -> AdaptiveComplexityConfig:
    """Get the global adaptive complexity configuration."""
    global _global_config
    if _global_config is None:
        _global_config = AdaptiveComplexityConfig()
    return _global_config


def set_config(config: AdaptiveComplexityConfig):
    """Set the global adaptive complexity configuration."""
    global _global_config
    _global_config = config


def reset_config():
    """Reset to default configuration."""
    global _global_config
    _global_config = AdaptiveComplexityConfig()
