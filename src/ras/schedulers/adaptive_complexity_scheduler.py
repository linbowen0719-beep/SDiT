"""
Adaptive Complexity Scheduler

Extends RASFlowMatchEulerDiscreteScheduler to support adaptive complexity-based sampling.
Provides helper methods for synchronization point detection and timestep management.
"""

from typing import List, Optional, Union
import torch
import numpy as np

from .ras_scheduling_flow_match_euler_discrete import RASFlowMatchEulerDiscreteScheduler
from ..utils.adaptive_complexity_config import AdaptiveComplexityConfig


class AdaptiveComplexityScheduler(RASFlowMatchEulerDiscreteScheduler):
    """
    Scheduler with adaptive complexity support.

    This scheduler extends the base RAS scheduler with methods to:
    - Identify synchronization points
    - Get intermediate timesteps between sync points
    - Manage adaptive timestep schedules for different patch groups
    """

    def __init__(
        self,
        *args,
        syn_timesteps: Optional[List[int]] = None,
        custom_timesteps: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Args:
            syn_timesteps: List of synchronization points
            custom_timesteps: Custom full timestep schedule
            *args, **kwargs: Passed to parent RASFlowMatchEulerDiscreteScheduler
        """
        super().__init__(*args, **kwargs)

        # Store adaptive complexity configuration
        self.syn_timesteps = syn_timesteps or AdaptiveComplexityConfig.DEFAULT_SYN_TIMESTEPS
        self.custom_timesteps = custom_timesteps
        self._adaptive_config = None

    def set_adaptive_config(self, config: AdaptiveComplexityConfig):
        """Set the adaptive complexity configuration."""
        self._adaptive_config = config
        self.syn_timesteps = config.syn_timesteps

    def is_sync_point(self, timestep: Union[int, float, torch.Tensor]) -> bool:
        """
        Check if the given timestep is a synchronization point.

        Args:
            timestep: Current timestep (can be tensor, int, or float)

        Returns:
            True if this is a sync point, False otherwise
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()

        timestep = int(round(float(timestep)))

        # Check if timestep matches any sync point (with tolerance)
        for sync_t in self.syn_timesteps:
            if abs(timestep - sync_t) <= 5:  # Tolerance of 5
                return True
        return False

    def is_last_sync_point(self, timestep: Union[int, float, torch.Tensor]) -> bool:
        """
        Check if the given timestep is the last synchronization point.

        Args:
            timestep: Current timestep

        Returns:
            True if this is the last sync point, False otherwise
        """
        if not self.is_sync_point(timestep):
            return False

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()

        timestep = int(round(float(timestep)))

        # Find nearest sync point
        nearest_sync = min(self.syn_timesteps, key=lambda t: abs(t - timestep))

        return nearest_sync == self.syn_timesteps[-1]

    def get_sync_index(self, timestep: Union[int, float, torch.Tensor]) -> int:
        """
        Get the index of the current sync point in syn_timesteps.

        Args:
            timestep: Current timestep

        Returns:
            Index in syn_timesteps, or -1 if not a sync point
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()

        timestep = int(round(float(timestep)))

        # Find nearest sync point
        min_dist = float('inf')
        best_idx = -1

        for idx, sync_t in enumerate(self.syn_timesteps):
            dist = abs(timestep - sync_t)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        if min_dist <= 5:  # Tolerance
            return best_idx
        return -1

    def get_next_sync_timestep(self, current_timestep: Union[int, float, torch.Tensor]) -> Optional[int]:
        """
        Get the next synchronization timestep after the current one.

        Args:
            current_timestep: Current timestep

        Returns:
            Next sync timestep, or None if at the last sync point
        """
        current_idx = self.get_sync_index(current_timestep)

        if current_idx == -1 or current_idx >= len(self.syn_timesteps) - 1:
            return None

        return self.syn_timesteps[current_idx + 1]

    def get_intermediate_timesteps(
        self,
        t_start: Union[int, float, torch.Tensor],
        t_end: Union[int, float, torch.Tensor]
    ) -> List[int]:
        """
        Get all timesteps between two sync points (exclusive of t_end).

        Args:
            t_start: Starting timestep (current sync point)
            t_end: Ending timestep (next sync point)

        Returns:
            List of timesteps in range (t_end, t_start], in descending order
        """
        if isinstance(t_start, torch.Tensor):
            t_start = t_start.item()
        if isinstance(t_end, torch.Tensor):
            t_end = t_end.item()

        t_start = int(round(float(t_start)))
        t_end = int(round(float(t_end)))

        # Use custom timesteps if provided, otherwise use scheduler timesteps
        if self.custom_timesteps is not None:
            available_timesteps = self.custom_timesteps
        else:
            # Convert scheduler timesteps to list of ints
            available_timesteps = [int(round(t.item() if isinstance(t, torch.Tensor) else t))
                                   for t in self.timesteps]

        # Filter timesteps in range (t_end, t_start]
        intermediate = [t for t in available_timesteps if t_end < t <= t_start]

        return sorted(intermediate, reverse=True)

    def get_all_intermediate_including_end(
        self,
        t_start: Union[int, float, torch.Tensor],
        t_end: Union[int, float, torch.Tensor]
    ) -> List[int]:
        """
        Get all timesteps between two sync points (inclusive of both ends).

        Args:
            t_start: Starting timestep (current sync point)
            t_end: Ending timestep (next sync point)

        Returns:
            List of timesteps in range [t_end, t_start], in descending order
        """
        if isinstance(t_start, torch.Tensor):
            t_start = t_start.item()
        if isinstance(t_end, torch.Tensor):
            t_end = t_end.item()

        t_start = int(round(float(t_start)))
        t_end = int(round(float(t_end)))

        # Use custom timesteps if provided
        if self.custom_timesteps is not None:
            available_timesteps = self.custom_timesteps
        else:
            available_timesteps = [int(round(t.item() if isinstance(t, torch.Tensor) else t))
                                   for t in self.timesteps]

        # Filter timesteps in range [t_end, t_start]
        intermediate = [t for t in available_timesteps if t_end <= t <= t_start]

        return sorted(intermediate, reverse=True)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        """
        Override set_timesteps to use custom timesteps if provided.

        Args:
            num_inference_steps: Number of inference steps
            device: Device to put timesteps on
            **kwargs: Additional arguments passed to parent
        """
        # If custom timesteps are provided, use them directly
        if self.custom_timesteps is not None:
            timesteps = np.array(self.custom_timesteps, dtype=np.float32)
            self.num_inference_steps = len(timesteps)

            # Convert to tensor
            self.timesteps = torch.from_numpy(timesteps).to(device=device)

            # Compute sigmas from timesteps
            sigmas = self.timesteps / self.config.num_train_timesteps

            # Apply shifting if needed
            if not self.config.use_dynamic_shifting:
                shift = self.shift if hasattr(self, 'shift') else self.config.shift
                sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

            # Reset step index
            self._step_index = None
            self._begin_index = None
        else:
            # Use parent implementation
            super().set_timesteps(num_inference_steps, device, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  num_train_timesteps={self.config.num_train_timesteps},\n"
            f"  num_inference_steps={self.num_inference_steps if hasattr(self, 'num_inference_steps') else 'not set'},\n"
            f"  syn_timesteps={self.syn_timesteps},\n"
            f"  custom_timesteps={'set' if self.custom_timesteps else 'not set'}\n"
            f")"
        )
