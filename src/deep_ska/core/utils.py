"""Utility functions used by the package."""

import os

import numpy as np


def create_state_space(
    n_species: int, lower_bound: int | list[int], upper_bound: int | list[int]
) -> np.ndarray:
    """Create a meshgrid over the state space defined by config boundaries."""
    # Ensure bounds are lists (a single value is broadcasted to all species):
    lower_bound = [lower_bound] * n_species if isinstance(lower_bound, int) else lower_bound
    upper_bound = [upper_bound] * n_species if isinstance(upper_bound, int) else upper_bound

    # Create the state space for each species based on the bounds:
    states = [np.arange(low, up) for low, up in zip(lower_bound, upper_bound, strict=True)]

    # Create the meshgrid and reshape it to a 2D array:
    return np.array(np.meshgrid(*states, indexing="ij"), dtype=np.float64).T.reshape(-1, n_species)


def get_available_cpus(user_override: int | None = None) -> int:
    """Determine the number of jobs to use for parallel processing."""
    if user_override is not None:
        return user_override

    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])

    return max(1, os.cpu_count() - 1)  # one CPU for the system
