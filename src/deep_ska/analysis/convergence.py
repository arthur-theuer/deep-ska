
"""Pipeline functions for convergence analysis."""
import logging

import numpy as np

from ..core.simulation import ExpectationGeneratorSSA
from ..logging.helpers import log_without_linebreak
from .analyzers import (
    ExpectationGeneratorExactCV,
    ExpectationGeneratorIS,
    FinalTimeSubnetAnalyzer,
)


def run_convergence_ssa_repeats(generator: ExpectationGeneratorSSA, t_stop: float, n_time_samples: int, n_trajectories: int, validation_trajectories: set, analyzer: FinalTimeSubnetAnalyzer, n_repeats: int, first_data: dict = None) -> tuple[list[dict], list[dict]]:
    """Run repeated convergence SSA estimates."""
    max_length = len(str(n_repeats))
    repeat_results = []
    repeat_data = []
    start_idx = 0

    # Reuse first repeat if provided:
    if first_data is not None:
        logging.info(f"Reusing convergence 'SSA' repeat {1:{max_length}d}/{n_repeats} ...")
        repeat_data.append(first_data)
        analyzer.compute_final_ssa_estimate(first_data, results_dict:={})
        repeat_results.append(results_dict)
        start_idx = 1

    # Do the remaining repeats:
    for i in range(start_idx, n_repeats):
        log_without_linebreak(f"Running convergence 'SSA' repeat {i+1:{max_length}d}/{n_repeats} ...")
        data = generator.sample_temporal_rtc_trajectories(t_stop, n_time_samples, n_trajectories, validation_trajectories)
        repeat_data.append(data)
        analyzer.compute_final_ssa_estimate(data, results_dict:={})
        repeat_results.append(results_dict)

    return repeat_data, repeat_results


def run_convergence_deep_cv_repeats(repeat_data: list[dict], analyzer: FinalTimeSubnetAnalyzer) -> list[dict]:
    """Run repeated convergence DeepCV estimates."""
    max_length = len(str(len(repeat_data)))
    repeat_results = []

    for i in range(0, len(repeat_data)):
        log_without_linebreak(f"Running convergence 'SSA with DeepCV' repeat {i+1:{max_length}d}/{len(repeat_data)} ...")
        data = repeat_data[i]
        analyzer.compute_final_deep_cv_estimate(data, results_dict:={})
        repeat_results.append(results_dict)

    return repeat_results


def run_convergence_exact_deep_cv_repeats(analyzer: FinalTimeSubnetAnalyzer, exact_cv_generator: ExpectationGeneratorExactCV, val_config: dict, n_repeats: int, first_results: dict = None) -> list[dict]:
    """Run repeated convergence exact DeepCV estimates."""
    max_length = len(str(n_repeats))
    repeat_results = []
    start_idx = 0

    # Extract validation configuration parameters:
    t_stop = val_config["conv_t_stop"]
    n_trajectories = int(val_config["conv_n_trajectories"] * val_config["conv_ssa_with_cv_fraction"])

    # Reuse first repeat if provided:
    if first_results is not None:
        logging.info(f"Reusing convergence 'SSA with DeepCV' repeat {1:{max_length}d}/{n_repeats} ...")
        repeat_results.append(first_results)
        start_idx = 1

    for i in range(start_idx, n_repeats):
        log_without_linebreak(f"Running convergence 'SSA with DeepCV' repeat {i+1:{max_length}d}/{n_repeats} ...")
        trajectories = exact_cv_generator.sample_final_exact_cv_trajectories(t_stop, n_trajectories)

        final_output = trajectories["final_output"]
        adjustment_term = trajectories["adjustment_term"]
        diff_estimate = final_output - adjustment_term
        # Compute mean, variance, CV and 95% CI of the importance sampling estimate:
        mean, var, cv, ci, tolerance, n_samples, var_ci_lower, var_ci_upper = analyzer.compute_estimate_statistics(diff_estimate)
        # Also compute the cumulative means:
        cummean = np.cumsum(diff_estimate, axis=0) / np.arange(1, diff_estimate.shape[0] + 1).reshape((-1,) + (1,) * (diff_estimate.ndim - 1))
        # Also compute the mean of the adjustment term (should be close to zero):
        adjustment_mean = np.mean(adjustment_term, axis=0)
        # Store the results in a dictionary:
        results_dict = {"diff_mean": mean, "diff_var": var, "diff_cv": cv, "diff_ci": ci, "diff_tolerance": tolerance, "diff_adjustment_mean": adjustment_mean, "diff_samples": n_samples, "diff_var_ci_lower": var_ci_lower, "diff_var_ci_upper": var_ci_upper, "diff_cummean": cummean}
        # Append the results dictionary to the list of repeat results:
        repeat_results.append(results_dict)

    return repeat_results


def run_convergence_deep_is_repeats(analyzer: FinalTimeSubnetAnalyzer, is_generator: ExpectationGeneratorIS, val_config: dict, active_trajectories: set, n_repeats: int, first_results: dict = None) -> list[dict]:
    """Run repeated convergence DeepIS estimates."""
    max_length = len(str(n_repeats))
    repeat_results = []
    start_idx = 0

    # Extract validation configuration parameters:
    t_stop = val_config["conv_t_stop"]
    n_time_samples = val_config["conv_n_time_samples"]
    n_trajectories = int(val_config["conv_n_trajectories"] * val_config["conv_ssa_with_is_fraction"])

    # Reuse first repeat if provided:
    if first_results is not None:
        logging.info(f"Reusing convergence 'SSA with DeepIS' repeat {1:{max_length}d}/{n_repeats} ...")
        repeat_results.append(first_results)
        start_idx = 1

    for i in range(start_idx, n_repeats):
        log_without_linebreak(f"Running convergence 'SSA with DeepIS' repeat {i+1:{max_length}d}/{n_repeats} ...")
        trajectories = is_generator.sample_final_deep_is_trajectories(t_stop, n_time_samples, n_trajectories, active_trajectories)

        if val_config["compute_is_estimate"] == False:  # changed to False if the simulation took too long
            return

        is_trajectories = trajectories["is_trajectories"][:, -1, :]
        # Compute mean, variance, CV and 95% CI of the importance sampling estimate:
        mean, var, cv, ci, _, n_samples, var_ci_lower, var_ci_upper = analyzer.compute_estimate_statistics(is_trajectories)
        # Also compute the cumulative means:
        cummean = np.cumsum(is_trajectories, axis=0) / np.arange(1, is_trajectories.shape[0] + 1).reshape((-1,) + (1,) * (is_trajectories.ndim - 1))
        # Store the results in a dictionary:
        results_dict = {"is_raw_mean": mean, "is_raw_var": var, "is_raw_cv": cv, "is_raw_ci": ci, "is_raw_samples": n_samples, "is_raw_var_ci_lower": var_ci_lower, "is_raw_var_ci_upper": var_ci_upper, "is_raw_cummean": cummean}
        # Append the results dictionary to the list of repeat results:
        repeat_results.append(results_dict)

    return repeat_results
