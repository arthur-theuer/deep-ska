"""Timing functions for test state analyses."""

import logging
import os
import re

import numpy as np

from ..core.initialization import RunContext
from ..logging.helpers import insert_blank_line, insert_subsection_marker


def test_state_log_file_analysis(run: RunContext, test_state: str = None) -> None:
    """Analyze log file to extract analysis timing information for a single test state."""
    insert_subsection_marker()

    if not os.path.exists(log_file_path := run.VAL_CONFIG["log_file_path"]):
        logging.warning(f"Log file not found: {log_file_path}")
        return

    with open(log_file_path) as f:
        log_content = f.read()

    # Extract analysis timing information:
    convergence_info, optimal_info, time_suboptimal_info, output_suboptimal_info = _extract_all_analysis_timing_info(log_content, test_state)

    if convergence_info:
        _analyze_convergence_timing(test_state, convergence_info)
    else:
        logging.info(f"No convergence timing information found for test state {test_state}.")

    insert_blank_line()

    if optimal_info:
        _analyze_sub_optimal_timing(test_state, optimal_info, type="optimal")
    else:
        logging.info(f"No optimal timing information found for test state {test_state}.")

    insert_blank_line()

    if output_suboptimal_info:
        _analyze_sub_optimal_timing(test_state, output_suboptimal_info, type="output")
    else:
        logging.info(f"No output-suboptimal timing information found for test state {test_state}.")

    insert_blank_line()

    if time_suboptimal_info:
        _analyze_sub_optimal_timing(test_state, time_suboptimal_info, type="time")
    else:
        logging.info(f"No time-suboptimal timing information found for test state {test_state}.")


def _analyze_convergence_timing(test_state: str, convergence_info: dict) -> None:
    """Analyze convergence timing information and log the results."""
    state_info = f" for test state {test_state}" if test_state else ""
    logging.info(f"Convergence timing analysis{state_info}:")

    insert_blank_line()
    # Compute dynamic widths for each column:
    max_method_w = max(len(m) for m in convergence_info.keys())
    max_total_time_w = max(len(f"{info['total_time']:.4f}") for info in convergence_info.values())
    max_traj_w = max(len(str(info["n_trajectories"])) for info in convergence_info.values())
    max_tpt_w = max(len(f"{info['time_per_trajectory']:.6f}") for info in convergence_info.values())

    # Log rows with column-based alignment:
    for method, info in convergence_info.items():
        total_time = info["total_time"]
        n_trajectories = info["n_trajectories"]
        time_per_trajectory = info["time_per_trajectory"]

        logging.info(
            f"{method:<{max_method_w}} : "
            f"{total_time:>{max_total_time_w}.4f} seconds for "
            f"{n_trajectories:>{max_traj_w}} trajectories → "
            f"{time_per_trajectory:>{max_tpt_w}.6f} seconds/trajectory"
        )

    insert_blank_line()
    # Print product of time per trajectory and variance if available:
    for method, info in convergence_info.items():
        time_variance_product = info.get("time_variance_product")
        if time_variance_product is not None:
            logging.info(
                f"{method:<{max_method_w}} : "
                f"{info['time_variance_product']} seconds/trajectory × variance"
            )


def _analyze_sub_optimal_timing(test_state: str, sub_optimal_info: dict, type: str) -> None:
    """Analyze (sub)optimal timing information and log the results."""
    state_info = f" for test state {test_state}" if test_state else ""
    if type == "optimal":
        logging.info(f"Optimal timing analysis{state_info}:")
    else:
        logging.info(f"{type.capitalize()}-suboptimal timing analysis{state_info}:")

    insert_blank_line()
    # Compute dynamic widths for each column:
    max_method_w = max(len(m) for m in sub_optimal_info.keys())
    max_total_time_w = max(len(f"{info['total_time']:.4f}") for info in sub_optimal_info.values())
    max_traj_w = max(len(str(info["n_trajectories"])) for info in sub_optimal_info.values())
    max_tpt_w = max(len(f"{info['time_per_trajectory']:.6f}") for info in sub_optimal_info.values())

    # Log rows with column-based alignment:
    for method, info in sub_optimal_info.items():
        total_time = info["total_time"]
        n_trajectories = info["n_trajectories"]
        time_per_trajectory = info["time_per_trajectory"]

        if type in ["optimal", "output"] and method in ["SSA+DeepCV", "SSA+DeepIS"]:
            trajectory_name = "time course"
            trajectories_name = "time courses"
        else:
            trajectory_name = "trajectory"
            trajectories_name = "trajectories"

        logging.info(
            f"{method:<{max_method_w}} : "
            f"{total_time:>{max_total_time_w}.4f} seconds for "
            f"{n_trajectories:>{max_traj_w}} {trajectories_name} → "
            f"{time_per_trajectory:>{max_tpt_w}.6f} seconds/{trajectory_name}"
        )

    insert_blank_line()
    # Print product of time per trajectory (per time grid point) and variance if available:
    for method, info in sub_optimal_info.items():
        time_variance_product = info.get("time_variance_product")

        if type in ["optimal", "output"] and method in ["SSA+DeepCV", "SSA+DeepIS"]:
            trajectory_name = "time course"
            trajectories_name = "time courses"
        else:
            trajectory_name = "trajectory"
            trajectories_name = "trajectories"

        if time_variance_product is not None:
            logging.info(
                f"{method:<{max_method_w}} : "
                f"{info['time_variance_product']} seconds/{trajectory_name} × mean variance over time"
            )


def _extract_all_analysis_timing_info(log_content: str, test_state: str = None) -> dict:
    """Extract convergence timing information for a specific test state from log content."""
    if test_state:
        # Find the section starting with the test state:
        test_state_pattern = rf"Running analysis for test state {re.escape(test_state)} \.\.\."
        test_state_match = re.search(test_state_pattern, log_content)

        if not test_state_match:
            logging.warning(f"Test state {test_state} not found in log file.")
            return {}

        # Find the next test state or end of file to limit the search scope:
        next_test_state_pattern = r"Running analysis for test state \["
        next_match = re.search(next_test_state_pattern, log_content[test_state_match.end():])

        if next_match:
            # Extract content only for this test state:
            relevant_content = log_content[test_state_match.start():test_state_match.end() + next_match.start()]
        else:
            # This is the last test state, take everything from here to the end:
            relevant_content = log_content[test_state_match.start():]
    else:
        # Use entire log content if no specific test state is requested:
        logging.info("No specific test state provided; analyzing entire log content and using the first match found.")
        relevant_content = log_content

    # Extract all available timing information from the relevant content:
    convergence_info = _extract_convergence_timing_info(relevant_content)
    optimal_info = _extract_sub_optimal_timing_info(relevant_content, type="optimal")
    output_suboptimal_info = _extract_sub_optimal_timing_info(relevant_content, type="output")
    time_suboptimal_info = _extract_sub_optimal_timing_info(relevant_content, type="time")

    return convergence_info, optimal_info, time_suboptimal_info, output_suboptimal_info


def _extract_convergence_timing_info(relevant_content: str) -> dict:
    """Extract convergence timing information from the relevant log content."""
    convergence_info = {}

    # Extract 'SSA' convergence timing:
    ssa_pattern = r"Generating convergence 'SSA' estimate using (\d+) trajectories at t=[\d.]+ \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds\."
    ssa_match = re.search(ssa_pattern, relevant_content)
    if ssa_match:
        n_trajectories = int(ssa_match.group(1))
        memory_usage = float(ssa_match.group(2).replace(",", ""))
        total_time = float(ssa_match.group(3))
        variance = _extract_variance_for_method(relevant_content, "Convergence final time", "SSA")

        time_per_trajectory = total_time / n_trajectories
        time_variance_product = time_per_trajectory * variance if variance is not None else None

        convergence_info["SSA"] = {
            "n_trajectories": n_trajectories,
            "memory_usage_mb": memory_usage,
            "total_time": total_time,
            "variance": variance,
            "time_per_trajectory": time_per_trajectory,
            "time_variance_product": time_variance_product
        }

    # Extract exact 'SSA with DeepCV' convergence timing:
    deepcv_pattern = r"Generating convergence 'SSA with DeepCV' expectation estimate for (\d+) trajectories at t=[\d.]+ \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds\."
    deepcv_match = re.search(deepcv_pattern, relevant_content)
    if deepcv_match:
        n_trajectories = int(deepcv_match.group(1))
        memory_usage = float(deepcv_match.group(2).replace(",", ""))
        variance = _extract_variance_for_method(relevant_content, "Convergence final time", "SSA with DeepCV")
        total_time = float(deepcv_match.group(3)) / len(variance)

        time_per_trajectory = total_time / n_trajectories
        time_variance_product = time_per_trajectory * variance if variance is not None else None

        convergence_info["SSA+DeepCV"] = {
            "n_trajectories": n_trajectories,
            "memory_usage_mb": memory_usage,
            "total_time": total_time,
            "variance": variance,
            "time_per_trajectory": time_per_trajectory,
            "time_variance_product": time_variance_product
        }

    # Extract 'SSA with DeepCV' convergence timing:
    deepcv_pattern = r"Computing convergence 'SSA with DeepCV' expectation estimate for (\d+) trajectories at t=[\d.]+ \.\.\. ([\d.]+) seconds\."
    deepcv_match = re.search(deepcv_pattern, relevant_content)
    if deepcv_match and ssa_match:
        n_trajectories = int(deepcv_match.group(1))
        variance = _extract_variance_for_method(relevant_content, "Convergence final time", "SSA with DeepCV")
        total_time = float(deepcv_match.group(2)) / len(variance) + convergence_info["SSA"]["time_per_trajectory"] * n_trajectories

        time_per_trajectory = total_time / n_trajectories
        time_variance_product = time_per_trajectory * variance if variance is not None else None

        convergence_info["SSA+DeepCV"] = {
            "n_trajectories": n_trajectories,
            "memory_usage_mb": convergence_info["SSA"]["memory_usage_mb"] * n_trajectories / convergence_info["SSA"]["n_trajectories"],
            "total_time": total_time,
            "variance": variance,
            "time_per_trajectory": time_per_trajectory,
            "time_variance_product": time_variance_product
        }

    # Extract 'SSA with DeepIS' convergence timing:
    deepis_pattern = r"Generating convergence 'SSA with DeepIS' expectation estimate using (\d+) trajectories per output function at t=[\d.]+ \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds\."
    deepis_match = re.search(deepis_pattern, relevant_content)
    if deepis_match:
        n_trajectories = int(deepis_match.group(1))
        memory_usage = float(deepis_match.group(2).replace(",", ""))
        variance = _extract_variance_for_method(relevant_content, "Convergence final time", "SSA with DeepIS")
        total_time = float(deepis_match.group(3)) / len(variance)

        time_per_trajectory = total_time / n_trajectories
        time_variance_product = time_per_trajectory * variance if variance is not None else None

        convergence_info["SSA+DeepIS"] = {
            "n_trajectories": n_trajectories,
            "memory_usage_mb": memory_usage,
            "total_time": total_time,
            "variance": variance,
            "time_per_trajectory": time_per_trajectory,
            "time_variance_product": time_variance_product
        }

    return convergence_info


def _extract_sub_optimal_timing_info(relevant_content: str, type: str) -> dict:
    """Extract (sub)optimal estimate timing information from the relevant log content."""
    sub_optimal_info = {}

    # Extract 'SSA' timing:
    ssa_pattern = r"Generating temporal 'SSA' estimate using (\d+) trajectories and (\d+) time points \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds\."
    ssa_match = re.search(ssa_pattern, relevant_content)
    if ssa_match:
        n_trajectories = int(ssa_match.group(1))
        n_time_points = int(ssa_match.group(2))
        memory_usage = float(ssa_match.group(3).replace(",", ""))
        total_time = float(ssa_match.group(4))
        variance = _extract_variance_for_method(relevant_content, "Mean", "SSA")

        time_per_trajectory = total_time / n_trajectories
        time_variance_product = time_per_trajectory * variance if variance is not None else None

        sub_optimal_info["SSA"] = {
            "n_trajectories": n_trajectories,
            "n_time_points": n_time_points,
            "memory_usage_mb": memory_usage,
            "total_time": total_time,
            "variance": variance,
            "time_per_trajectory": time_per_trajectory,
            "time_variance_product": time_variance_product
        }

    if type == "optimal":
        # Extract optimal 'SSA with DeepCV' timing:
        deepcv_pattern = r"Computing temporal \(and output-suboptimal\) 'SSA with DeepCV' expectation estimate for (\d+) time courses and (\d+) time points \.\.\. ([\d.]+) seconds\."
        deepcv_type = "Mean optimal"
    elif type == "output":
        # Extract output-suboptimal 'SSA with DeepCV' timing:
        deepcv_pattern = r"Computing temporal \(and output-suboptimal\) 'SSA with DeepCV' expectation estimate for (\d+) time courses and (\d+) time points \.\.\. ([\d.]+) seconds\."
        deepcv_type = "Mean output-suboptimal"
    elif type == "time":
        # Extract time-suboptimal 'SSA with DeepCV' timing:
        deepcv_pattern = r"Computing time-suboptimal temporal 'SSA with DeepCV' expectation estimate for (\d+) trajectories and (\d+) time points \.\.\. ([\d.]+) seconds\."
        deepcv_type = "Mean time-suboptimal"
    deepcv_match = re.search(deepcv_pattern, relevant_content)
    if deepcv_match and ssa_match:
        n_trajectories = int(deepcv_match.group(1))
        n_time_points = int(deepcv_match.group(2))
        variance = _extract_variance_for_method(relevant_content, deepcv_type, "SSA with DeepCV")
        total_time = float(deepcv_match.group(3)) / len(variance) + sub_optimal_info["SSA"]["time_per_trajectory"] * n_trajectories

        time_per_trajectory = total_time / n_trajectories
        time_variance_product = time_per_trajectory * variance if variance is not None else None

        sub_optimal_info["SSA+DeepCV"] = {
            "n_trajectories": n_trajectories,
            "n_time_points": n_time_points,
            "memory_usage_mb": sub_optimal_info["SSA"]["memory_usage_mb"] * n_trajectories / sub_optimal_info["SSA"]["n_trajectories"],
            "total_time": total_time,
            "variance": variance,
            "time_per_trajectory": time_per_trajectory,
            "time_variance_product": time_variance_product
        }

    if type == "optimal":
        # Extract optimal 'SSA with DeepIS' timing:
        deepis_pattern = r"Generating temporal \(and output-suboptimal\) 'SSA with DeepIS' expectation estimate using (\d+) time courses per output function and (\d+) time points \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds\."
        deepis_type = "Mean optimal"
    # Extract output-suboptimal 'SSA with DeepIS' timing:
    elif type == "output":
        deepis_pattern = r"Generating temporal \(and output-suboptimal\) 'SSA with DeepIS' expectation estimate using (\d+) time courses per output function and (\d+) time points \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds\."
        deepis_type = "Mean output-suboptimal"
    elif type == "time":
        # Extract time-suboptimal 'SSA with DeepIS' timing:
        deepis_pattern = r"Generating time-suboptimal temporal 'SSA with DeepIS' expectation estimate using (\d+) trajectories per output function and (\d+) time points \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds\."
        deepis_type = "Mean time-suboptimal"
    deepis_match = re.search(deepis_pattern, relevant_content)
    if deepis_match:
        n_trajectories = int(deepis_match.group(1))
        n_time_points = int(deepis_match.group(2))
        memory_usage = float(deepis_match.group(3).replace(",", ""))
        variance = _extract_variance_for_method(relevant_content, deepis_type, "SSA with DeepIS")
        total_time = float(deepis_match.group(4)) / len(variance)

        time_per_trajectory = total_time / n_trajectories
        time_variance_product = time_per_trajectory * variance if variance is not None else None

        sub_optimal_info["SSA+DeepIS"] = {
            "n_trajectories": n_trajectories,
            "n_time_points": n_time_points,
            "memory_usage_mb": memory_usage,
            "total_time": total_time,
            "variance": variance,
            "time_per_trajectory": time_per_trajectory,
            "time_variance_product": time_variance_product
        }

    return sub_optimal_info


def _extract_variance_for_method(log_content: str, type: str, method: str) -> np.ndarray | None:
    """Extract variance information for a specific method from log content."""
    escaped_type = re.escape(type)
    escaped_method = re.escape(method)
    # Pattern example 1: "Convergence final time 'SSA' variance: [9.21573186e+01 6.28486280e+05]"
    # Pattern example 2: "Mean time-suboptimal 'SSA with DeepCV' variance over time: [7.23583084e+01 4.24466336e+06]"
    variance_pattern = rf"{escaped_type} '{escaped_method}' variance(?: over time)?: \[([^\]]+)\]"

    variance_match = re.search(variance_pattern, log_content)
    if variance_match:
        # Extract the variance values string and parse it:
        variance_str = variance_match.group(1)
        # Split by whitespace and convert to floats:
        variance_values = [float(x) for x in variance_str.split()]
        return np.array(variance_values)

    return None


if __name__ == "__main__":
    pass
