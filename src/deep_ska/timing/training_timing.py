"""Timing functions for model training."""

import logging
import os
import re

from ..core.initialization import RunContext
from ..logging.helpers import insert_blank_line, insert_section_marker


def training_log_file_analysis(run: RunContext) -> None:
    """Analyze log file to extract timing information and compute training time breakdown."""
    insert_section_marker()

    if not os.path.exists(log_file_path := run.VAL_CONFIG["log_file_path"]):
        logging.warning(f"Log file not found: {log_file_path}")
        return

    with open(log_file_path) as f:
        log_content = f.read()

    # Extract timing information:
    timing_info = _extract_training_timing_info(log_content)

    # Analyze early stopping timing if available:
    _analyze_training_timing(timing_info)


def _analyze_training_timing(timing_info: dict) -> None:
    """Analyze early stopping timing information and log the results."""
    if timing_info["training_routine_time"]:
        training_routine_time = timing_info["training_routine_time"]
    else:
        logging.warning("Could not find training routine time information in log file.")
        return

    if timing_info["ssa_training_total_time"]:
        if len(timing_info["ssa_training_total_time"]) > 1:
            logging.warning("Multiple SSA training total time entries found in the log; using the first one.")
        ssa_training_time = timing_info["ssa_training_total_time"][0]
    else:
        logging.warning("Could not find SSA training total time information in log file. Setting SSA training time to 0.")
        ssa_training_time = 0

    if timing_info["ssa_validation_total_time"]:
        if len(timing_info["ssa_validation_total_time"]) > 1:
            logging.warning("Multiple SSA validation total time entries found in the log; using the first one.")
        ssa_validation_time = timing_info["ssa_validation_total_time"][0]
    else:
        logging.info("Could not find SSA validation total time information in log file. Setting SSA validation time to 0.")
        ssa_validation_time = 0

    if timing_info["ergodic_mean_total_time"]:
        if len(timing_info["ergodic_mean_total_time"]) > 1:
            logging.warning("Multiple ergodic mean total time entries found in the log; using the first one.")
        ergodic_mean_time = timing_info["ergodic_mean_total_time"][0]
    else:
        logging.info("Could not find ergodic mean total time information in log file. Setting ergodic mean time to 0.")
        ergodic_mean_time = 0

    if timing_info["is_resampling_training_total_time"]:
        is_resampling_training_time = sum(timing_info["is_resampling_training_total_time"])
    else:
        logging.info("No DeepIS resampling training timing information found in log file. Setting DeepIS resampling training time to 0.")
        is_resampling_training_time = 0

    if timing_info["is_resampling_validation_total_time"]:
        is_resampling_validation_time = sum(timing_info["is_resampling_validation_total_time"])
    else:
        logging.info("No DeepIS resampling validation timing information found in log file. Setting DeepIS resampling validation time to 0.")
        is_resampling_validation_time = 0

    if timing_info["early_stopping_times"]:
        early_stopping_time = sum(timing_info["early_stopping_times"])
    else:
        logging.info("No early stopping timing information found in log file. Setting early stopping time to 0.")
        early_stopping_time = 0

    if any(value == 0 for value in [ssa_training_time, ssa_validation_time, ergodic_mean_time, is_resampling_training_time, is_resampling_validation_time, early_stopping_time]):
        insert_blank_line()

    core_training_time = training_routine_time - is_resampling_validation_time - early_stopping_time
    total_solving_time = ssa_training_time + ergodic_mean_time + core_training_time

    # Log the analysis results:
    logging.info(f"Total solving time: {total_solving_time:.2f} seconds")
    if ssa_training_time > 0:
        logging.info(f"> SSA simulation time (training): {ssa_training_time:.2f} seconds ({(ssa_training_time / total_solving_time):.1%})")
    if is_resampling_training_time > 0:
        logging.info(f"> DeepIS resampling time (training): {is_resampling_training_time:.2f} seconds ({(is_resampling_training_time / total_solving_time):.1%})")
    if ergodic_mean_time > 0:
        logging.info(f"> Ergodic mean computation time: {ergodic_mean_time:.2f} seconds ({(ergodic_mean_time / total_solving_time):.1%})")
    logging.info(f"> Training time: {core_training_time:.2f} seconds ({(core_training_time / total_solving_time):.1%})")

    insert_blank_line()

    if ssa_validation_time > 0:
        logging.info(f"SSA simulation time (validation): {ssa_validation_time:.2f} seconds (+ {(ssa_validation_time / total_solving_time):.1%})")
    if is_resampling_validation_time > 0:
        logging.info(f"DeepIS resampling time (validation): {is_resampling_validation_time:.2f} seconds (+ {(is_resampling_validation_time / total_solving_time):.1%})")
    if early_stopping_time > 0:
        logging.info(f"Early stopping analysis time: {early_stopping_time:.2f} seconds (+ {(early_stopping_time / total_solving_time):.1%})")
    if timing_info["early_stopping_times"]:
        logging.info(f"Number of early stopping evaluations: {len(timing_info['early_stopping_times'])}")
        logging.info(f"Average time per early stopping evaluation: {early_stopping_time / len(timing_info['early_stopping_times']):.2f} seconds")


def _extract_training_timing_info(log_content: str) -> dict:
    """Extract all timing information from log file content."""
    timing_info = {}

    # Extract SSA simulation time (training trajectories):
    timing_info.update(_extract_simulation_times(log_content, "ssa_training"))
    # Extract SSA simulation time (validation trajectories):
    timing_info.update(_extract_simulation_times(log_content, "ssa_validation"))

    # Extract ergodic mean computation time:
    timing_info.update(_extract_simulation_times(log_content, "ergodic_mean"))

    # Extract DeepIS resampling times (training trajectories):
    timing_info.update(_extract_simulation_times(log_content, "is_resampling_training"))
    # Extract DeepIS resampling times (validation trajectories):
    timing_info.update(_extract_simulation_times(log_content, "is_resampling_validation"))

    # Extract total training time:
    timing_info.update(_extract_training_routine_time(log_content))

    # Extract early stopping times:
    timing_info.update(_extract_early_stopping_times(log_content))

    return timing_info


def _extract_simulation_times(log_content: str, type: str) -> dict:
    """Extract DeepIS resampling timing information from log content."""
    if type == "ssa_training":
        # Pattern example: "Generating 20000 training samples ... 1,917.061 MB / 21.0552 seconds."
        pattern = r"Generating (\d+) training samples \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds."
    elif type == "ssa_validation":
        # Pattern example: "Generating 10000 validation samples ... 1,001.431 MB / 11.7623 seconds."
        pattern = r"Generating (\d+) validation samples \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds."

    elif type == "ergodic_mean":
        # Pattern example: "Computing the 'Ergodic mean' for model initialization using 5 trajectories ... 22.5862 seconds."
        pattern = r"Computing the 'Ergodic mean' for model initialization using (\d+) trajectories \.\.\. ([\d.]+) seconds."

    elif type == "is_resampling_training":
        # Pattern example: "Generating 20000 DeepIS training samples per output function ... 2,138.374 MB / 1969.9833 seconds."
        pattern = r"Generating (\d+) DeepIS training samples per output function \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds."
    elif type == "is_resampling_validation":
        # Pattern example: "Generating 20000 DeepIS validation samples per output function ... 2,138.374 MB / 1969.9833 seconds."
        pattern = r"Generating (\d+) DeepIS validation samples per output function \.\.\. (\d{1,3}(?:,\d{3})*\.\d{3}) MB / ([\d.]+) seconds."

    matches = re.findall(pattern, log_content)
    if matches:
        if type == "ergodic_mean":
            n_trajectories = [int(match[0]) for match in matches]
            memory_usage = None
            total_time = [float(match[1]) for match in matches]
        else:
            n_trajectories = [int(match[0]) for match in matches]
            memory_usage = [float(match[1].replace(",", "")) for match in matches]
            total_time = [float(match[2]) for match in matches]
    else:
        n_trajectories = None
        memory_usage = None
        total_time = None

    return {
        f"{type}_n_trajectories": n_trajectories,
        f"{type}_memory_usage_mb": memory_usage,
        f"{type}_total_time": total_time
    }


def _extract_training_routine_time(log_content: str) -> dict:
    """Extract total training time from log content."""
    # Pattern example: "The training routine took 12739.048391819 seconds."
    pattern = r"The training routine took ([\d.]+) seconds\."

    matches = re.findall(pattern, log_content)

    if len(matches) > 1:
        logging.warning("Multiple training routine time entries found in the log; using the first one.")

    total_time = float(matches[0]) if matches else None

    return {"training_routine_time": total_time}


def _extract_early_stopping_times(log_content: str) -> dict:
    """Extract early stopping timing information from log content."""
    # Pattern example: "Computing temporal 'SSA with DeepCV' estimate for early stopping ... 6.2798 seconds."
    pattern = r"Computing temporal '[^']*' estimate for early stopping \.\.\. ([\d.]+) seconds\."

    matches = re.findall(pattern, log_content)
    times = [float(match) for match in matches] if matches else None

    return {"early_stopping_times": times}


if __name__ == "__main__":
    pass
