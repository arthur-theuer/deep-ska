"""Decorators and generic logging and timing utilities."""

import logging
import subprocess
import time
from collections.abc import Callable
from functools import wraps

import numpy as np


def training_timer(func: Callable) -> Callable:
    """Decorator to measure the runtime of a function, especially for training the models."""

    def wrapper(*args: tuple, **kwargs: dict) -> any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        insert_blank_line()
        logging.info(f"The training routine took {elapsed_time} seconds.")
        return result

    return wrapper


def execution_timer(func: Callable) -> Callable:
    """Measure the runtime of a function and add it to the previous logging message.

    This decorator measures the execution time of a function and appends it to
    the previous logging message. The previous message should not end with a line break
    to ensure proper formatting.
    """

    @wraps(func)
    def wrapper(*args: tuple, **kwargs: dict) -> object:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Update previous log message with the elapsed time:
        logger = logging.getLogger()
        for handler in logger.handlers:
            if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
                handler.stream.write(f" {elapsed_time:.4f} seconds.\n")
        return result

    return wrapper


def array_memory_logger(func: Callable) -> Callable:
    """Measure total memory size (in MB) of NumPy arrays in the returned dict and log it."""

    @wraps(func)
    def wrapper(*args: tuple, **kwargs: dict) -> object:
        result = func(*args, **kwargs)
        # Only log memory if the result is a dictionary:
        if isinstance(result, dict):
            try:
                total_bytes = sum(
                    arr.nbytes for arr in result.values() if isinstance(arr, np.ndarray)
                )
                total_mb = total_bytes / (1024**2)
                # Append memory info to previous log message:
                logger = logging.getLogger()
                for handler in logger.handlers:
                    if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
                        handler.stream.write(f" {total_mb:,.3f} MB /")
            except Exception as e:
                logger = logging.getLogger()
                logger.warning(f"[memory_of_arrays] Failed to measure memory: {e}")

        return result

    return wrapper


def insert_blank_line(n_lines: int = 1) -> None:
    """Write a blank line directly to all handlers' streams.

    Args:
        n_lines: Number of blank lines to insert
    """
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
            for _ in range(n_lines):
                handler.stream.write("\n")
            handler.flush()  # ensure the blank line is written immediately


def insert_section_marker() -> None:
    """Write a section marker directly to all handlers' streams."""
    m = "------------------------------------------------------------------------------------------"
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
            handler.stream.write(f"\n{m}\n\n")
            handler.flush()  # ensure the marker is written immediately


def insert_subsection_marker() -> None:
    """Write a subsection marker directly to all handlers' streams."""
    m = "- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -"
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
            handler.stream.write(f"\n{m}\n\n")
            handler.flush()  # ensure the marker is written immediately


def log_without_linebreak(message: str) -> None:
    """Log an info message without a line break to append additional information to the line afterwards.

    This function is typically used together with execution_timer to append timing
    information to the log message.

    Args:
        message: The message to log without a linebreak
    """
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.terminator = ""
    logging.info(message)
    for handler in logger.handlers:
        handler.terminator = "\n"


def format_mean_ci_arrays(mean: np.ndarray, ci: np.ndarray) -> str:
    """Format an array of means and confidence intervals into a readable format.

    Args:
        mean: Array of mean values
        ci: Array of confidence intervals

    Returns:
        A formatted string representation of the means and confidence intervals

    Raises:
        ValueError: If the shapes of the mean and confidence interval arrays don't match
        TypeError: If inputs are not numpy arrays
    """
    if not isinstance(mean, np.ndarray) or not isinstance(ci, np.ndarray):
        raise TypeError("Both mean and ci must be numpy arrays.")

    if mean.shape != ci.shape:
        raise ValueError("Shapes of means and confidence intervals must match.")

    def format_element(m: float, c: float) -> str:
        return f"{m:.6g} ± {c:.6g}"  # floating-point notation

    with np.printoptions(linewidth=np.inf):  # to avoid unwanted line breaks
        return np.array2string(
            np.vectorize(format_element)(mean, ci),
            separator=", ",
            formatter={"all": lambda x: str(x)},
        )


def get_git_commit_short_hash() -> str:
    """Get the short hash of the current git commit."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        )
    except Exception:
        return "unknown"
