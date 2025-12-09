"""Helper functions to more easily plot lines."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .registry import register_plot


@register_plot("hline")
def hline_plot(ax: Axes, y_data: np.ndarray, color: str) -> Line2D:
    """Helper function to plot a dashed hline."""
    lines = ax.axhline(y_data, color=color, linestyle="--")
    return lines  # only return first line object for legend


@register_plot("plot")
def plot_plot(ax: Axes, x_data: np.ndarray, y_data: np.ndarray, color: str) -> Line2D:
    """Helper function to plot a simple line."""
    lines = ax.plot(x_data, y_data, color=color, linestyle="-")  # data line
    return lines[0]  # only return first line object for legend


@register_plot("rdot")
def rdot_plot(ax: Axes, x_data: np.ndarray, y_data: np.ndarray) -> Line2D:
    """Helper function to plot a simple red dot plot."""
    lines = ax.plot(x_data, y_data, color="tab:red", marker="o", linestyle="None")  # data points
    return lines[0]


@register_plot("scat")
def scat_plot(ax: Axes, x_data: np.ndarray, y_data: np.ndarray, color: str) -> Line2D:
    """Helper function to plot a simple scatter plot."""
    lines = ax.plot(x_data, y_data, color=color, marker="o", markerfacecolor="white")  # data points
    return lines[0]


@register_plot("step")
def step_plot(ax: Axes, x_data: np.ndarray, y_data: np.ndarray, color: str, where: str) -> Line2D:
    """Helper function to plot step lines."""
    lines = ax.step(x_data, y_data, color=color, alpha=0.3, where=where)
    return lines[0]  # only return first line object for legend


@register_plot("temp")
def temp_plot(ax: Axes, x_data: np.ndarray, y_data: np.ndarray, idx: int, t_stop_train: float, t_stop_valid: float, color: str) -> Line2D:
    """Helper function to plot the training and validation components in different styles."""
    lines = ax.plot(x_data[:idx], y_data[:idx], color=color, linestyle="-")  # training data line
    if t_stop_valid > t_stop_train:
        ax.plot(x_data[idx:], y_data[idx:], color=color, linestyle="--")  # validation data line
    return lines[0]  # only return first line object for legend


@register_plot("traj")
def traj_plot(ax: Axes, x_data: np.ndarray, y_data: np.ndarray, color: str) -> Line2D:
    """Helper function to plot trajectories."""
    lines = ax.plot(x_data, y_data, color=color, alpha=0.3, linestyle="-")
    return lines[0]  # only return first line object for legend
