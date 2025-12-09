"""Generalized logic for plotting panels from stored figure data."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

from ..core.initialization import RunContext
from .common import fill_wrappers, line_wrappers  # noqa: F401
from .common.dispatch import render_instruction

naming_dict = {
    "A_TemporalTrajectories_": "C_TemporalTrajectories_",
    "B_TemporalExpectation_": "D_TemporalExpectation_",
    "C_TemporalExpectationVariance_": "E_TemporalExpectationVariance_",
    "D_TemporalExpectationVarianceLog_": "F_TemporalExpectationVarianceLog_",
    "E_ExpectationErrorConvergence_": "G_ExpectationErrorConvergence_",
    "F_ExpectationErrorConvergenceCI_": "H_ExpectationErrorConvergenceCI_",
}


def _normalize_axes_shape(panel_axes: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Ensure panel_axes is always a 2D numpy array."""
    if n_rows == 1 and n_cols == 1:
        return np.array([[panel_axes]])
    elif n_rows == 1:
        return panel_axes.reshape(1, -1)
    elif n_cols == 1:
        return panel_axes.reshape(-1, 1)
    else:
        return np.atleast_2d(panel_axes)


def _fill_panel_cell(ax: plt.Axes, curr_data: dict, row: int, col: int, n_rows: int) -> None:
    """Render a single subplot cell with standard formatting."""
    ax.set_box_aspect(1)

    if row == 0:
        title_color = "black" if np.all(curr_data["state"] >= 0) else "tab:red"  # highlight negative states in red
        ax.set_title(curr_data["title"], color=title_color)
    if col == 0:
        ax.set_ylabel(curr_data["ylabel"])
    if row == n_rows - 1:
        ax.set_xlabel(curr_data["xlabel"])

    # Replay stored plot elements (fills, lines, error bars) in their original order:
    for instr in curr_data["elements"]:
        render_instruction(ax, instr)


def _add_panel_legend(fig: plt.Figure, panel_axes: np.ndarray, curr_data: dict, n_cols: int) -> None:
    """Position and draw legend below bottom row."""
    # Get the bottom row of axes:
    if isinstance(panel_axes, np.ndarray) and panel_axes.ndim == 2:
        bottom_axes = panel_axes[-1, :]
    else:
        bottom_axes = np.atleast_1d(panel_axes)
    # Compute the union of their bounding boxes:
    bboxes = [ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()) for ax in bottom_axes]
    union = Bbox.union(bboxes)
    # Place legend just below this union box:
    n_legend_cols = len(curr_data["labels"]) if n_cols > 2 else 1
    fig.legend(curr_data["handles"][::-1], curr_data["labels"][::-1], loc="upper center", bbox_to_anchor=(0.5, union.y0), ncol=n_legend_cols, frameon=False)


def _save_legend_figure(curr_data: dict, run: RunContext, figure_name: str) -> None:
    """Save a separate legend-only figure."""
    fig = plt.figure(figsize=(10, 1))
    n_legend_cols = len(curr_data["labels"])
    fig.legend(curr_data["handles"][::-1], curr_data["labels"][::-1], loc="center", ncol=n_legend_cols, frameon=False)
    fig.savefig(f"{run.results_subdir}/{figure_name}_legend.pdf", bbox_inches="tight")
    plt.close(fig)


def _draw_panel_grid(grouped_figures: dict, run: RunContext, special_render_fn: Callable[[plt.Axes, dict, str], None] | None = None, layout_mode: str = "panel", index: int | str = "_", show_legend: bool = True) -> None:
    """Generalized panel plotter for multi-panel figures."""
    if not grouped_figures:
        return

    for names_key, row_groups in grouped_figures.items():
        if not row_groups:
            continue

        # Detect dictionary nesting:
        first_val = next(iter(row_groups.values()))
        is_nested = isinstance(first_val, dict) and any(isinstance(v, dict) for v in first_val.values())

        # Determine subplot geometry:
        if layout_mode == "panel" and is_nested:
            # Get panel dimensions:
            n_rows = len(row_groups)
            n_cols = max(len(col_group) for col_group in row_groups.values())
            # Create 2D panel grid:
            sharey = "row" if run.PLT_CONFIG.get("share_panel_y_axes", False) else None
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8*n_cols, 2.8*n_rows), sharey=sharey)
            # Handle different subplot arrangements:
            axes = _normalize_axes_shape(axes, n_rows, n_cols)

            for i, row_key in enumerate(sorted(row_groups.keys())):
                for j, col_key in enumerate(sorted(row_groups[row_key].keys())):
                    curr_data = row_groups[row_key][col_key]
                    ax = axes[i, j]

                    _fill_panel_cell(ax, curr_data, i, j, n_rows)

                    # Optional domain-specific behavior (log scaling):
                    if special_render_fn:
                        special_render_fn(ax, curr_data, names_key)

            fig.tight_layout()  # only for 2D panels to avoid overlap

        elif layout_mode == "column" and not is_nested:
            # Get panel dimensions:
            n_rows = len(row_groups)
            n_cols = 1
            # Create 1D column grid:
            sharey = run.PLT_CONFIG.get("share_panel_y_axes", False)
            fig, axes = plt.subplots(n_rows, 1, figsize=(2.8, 2.8*n_rows), sharey=sharey)
            # Make sure axes is always a 1D array:
            axes = np.atleast_1d(axes)

            for i, key in enumerate(sorted(row_groups.keys())):
                curr_data = row_groups[key]
                ax = axes[i]

                _fill_panel_cell(ax, curr_data, i, 0, n_rows)

                # Optional domain-specific behavior (log scaling):
                if special_render_fn:
                    special_render_fn(ax, curr_data, names_key)

        elif layout_mode == "row" and not is_nested:
            # Get panel dimensions:
            n_rows = 1
            n_cols = len(row_groups)
            # Create 1D row grid:
            fig, axes = plt.subplots(1, n_cols, figsize=(2.8*n_cols, 2.8))
            # Make sure axes is always a 1D array:
            axes = np.atleast_1d(axes)

            for j, key in enumerate(sorted(row_groups.keys())):
                curr_data = row_groups[key]
                ax = axes[j]

                _fill_panel_cell(ax, curr_data, 0, j, 1)

                # Optional domain-specific behavior (log scaling):
                if special_render_fn:
                    special_render_fn(ax, curr_data, names_key)

        else:
            continue  # skip malformed cases

        fig.canvas.draw()

        # Add legend if requested:
        if show_legend:
            _add_panel_legend(fig, axes, curr_data, n_cols)

        figure_name = f"{run.timestamp}_{index}_{names_key}"

        fig.align_ylabels()
        fig.savefig(f"{run.results_subdir}/{figure_name}.pdf", bbox_inches="tight")
        plt.close(fig)

        # Create a figure just for the legend:
        if show_legend:
            _save_legend_figure(curr_data, run, figure_name)


def special_render_temporal(ax: plt.Axes, curr_data: dict, names_key: str) -> None:
    """Add log scales for convergence/variance plots."""
    _ = curr_data  # unused parameter
    if any(sub in names_key for sub in ["E_ExpectationErrorConvergence_", "F_ExpectationErrorConvergenceCI_"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
    if "D_TemporalExpectationVarianceLog_" in names_key:
        ax.set_yscale("log")


def group_temporal_panel_data(figure_data: dict) -> dict[str, dict[object, dict[object, dict]]]:
    """Group figure data for temporal panels."""
    grouped = {}
    for fig_name, curr_data in figure_data.items():
        figure_name = None
        for key in ["A_TemporalTrajectories_", "B_TemporalExpectation_", "C_TemporalExpectationVariance_", "D_TemporalExpectationVarianceLog_", "E_ExpectationErrorConvergence_", "F_ExpectationErrorConvergenceCI_"]:
            if f"_{key}" in fig_name or fig_name.startswith(key):
                figure_name = naming_dict[key]
                break
        if not figure_name:
            continue

        names_key = figure_name + "_".join(curr_data["names"])
        grouped.setdefault(names_key, {}).setdefault(curr_data["output_idx"], {})[curr_data["index"]] = curr_data

    return grouped


def group_temporal_column_data(figure_data: dict) -> dict[str, dict[object, dict]]:
    """Group temporal expectation data for 1D columns (single test state)."""
    grouped = {}
    for fig_name, curr_data in figure_data.items():
        figure_name = None
        for key in ["A_TemporalTrajectories_", "B_TemporalExpectation_", "C_TemporalExpectationVariance_", "D_TemporalExpectationVarianceLog_", "E_ExpectationErrorConvergence_", "F_ExpectationErrorConvergenceCI_"]:
            if key in fig_name:
                figure_name = key
                break
        if not figure_name:
            continue

        names_key = figure_name + "_".join(curr_data["names"])
        grouped.setdefault(names_key, {})[curr_data["output_idx"]] = curr_data
    return grouped


def plot_temporal_column(figure_data: dict, run: RunContext, index: int) -> None:
    """Create 1D columns combining temporal plots with output functions as rows."""
    _draw_panel_grid(
        grouped_figures=group_temporal_column_data(figure_data),
        run=run,
        special_render_fn=special_render_temporal,
        layout_mode="column",
        index=index,
        show_legend=False
    )


def plot_temporal_panel(figure_data: dict, run: RunContext) -> None:
    """Create 2D panels combining temporal plots with output functions as rows and test states as columns."""
    _draw_panel_grid(
        grouped_figures=group_temporal_panel_data(figure_data),
        run=run,
        special_render_fn=special_render_temporal,
        layout_mode="panel"
    )
