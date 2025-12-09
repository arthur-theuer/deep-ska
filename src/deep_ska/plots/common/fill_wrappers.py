"""Generic fill and error bar plots."""

from matplotlib.axes import Axes

from .registry import register_plot


@register_plot("fill")
def fill_fill(ax: Axes, *args: object, **kwargs: object) -> object:
    """Helper function to plot a filled area."""
    return ax.fill_between(*args, **kwargs)


@register_plot("error")
def error_fill(ax: Axes, *args: object, **kwargs: object) -> object:
    """Helper function to plot error bars."""
    return ax.errorbar(*args, **kwargs)
