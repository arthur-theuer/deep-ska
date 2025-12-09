""""Dispatch functions to call registered plot functions by kind."""
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .instruction import PlotInstruction
from .registry import PLOT_REGISTRY


def dispatch_plot(ax: Axes, kind: str, *args: object, **kwargs: object) -> Line2D:
    """Call the registered plot function by 'kind' with the given args/kwargs."""
    try:
        func = PLOT_REGISTRY[kind]
    except KeyError as e:
        raise ValueError(f"Unknown plot type '{kind}'. Available: {list(PLOT_REGISTRY)}") from e
    return func(ax, *args, **kwargs)


def render_instruction(ax: Axes, instr: PlotInstruction) -> Line2D:
    """Render a single PlotInstruction onto an axis."""
    return dispatch_plot(ax, instr.kind, *instr.args, **instr.kwargs)
