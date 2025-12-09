"""Module defining the PlotInstruction dataclass for plotting instructions."""

from dataclasses import dataclass, field


@dataclass
class PlotInstruction:
    """Dataclass representing a plotting instruction."""
    kind: str
    args: tuple[object, ...]
    kwargs: dict[str, object] = field(default_factory=dict)
