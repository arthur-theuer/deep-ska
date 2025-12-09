"""DeepSKA provides SDnet, a spectral-decomposition neural architecture, together with DLMC estimators for fast and reliable inference in stochastic reaction networks."""

from .core.initialization import RunContext
from .main import main

__all__ = ['RunContext', 'main']
