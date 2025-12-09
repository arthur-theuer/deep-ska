"""Common helper functions and dictionaries for plotting."""

import numpy as np

architecture_dict = {
    "naive": "SFF arch.",
    "features": "FFTF arch.",
    "spectral_matched_lumped": "SDnet",
    "spectral_matched_simplified": "SDnet",
    "spectral_matched_full": "SDnet",
    "spectral_complex": "SDnet"
}

color_dict = {
    "loss_expectation_dark": "#BF00BF",
    "loss_expectation_light": "#FF00FF",
    "loss_sensitivity_dark": "#008080",
    "loss_sensitivity_light": "#00BFBF",

    "decay_expectation_dark": "#BF00BF",
    "decay_expectation_light": "#FF00FF",

    "exact": "#FF5722",
    "nn_direct": "#9575CD",
    "nn_indirect": "#D1C4E9",

    "sim": "#4CAF50",
    "sim_old": "#A5D6A7",
    "sim_small": "#BCAAA4",
    "sim_tolerance": "#BDBDBD",

    "sim_cv": "#2196F3",
    "sim_is": "#FFB300",

    "deep_ipa": "#FFAB91",
    "deep_ipa_cv": "#F4511E",

    "bpa": "#F8BBD0",
    "bpa_cv": "#F06292",
}

notation_dict = {
    "features": "\\ell",
    "spectral_matched_lumped": "m",
    "spectral_matched_simplified": "m",
    "spectral_matched_full": "m",
    "spectral_complex": "\\ell"
}


def create_centered_array(n: int, step_size: float = 1) -> np.ndarray:
    """Creates an array with n values centered around 0 with a given step size."""
    if n % 2 == 0:  # even number of elements
        start = -(n // 2) * step_size + step_size / 2
        end = (n // 2) * step_size - step_size / 2
    else:  # odd number of elements
        start = -(n // 2) * step_size
        end = (n // 2) * step_size
    return np.linspace(start, end, n)
