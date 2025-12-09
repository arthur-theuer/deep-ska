"""Implements subnet architecture placeholder for learning S."""

from torch import nn


class PlaceholderSensitivitySubnet(nn.Module):
    """Spectral decomposition sensitivity subnet placeholder."""

    def __init__(self, n_species: int, n_reactions: int, n_params: int, nn_config: dict, int_net: bool = False) -> None:
        """Initialize the spectral decomposition sensitivity subnet placeholder."""
        super().__init__()
        # Get network size from configuration file:
        n_nodes = nn_config["n_nodes_per_layer"]
        n_hidden_layers = nn_config["n_hidden_layers"]
        n_spectral_terms = nn_config["v"]["n_spectral_terms"]
        # Create placeholder feedforward layers:
        layers = [nn.Linear(n_species, n_nodes), nn.ReLU()]
        layers.extend([layer for _ in range(n_hidden_layers) for layer in (nn.Linear(n_nodes, n_nodes), nn.ReLU())])
        layers.append(nn.Linear(n_nodes, 2*n_spectral_terms*n_params) if not int_net else nn.Linear(n_nodes, 2*n_spectral_terms*n_params*n_reactions))
        # Create the feedforward network from the layers:
        self.feedforward = nn.Sequential(*layers)
