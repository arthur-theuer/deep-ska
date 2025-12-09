"""Specific logging utilities for the package."""

import logging

import numpy as np

from ..core.model import ExpectationModel


def log_stoichiometry_matrix(matrix: np.ndarray, species_names: list[str]) -> None:
    """Log the transposed stoichiometry matrix (species x reactions) in a readable format."""
    logger = logging.getLogger()

    for handler in logger.handlers:
        if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
            handler.stream.write("\n")
            transposed = matrix.T  # shape: [n_species, n_reactions]
            col_width = 4  # spacing for each reaction column

            # Compute max width for species names:
            max_label_len = max((len(name) for name in species_names), default=10)
            label_field_width = max(max_label_len + 2, 14)

            # Header row: R0 R1 R2 ...
            header = " " * label_field_width + "".join(f"R{i:<{col_width-1}}" for i in range(transposed.shape[1]))
            handler.stream.write(header + "\n")

            # Each species row:
            for i, row in enumerate(transposed):
                species_label = species_names[i] if i < len(species_names) else f"S{i}"
                species_label = species_label.ljust(label_field_width)
                row_str = "".join(f"{v:<{col_width}}" for v in row)
                handler.stream.write(f"{species_label}{row_str}\n")

            handler.flush()


def log_reaction_dict(reaction_dict: dict[int, list[str]]) -> None:
    """Write the reaction dictionary to the log file in a readable, aligned format."""
    logger = logging.getLogger()

    for handler in logger.handlers:
        if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
            handler.stream.write("\n")
            max_key_len = max(len(str(k)) for k in reaction_dict)
            max_type_len = max(len(v[0]) for v in reaction_dict.values())

            for k, v in reaction_dict.items():
                reaction_type = v[0]
                rest = ", ".join(str(x) for x in v[1:])
                handler.stream.write(f"  {str(k).rjust(max_key_len)}: {reaction_type.ljust(max_type_len)} - {rest}\n")

            handler.flush()


def log_model_parameters(model: ExpectationModel) -> None:
    """Write model parameters to the log file without logging prefixes, formatted for readability."""
    logger = logging.getLogger()

    for handler in logger.handlers:
        if isinstance(handler, (logging.FileHandler | logging.StreamHandler)):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_str = str(param.data)
                    if "\n" in param_str:  # for multi-line tensors
                        handler.stream.write(f"\n{name}:\n")
                        indented_param_str = param_str.replace("\n", "\n    ")  # indent new lines
                        handler.stream.write(f"    {indented_param_str}\n")  # add extra space after each parameter
                    else:  # for single-line tensors
                        handler.stream.write(f"\n{name}: {param_str}\n")  # add extra space after each parameter
            handler.flush()  # ensure that the parameters are written immediately
