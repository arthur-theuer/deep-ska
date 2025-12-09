"""Main script for orchestrating the training and analysis of a model."""

import argparse
import logging

from .analysis import analyze_test_states, create_test_states
from .core.initialization import RunContext
from .logging.helpers import insert_blank_line, insert_section_marker
from .logging.utils import log_reaction_dict, log_stoichiometry_matrix
from .plots import model_plots
from .reaction_networks import examples
from .reaction_networks.definition import ReactionNetwork
from .training import ExpectationModelTrainer, train_or_load_model


def main(run: RunContext) -> None:
    """Main function for running the training and analysis pipeline."""
    # Create the reaction network:
    logging.info(f"Creating the {run.RN_CONFIG['reaction_network_abbreviation']} reaction network ...")
    network_class = getattr(examples, run.RN_CONFIG["reaction_network"])
    network: ReactionNetwork = network_class(rn_config=run.RN_CONFIG, init_type=run.RN_CONFIG["init_type"])
    logging.info(f"Using {network.out_function_size} output function(s) for a network with {network.param_dict_size} parameter(s).")
    insert_blank_line()
    # Log reaction network details:
    logging.info("Reaction network reactant matrix (species x reactions):")
    log_stoichiometry_matrix(network.reactant_matrix, network.species_labels)
    insert_blank_line()
    logging.info("Reaction network product matrix (species x reactions):")
    log_stoichiometry_matrix(network.product_matrix, network.species_labels)
    insert_blank_line()
    logging.info("Reaction network propensities:")
    log_reaction_dict(network.reaction_dict)
    insert_section_marker()

    # Initialize the model trainer:
    trainer = ExpectationModelTrainer(network, run)

    # PART I - MODEL TRAINING OR LOADING:
    train_or_load_model(trainer, run)
    # Plots for the decay modes of the model:
    if run.NN_CONFIG["v"]["subnet_architecture"] != "naive":
        model_plots.plot_expectation_decay_modes(trainer.model.decay_real, trainer.model.decay_imag, trainer.model.v_stationary_mean, run)

    # PART II - MODEL ANALYSIS WITH TEST STATES:
    if run.NN_CONFIG["v"]["subnet_architecture"] == "spectral_complex":  # update spectral subnet attributes for analysis (if applicable)
        attr, value = ("pairing_mode", "paired") if run.NN_CONFIG["v"]["pairing_mode"] == "paired" else ("allow_complex", False)
        for subnets in ["main_net", "diff_net", "quot_net"]:
            if (subnet := getattr(trainer.model, subnets, None)):
                setattr(subnet, attr, value)

    if run.VAL_CONFIG["run_test_state_analysis"]:
        test_states = create_test_states(trainer, run.VAL_CONFIG["test_states"], run.VAL_CONFIG["n_random_states"], run.VAL_CONFIG["analysis_seed"])
        analyze_test_states(test_states, trainer, run)

    insert_section_marker()
    logging.info("DeepSKA run finished successfully.")
    insert_section_marker()


def cli() -> None:
    """Command-line entrypoint for the package."""
    # Set up the command line argument parser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s <config_file> [--config_dir <config_dir>]",
        description="Main script for orchestrating the training and evaluation of a DeepSKA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="filename of the configuration file"
    )

    parser.add_argument(
        "-d", "--config_dir",
        default="configs",
        type=str,
        metavar="",
        help="directory where the configuration file is located"
    )

    # Run the main function:
    main(RunContext(parser.parse_args()))
