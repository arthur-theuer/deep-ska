"""Train or load the model based on the configuration."""
import json
import logging

import torch

from ..core.initialization import RunContext
from ..logging.helpers import insert_blank_line, insert_section_marker
from ..logging.utils import log_model_parameters
from ..plots import training_plots
from ..timing.training_timing import training_log_file_analysis
from .trainers import ExpectationModelTrainer


def count_trainable_parameters(trainer: ExpectationModelTrainer) -> int:
    """Count the number of trainable parameters in different categories."""
    param_to_name = {p: n for n, p in trainer.model.named_parameters()}
    cat_totals = {"eigenfunctions (main_net)": 0, "eigenfunctions (diff_net)": 0, "function coordinates": 0, "decay modes": 0, "other": 0}
    for group in trainer.optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                name = param_to_name.get(p, "<unnamed>")
                numel = p.numel()
                if name.startswith("main_net"):
                    cat_totals["eigenfunctions (main_net)"] += numel
                elif name.startswith("diff_net"):
                    cat_totals["eigenfunctions (diff_net)"] += numel
                elif name.startswith("func_coord"):
                    cat_totals["function coordinates"] += numel
                elif name.startswith("decay"):
                    cat_totals["decay modes"] += numel
                else:
                    cat_totals["other"] += numel

    return cat_totals


def train_or_load_model(trainer: ExpectationModelTrainer, run: RunContext) -> dict:
    """Train or load the model depending on the configuration."""
    # Print untrained (just initialized) model parameters:
    insert_section_marker()
    logging.info("INITIALIZED MODEL PARAMETERS:")
    log_model_parameters(trainer.model)
    insert_section_marker()

    if run.NN_CONFIG["model_training_needed"]:
        if run.NN_CONFIG["use_previous_training_weights"]:
            if run.NN_CONFIG["trained_model_timestamp"]:
                model_path = f"{run.results_dir}/{run.NN_CONFIG['trained_model_timestamp']}_{run.config_file_name}/{run.NN_CONFIG['trained_model_timestamp']}_{run.RN_CONFIG['reaction_network_abbreviation']}_model_weights.pth"
            elif run.NN_CONFIG["trained_model_path"]:
                model_path = run.NN_CONFIG["trained_model_path"]
            else:
                raise ValueError("Either 'trained_model_timestamp' or 'trained_model_path' must be specified in the configuration file to load a pre-trained model.")

            logging.info(f"Loading model weights from '{model_path}' ...")
            state_dict = torch.load(model_path)

            if run.NN_CONFIG["v"]["only_train_func_dependence"]:
                # Remove some keys before loading so we can retrain the function coordinates in this special mode:
                keys_to_remove = ["v_stationary_mean", "func_coord_real", "func_coord_imag", "s_stationary_mean", "der_func_coord_real", "der_func_coord_imag"]
                removed_keys = [key for key in keys_to_remove if state_dict.pop(key, None) is not None]
                if removed_keys:
                    logging.info(f"Removed {len(removed_keys)} keys from the loaded state dictionary: {removed_keys}")

            trainer.model.load_state_dict(state_dict, strict=False)
            # Print loaded model parameters:
            insert_section_marker()
            logging.info("PREVIOUS MODEL PARAMETERS USED AS STARTING POINT:")
            log_model_parameters(trainer.model)
            insert_section_marker()

        # Count the number of trainable parameters in different categories:
        cat_totals = count_trainable_parameters(trainer)
        # Log the counts of trainable parameters:
        logging.info(f"Beginning to solve the '{run.RN_CONFIG['reaction_network_abbreviation']}' reaction network with {sum(cat_totals.values())} parameters ...")
        for k, v in ((k, v) for k, v in cat_totals.items() if v > 0):
            logging.info(f"> {k}: {v} parameters")

        insert_blank_line()
        history = trainer.train()
        insert_blank_line()
        # Save the neural network state dictionary:
        state_dict_path = f"{run.results_subdir}/{run.timestamp}_{run.RN_CONFIG['reaction_network_abbreviation']}_model_weights.pth"
        torch.save(trainer.model.state_dict(), state_dict_path)
        logging.info(f"Model state dictionary saved to '{state_dict_path}'.")
        # Save the training history data:
        history_path = f"{run.results_subdir}/{run.timestamp}_{run.RN_CONFIG['reaction_network_abbreviation']}_training_history.json"
        with open(history_path, "w") as json_file:
            json.dump(history, json_file, indent=4)
        logging.info(f"Training history saved to '{history_path}'.")

    else:
        # Get the model and history paths based on the configuration:
        if run.NN_CONFIG["trained_model_timestamp"]:
            model_path = f"{run.results_dir}/{run.NN_CONFIG['trained_model_timestamp']}_{run.config_file_name}/{run.NN_CONFIG['trained_model_timestamp']}_{run.RN_CONFIG['reaction_network_abbreviation']}_model_weights.pth"
            history_path = f"{run.results_dir}/{run.NN_CONFIG['trained_model_timestamp']}_{run.config_file_name}/{run.NN_CONFIG['trained_model_timestamp']}_{run.RN_CONFIG['reaction_network_abbreviation']}_training_history.json"
        elif run.NN_CONFIG["trained_model_path"]:
            model_path = run.NN_CONFIG["trained_model_path"]
            if run.NN_CONFIG["training_history_path"]:
                history_path = run.NN_CONFIG["training_history_path"]
            else:
                raise ValueError("When 'trained_model_path' is specified, 'training_history_path' must also be provided to load the training history.")
        else:
            raise ValueError("Either 'trained_model_timestamp' or 'trained_model_path' must be specified in the configuration file to load a pre-trained model.")

        # Load the model weights from a file:
        logging.info(f"Loading model weights from '{model_path}' ...")
        trainer.model.load_state_dict(torch.load(model_path), strict=False)
        # Load the training history data from a file:
        logging.info(f"Loading training history from '{history_path}' ...")
        with open(history_path) as json_file:
            history = json.load(json_file)

    # Print trained model parameters:
    insert_section_marker()
    logging.info("TRAINED MODEL PARAMETERS:")
    log_model_parameters(trainer.model)

    # Save the training history plot:
    training_plots.plot_training_history(history, run)

    # This computes relevant statistics using the times logged in the log file:
    training_log_file_analysis(run)

    return history

