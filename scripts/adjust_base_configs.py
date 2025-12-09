"""Generates modified versions of base configuration files by manipulating various fields."""

import logging
import os
from collections.abc import Callable
from copy import deepcopy

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

yaml = YAML()


def load_yaml(file_path: str) -> CommentedMap:
    """Loads a YAML file and returns its content as a dictionary."""
    with open(file_path) as file:
        return yaml.load(file)


def save_yaml(data: CommentedMap, file_path: str) -> None:
    """Saves a CommentedMap to a YAML file with custom formatting."""
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def set_flow_style_for_nested_lists(parent_list: CommentedSeq) -> None:
    """Helper function to set flow style for all nested lists."""
    parent_list.fa.set_flow_style()
    for item in parent_list:
        if hasattr(item, 'fa'):
            item.fa.set_flow_style()


def v_part1a_usual_crns(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1: Usual CRNs with identity relationship and x -> x^2."""
    data['reaction_network_config']['output_function_arguments']['monomial_orders'].clear()
    data['reaction_network_config']['output_function_arguments']['monomial_orders'].extend([1, 2])

    data['neural_network_config']['v']['stationary_mean_initialization'] = 'estimate'

    data['validation_config']['compute_is_estimate'] = True
    data['validation_config']['compute_deep_var_estimates'] = True

    return data


def v_part1a_usual_crns_train_only(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1: Usual CRNs with identity relationship and x -> x^2, training only."""
    data = v_part1a_usual_crns(data)

    data['validation_config']['compute_cv_estimate'] = False
    data['validation_config']['compute_is_estimate'] = False
    data['validation_config']['compute_deep_var_estimates'] = False
    data['validation_config']['em_validation_needed'] = False

    if data['reaction_network_config']['reaction_network'] == 'ConstitutiveGeneExpression':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'ReferenceBasedAntitheticIntegralController':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'SusceptibleInfectedRecoveredNetwork':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'SelfRegulatoryGeneExpression':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'Repressilator':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'GeneticToggleSwitch':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    set_flow_style_for_nested_lists(data['validation_config']['conv_test_states'])

    return data


def v_part1a_usual_crns_dlmc_only(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1: Usual CRNs with identity relationship and x -> x^2, DLMC methods only."""
    data = v_part1a_usual_crns(data)

    data['neural_network_config']['use_previous_training_weights'] = True
    data['neural_network_config']['model_training_needed'] = False
    data['neural_network_config']['train_size'] = 5
    data['neural_network_config']['valid_size'] = 5

    if data['reaction_network_config']['reaction_network'] == 'ReferenceBasedAntitheticIntegralController':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'ConstitutiveGeneExpression':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'SusceptibleInfectedRecoveredNetwork':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'SelfRegulatoryGeneExpression':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'Repressilator':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'GeneticToggleSwitch':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-45'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    set_flow_style_for_nested_lists(data['validation_config']['conv_test_states'])

    return data


def v_part1a_usual_crns_conv_only(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1: Usual CRNs with identity relationship and x -> x^2, convergence only."""
    data = v_part1a_usual_crns(data)

    data['neural_network_config']['use_previous_training_weights'] = True
    data['neural_network_config']['model_training_needed'] = False
    data['neural_network_config']['train_size'] = 5
    data['neural_network_config']['valid_size'] = 5

    data['validation_config']['ssa_with_cv_fraction'] = 0.0001
    data['validation_config']['ssa_with_is_fraction'] = 0.0001
    data['validation_config']['compute_deep_var_estimates'] = False
    data['validation_config']['em_validation_needed'] = False

    if data['reaction_network_config']['reaction_network'] == 'ReferenceBasedAntitheticIntegralController':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
    elif data['reaction_network_config']['reaction_network'] == 'ConstitutiveGeneExpression':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
    elif data['reaction_network_config']['reaction_network'] == 'SusceptibleInfectedRecoveredNetwork':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
    elif data['reaction_network_config']['reaction_network'] == 'SelfRegulatoryGeneExpression':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
    elif data['reaction_network_config']['reaction_network'] == 'Repressilator':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-28'
    elif data['reaction_network_config']['reaction_network'] == 'GeneticToggleSwitch':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-22_22-44-45'

    return data


def v_part1b_unusual_crns(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1b: Unusual CRNs with identity relationship and x -> x^2."""
    data['reaction_network_config']['output_function_arguments']['monomial_orders'].clear()
    data['reaction_network_config']['output_function_arguments']['monomial_orders'].extend([1, 2])

    data['neural_network_config']['v']['stationary_mean_initialization'] = 'estimate'

    data['validation_config']['compute_is_estimate'] = True
    data['validation_config']['compute_deep_var_estimates'] = True

    return data


def v_part1b_unusual_crns_train_only(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1b: Unusual CRNs with identity relationship and x -> x^2, training only."""
    data = v_part1b_unusual_crns(data)

    data['validation_config']['compute_cv_estimate'] = False
    data['validation_config']['compute_is_estimate'] = False
    data['validation_config']['compute_deep_var_estimates'] = False
    data['validation_config']['em_validation_needed'] = False

    if data['reaction_network_config']['reaction_network'] == 'SensorBasedAntitheticIntegralController':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'NonlinearConversionCascade':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'LinearConversionCascadeWithFeedback':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    set_flow_style_for_nested_lists(data['validation_config']['conv_test_states'])

    return data


def v_part1b_unusual_crns_dlmc_only(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1b: Unusual CRNs with identity relationship and x -> x^2, DLMC methods only."""
    data = v_part1a_usual_crns(data)

    data['neural_network_config']['use_previous_training_weights'] = True
    data['neural_network_config']['model_training_needed'] = False
    data['neural_network_config']['train_size'] = 5
    data['neural_network_config']['valid_size'] = 5

    if data['reaction_network_config']['reaction_network'] == 'SensorBasedAntitheticIntegralController':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-27_21-02-57'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'NonlinearConversionCascade':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-27_21-02-54'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'LinearConversionCascadeWithFeedback':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-27_21-02-57'
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    set_flow_style_for_nested_lists(data['validation_config']['conv_test_states'])

    return data


def v_part1b_unusual_crns_conv_only(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1b: Unusual CRNs with identity relationship and x -> x^2, convergence only."""
    data = v_part1b_unusual_crns(data)

    data['neural_network_config']['use_previous_training_weights'] = True
    data['neural_network_config']['model_training_needed'] = False
    data['neural_network_config']['train_size'] = 5
    data['neural_network_config']['valid_size'] = 5

    data['validation_config']['n_trajectories'] = 100
    data['validation_config']['ssa_with_cv_fraction'] = 0.05
    data['validation_config']['ssa_with_is_fraction'] = 0.05
    data['validation_config']['compute_deep_var_estimates'] = False
    data['validation_config']['em_validation_needed'] = False

    if data['reaction_network_config']['reaction_network'] == 'SensorBasedAntitheticIntegralController':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-27_21-02-57'
    elif data['reaction_network_config']['reaction_network'] == 'NonlinearConversionCascade':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-27_21-02-54'
    elif data['reaction_network_config']['reaction_network'] == 'LinearConversionCascadeWithFeedback':
        data['neural_network_config']['trained_model_timestamp'] = '2025-09-27_21-02-57'

    return data


def v_part1c_naive_architecture(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1c: Naive approach."""
    data = v_part1a_usual_crns(data)
    data['subnet_architecture'] = 'naive'
    return data


def v_part1d_features_architecture(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 1d: Features architecture."""
    data = v_part1a_usual_crns(data)
    data['subnet_architecture'] = 'features'
    return data


def v_part2a_pinn_relationship(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 3: PINN relationship."""
    data = v_part1a_usual_crns(data)
    data['neural_network_config']['v']['loss_function'] = 'pinn_loss'
    data['neural_network_config']['v']['exact_time_derivative'] = True

    data['validation_config']['compute_cv_estimate'] = False
    data['validation_config']['compute_is_estimate'] = False
    data['validation_config']['compute_deep_var_estimates'] = False
    data['validation_config']['em_validation_needed'] = False

    if data['reaction_network_config']['reaction_network'] == 'ReferenceBasedAntitheticIntegralController':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'ConstitutiveGeneExpression':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'SusceptibleInfectedRecoveredNetwork':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'SelfRegulatoryGeneExpression':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'Repressilator':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    elif data['reaction_network_config']['reaction_network'] == 'GeneticToggleSwitch':
        data['validation_config']['conv_test_states'] = CommentedSeq([])
    set_flow_style_for_nested_lists(data['validation_config']['test_states'])

    return data


def v_part2b_log_relationship(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PART 4: Logarithmic relationship."""
    data = v_part2a_pinn_relationship(data)
    data['reaction_network_config']['as_relationship'] = 'logarithm'
    data['neural_network_config']['v']['loss_function'] = 'pinn_loss'
    return data


def v_part2c_id_pinn_relationship(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - Multiobjective identity-PINN relationship."""
    data = v_part2a_pinn_relationship(data)
    data['neural_network_config']['v']['loss_function'] = 'id_pinn_loss'
    return data


def v_part2d_id_pinn_relationship_adapative(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - Multiobjective identity-PINN relationship with adaptive weights."""
    data = v_part2a_pinn_relationship(data)
    data['neural_network_config']['v']['loss_function'] = 'id_pinn_loss'
    data['neural_network_config']['v']['adaptive_weights'] = True
    return data


def v_part2e_identity_relationship_no_sim(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - Identity relationship without simulation data."""
    data = v_part2a_pinn_relationship(data)
    data['reaction_network_config']['lower_bound'] = 0
    data['reaction_network_config']['upper_bound'] = 30
    data['neural_network_config']['v']['loss_function'] = 'identity_loss'
    data['neural_network_config']['use_ssa_samples'] = False
    return data


def v_part2f_pinn_relationship_no_sim(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - PINN relationship without simulation data."""
    data = v_part2a_pinn_relationship(data)
    data['reaction_network_config']['lower_bound'] = 0
    data['reaction_network_config']['upper_bound'] = 30
    data['neural_network_config']['v']['loss_function'] = 'pinn_loss'
    data['neural_network_config']['use_ssa_samples'] = False
    return data


def v_part2g_id_pinn_relationship_no_sim(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - Multiobjective identity-PINN relationship without simulation data."""
    data = v_part2a_pinn_relationship(data)
    data['reaction_network_config']['lower_bound'] = 0
    data['reaction_network_config']['upper_bound'] = 30
    data['neural_network_config']['v']['loss_function'] = 'id_pinn_loss'
    data['neural_network_config']['use_ssa_samples'] = False
    return data


def v_part2h_id_pinn_relationship_adapative_no_sim(data: CommentedMap) -> CommentedMap:
    """V(x,f,t) - Multiobjective identity-PINN relationship with adaptive weights, without simulation data."""
    data = v_part2a_pinn_relationship(data)
    data['reaction_network_config']['lower_bound'] = 0
    data['reaction_network_config']['upper_bound'] = 30
    data['neural_network_config']['v']['loss_function'] = 'id_pinn_loss'
    data['neural_network_config']['v']['adaptive_weights'] = True
    data['neural_network_config']['use_ssa_samples'] = False
    return data


def generate_version(modification_function: Callable, config_name: str, output_dir: str = ".") -> None:
    """Generates modified versions of a base configuration file by applying specified modifications."""
    # Loop through each modification function and generate a version
    data = load_yaml(f"configs/{config_name}.ska.yaml")
    # Create a new version by applying the modification function
    modified_data = modification_function(deepcopy(data))  # .copy() ensures the original data remains unchanged
    # Save the modified data to a new file
    modification_function_name = modification_function.__name__
    if modification_function_name in ['v_part1a_usual_crns_dlmc_only', 'v_part1a_usual_crns_conv_only']:
        output_file = f'v_part1a_usual_crns_train_only_{config_name}.ska.yaml'
    elif modification_function_name in ['v_part1b_unusual_crns_dlmc_only', 'v_part1b_unusual_crns_conv_only']:
        output_file = f'v_part1b_unusual_crns_train_only_{config_name}.ska.yaml'
    else:
       output_file = f'{modification_function_name}_{config_name}.ska.yaml'

    # Create a subdirectory named after the modification function
    function_output_dir = os.path.join(output_dir, modification_function_name)
    os.makedirs(function_output_dir, exist_ok=True)

    # Save the modified data to a new file in the function-specific directory
    full_output_path = os.path.join(function_output_dir, output_file)
    save_yaml(modified_data, full_output_path)
    logging.info(f"Modification '{modification_function_name}' for {config_name} saved in '{function_output_dir}'.")


def main() -> None:
    """Main function to generate modified configuration files."""
    # Configure YAML settings:
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 1000
    yaml.preserve_quotes = True
    yaml.default_flow_style = False

    # Dictionary of configuration files to generate versions for:
    version_modifications = {
        v_part1a_usual_crns: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part1a_usual_crns_train_only: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part1a_usual_crns_dlmc_only: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part1a_usual_crns_conv_only: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part1b_unusual_crns: ["sAIC", "NCC", "LCF"],
        v_part1b_unusual_crns_train_only: ["sAIC", "NCC", "LCF"],
        v_part1b_unusual_crns_dlmc_only: ["sAIC", "NCC", "LCF"],
        v_part1b_unusual_crns_conv_only: ["sAIC", "NCC", "LCF"],
        v_part1c_naive_architecture: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR", "sAIC", "NCC", "LCF"],
        v_part1d_features_architecture: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR", "sAIC", "NCC", "LCF"],
        v_part2a_pinn_relationship: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part2b_log_relationship: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part2c_id_pinn_relationship: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part2d_id_pinn_relationship_adapative: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part2e_identity_relationship_no_sim: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part2f_pinn_relationship_no_sim: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part2g_id_pinn_relationship_no_sim: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
        v_part2h_id_pinn_relationship_adapative_no_sim: ["CGE", "SLF", "TSW", "REP", "rAIC", "SIR"],
    }

    # Specify the output directory for generated config files:
    output_directory = "cluster_configs"

    for modification_function, config_list in version_modifications.items():
        for config_name in config_list:
            generate_version(modification_function, config_name, output_directory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
