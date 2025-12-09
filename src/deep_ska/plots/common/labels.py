"""Common label generation functions for plotting."""

def generate_output_function_labels(out_function_size: int, use_theta: bool = False) -> list:
    """Generate the labels for the output functions based on the output function size."""
    theta = "_{{\\theta}}" if use_theta else ""
    if out_function_size == 1:
        return [f"$E_x[f(X{theta}(t))]$"]
    return [f"$E_x[f_{i+1}(X{theta}(t))]$" for i in range(out_function_size)]


def generate_stationary_output_function_labels(out_function_size: int) -> list:
    """Generate the stationarity labels for the output functions based on the output function size."""
    if out_function_size == 1:
        return ["$f$"]
    return [f"$f_{i+1}$" for i in range(out_function_size)]


def generate_trajectory_labels(out_function_size: int, use_theta: bool = False) -> list:
    """Generate the labels for the trajectories based on the output function size."""
    theta = "_{{\\theta}}" if use_theta else ""
    if out_function_size == 1:
        return [f"Traj. for $E_x[f(X{theta}(t))]$"]
    return [f"Traj. for $E_x[f_{i+1}(X{theta}(t))]$" for i in range(out_function_size)]


def generate_variance_output_function_labels(out_function_size: int, use_theta: bool = False) -> list:
    """Generate the variance labels for the output functions based on the output function size."""
    theta = "_{{\\theta}}" if use_theta else ""
    if out_function_size == 1:
        return [f"$\\text{{Var}}_x[f(X{theta}(t))]$"]
    return [f"$\\text{{Var}}_x[f_{i+1}(X{theta}(t))]$" for i in range(out_function_size)]
