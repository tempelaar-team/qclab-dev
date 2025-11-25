from qclab import Simulation


def action_variable_specs(
    sim: Simulation, state: dict, parameters: dict, kwarg1: str = "default_value"
):
    """
    One line description

    Extended description

    Parameters
    ----------

    Optional Keyword Arguments
    --------------------------
    kwarg1:
        Description of kwarg1.

    Requires
    --------
    sim.model.constants.constant_1: ndarray of shape (B, C), dtype=complex128
        Description of constant_1.
    sim.algorithm.settings.algo_setting_1: type
        Description of algo_setting_1.
    sim.model.ingredients.get('ingredient_name'): IngredientType
        Description of ingredient_name.
    sim.settings.setting_1: type
        Description of setting_1.

    Reads
    -----
    state["test"]: ndarray of shape (B, C), dtype=complex128
        Description of state["test"],
    parameters["test2"]: ndarray of shape (B, C), dtype=complex128
        Description of parameters["test2"]

    Writes
    ------
    state["test3"]: ndarray of shape (B, C), dtype=complex128
        Description of state["test3"],
    parameters["test4"]: ndarray of shape (B, C), dtype=complex128
        Description of parameters["test4"]

    Raises
    ------

    Notes
    -----
    Symbols: B = sim.settings.batch_size, C = sim.model.constants.num_classical_coordinates.

    See Also
    --------
    """
    return state, parameters

