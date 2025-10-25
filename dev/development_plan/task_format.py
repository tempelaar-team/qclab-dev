from qclab import Simulation


def action_variable_specs(
    sim: Simulation, state: dict, parameters: dict, kwarg1: str = "default_value"
):
    """
    One line description

    Extended description

    Parameters
    ----------
    sim:
        The simulation object.
    state:
        The state object.
    parameters:
        The parameters object.

    Other Parameters
    ----------------
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

    Writes
    ------

    Returns
    -------
    (state, parameters) : tuple(dict, dict)
        The updated state and parameters objects.

    Raises
    ------

    Notes
    -----
    Symbols: B = sim.settings.batch_size, C = sim.model.constants.num_classical_coordinates.

    See Also
    --------
    """
    return state, parameters


def action_variable_specs2(
    sim: Simulation, state: dict, parameters: dict, kwarg1: str = "default_value"
):
    """
    One line description

    Extended description

    Parameters
    ----------
    sim:
        The simulation object.
    state:
        The state object.
    parameters:
        The parameters object.

    Other Parameters
    ----------------
    kwarg1:
        Description of kwarg1.

    Requires
    --------
    :attr:`sim.model.constants.constant_1` : :class:`numpy.ndarray`, shape (B, C), dtype complex128
    Description of constant_1. Broadcastable to (B, C). Units: eV.

    :attr:`sim.algorithm.settings.algo_setting_1` : :class:`qclab.AlgoSetting`
        Description of algo_setting_1. Default: 3.2.

    :attr:`sim.model.ingredients['ingredient_name']` : :class:`qclab.Ingredient`
        Ingredient registered under key ``'ingredient_name'``. Must implement
        :meth:`qclab.Ingredient.__call__`.

    :attr:`sim.settings.setting_1` : :class:`int`
        Description of setting_1. Allowed: {0, 1}.

    Reads
    -----

    Writes
    ------

    Returns
    -------
    (state, parameters) : tuple(dict, dict)
        The updated state and parameters objects.

    Raises
    ------

    Notes
    -----
    Symbols: B = sim.settings.batch_size, C = sim.model.constants.num_classical_coordinates.

    See Also
    --------
    """
    return state, parameters