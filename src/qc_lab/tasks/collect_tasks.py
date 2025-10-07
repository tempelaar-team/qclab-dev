"""
This module contains tasks that are used to collect data from the state or
parameters objects into the output dictionary of the state object.
"""


def collect_t(sim, state, parameters):
    """
    Collect the time in the state object.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.output_dict["t"] : ndarray
        stores the current time in each trajectory.
    """
    state.output_dict["t"] = state.t
    return state, parameters


def collect_dm_db(sim, state, parameters):
    """
    Collect the diabatic density matrix in the state object.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.output_dict["dm_db"] : ndarray
        stores the diabatic density matrix.
    """
    state.output_dict["dm_db"] = state.dm_db
    return state, parameters


def collect_classical_energy(sim, state, parameters):
    """
    Collect the classical energy in the state object.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.output_dict["classical_energy"] : ndarray
        stores the classical energy.
    """
    state.output_dict["classical_energy"] = state.classical_energy
    return state, parameters


def collect_quantum_energy(sim, state, parameters):
    """
    Collect the quantum energy in the state object.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.output_dict["quantum_energy"] : ndarray
        stores the quantum energy.
    """
    state.output_dict["quantum_energy"] = state.quantum_energy
    return state, parameters
