"""
This module contains the collect tasks which are used to collect data from the state
or parameters objects into the output dictionary of the state object.
"""


def collect_t(algorithm, sim, parameters, state):
    """
    Collect the time in the state object.

    Required constants:
        - None.
    """
    state.output_dict["t"] = state.t
    return parameters, state


def collect_dm_db(algorithm, sim, parameters, state):
    """
    Collect the diabatic density matrix in the state object.

    Required constants:
        - None.
    """
    state.output_dict["dm_db"] = state.dm_db
    return parameters, state


def collect_classical_energy(algorithm, sim, parameters, state):
    """
    Collect the classical energy in the state object.

    Required constants:
        - None.
    """
    state.output_dict["classical_energy"] = state.classical_energy
    return parameters, state


def collect_quantum_energy(algorithm, sim, parameters, state):
    """
    Collect the quantum energy in the state object.

    Required constants:
        - None.
    """
    state.output_dict["quantum_energy"] = state.quantum_energy
    return parameters, state
