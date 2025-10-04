"""
This module runs the models and algorithms in QC Lab for a given simulation setting and
compares them against a reference dataset.
"""

import pytest
import os
import numpy as np
from qc_lab import Simulation, Data
from qc_lab.models import (
    SpinBoson,
    HolsteinLattice,
    HolsteinLatticeReciprocalSpace,
    FMOComplex,
    TullyProblemOne,
    TullyProblemTwo,
    TullyProblemThree,
)
from qc_lab.algorithms import MeanField, FewestSwitchesSurfaceHopping
from qc_lab.dynamics import serial_driver, parallel_driver_multiprocessing
from qc_lab.numerical_constants import INVCM_TO_300K


model_sim_settings = {
    "SpinBoson": {
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "HolsteinLattice": {
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "HolsteinLatticeReciprocalSpace": {
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "FMOComplex": {
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "TullyProblemOne": {
        "tmax": 5000,
        "dt_update": 0.5,
        "dt_collect": 10,
        "progress_bar": False,
    },
    "TullyProblemTwo": {
        "tmax": 5000,
        "dt_update": 0.5,
        "dt_collect": 10,
        "progress_bar": False,
    },
    "TullyProblemThree": {
        "tmax": 5000,
        "dt_update": 0.5,
        "dt_collect": 10,
        "progress_bar": False,
    },
}


model_settings = {
    "SpinBoson": {
        "kBT": 1.0,
        "V": 0.5,
        "E": 0.5,
        "A": 100,
        "W": 0.1,
        "l_reorg": 0.005,
        "boson_mass": 1.0,
    },
    "HolsteinLattice": {
        "kBT": 1.0,
        "g": 0.5,
        "w": 0.5,
        "N": 10,
        "J": 1.0,
        "phonon_mass": 1.0,
        "periodic": True,
    },
    "HolsteinLatticeReciprocalSpace": {
        "kBT": 1.0,
        "g": 0.5,
        "w": 0.5,
        "N": 10,
        "J": 1.0,
        "phonon_mass": 1.0,
        "periodic": True,
    },
    "FMOComplex": {
        "kBT": 1.0,
        "mass": 1.0,
        "l_reorg": 35.0 * INVCM_TO_300K,
        "w_c": 106.14 * INVCM_TO_300K,
        "N": 200,
    },
    "TullyProblemOne": {
        "init_momentum": 10.0,
        "init_position": -5.0,
        "mass": 2000.0,
        "A": 0.01,
        "B": 1.6,
        "C": 0.005,
        "D": 1.0,
    },
    "TullyProblemTwo": {
        "init_momentum": 10.0,
        "init_position": -5.0,
        "mass": 2000.0,
        "A": 0.1,
        "B": 0.28,
        "C": 0.015,
        "D": 0.06,
        "E_0": 0.05,
    },
    "TullyProblemThree": {
        "init_momentum": 10.0,
        "init_position": -5.0,
        "mass": 2000.0,
        "A": 0.0006,
        "B": 0.1,
        "C": 0.9,
    },
}



def test_output_serial():
    """
    This test verifies that the serial driver produces the expected results.
    """
    local_settings = {"num_trajs": 10, "batch_size": 10}
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [MeanField, FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(
                {**model_sim_settings[model_class.__name__], **local_settings}
            )
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class()
            sim.initial_state.wf_db = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state.wf_db[0] = 1j
            data = serial_driver(sim)
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


def test_output_multiprocessing():
    """
    This test verifies that the multiprocessing driver produces the same results
    as the serial driver.
    """
    local_settings = {"num_trajs": 10, "batch_size": 5}
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [MeanField, FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(
                {**model_sim_settings[model_class.__name__], **local_settings}
            )
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class()
            sim.initial_state.wf_db = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state.wf_db[0] = 1j
            data = parallel_driver_multiprocessing(sim)
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


def test_output_different_h():
    """
    This test ensures that changing the value of the ``classical_coordinate_weight`` does
    not change the result of the simulation.
    """
    local_settings = {"num_trajs": 10, "batch_size": 10}
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [MeanField, FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(
                {**model_sim_settings[model_class.__name__], **local_settings}
            )
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            np.random.seed(10)
            # Adjust the weights to be a random positive number between 1 and 2.
            sim.model.constants.classical_coordinate_weight = (
                np.random.rand(sim.model.constants.num_classical_coordinates) + 1.0
            )
            sim.algorithm = algorithm_class()
            sim.initial_state.wf_db = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state.wf_db[0] = 1j
            data = serial_driver(sim)
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


def test_output_fssh_gauge_fixing():
    """
    This test ensures that the initial choice of phase in FSSH does not affect
    the final result of a calculation.

    The test proceeds by adding a random phase to the eigenvectors computed at the
    outset of the simulation. The relative phases of these vectors are fixed so that
    the derivative couplings are real-valued ("phase_der_couple"). At subsequent
    timesteps, the relative phase of the new eigenvectors is determined from maximizing the
    real part of their overlap with the previous set of eigenvectors ("phase_overlap").

    Note that this test will only work if the ``add_random_phase`` task is between the
    ``diagonalize_matrix`` and ``gauge_fix_eigs`` tasks in the initialization recipe.

    """
    local_settings = {"num_trajs": 10, "batch_size": 10}
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    my_FSSH = FewestSwitchesSurfaceHopping({"gauge_fixing": "phase_overlap"})

    def add_random_phase(sim, state, parameters):
        np.random.seed(10)
        random_phases = np.exp(
            1j
            * 2.0
            * np.pi
            * np.random.rand(
                sim.settings.batch_size, sim.model.constants.num_quantum_states
            )
        )
        state.eigvecs = random_phases[:, np.newaxis, :] * state.eigvecs
        return state, parameters

    my_FSSH.initialization_recipe.insert(6, add_random_phase)
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [my_FSSH]:
            print(f"Testing {model_class.__name__} with FewestSwitchesSurfaceHopping")
            sim = Simulation(
                {**model_sim_settings[model_class.__name__], **local_settings}
            )
            model_name = model_class.__name__
            algorithm_name = "FewestSwitchesSurfaceHopping"
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class
            sim.initial_state.wf_db = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state.wf_db[0] = 1j
            data = serial_driver(sim)
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                print("Comparing ", key)
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


def test_output_fssh_deterministic():
    """
    This test verifies the deterministic version of FSSH.
    """
    local_settings = {"num_trajs": 10, "batch_size": 10}
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        algorithm_class = FewestSwitchesSurfaceHopping
        print(f"Testing {model_class.__name__} with deterministic {algorithm_class.__name__}")

        sim = Simulation({**model_sim_settings[model_class.__name__], **local_settings})
        model_name = model_class.__name__
        algorithm_name = algorithm_class.__name__
        sim.model = model_class(model_settings[model_class.__name__])
        sim.model.initialize_constants()
        sim.settings.batch_size *= sim.model.constants.num_quantum_states
        sim.settings.num_trajs *= sim.model.constants.num_quantum_states
        sim.algorithm = algorithm_class({"fssh_deterministic": True})
        sim.initial_state.wf_db = np.zeros(
            sim.model.constants.num_quantum_states, dtype=complex
        )
        sim.initial_state.wf_db[0] = 1j
        data = serial_driver(sim)
        data_correct = Data().load(
            os.path.join(
                reference_folder, f"{model_name}_{algorithm_name}_deterministic.h5"
            )
        )
        for key, val in data.data_dict.items():
            np.testing.assert_allclose(
                val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
            )
    return


if __name__ == "__main__":
    test_output_serial()
    test_output_multiprocessing()
    test_output_different_h()
    test_output_fssh_gauge_fixing()
