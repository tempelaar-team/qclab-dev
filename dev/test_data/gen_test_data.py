"""
This module runs the models and algorithms in QC Lab for a given simulation setting and
compares them against a reference dataset.
"""

import pytest
import os
import numpy as np
from qclab import Simulation, Data
from qclab.models import (
    SpinBoson,
    HolsteinLattice,
    HolsteinLatticeReciprocalSpace,
    FMOComplex,
    TullyProblemOne,
    TullyProblemTwo,
    TullyProblemThree,
)
from qclab.algorithms import MeanField, FewestSwitchesSurfaceHopping
from qclab.dynamics import serial_driver, parallel_driver_multiprocessing
from qclab.numerical_constants import INVCM_TO_300K


model_sim_settings = {
    "SpinBoson": {
        "num_trajs": 100,
        "batch_size": 100,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "HolsteinLattice": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "HolsteinLatticeReciprocalSpace": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "FMOComplex": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "TullyProblemOne": {
        "num_trajs": 100,
        "batch_size": 100,
        "tmax": 5000,
        "dt_update": 0.5,
        "dt_collect": 10,
        "progress_bar": False,
    },
    "TullyProblemTwo": {
        "num_trajs": 100,
        "batch_size": 100,
        "tmax": 5000,
        "dt_update": 0.5,
        "dt_collect": 10,
        "progress_bar": False,
    },
    "TullyProblemThree": {
        "num_trajs": 100,
        "batch_size": 100,
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
        "l_reorg": 35.0 * INVCM_TO_300K,  # reorganization energy
        "w_c": 106.14 * INVCM_TO_300K,  # characteristic frequency
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

def gen_test_data():
    reference_folder = os.path.join(os.path.dirname(__file__), "data_noh5/")
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

            sim = Simulation(model_sim_settings[model_class.__name__])
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class()
            sim.initial_state["wf_db"] = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state["wf_db"][0] = 1j
            data = serial_driver(sim)
            data.save(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}")
            )
    return

def gen_test_data_deterministic():
    reference_folder = os.path.join(os.path.dirname(__file__), "data_noh5/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
        HolsteinLatticeReciprocalSpace,
        FMOComplex,
        TullyProblemOne,
        TullyProblemTwo,
        TullyProblemThree,
    ]:
        for algorithm_class in [FewestSwitchesSurfaceHopping]:
            print(f"Testing {model_class.__name__} with {algorithm_class.__name__}")

            sim = Simulation(model_sim_settings[model_class.__name__])
            model_name = model_class.__name__
            algorithm_name = algorithm_class.__name__
            sim.model = model_class(model_settings[model_class.__name__])
            sim.model.initialize_constants()
            sim.settings.batch_size *= sim.model.constants.num_quantum_states
            sim.settings.num_trajs *= sim.model.constants.num_quantum_states
            sim.algorithm = algorithm_class({"fssh_deterministic":True})
            sim.initial_state["wf_db"] = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.initial_state["wf_db"][0] = 1j
            data = serial_driver(sim)
            data.save(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}_deterministic")
            )
    return



if __name__ == "__main__":
    gen_test_data_deterministic()
    gen_test_data()
    #check_test_data()