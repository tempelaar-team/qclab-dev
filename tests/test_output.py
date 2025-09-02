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
    FMOComplex,
    TullyProblemOne,
    TullyProblemTwo,
    TullyProblemThree,
)
from qc_lab.algorithms import MeanField, FewestSwitchesSurfaceHopping
from qc_lab.dynamics import serial_driver


model_sim_settings = {
    "SpinBoson": {
        "num_trajs": 10,
        "batch_size": 10,
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
    "FMOComplex": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 10,
        "dt_update": 0.01,
        "dt_collect": 0.1,
        "progress_bar": False,
    },
    "TullyProblemOne": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 5000,
        "dt_update": 0.01,
        "dt_collect": 10,
        "progress_bar": False,
    },
    "TullyProblemTwo": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 5000,
        "dt_update": 0.01,
        "dt_collect": 10,
        "progress_bar": False,
    },
    "TullyProblemThree": {
        "num_trajs": 10,
        "batch_size": 10,
        "tmax": 5000,
        "dt_update": 0.01,
        "dt_collect": 10,
        "progress_bar": False,
    },
}


def test_output():
    reference_folder = os.path.join(os.path.dirname(__file__), "reference/")
    for model_class in [
        SpinBoson,
        HolsteinLattice,
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
            sim.model = model_class()
            sim.model.initialize_constants()
            sim.algorithm = algorithm_class()
            sim.state.wf_db = np.zeros(
                sim.model.constants.num_quantum_states, dtype=complex
            )
            sim.state.wf_db[0] = 1j
            data = serial_driver(sim)
            data_correct = Data().load(
                os.path.join(reference_folder, f"{model_name}_{algorithm_name}.h5")
            )
            for key, val in data.data_dict.items():
                np.testing.assert_allclose(
                    val, data_correct.data_dict[key], rtol=1e-5, atol=1e-8
                )
    return


if __name__ == "__main__":
    test_output()
