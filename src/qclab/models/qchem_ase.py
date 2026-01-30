"""
This module contains the Q-Chem ASE Model class.
"""

import numpy as np
from qclab.functions import (
    vectorize_ingredient,
    make_ingredient_sparse,
    z_to_q,
    z_to_p,
    qp_to_z,
    dqdp_to_dzc,
)
from qclab.model import Model, Constants
from qclab import ingredients
from qclab.numerical_constants import (
    INVCM_TO_HA,
    ANGSTROM_TO_BOHR,
    AMU_TO_EMASS,
)
from qclab.interfaces import QCLabQChemInterface
import copy


class QChemASE(Model):
    """
    Model class that uses the Q-Chem interface of the Atomic Simulation
    Environment (ASE) to perform ab initio quantum chemistry calculations.

    It is compatible with the adiabatic algorithms implemented in QC Lab.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "ase_atoms_object": None,
            "qchem_args": None,
            "qchem_tddft_args": None,
            "num_quantum_states": None,
            "kBT": 1.0,
            "seed": 0,
        }
        self.update_dh_qc_dzc = True
        self.update_h_q = False
        self.properties = {}
        self.calculator = QCLabQChemInterface
        super().__init__(self.default_constants, constants)

    def _init_model(self, parameters, **kwargs):
        mol = self.constants.ase_atoms_object
        atom_masses = mol.get_masses() * AMU_TO_EMASS
        num_atoms = len(atom_masses)
        self.constants.num_classical_coordinates = num_atoms * 3
        self.constants.classical_coordinate_mass = (
            atom_masses[np.newaxis] * np.ones((3, num_atoms))
        ).flatten()
        self.constants.classical_coordinate_weight = np.ones(
            self.constants.num_classical_coordinates
        )
        if not "energy_offset" in self.constants.__dict__:
            # The print statements in this function are just
            # for debugging and will be removed.
            print("Calculating energy offset")
            calc = self.calculator(
                atoms=mol,
                folder_scratch="qclab_job_" + str(self.constants.seed),
                **{
                    **self.constants.qchem_args,
                    **self.constants.qchem_tddft_args,
                    "seed": None,
                },
            )
            calc.label += "_energy_" + str(self.constants.seed)
            print("writing input"),
            calc.write_input(properties=["energy"],
                excited_amplitudes=False)
            print('execute')
            calc.execute()
            print('read')
            calc.read_results(properties=["energy"],
                excited_amplitudes=False)
            self.constants.energy_offset = calc.results["energy"][0]  # In Hartrees
            print("Energy offset in Hartrees:", self.constants.energy_offset)
        if not "harmonic_frequency" in self.constants.__dict__:
            print("Calculating harmonic frequencies and normal modes")
            calc = self.calculator(
                atoms=mol,
                folder_scratch="qclab_job_" + str(self.constants.seed),
                **{**self.constants.qchem_args, "seed": None},
            )
            calc.label += "_freq_" + str(self.constants.seed)
            calc.write_input(properties=["frequency"])
            calc.execute()
            calc.read_results(properties=["frequency"])
            self.constants.harmonic_frequency = calc.results["frequency"] * INVCM_TO_HA
            print("frequencies in Hartrees:", self.constants.harmonic_frequency)
            if np.any(self.constants.harmonic_frequency < 0):
                raise ValueError("Negative harmonic frequencies found.")
            self.constants.mass_weighted_normal_modes = (
                calc.results["normal_mode"]
                .reshape((len(self.constants.harmonic_frequency), num_atoms * 3))
                .T
            )
        self.constants.init_position = mol.get_positions().flatten() * ANGSTROM_TO_BOHR
        self.constants.finite_difference_delta = 1e-2

    def init_classical(self, parameters, **kwargs):
        # temporarily set the masses to 1 for mass-weighted normal mode sampling.
        normal_modes = self.constants.mass_weighted_normal_modes / np.sqrt(
            self.constants.classical_coordinate_mass[:, np.newaxis]
        )
        old_constants = copy.deepcopy(self.constants)
        self.constants = Constants()
        self.constants.num_classical_coordinates = len(
            old_constants.harmonic_frequency
        )  # set to number of normal modes
        self.constants.classical_coordinate_mass = np.ones(
            self.constants.num_classical_coordinates
        )
        self.constants.harmonic_frequency = old_constants.harmonic_frequency
        self.constants.classical_coordinate_weight = self.constants.harmonic_frequency
        self.constants.kBT = old_constants.kBT
        z_mwnm = ingredients.init_classical_wigner_harmonic(self, parameters, **kwargs)
        # convert back to mass-weighted normal coordinates
        q_mwnm = z_to_q(
            z_mwnm,
            self.constants.classical_coordinate_mass[np.newaxis],
            self.constants.classical_coordinate_weight[np.newaxis],
        )
        p_mwnm = z_to_p(
            z_mwnm,
            self.constants.classical_coordinate_mass[np.newaxis],
            self.constants.classical_coordinate_weight[np.newaxis],
        )
        # convert back to Cartesian coordinates.
        q = np.einsum("tm, cm ->tc", q_mwnm, normal_modes) + old_constants.init_position
        p = np.einsum("tm, cm ->tc", p_mwnm, normal_modes)
        self.constants = old_constants
        z = qp_to_z(
            q,
            p,
            self.constants.classical_coordinate_mass[np.newaxis],
            self.constants.classical_coordinate_weight[np.newaxis],
        )
        return z

    def h_q(self, parameters, **kwargs):
        batch_size = kwargs["batch_size"]
        num_quantum_states = self.constants.num_quantum_states
        out = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )
        return out

    @vectorize_ingredient
    def h_qc(self, parameters, **kwargs):
        z = kwargs["z"]
        num_quantum_states = self.constants.num_quantum_states
        traj_ind = kwargs["traj_ind"]
        has_ab_initio_properties = "ab_initio_properties" in parameters.keys()
        if has_ab_initio_properties:
            properties = parameters["ab_initio_properties"][traj_ind]
            if "energy" in properties.keys():
                diag_h_qc = (
                    properties["energy"][:num_quantum_states]
                    - self.constants.energy_offset
                )
                print("loaded h_qc from parameters")
                needs_energy = False
            else:
                needs_energy = True
        if not (has_ab_initio_properties) or needs_energy:
            print("running h_qc")
            property_dict={
                "energy": {"z": z, "excited_amplitudes": True},
            },
            ab_initio_properties_calculator, has_ab_intio_property_calculator = self.get(
                "ab_initio_properties_calculator"
            )
            if has_ab_intio_property_calculator:
                print("running ab initio property calculator")
                properties = ab_initio_properties_calculator(
                    self,
                    parameters,
                    batch_size=1,
                    property_dict=property_dict,
                )


            # 
            # num_classical_coordinates = self.constants.num_classical_coordinates
            # num_quantum_states = self.constants.num_quantum_states
            # m = self.constants.classical_coordinate_mass
            # h = self.constants.classical_coordinate_weight
            # mol = self.constants.ase_atoms_object
            # qchem_args = self.constants.qchem_args
            # qchem_tddft_args = self.constants.qchem_tddft_args
            # q = z_to_q(
            #     z,
            #     m,
            #     h,
            # )
            # mol.set_positions(
            #     q.reshape((num_classical_coordinates // 3, 3)) / ANGSTROM_TO_BOHR
            # )
            # calc = self.calculator(
            #     atoms=mol,
            #     folder_scratch="qclab_job_" + str(self.constants.seed),
            #     **{**qchem_args, **qchem_tddft_args},
            # )
            # calc.label += "_energy"
            # calc.label += "_" + str(self.constants.seed)
            # calc.write_input(properties=["energy"])
            # print("Calculating energies")
            # calc.execute()
            # calc.read_results(properties=["energy"])
            # properties = calc.results
            diag_h_qc = (
                properties["energy"][:num_quantum_states] - self.constants.energy_offset
            )
        assert (
            len(diag_h_qc) == num_quantum_states
        ), "Number of quantum states mismatch." + str(diag_h_qc)
        if np.any(np.diff(diag_h_qc) < 0):
            print("Error: excited states are lower in energy than the ground state.")
            print(diag_h_qc)
        if np.any(np.diff(diag_h_qc) == 0):
            print("Error: degenerate states.")
            print(diag_h_qc)
        return np.diag(diag_h_qc)

    @make_ingredient_sparse
    @vectorize_ingredient
    def dh_qc_dzc(self, parameters, **kwargs):
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        out = np.zeros(
            (num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        traj_ind = kwargs["traj_ind"]
        has_ab_initio_properties = "ab_initio_properties" in parameters.keys()
        if has_ab_initio_properties:
            properties = parameters["ab_initio_properties"][traj_ind]
            if "gradient" in properties.keys():
                for state_ind in range(num_quantum_states):
                    # Convert to derivative w.r.t. zc.
                    out[:, state_ind, state_ind] = dqdp_to_dzc(
                        properties["gradient"][:, :, state_ind].flatten(),
                        None,
                        m,
                        h,
                    )
                print("loaded dh_qc_dzc from parameters")
                needs_gradient = False
            else:
                needs_gradient = True
        if not (has_ab_initio_properties) or needs_gradient:
            print("running dh_qc_dzc")
            z = kwargs["z"]
            mol = self.constants.ase_atoms_object
            qchem_args = self.constants.qchem_args
            qchem_tddft_args = self.constants.qchem_tddft_args
            q = z_to_q(z, m, h)
            mol.set_positions(
                q.reshape((num_classical_coordinates // 3, 3)) / ANGSTROM_TO_BOHR
            )

            calc = QCLabQChemCalculator(
                atoms=mol,
                folder_scratch="qclab_job_" + str(self.constants.seed),
                **{
                    **qchem_args,
                    **qchem_tddft_args,
                    "seed": None,
                },
            )

            calc.label += "_" + "g"
            calc.label += "_" + str(self.constants.seed)
            mol.calc = calc
            mol.calc.write_input(properties=["gradient"])
            mol.calc.execute()
            mol.calc.read_results(properties=["gradient"])

            for state_ind in range(num_quantum_states):
                # Convert to derivative w.r.t. zc.
                out[:, state_ind, state_ind] = dqdp_to_dzc(
                    mol.calc.results["gradient"][:, :, state_ind].flatten(),
                    None,
                    m,
                    h,
                )
        return out

    @vectorize_ingredient
    def derivative_coupling_dzc(self, parameters, **kwargs):
        print("derivative_coupling_dzc")
        z = kwargs["z"]
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        out = np.zeros(
            (num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        traj_ind = kwargs["traj_ind"]
        has_ab_initio_properties = "ab_initio_properties" in parameters.keys()
        if has_ab_initio_properties:
            properties = parameters["ab_initio_properties"][traj_ind]
            if "derivative_coupling" in properties.keys():
                derivative_coupling_dq = properties["derivative_coupling"]
                for key, val in derivative_coupling_dq.items():
                    out[:, key[0], key[1]] = (
                        dqdp_to_dzc(val.flatten(), None, m, h) / ANGSTROM_TO_BOHR
                    )  # convert from 1/A to 1/Bohr
                    out[:, key[1], key[0]] = -np.conj(out[:, key[0], key[1]])
                print(np.sum(np.abs(out), axis=0))
                print("loaded derivative_coupling from parameters")
                needs_derivative_coupling = False
            else:
                needs_derivative_coupling = True
        if not (has_ab_initio_properties) or needs_derivative_coupling:
            print("running derivative_coupling")
            mol = self.constants.ase_atoms_object
            qchem_args = self.constants.qchem_args
            qchem_tddft_args = self.constants.qchem_tddft_args
            q = z_to_q(z, m, h)
            mol.set_positions(
                q.reshape((num_classical_coordinates // 3, 3)) / ANGSTROM_TO_BOHR
            )
            if num_quantum_states < 2:
                return out
            calc = self.calculator(
                atoms=mol,
                folder_scratch="qclab_job_" + str(self.constants.seed),
                **{
                    **qchem_args,
                    **qchem_tddft_args,
                    "CALC_NAC": "True",
                    "CIS_DER_NUMSTATE": str(num_quantum_states),
                    "seed": None,
                },
            )
            calc.label += "_" + "derivative_coupling"
            calc.label += "_" + str(self.constants.seed)
            calc.write_input(
                properties=["derivative_coupling"],
                state_inds_derivative_couplings=[
                    i for i in range(self.constants.num_quantum_states)
                ],
            )
            calc.execute()
            print("Calculating derivative couplings")
            calc.read_results(properties=["derivative_coupling"])
            derivative_coupling_dq = calc.results["derivative_coupling"]
            for key, val in derivative_coupling_dq.items():
                out[:, key[0], key[1]] = (
                    dqdp_to_dzc(val.flatten(), None, m, h) / ANGSTROM_TO_BOHR
                )  # convert from 1/A to 1/Bohr
                out[:, key[1], key[0]] = -np.conj(out[:, key[0], key[1]])
            print(np.sum(np.abs(out), axis=0))
            print("finished derivative_coupling_dzc")
        return out

    

    ingredients = [
        ("ab_initio_properties_calculator", ingredients.ab_initio_properties_calculator_qchem),
        ("h_q", h_q),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_free),
        ("dh_c_dzc", ingredients.dh_c_dzc_free),
        ("dh_qc_dzc", dh_qc_dzc),
        ("hop", ingredients.hop_free),
        ("init_classical", init_classical),
        ("derivative_coupling_dzc", derivative_coupling_dzc),
        ("_init_model", _init_model),
    ]