"""
This module contains the Q-Chem ASE Model class.
"""

import numpy as np
import os
from copy import deepcopy
from ase.calculators.calculator import FileIOCalculator
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
        self.calculator = QCLabQChemCalculator
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
            calc.write_input(properties=["energy"])
            calc.execute()
            calc.read_results(properties=["energy"])
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
            z = kwargs["z"]
            num_classical_coordinates = self.constants.num_classical_coordinates
            num_quantum_states = self.constants.num_quantum_states
            m = self.constants.classical_coordinate_mass
            h = self.constants.classical_coordinate_weight
            mol = self.constants.ase_atoms_object
            qchem_args = self.constants.qchem_args
            qchem_tddft_args = self.constants.qchem_tddft_args
            q = z_to_q(
                z,
                m,
                h,
            )
            mol.set_positions(
                q.reshape((num_classical_coordinates // 3, 3)) / ANGSTROM_TO_BOHR
            )
            calc = self.calculator(
                atoms=mol,
                folder_scratch="qclab_job_" + str(self.constants.seed),
                **{**qchem_args, **qchem_tddft_args},
            )
            calc.label += "_energy"
            calc.label += "_" + str(self.constants.seed)
            calc.write_input(properties=["energy"])
            print("Calculating energies")
            calc.execute()
            calc.read_results(properties=["energy"])
            properties = calc.results
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

    @vectorize_ingredient
    def ab_initio_properties_calculator(self, parameters, **kwargs):
        property_dict = kwargs["property_dict"]
        traj_ind = kwargs["traj_ind"]
        properties = {}
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        mol = self.constants.ase_atoms_object
        qchem_args = self.constants.qchem_args
        qchem_tddft_args = self.constants.qchem_tddft_args
        for property in property_dict.keys():
            property_args = copy.deepcopy(property_dict[property])
            for arg_key in property_args.keys():
                if type(property_args[arg_key]) is np.ndarray:
                    property_args[arg_key] = property_args[arg_key][traj_ind]
            if property == "energy":
                print("apc energy")
                z = property_args["z"]
                q = z_to_q(z, m, h)
                mol.set_positions(
                    q.reshape((num_classical_coordinates // 3, 3)) / ANGSTROM_TO_BOHR
                )
                calc = self.calculator(
                    atoms=mol,
                    folder_scratch="qclab_job_" + str(self.constants.seed),
                    **{**qchem_args, **qchem_tddft_args},
                )
                calc.label += "_energy"
                calc.label += "_" + str(self.constants.seed)
                calc.write_input(
                    properties=["energy"],
                    excited_amplitudes=property_args["excited_amplitudes"],
                )
                calc.execute()
                calc.read_results(
                    properties=["energy"],
                    excited_amplitudes=property_args["excited_amplitudes"],
                )
                for key in calc.results.keys():
                    properties[key] = calc.results[key]
                # properties["energy"] = calc.results["energy"]
            if property == "gradient":
                print("apc gradient")
                properties["gradient"] = []
                state_inds_gradient = property_args["state_inds_gradient"]
                if isinstance(state_inds_gradient, (int, np.integer)):
                    state_inds_gradient = [state_inds_gradient]
                z = property_args["z"]
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
                    },
                )
                calc.label += "_" + "g"
                calc.label += "_" + str(self.constants.seed)
                mol.calc = calc
                mol.calc.write_input(
                    properties=["gradient"], state_inds_gradient=state_inds_gradient
                )
                mol.calc.execute()
                mol.calc.read_results(properties=["gradient"])
                for key in calc.results.keys():
                    properties[key] = calc.results[key]
                # properties["gradient"] = calc.results["gradient"]
            if property == "derivative_coupling":
                print("apc derivative_coupling")
                state_inds_derivative_couplings = property_args[
                    "state_inds_derivative_couplings"
                ]
                z = property_args["z"]
                q = z_to_q(z, m, h)
                mol.set_positions(
                    q.reshape((num_classical_coordinates // 3, 3)) / ANGSTROM_TO_BOHR
                )
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
                    state_inds_derivative_couplings=state_inds_derivative_couplings,
                )
                calc.execute()
                calc.read_results(properties=["derivative_coupling"])
                # properties["derivative_coupling"] = calc.results["derivative_coupling"]
                for key in calc.results.keys():
                    properties[key] = calc.results[key]
            if property == "wf_overlaps":
                print("apc wf_overlaps")
                z = property_args["z"]
                q = z_to_q(z, m, h)
                mol.set_positions(
                    q.reshape((num_classical_coordinates // 3, 3)) / ANGSTROM_TO_BOHR
                )
                calc = self.calculator(
                    atoms=mol,
                    folder_scratch="qclab_job_" + str(self.constants.seed),
                    **{
                        **qchem_args,
                        **qchem_tddft_args,
                    },
                )
                calc.label += "_" + "wf"
                calc.label += "_" + str(self.constants.seed)
                mol_previous = deepcopy(mol)
                z_previous = property_args["z_previous"]
                q_previous = z_to_q(z_previous, m, h)
                mol_previous.set_positions(
                    q_previous.reshape((num_classical_coordinates // 3, 3))
                    / ANGSTROM_TO_BOHR
                )
                calc.write_input(
                    properties=["wf_overlaps"],
                    atoms_previous=mol_previous,
                )
                calc.execute()
                # calc.num_excited_states = self.constants.num_quantum_states - 1
                print(
                    np.sum(
                        np.abs(property_args["current_amplitudes"]["x"]) ** 2,
                        axis=(1, 2),
                    )
                )
                print(
                    np.sum(
                        np.abs(property_args["current_amplitudes"]["x"]) ** 2,
                        axis=(1, 2),
                    )
                )
                calc.read_results(
                    properties=["wf_overlaps"],
                    current_amplitudes=property_args["current_amplitudes"],
                    previous_amplitudes=property_args["previous_amplitudes"],
                )
                # properties["wf_overlaps"] = calc.results["wf_overlaps"]
                for key in calc.results.keys():
                    properties[key] = calc.results[key]
                print("overlaps")
                print(calc.results["wf_overlaps"])
        return properties

    ingredients = [
        ("ab_initio_properties_calculator", ab_initio_properties_calculator),
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




class QCLabQChemCalculator(FileIOCalculator):
    """
    Q-Chem ASE calculator for QC Lab calculations.

    Based on the ASE Q-Chem calculator:
    https://wiki.fysik.dtu.dk/ase/ase/calculators/qchem
    """

    # name = "QChem"
    # implemented_properties = ["energy", "gradient", "derivative_coupling", "frequency"]
    _legacy_default_command = "qchem PREFIX.inp PREFIX.out"
    # default_parameters = {"jobtype": None}

    def __init__(
        self,
        atoms,
        label="qchem",
        folder_scratch="QCLab_job",
        nt=1,
        np=1,
        **kwargs,
    ):
        ignore_bad_restart_file = FileIOCalculator._deprecated
        restart = None
        ######
        ##Ensuring that all kwargs are lowercase
        kwargs = {str(k).lower(): v for k, v in kwargs.items()}
        #######

        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, **kwargs
        )
        ## Store objects globally
        self.atoms = atoms
        self.folder_scratch = folder_scratch
        # Build command
        self.command = "qchem "
        if np != 1:
            self.command += f"-np {np} "
        if nt != 1:
            self.command += f"-nt {nt} "
        self.command += "PREFIX.inp PREFIX.out"
        self.command += "\t" + folder_scratch
        self.command += " > /dev/null 2>&1"
        ###########
        ### implemented methods
        self.implemented_properties = [
            "energy",
            "gradient",
            "derivative_coupling",
            "frequency",
            "wf_overlaps",
        ]

    def read(self, label):
        raise NotImplementedError

    ##############################
    ###antonio modifications
    # ------------------------------------------------------------
    # property list -> job specs (for multi-job inputs)
    # ------------------------------------------------------------
    def _build_job_specs(self, properties):
        """
        Map ASE properties -> list of Q-Chem jobs.

        Each job spec is a dict with:
            name  : label (e.g. 'gradient', 'derivative_coupling', 'frequency', 'energy')
            jobtype : Q-Chem JOBTYPE ('FORCE', 'FREQ', 'SP')
            write_derivative_coupling : bool, whether to emit $derivative_coupling
        """
        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        props = set(properties)
        ####check if the asked properties are implemented
        self._check_requested_properties(properties)
        job_specs = []
        for prop in props:
            if "energy" in prop:
                job_specs.append(
                    {
                        "name": "energy",
                        "jobtype": "SP",
                        "write_derivative_coupling": False,
                    }
                )
            if "gradient" in prop:
                job_specs.append(
                    {
                        "name": "gradient",
                        "jobtype": "FORCE",
                        "write_derivative_coupling": False,
                    }
                )

            if "frequency" in prop:
                job_specs.append(
                    {
                        "name": "frequency",
                        "jobtype": "FREQ",
                        "write_derivative_coupling": False,
                    }
                )

            if "derivative_coupling" in prop:
                job_specs.append(
                    {
                        "name": "derivative_coupling",
                        "jobtype": "SP",
                        "write_derivative_coupling": True,
                    }
                )
            if "wf_overlaps" in prop:
                job_specs.append(
                    {
                        "name": "wf_overlaps",
                        "jobtype": "SP",
                        "write_derivative_coupling": False,
                    }
                )
        return job_specs

    ########
    ##This function checks if the requested properties are implemented
    def _check_requested_properties(self, job_resquested):
        for job in job_resquested:
            if job not in self.implemented_properties:
                raise ValueError(f"Requested property '{job}' is not implemented.")

    ########
    def _write_job(
        self,
        fileobj,
        job_spec,
        state_inds_gradient=None,
        state_inds_derivative_couplings=None,
        atoms_previous=None,
        excited_amplitudes=False,
    ):
        """
        Write  Q-Chem jobs (single job or multiple jobs that share same geometry).
        """
        for i, job in enumerate(job_spec):
            if "gradient" in job["name"]:
                # This determines the gradient job belongs multiple jobs or single one
                if len(job_spec) == 1:
                    flag = 0
                else:
                    if i == 0:
                        flag = 0
                    else:
                        flag = i + 1
                self._write_gradient_jobs(fileobj, job, flag, state_inds_gradient)

            elif "wf_overlaps" in job["name"]:
                if len(job_spec) == 1:
                    flag = 0
                else:
                    if i == 0:
                        flag = 0
                    else:
                        flag = i + 1
                if atoms_previous == None:
                    raise ValueError(
                        "previous geometry must be provided"
                        "when requesting wf_overlaps"
                    )
                self._write_wf_overlaps_jobs(fileobj, job, atoms_previous, flag)
            else:
                #######
                ## comment
                self._write_comments_qchem(fileobj, job["name"], indx=i)
                #####
                ## geomentry definition
                self._write_geometry_qchem(fileobj, indx=i)
                #####
                ## job definitnion
                self._write_job_defi_qchem(
                    fileobj, job, excited_amplitudes=excited_amplitudes
                )
                #####
                ###Derivative coupling
                if job["write_derivative_coupling"]:
                    self._write_derivative_coupling_qchem(
                        fileobj, state_inds_derivative_couplings
                    )

    #########
    ####This function handles the writing for gradient jobs requested
    def _write_gradient_jobs(self, fileobj, job_spec, flag=0, state_inds_gradient=None):
        if state_inds_gradient == None:
            n_s = int(self.parameters.get("cis_n_roots", 1)) + 1
            state_inds_gradient = [i for i in range(n_s)]
        elif isinstance(state_inds_gradient, int):
            state_inds_gradient = [i for i in range(state_inds_gradient)]

        for j, i in enumerate(state_inds_gradient):
            if i > 0:
                self.parameters["cis_state_deriv"] = str(i)
            if flag == 0:
                indx = j
            else:
                indx = flag
            #######
            ## comment
            self._write_comments_qchem(fileobj, f"gradient state {i}", indx)
            #####
            ## geomentry definition
            self._write_geometry_qchem(fileobj, indx)
            #####
            ## job definitnion
            self._write_job_defi_qchem(fileobj, job_spec, flag=i)

    #########
    ####This function handles the writing for wavefunction couplings jobs requested
    def _write_wf_overlaps_jobs(self, fileobj, job_spec, atoms_previous, flag=0):
        indx = flag
        ##creating the input for the previous geometry
        self._write_comments_qchem(fileobj, job_spec["name"] + " previous step", indx)
        ####write geometry
        fileobj.write("$molecule\n")
        if (
            "multiplicity" not in self.parameters
            or self.parameters["multiplicity"] is None
        ):
            tot_magmom = self.atoms.get_initial_magnetic_moments().sum()
            mult = int(tot_magmom) + 1
        else:
            mult = int(self.parameters["multiplicity"])
        charge = int(self.parameters.get("charge", 0))
        fileobj.write("   %d %d\n" % (charge, mult))

        for a in atoms_previous:
            fileobj.write("   {}  {:f}  {:f}  {:f}\n".format(a.symbol, a.x, a.y, a.z))
        fileobj.write("$end\n\n")
        #####
        ## job definitnion
        keys_for_wf_overlaps = ["mo_overlaps_two_geoms"]
        if all(k in self.parameters for k in keys_for_wf_overlaps):
            pass
        else:
            self.parameters["mo_overlaps_two_geoms"] = 1
        if self.parameters.get("mo_overlaps_two_geoms") != 1:
            self.parameters["mo_overlaps_two_geoms"] = 1
        #####
        self._write_job_defi_qchem(fileobj, job_spec)

        #####
        ###creating input for the current geometry
        self._write_comments_qchem(
            fileobj, job_spec["name"] + " current step", indx + 1
        )
        ####write geometry
        self._write_geometry_qchem(fileobj, indx=0)
        #####job definition
        self.parameters["mo_overlaps_two_geoms"] = 2
        self._write_job_defi_qchem(fileobj, job_spec)

    ########
    ## function that are used to write qchem input file
    ####Comments and job separator
    def _write_comments_qchem(self, fileobj, job, indx=0):
        if indx != 0:
            fileobj.write("@@@\n\n")
        fileobj.write("$comment\n")
        fileobj.write(f"QC Lab generated input file ({job} job)\n")
        fileobj.write("$end\n\n")

    #####Geometry definition
    def _write_geometry_qchem(self, fileobj, indx=0):
        if indx == 0:
            fileobj.write("$molecule\n")
            if (
                "multiplicity" not in self.parameters
                or self.parameters["multiplicity"] is None
            ):
                tot_magmom = self.atoms.get_initial_magnetic_moments().sum()
                mult = int(tot_magmom) + 1
            else:
                mult = int(self.parameters["multiplicity"])
            charge = int(self.parameters.get("charge", 0))
            fileobj.write("   %d %d\n" % (charge, mult))

            for a in self.atoms:
                fileobj.write(
                    "   {}  {:f}  {:f}  {:f}\n".format(a.symbol, a.x, a.y, a.z)
                )
            fileobj.write("$end\n\n")
        else:
            fileobj.write("$molecule\n")
            fileobj.write("read\n")
            fileobj.write("$end\n\n")

    ####Job definition
    def _write_job_defi_qchem(
        self, fileobj, job_spec, flag=1, excited_amplitudes=False
    ):
        fileobj.write("$rem\n")
        fileobj.write(f"   JOBTYPE           {job_spec['jobtype']}\n")
        ##Writing excited amplitudes if requested
        if job_spec["name"] == "energy" and excited_amplitudes:
            self.parameters["GUI"] = "2"
            self.parameters["IQMOL_FCHK"] = "TRUE"
        ###This is to handle the parameters for gradient job at the GS
        if (
            flag == 0
            and job_spec["name"] == "gradient"
            or job_spec["name"] == "wf_overlaps"
        ):
            for prm, val in self.parameters.items():
                if prm.lower() in [
                    "charge",
                    "multiplicity",
                    "jobtype",
                    "cis_state_deriv",
                    "cis_n_roots",
                    "cis_singlets",
                    "cis_triplets",
                ]:
                    continue
                if val is None:
                    continue
                if isinstance(val, str):
                    v_str = val.upper()
                else:
                    v_str = str(val)
                fileobj.write("   %-25s   %s\n" % (prm.upper(), v_str))
        else:
            for prm, val in self.parameters.items():
                if prm.lower() in ["charge", "multiplicity", "jobtype"]:
                    continue
                if val is None:
                    continue
                if isinstance(val, str):
                    v_str = val.upper()
                else:
                    v_str = str(val)
                fileobj.write("   %-25s   %s\n" % (prm.upper(), v_str))

        # Always ignore symmetry; otherwise, the coordinates will change the previously established origin.
        fileobj.write("   %-25s   %s\n" % ("SYM_IGNORE", "TRUE"))

        fileobj.write("$end\n\n")

    def _write_derivative_coupling_qchem(
        self, fileobj, state_inds_derivative_couplings=None
    ):
        if state_inds_derivative_couplings is None:
            n_s = int(self.parameters.get("cis_der_numstate", 1))
            state_inds_derivative_couplings = [i for i in range(n_s)]
            print(
                "state_inds_derivative_couplings has not been provided.\n"
                f"Derivative couplings for states 0 to {n_s-1} will be computed."
            )

        fileobj.write("$derivative_coupling\n")
        fileobj.write(f"{state_inds_derivative_couplings[0]} is the reference state\n")
        fileobj.write(" ".join(str(s) for s in state_inds_derivative_couplings) + "\n")
        fileobj.write("$end\n\n")

    ############################################################################################

    def read_results(
        self,
        properties=None,
        state_inds_gradient=None,
        excited_amplitudes=False,
        previous_amplitudes=None,
        current_amplitudes=None,
    ):
        ###This makes easy to ask for energy results
        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        #####
        num_atoms = len(self.atoms)
        filename = self.label + ".out"
        with open(filename, "r") as fileobj:
            file_content = fileobj.read()
        fileobj.close()
        #####check for qchem normal termination
        self._check_qchem_normal_termination(
            file_content, properties, state_inds_gradient
        )
        ######
        ###pulling data from the output file
        self._organize_data_qchem_file(
            file_content,
            properties,
            state_inds_gradient,
            num_atoms,
            excited_amplitudes=excited_amplitudes,
            previous_amplitudes=previous_amplitudes,
            current_amplitudes=current_amplitudes,
        )

    ############
    ###check for qchem normal termination
    def _check_qchem_normal_termination(
        self, file_content, properties, state_inds_gradient
    ):
        ###checking qchem normal termination
        normal_termination = "for using Q-Chem"
        if "gradient" in properties:
            if state_inds_gradient is None:
                n_s = int(self.parameters.get("cis_n_roots", 1)) + len(properties)
            elif isinstance(state_inds_gradient, int):
                n_s = state_inds_gradient + len(properties) - 1
            else:
                n_s = len(state_inds_gradient) + len(properties) - 1
            if file_content.count(normal_termination) < n_s:
                raise ValueError("Q-Chem did not terminate normally.")
        elif "wf_overlaps" in properties:
            n_s = len(properties) + 1
            if file_content.count(normal_termination) < n_s:
                raise ValueError("Q-Chem did not terminate normally.")
        else:
            if file_content.count(normal_termination) < len(properties):
                raise ValueError("Q-Chem did not terminate normally.")

    ############
    ###organizing the data form qchem output file to extract the needed properties
    def _organize_data_qchem_file(
        self,
        file_content,
        properties,
        state_inds_gradient,
        num_atoms,
        excited_amplitudes=False,
        previous_amplitudes=None,
        current_amplitudes=None,
    ):
        normal_termination = "for using Q-Chem"
        for job in properties:
            if "gradient" in job:
                job_comment = f"QC Lab generated input file (gradient state"
                temp = file_content.split(job_comment)
                if state_inds_gradient is None:
                    n = int(self.parameters.get("cis_n_roots", 1)) + 1
                elif isinstance(state_inds_gradient, int):
                    n = state_inds_gradient
                else:
                    n = len(state_inds_gradient)
                gradient_files = temp[1:n]
                gradient_files.append(temp[n].split(normal_termination)[0])
                self._pull_data(
                    job,
                    gradient_files,
                    num_atoms,
                    state_inds_gradient=state_inds_gradient,
                )
            elif "wf_overlaps" in job:
                job_comment = f"QC Lab generated input file (wf_overlaps"
                temp = file_content.split(job_comment)
                coupling_files = temp[2].split(normal_termination)[0]
                self._pull_data(
                    job,
                    coupling_files.splitlines(),
                    num_atoms,
                    previous_amplitudes=previous_amplitudes,
                    current_amplitudes=current_amplitudes,
                )
            else:
                job_comment = f"QC Lab generated input file ({job}"
                temp = file_content.split(job_comment)
                job_file = temp[1].split(normal_termination)[0]
                self._pull_data(
                    job,
                    job_file.splitlines(),
                    num_atoms,
                    excited_amplitudes=excited_amplitudes,
                )

    ####pulling the data that is needed
    def _pull_data(
        self,
        property,
        input_file,
        num_atoms,
        state_inds_gradient=None,
        excited_amplitudes=False,
        previous_amplitudes=None,
        current_amplitudes=None,
    ):
        if "energy" in property:
            self._pull_exst_energy_qchem(
                input_file, excited_amplitudes=excited_amplitudes
            )
        elif "gradient" in property:
            self._pull_gradient_results_qchem(
                input_file, num_atoms, state_inds_gradient
            )
        elif "frequency" in property:
            self._pull_vibration_results_qchem(input_file, num_atoms)
        elif "derivative_coupling" in property:
            self._pull_derivative_coupling_results_qchem(input_file, num_atoms)
        elif "wf_overlaps" in property:
            self._pull_overlaps_qchem(
                previous_amplitudes=previous_amplitudes,
                current_amplitudes=current_amplitudes,
            )
        else:
            raise ValueError("This type of calculation has not been implemented yet")

    ####Gathering results from vibrational calculations
    ####the input_file is a list of lines that contains the qchem ouputfile
    def _pull_vibration_results_qchem(self, input_file, num_atoms):

        ###Get the position of the modes and frequencies
        indx_modes = []
        for i, line in enumerate(input_file):
            if "Mode:" in line:
                indx_modes.append(i)

        # frequencies
        freqs = []
        for i in indx_modes:
            freqs.append(input_file[i + 1].split()[1:])
        freqs = np.array(freqs, dtype=float).flatten()

        # modes/displacements
        modes = np.zeros((len(freqs), num_atoms, 3))
        ii = 0
        nlines = 8  # Number of lines between the Mode: line and the displacement-matrix line in a Q-Chem output file.
        for i in indx_modes:
            for j in range(num_atoms):
                temp = input_file[i + nlines + j].split()[1:]
                ss = int(len(temp) / 3)
                for jj in range(ss):
                    modes[ii + jj, j, :] = temp[jj * 3 : jj * 3 + 3]
            ii += ss

        self.results["frequency"] = freqs
        self.results["normal_mode"] = modes

    #####Gathering results from excited state energy calculations
    ####the input_file is a list of lines that contains the qchem ouputfile
    def _pull_exst_energy_qchem(self, input_file, excited_amplitudes=False):
        ####determining the number of states
        if self.parameters.get("cis_n_roots") is None:
            nt_states = 1
        else:
            nt_states = int(self.parameters.get("cis_n_roots")) + 1

        energy = np.zeros(nt_states)
        i = 0
        gs_found = False
        for j, line in enumerate(input_file):
            ##Energy GS
            if "SCF time:" in line:
                print(line)
                energy[i] = float(input_file[j + 2].split()[-1])
                i += 1
                gs_found = True
            elif "Timing for Total SCF" in line and not(gs_found):
                try:
                    energy[i] = float(input_file[j+2].split()[-1])
                    i += 1
                    gs_found = True
                except:
                    pass
            elif "Total energy in the final basis set" in line and not(gs_found):
                energy[i] = float(line.split()[-1])
                i += 1
                gs_found = True

            ##Energy Excited State
            elif "Total energy for state" in line:
                print(line)
                energy[i] = float(line.split()[-2])
                i += 1
        self.results["energy"] = energy
        ###extracting excited state amplitudes if requested
        if excited_amplitudes:
            self.results["excited_state_amplitudes"] = {}
            (
                X_alpha,
                Y_alpha,
                num_basis_functions,
                num_alpha_electrons,
                num_excited_states,
            ) = self._pull_excited_state_amplitudes_qchem()
            self.results["excited_state_amplitudes"]["x"] = X_alpha
            self.results["excited_state_amplitudes"]["y"] = Y_alpha
            self.results["excited_state_amplitudes"][
                "num_basis_functions"
            ] = num_basis_functions
            self.results["excited_state_amplitudes"][
                "num_alpha_electrons"
            ] = num_alpha_electrons
            self.results["excited_state_amplitudes"][
                "num_excited_states"
            ] = num_excited_states

    #####Gathering results from force/gradfient calculations
    ####the input_file is a list of lines that contains the qchem ouputfile
    def _pull_gradient_results_qchem(
        self, input_file, num_atoms, state_inds_gradient=None
    ):

        if state_inds_gradient is None:
            n_s = int(self.parameters.get("cis_n_roots", 1)) + 1
            state_inds_gradient = [i for i in range(n_s)]
        elif isinstance(state_inds_gradient, int):
            state_inds_gradient = [i for i in range(state_inds_gradient)]

        gradient = np.zeros((num_atoms, 3, len(state_inds_gradient)))
        flag_words_grad = ["Gradient of the state energy", "Gradient of SCF Energy"]
        flag_words_out_grad = ["Gradient time", "Max gradient component"]
        for state_indx in range(len(state_inds_gradient)):
            output_data = input_file[state_indx].splitlines()
            for indx, line in enumerate(output_data):
                if any(word in line for word in flag_words_grad):
                    ii = indx + 1
                    line2 = output_data[ii]
                    while not any(word in line2 for word in flag_words_out_grad):
                        jndx = np.array(line2.split(), dtype=int) - 1
                        gradient[jndx, 0, state_indx] = np.array(
                            output_data[ii + 1].split()[1:], dtype=float
                        )
                        gradient[jndx, 1, state_indx] = np.array(
                            output_data[ii + 2].split()[1:], dtype=float
                        )
                        gradient[jndx, 2, state_indx] = np.array(
                            output_data[ii + 3].split()[1:], dtype=float
                        )
                        ii += 4
                        line2 = output_data[ii]
                    break
        self.results["gradient"] = gradient

    #####Gathering results from Derivative Coupling calculations

    def _pull_derivative_coupling_results_qchem(self, input_file, num_atoms, ETF=True):
        ##Locating the values of interest
        l_states = []  ##Line that identifies the states involved in Derivative_coupling
        l_ETF = (
            []
        )  ## Line that identifies the location of Derivative_coupling's values with ETF corrections
        l_noETF = (
            []
        )  ## Line that identifies the location of Derivative_coupling's values without ETF corrections
        for i, line in enumerate(input_file):
            if "between states" in line:
                l_states.append(i)
            elif "with ETF" in line:
                l_ETF.append(i)
            elif "without ETF" in line:
                l_noETF.append(i)

        # gathering the data
        DC = np.zeros((num_atoms, 3))  ## Derivative coupling matrix
        derivative_coupling = (
            {}
        )  ### dictionary where will be store Derivative coupling matrix
        for i, sl in enumerate(l_states):
            temp = input_file[sl].split()
            i_st = int(temp[-3])  ## stetes involved
            j_st = int(temp[-1])  ## in Derivative_coupling

            base = l_ETF[i] if ETF else l_noETF[i]  ### choosing the NAC matrix

            for j in range(num_atoms):
                DC[j, :] = np.array(input_file[base + 3 + j].split()[1:], dtype=float)

            derivative_coupling[(i_st, j_st)] = DC.copy()
        self.results["derivative_coupling"] = derivative_coupling

    #####Gathering results from Wavefunction overlaps calculations
    def _pull_overlaps_qchem(self, previous_amplitudes=None, current_amplitudes=None):
        if previous_amplitudes is None:
            raise ValueError(
                "previous excited state amplitudes must be provided"
                "to compute wavefunction overlaps."
            )
        if current_amplitudes is None:
            raise ValueError(
                "current excited state amplitudes must be provided"
                "to compute wavefunction overlaps."
            )
        #####creating global variables needed for computing overlaps
        self.num_basis_functions = current_amplitudes["num_basis_functions"]
        self.num_alpha_electrons = current_amplitudes["num_alpha_electrons"]
        self.num_excited_states = current_amplitudes["num_excited_states"]
        ###MO overlaps
        MO_overlaps = self._pull_mo_overlaps_qchem()
        ####computing overlaps matrices
        ###### TDDFT
        self._get_overlaps_TDDFT(
            previous_amplitudes["x"],
            previous_amplitudes["y"],
            current_amplitudes["x"],
            current_amplitudes["y"],
            MO_overlaps,
        )

    ################
    #####overlaps with TDDFT
    def _get_overlaps_TDDFT(self, x_prev, y_prev, x_curr, y_curr, MO_overlaps):
        ######
        ##overlaps matrices
        S_oo = MO_overlaps[0 : self.num_alpha_electrons, 0 : self.num_alpha_electrons]
        S_ov = MO_overlaps[0 : self.num_alpha_electrons, self.num_alpha_electrons :]
        S_vo = MO_overlaps[self.num_alpha_electrons :, 0 : self.num_alpha_electrons]
        S_vv = MO_overlaps[self.num_alpha_electrons :, self.num_alpha_electrons :]
        ###computing <GS^(1)|GS^(2)>
        GS_overlap = np.linalg.det(S_oo)
        ###computing <GS^(1)|ES_i^(2)>
        overlaps_gs_ex_x = self._compute_overlap_gs_ex(x_curr, S_oo, S_ov)
        overlaps_gs_ex_y = self._compute_overlap_gs_ex(y_curr, S_oo, S_ov)
        overlaps_gs_ex = overlaps_gs_ex_x - overlaps_gs_ex_y
        ###computing <ES_i^(1)|GS^(2)>
        overlaps_ex_gs_x = self._compute_overlap_ex_gs(x_prev, S_oo, S_vo)
        overlaps_ex_gs_y = self._compute_overlap_ex_gs(y_prev, S_oo, S_vo)
        overlaps_ex_gs = overlaps_ex_gs_x - overlaps_ex_gs_y
        ###computing <ES_i^(1)|ES_j^(2)>
        overlaps_ex_ex_x_x = self._compute_overlap_ex_ex(
            x_prev, x_curr, S_oo, S_ov, S_vo, S_vv
        )
        overlaps_ex_ex_x_y = self._compute_overlap_ex_ex(
            x_prev, y_curr, S_oo, S_ov, S_vo, S_vv
        )
        overlaps_ex_ex_y_x = self._compute_overlap_ex_ex(
            y_prev, x_curr, S_oo, S_ov, S_vo, S_vv
        )
        overlaps_ex_ex_y_y = self._compute_overlap_ex_ex(
            y_prev, y_curr, S_oo, S_ov, S_vo, S_vv
        )
        overlaps_ex_ex = (
            overlaps_ex_ex_x_x
            - overlaps_ex_ex_x_y
            - overlaps_ex_ex_y_x
            + overlaps_ex_ex_y_y
        )
        ###storing results
        ###storing results
        self.results["wf_overlaps"] = np.zeros(
            (self.num_excited_states + 1, self.num_excited_states + 1)
        )
        self.results["wf_overlaps"][0, 0] = GS_overlap
        self.results["wf_overlaps"][0, 1:] = 2.0 * overlaps_gs_ex
        self.results["wf_overlaps"][1:, 0] = 2.0 * overlaps_ex_gs
        self.results["wf_overlaps"][1:, 1:] = 2.0 * overlaps_ex_ex

    ####computing overlaps
    def _compute_overlap_gs_ex(self, x, S_oo, S_ov):
        A = S_oo
        Sov = S_ov
        sign, logdet = np.linalg.slogdet(A)
        detA = sign * np.exp(logdet)
        Ww = np.linalg.solve(A, Sov)
        Xx = np.asarray(x)
        overlaps_gs_ex = detA * np.einsum("eia,ia->e", Xx, Ww)
        return overlaps_gs_ex

    def _compute_overlap_ex_gs(self, x, S_oo, S_vo):
        A = S_oo
        Svo = S_vo
        sign, logdet = np.linalg.slogdet(A)
        detA = sign * np.exp(logdet)
        Vv = np.linalg.solve(A.T, Svo.T).T
        Xx = np.asarray(x)
        overlaps_ex_gs = detA * np.einsum("eia,ai->e", Xx, Vv)
        return overlaps_ex_gs

    def _compute_overlap_ex_ex(self, x, x2, S_oo, S_ov, S_vo, S_vv):
        A = S_oo
        nocc = A.shape[0]
        nb = S_vo.shape[0]
        na = S_ov.shape[1]
        X = np.asarray(x)
        X2 = np.asarray(x2)
        sign, logdet = np.linalg.slogdet(A)
        detA = sign * np.exp(logdet)

        invA = np.linalg.solve(A, np.eye(nocc))
        G = invA @ S_ov

        overlap = np.zeros((X.shape[0], X2.shape[0]))

        for j in range(nocc):
            Aj = A[j, :]
            colj = invA[:, j]

            Delta = S_vo - Aj[None, :]

            Delta_dot_colj = Delta @ colj
            Delta_dot_G = Delta @ G

            for i in range(nocc):
                q = invA[i, j]
                Gi = G[i, :]
                Aji = A[j, i]

                Delta_i = Delta[:, i]
                alpha = S_vv - S_vo[:, i][:, None] - S_ov[j, :][None, :] + Aji
                B21 = (Delta_dot_G - Delta_i[:, None]) + alpha * (Gi[None, :] - 1.0)
                B22 = 1.0 + Delta_dot_colj[:, None] + alpha * q
                detM = detA * (Gi[None, :] * B22 - q * B21)
                conjx = np.conj(X[:, j, :])
                x2ia = X2[:, i, :]
                overlap += np.einsum("eb,fa,ba->ef", conjx, x2ia, detM, optimize=True)

        return overlap

    ####This function extracts MO overlaps from qchem output file
    def _pull_mo_overlaps_qchem(self):
        qcscratch = os.environ["QCSCRATCH"]  ##Qchem scratch folder
        file_mo_overlap = (
            qcscratch
            + "/"
            + self.folder_scratch
            + "/"
            + "MO-overlaps/MO-overlap-TwoGeoms.txt"
        )
        with open(file_mo_overlap, "r") as file:
            data = file.readlines()
        overlaps_init = []
        for line in data:
            temp_values = np.array(line.split(), dtype=float)
            if len(temp_values) == 8:
                overlaps_init.append(temp_values[:-1])
            elif len(temp_values) == 0:
                continue
            else:
                overlaps_init.append(temp_values)

        length = len(np.concatenate(overlaps_init))
        if length != (self.num_basis_functions * self.num_basis_functions):
            raise ValueError("The number of overlap integrals read is not correct")

        MO_overlaps = np.zeros((self.num_basis_functions, self.num_basis_functions))
        num_blocks = int(len(overlaps_init) / self.num_basis_functions)
        for i in range(self.num_basis_functions):
            temp_values = np.concatenate(
                [
                    overlaps_init[self.num_basis_functions * j + i]
                    for j in range(num_blocks)
                ]
            )
            MO_overlaps[i, :] = temp_values

        MO_overlaps = MO_overlaps.T
        return MO_overlaps

    ####extracting excited state amplitudes
    def _pull_excited_state_amplitudes_qchem(self):
        file_fchk = self.label + ".fchk"
        with open(file_fchk, "r") as f:
            data = f.readlines()
        keywords_to_find = [
            "Number of alpha electrons",
            "Number of beta electrons",
            "Number of basis functions",
            "Number of Excited States",
            "Alpha X Amplitudes",
            "Alpha Y Amplitudes",
            "Alpha Amplitudes",
        ]
        keyword_to_stop = ["Beta X Amplitudes", "Beta Y Amplitudes", "Beta Amplitudes"]

        for i, line in enumerate(data):
            if keywords_to_find[0] in line:
                num_alpha_electrons = int(line.split()[-1])
            if keywords_to_find[1] in line:
                num_beta_electrons = int(line.split()[-1])
            if keywords_to_find[2] in line:
                num_basis_functions = int(line.split()[-1])
            if keywords_to_find[3] in line:
                num_excited_states = int(line.split()[-1])
                index_restart = i

        if num_alpha_electrons != num_beta_electrons:
            raise ValueError("The software only supports closed-shell calculations.")
        dimension_amplitudes = (
            num_excited_states
            * (num_basis_functions - num_alpha_electrons)
            * num_alpha_electrons
        )
        X_alpha = []
        Y_alpha = []
        for i, line in enumerate(data[index_restart:]):
            if keywords_to_find[4] in line:
                j = i + 1 + index_restart
                while keyword_to_stop[0] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    X_alpha.append(temp_values)
                    j = j + 1
            if keywords_to_find[5] in line:
                j = i + 1 + index_restart
                while keyword_to_stop[1] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    Y_alpha.append(temp_values)
                    j = j + 1
            if keywords_to_find[6] in line:
                j = i + 1 + index_restart
                while keyword_to_stop[2] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    X_alpha.append(temp_values)
                    Y_alpha.append(np.zeros_like(temp_values))
                    j = j + 1
        X_alpha = np.concatenate(X_alpha)
        X_alpha = np.reshape(
            X_alpha,
            (
                num_excited_states,
                num_alpha_electrons,
                (num_basis_functions - num_alpha_electrons),
            ),
        )
        Y_alpha = np.concatenate(Y_alpha)
        Y_alpha = np.reshape(
            Y_alpha,
            (
                num_excited_states,
                num_alpha_electrons,
                (num_basis_functions - num_alpha_electrons),
            ),
        )
        return (
            X_alpha,
            Y_alpha,
            num_basis_functions,
            num_alpha_electrons,
            num_excited_states,
        )

    ##############################

    def write_input(
        self,
        properties=None,
        system_changes=None,
        state_inds_gradient=None,
        state_inds_derivative_couplings=None,
        atoms_previous=None,
        excited_amplitudes=False,
    ):
        FileIOCalculator.write_input(self, self.atoms, properties, system_changes)
        filename = self.label + ".inp"
        job_specs = self._build_job_specs(properties)
        with open(filename, "w") as fileobj:
            self._write_job(
                fileobj,
                job_specs,
                state_inds_gradient=state_inds_gradient,
                state_inds_derivative_couplings=state_inds_derivative_couplings,
                atoms_previous=atoms_previous,
                excited_amplitudes=excited_amplitudes,
            )
