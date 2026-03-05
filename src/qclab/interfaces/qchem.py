"""
This module contains the Q-Chem interface for ab initio calculations.
"""

import os
import numpy as np
from ase.calculators.calculator import FileIOCalculator
from ase import Atoms
from qclab import numerical_constants


class QCLabQChemInterface(FileIOCalculator):
    """
    Q-Chem ASE interface for QC Lab calculations.

    Based on the ASE Q-Chem calculator:
    https://wiki.fysik.dtu.dk/ase/ase/calculators/qchem
    """

    _legacy_default_command = "qchem PREFIX.inp PREFIX.out"

    def __init__(
        self,
        atom_positions,  # In units of Bohr.
        atom_masses,  # In units of electron mass.
        atom_names,
        method_ex="tddft",
        label="qchem",
        folder_scratch="qclab_job",
        nt=1,
        np=1,
        **kwargs,
    ):
        ignore_bad_restart_file = FileIOCalculator._deprecated
        restart = None
        # Set default parameters.
        if kwargs.get("method", None) is None:
            kwargs["method"] = "B3LYP"
            kwargs["exchange"] = "B3LYP"
            self.method_ex = "tddft"
        if kwargs.get("basis", None) is None:
            kwargs["basis"] = "6-31G*"
        # Ensuring that all kwargs are lowercase.
        kwargs = {str(k).lower(): v for k, v in kwargs.items()}
        atoms = Atoms(
            symbols=atom_names,
            positions=atom_positions / numerical_constants.ANGSTROM_TO_BOHR,
        )
        atoms.set_masses(atom_masses / numerical_constants.AMU_TO_EMASS)
        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, **kwargs
        )
        # Store objects globally.
        self.atom_names = atom_names
        self.atom_masses = atom_masses
        self.atom_positions = atom_positions
        self.atoms = atoms
        self.folder_scratch = folder_scratch
        self.method_ex = method_ex
        # Build command.
        self.command = "qchem"
        if np != 1:
            self.command += f" -np {np}"
        if nt != 1:
            self.command += f" -nt {nt}"
        self.command += " PREFIX.inp PREFIX.out"
        self.command += "\t" + folder_scratch
        self.command += " > /dev/null 2>&1"
        # Properties implemented.
        self.implemented_properties = [
            "energy",
            "gradient",
            "derivative_coupling",
            "frequency",
            "wf_overlaps",
        ]
        # Job specs
        self.job_templates = {
                        "energy": {"name": "energy", "jobtype": "SP", "write_derivative_coupling": False},
                        "gradient": {"name": "gradient", "jobtype": "FORCE", "write_derivative_coupling": False},
                        "frequency": {"name": "frequency", "jobtype": "FREQ", "write_derivative_coupling": False},
                        "derivative_coupling": {"name": "derivative_coupling", "jobtype": "SP", "write_derivative_coupling": True},
                        "wf_overlaps": {"name": "wf_overlaps", "jobtype": "SP", "write_derivative_coupling": False},
                    }

    def read(self, label):           
        raise NotImplementedError

    def _build_job_specs(self, properties):
        """
            Each job spec is a dict with:
            name  : label (e.g. 'gradient', 'derivative_coupling', 'frequency', 'energy')
            jobtype : Q-Chem JOBTYPE ('FORCE', 'FREQ', 'SP')
            write_derivative_coupling : bool, whether to emit $derivative_coupling
        """
        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        # Check if the asked properties are implemented.
        self._check_requested_properties(properties)
        properties = list(dict.fromkeys(properties))
        return [self.job_templates[p].copy() for p in properties]
    
    def _check_requested_properties(self, job_requested):
        """
        Checks if the requested properties are implemented.
        """
        for job in job_requested:
            if job not in self.implemented_properties:
                raise ValueError(f"Requested property '{job}' is not implemented.")

    def _compute_flag(self,num_jobs,indx):
        """
        gradient/wf_overlaps: flag=0 for single-job input or first job,
        otherwise offset by +1
        """
        return 0 if (num_jobs == 1 or indx == 0) else (indx + 1)

    def _write_job(self, file_obj, job_spec, **kwargs):
        """
        Write  Q-Chem jobs (single job or multiple jobs that share same geometry).
        """
        num_jobs = len(job_spec)
        for i, job in enumerate(job_spec):
            flag = self._compute_flag(num_jobs, i)
            if "gradient" in job["name"]:
                self._write_gradient_jobs(
                    file_obj,
                    job,
                    flag,
                    kwargs["gradient"].get("state_inds_gradient", None),
                )
                continue
            elif "wf_overlaps" in job["name"]:
                if kwargs["wf_overlaps"]["atom_positions_previous"] is None:
                    raise ValueError(
                        "Previous geometry must be provided"
                        "when requesting wf_overlaps."
                    )
                atoms_previous = Atoms(
                    symbols=self.atom_names,
                    positions=kwargs["wf_overlaps"]["atom_positions_previous"]
                    / numerical_constants.ANGSTROM_TO_BOHR,
                )
                atoms_previous.set_masses(
                    self.atom_masses / numerical_constants.AMU_TO_EMASS
                )
                self._write_wf_overlaps_jobs(file_obj, job, atoms_previous, flag)
                continue
            else:
                self._write_comments(file_obj, job["name"], ind=i)
                self._write_molecule_section(file_obj, ind=i)
                excited_amplitudes_flag=(
                    kwargs["energy"]["excited_amplitudes"] if "energy" in job["name"] else False
                )
                self._write_job_defi(
                    file_obj,
                    job,
                    excited_amplitudes=excited_amplitudes_flag,
                    state_inds_derivative_coupling=kwargs.get(
                        "derivative_coupling", {}
                    ).get("state_inds_derivative_coupling", None),
                )
                if job["write_derivative_coupling"]:
                    self._write_derivative_coupling(
                        file_obj,
                        kwargs.get("derivative_coupling", {}).get(
                            "state_inds_derivative_coupling", None
                        ),
                    )

    def _get_state_inds_gradient(self, state_inds_gradient):
        if state_inds_gradient is None:
            num_states = int(self.parameters.get("cis_n_roots", 1)) + 1
            return [i for i in range(num_states)]
        elif isinstance(state_inds_gradient, (int, np.integer)):
            return [state_inds_gradient]
        else:
            return state_inds_gradient
        
    def _write_gradient_jobs(
        self, file_obj, job_spec, flag=0, state_inds_gradient=None
    ):
        """
        Handles the writing for gradient jobs requested.
        """
        state_inds_gradient = self._get_state_inds_gradient(state_inds_gradient)
        for j, i in enumerate(state_inds_gradient):
            if i > 0:
                self.parameters["cis_state_deriv"] = str(i)
            if flag == 0:
                ind = j
            else:
                ind = flag
            self._write_comments(file_obj, f"gradient state {i}", ind)
            self._write_molecule_section(file_obj, ind)
            self._write_job_defi(file_obj, job_spec, flag=i)

    def _write_wf_overlaps_jobs(self, file_obj, job_spec, atoms_previous, flag=0):
        """
        Handles the writing for wavefunction overlaps jobs requested.
        """
        ind = flag
        # Create the input for the previous geometry.
        self._write_comments(file_obj, job_spec["name"] + " previous step", ind)
        # Write geometry.
        file_obj.write("$molecule\n")
        charge, mult = self._get_charge_and_multiplicity()
        file_obj.write("   %d %d\n" % (charge, mult))
        self._write_geometry(file_obj,atoms_previous)
        file_obj.write("$end\n\n")
        # Job definition.
        keys_for_wf_overlaps = ["mo_overlaps_two_geoms"]
        if all(key in self.parameters for key in keys_for_wf_overlaps):
            pass
        else:
            self.parameters["mo_overlaps_two_geoms"] = 1
        if self.parameters.get("mo_overlaps_two_geoms") != 1:
            self.parameters["mo_overlaps_two_geoms"] = 1
        self._write_job_defi(file_obj, job_spec)

        # Creating input for the current geometry.
        self._write_comments(
            file_obj, job_spec["name"] + " current step", ind + 1
        )
        self._write_molecule_section(file_obj, ind=0)
        self.parameters["mo_overlaps_two_geoms"] = 2
        self._write_job_defi(file_obj, job_spec)

    def _write_comments(self, file_obj, job, ind=0):
        """
        Write comments and job separator for Q-Chem input file.
        """
        if ind != 0:
            file_obj.write("@@@\n\n")
        file_obj.write("$comment\n")
        file_obj.write(f"QC Lab generated input file ({job} job)\n")
        file_obj.write("$end\n\n")

    def _get_charge_and_multiplicity(self):
        charge=self.parameters.get("charge",None)
        multiplicity=self.parameters.get("multiplicity", None)
        if multiplicity is None or charge is None:
            raise ValueError(
                "Charge and multiplicity must be provided."
            )
        else:
            multiplicity = int(multiplicity)
            charge = int(charge)
        return charge, multiplicity
    
    def _write_geometry(self, file_obj,geometry):
        for a in geometry:
            file_obj.write(
            "   {}  {:f}  {:f}  {:f}\n".format(a.symbol, a.x, a.y, a.z)
                )
    
    def _write_molecule_section(self, file_obj, ind=0):
        file_obj.write("$molecule\n")
        if ind == 0:
            charge, multiplicity = self._get_charge_and_multiplicity()
            file_obj.write("   %d %d\n" % (charge, multiplicity))
            self._write_geometry(file_obj,self.atoms)
        else:
            file_obj.write("   read\n")
        file_obj.write("$end\n\n")

    def _write_job_defi(
        self,
        file_obj,
        job_spec,
        flag=1,
        excited_amplitudes=False,
        state_inds_derivative_coupling=None,
    ):
        file_obj.write("$rem\n")
        file_obj.write(f"   JOBTYPE           {job_spec['jobtype']}\n")
        parameters = self._get_job_specific_parameters(
            job=job_spec["name"],
            flag=flag,
            excited_amplitudes=excited_amplitudes,
            state_inds_derivative_coupling=state_inds_derivative_coupling,
        )
        for parameter, v_str in parameters.items():
            file_obj.write("   %-25s   %s\n" % (parameter.upper(), v_str))
            # Always ignore symmetry; otherwise, the coordinates will change the previously established origin.
        file_obj.write("   %-25s   %s\n" % ("SYM_IGNORE", "TRUE"))
        file_obj.write("$end\n\n")

    def _get_avoid_keywords_for_job(self, **kwargs):
        keywords_to_avoid={
            "energy":[  "calc_nac", "cis_der_numstate", "cis_state_deriv",
                "mo_overlaps_two_geoms", "charge", "multiplicity",
                "jobtype"],
            "gradient":{ "0":[ "calc_nac", "cis_der_numstate", "mo_overlaps_two_geoms",
                               "charge", "multiplicity", "jobtype",
                               "cis_state_deriv", "cis_n_roots", "cis_singlets",
                               "cis_triplets", "internal_stability", "gui",
                               "iqmol_fchk",
                            ], 
                         "1":[ "calc_nac", "cis_der_numstate", "mo_overlaps_two_geoms",
                               "charge", "multiplicity", "jobtype",
                               "internal_stability", "gui", "iqmol_fchk",
                             ]
                        },
            "derivative_coupling":[ "charge", "multiplicity", "jobtype",
                                    "internal_stability", "mo_overlaps_two_geoms", "gui",
                                    "iqmol_fchk",
                                  ],
            "wf_overlaps":[ "calc_nac", "cis_der_numstate", "charge",
                            "multiplicity", "jobtype", "cis_state_deriv",
                            "cis_n_roots", "cis_singlets", "cis_triplets",
                            "internal_stability", "gui", "iqmol_fchk",
                          ],
            "frequency":[ "calc_nac", "cis_der_numstate", "mo_overlaps_two_geoms",
                          "charge", "multiplicity", "jobtype", 
                          "gui", "iqmol_fchk",
                        ],
            }
        
        if kwargs["job"]=="gradient":
            if kwargs["flag"] != 0:
                kwargs["flag"] = 1  
            return keywords_to_avoid["gradient"][str(kwargs["flag"])]
        else:
            return keywords_to_avoid[kwargs["job"]]

    def _get_job_specific_parameters(self, **kwargs):
        parameters = {}
        keywords_to_avoid = self._get_avoid_keywords_for_job(**kwargs)
        if kwargs["job"] == "energy" and kwargs["excited_amplitudes"]:
            self.parameters["GUI"] = "2"
            self.parameters["IQMOL_FCHK"] = "TRUE"
        elif kwargs["job"] == "derivative_coupling":
            self.parameters["calc_nac"] = True
            if kwargs["state_inds_derivative_coupling"] is None:
                self.parameters["cis_der_numstate"] = (
                    int(self.parameters["cis_n_roots"]) + 1
                )
        for param, value in self.parameters.items():
            if param.lower() in keywords_to_avoid:
                continue
            if value is None:
                continue
            if isinstance(value, str):
                v_str = value.upper()
            else:
                v_str = str(value)
            parameters[param] = v_str
        return parameters

    def _write_derivative_coupling(
        self, file_obj, state_inds_derivative_coupling=None
    ):
        if state_inds_derivative_coupling is None:
            n_s = int(self.parameters.get("cis_der_numstate", 1))
            state_inds_derivative_coupling = [i for i in range(n_s)]
        elif isinstance(state_inds_derivative_coupling, (int, np.integer)):
            state_inds_derivative_coupling = [
                i for i in range(state_inds_derivative_coupling + 1)
            ]
        file_obj.write("$derivative_coupling\n")
        file_obj.write(f"{state_inds_derivative_coupling[0]} is the reference state\n")
        file_obj.write(" ".join(str(s) for s in state_inds_derivative_coupling) + "\n")
        file_obj.write("$end\n\n")

    def read_results(self, **kwargs):
        properties = kwargs.keys()
        if properties is None:
            properties = ["energy"]
        properties = [prop.lower() for prop in properties]
        num_atoms = len(self.atoms)
        filename = self.label + ".out"
        with open(filename, "r") as file_obj:
            file_content = file_obj.read()
        state_inds_gradient = self._get_state_inds_gradient(
                              kwargs.get("gradient", {}).get("state_inds_gradient", None)
                            )
 
        self._check_normal_termination(
            file_content, properties, state_inds_gradient
        )
        self._organize_data_file(file_content, properties, num_atoms, **kwargs)

    def _get_number_of_jobs(self, properties, state_inds_gradient):
        num_jobs=len(properties)
        if "gradient" in properties:
            num_jobs+=len(state_inds_gradient)-1
        if "wf_overlaps" in properties:
            num_jobs += 1
        return num_jobs
    
    def _check_normal_termination(
        self, file_content, properties, state_inds_gradient
    ):
        """
        Check Q-Chem normal termination.
        """
        normal_termination = "for using Q-Chem"
        num_jobs=self._get_number_of_jobs(properties, state_inds_gradient)
        if file_content.count(normal_termination) != num_jobs:
            raise ValueError("Q-Chem did not terminate normally.")
        
    def _split_file_by_job(self, file_content, properties, **kwargs):
        state_inds_gradient = self._get_state_inds_gradient(
                              kwargs.get("gradient", {}).get("state_inds_gradient", None)
                             )
        num_jobs=self._get_number_of_jobs(properties, state_inds_gradient)
        spliting_flag = "Running Job"
        if num_jobs == 1:
            return [file_content]
        else:
            return file_content.split(spliting_flag)[1:num_jobs+1] 

    def _organize_data_file(self, file_content, properties, num_atoms, **kwargs):
        """
        Organize Q-Chem data according to the requested properties.
        """
        state_inds_gradient = self._get_state_inds_gradient(
                              kwargs.get("gradient", {}).get("state_inds_gradient", None)
                              )
        split_files=self._split_file_by_job(file_content, properties, **kwargs)
        i=0
        for job in properties:
            if "gradient" in job:
                gradient_files=split_files[i:i+len(state_inds_gradient)]
                i+=len(state_inds_gradient)
                self._pull_data(
                    job,
                    gradient_files,
                    num_atoms,
                    state_inds_gradient=state_inds_gradient,
                )
            elif "wf_overlaps" in job:
                overlaps_files=split_files[i:i+1]
                i+=2
                self._pull_data(
                    job,
                    overlaps_files,
                    num_atoms,
                    amplitudes_previous=kwargs["wf_overlaps"].get(
                        "amplitudes_previous", None
                    ),
                    amplitudes_current=kwargs["wf_overlaps"].get(
                        "amplitudes_current", None
                    ),
                )
            else:
                job_file = split_files[i]
                i+=1
                excited_amplitudes = (
                    kwargs["energy"].get("excited_amplitudes", False) if "energy" in job else False
                )
                self._pull_data(
                    job,
                    job_file.splitlines(),
                    num_atoms,
                    excited_amplitudes=excited_amplitudes,
                )

    def _pull_data(self, property, file_obj, num_atoms, **kwargs):
        if "energy" in property:
            self._pull_energy(
                file_obj, excited_amplitudes=kwargs.get("excited_amplitudes", False)
            )
        elif "gradient" in property:
            self._pull_gradient(
                file_obj, num_atoms, kwargs.get("state_inds_gradient", None)
            )
        elif "frequency" in property:
            self._pull_vibration(file_obj, num_atoms)
        elif "derivative_coupling" in property:
            self._pull_derivative_coupling(file_obj, num_atoms)
        elif "wf_overlaps" in property:
            self._pull_overlaps(
                amplitudes_previous=kwargs.get("amplitudes_previous", None),
                amplitudes_current=kwargs.get("amplitudes_current", None),
            )
        else:
            raise ValueError("This type of calculation has not been implemented yet.")

    def _pull_vibration(self, file_obj, num_atoms):
        ind_modes = []
        for i, line in enumerate(file_obj):
            if "Mode:" in line:
                ind_modes.append(i)
        freqs = []
        for i in ind_modes:
            freqs.append(file_obj[i + 1].split()[1:])
        freqs = np.array(freqs, dtype=float).flatten()
        modes = np.zeros((len(freqs), num_atoms, 3))
        i_count = 0
        num_lines = 8  # Number of lines between the Mode: line and the displacement-matrix line in a Q-Chem output file.
        for i in ind_modes:
            for j in range(num_atoms):
                temporal_data = file_obj[i + num_lines + j].split()[1:]
                num_modes = int(len(temporal_data) / 3)
                for k in range(num_modes):
                    modes[i_count + k, j, :] = temporal_data[k * 3 : k * 3 + 3]
            i_count += num_modes

        self.results["frequency"] = freqs
        self.results["normal_mode"] = modes

    def _pull_energy(self, file_obj, excited_amplitudes=False):
        # Get the number of states to be extracted from the output file.
        if self.parameters.get("cis_n_roots") is None:
            nt_states = 1   
        else:
            nt_states = int(self.parameters.get("cis_n_roots")) + 1
    
        keywords_for_gs_energy = [
            "SCF time:",
            "Timing for Total SCF",
            "Total energy in the final basis set",
        ]
        keywords_for_excited_energy = ["Total energy for state"]

        i=0
        energy=np.zeros(nt_states)
        for j, line in enumerate(file_obj):
            # Extract ground state energy.
            if any(word in line for word in keywords_for_gs_energy):
                try:
                    energy[i] = float(file_obj[j + 2].split()[-1])
                    i += 1
                except:
                    pass    
            # Extract excited state energies.
            elif any(word in line for word in keywords_for_excited_energy):
                energy[i] = float(line.split()[-2])
                i += 1
        self.results["energy"] = energy
        # Extract excited state amplitudes if requested.
        if excited_amplitudes:
            self.results["excited_amplitudes"] = {}
            (
                x_alpha,
                y_alpha,
                num_basis_functions,
                num_alpha_electrons,
                num_excited_states,
            ) = self._pull_excited_amplitudes()
            self.results["excited_amplitudes"]["x"] = x_alpha
            self.results["excited_amplitudes"]["y"] = y_alpha
            self.results["excited_amplitudes"][
                "num_basis_functions"
            ] = num_basis_functions
            self.results["excited_amplitudes"][
                "num_alpha_electrons"
            ] = num_alpha_electrons
            self.results["excited_amplitudes"][
                "num_excited_states"
            ] = num_excited_states



    def _pull_gradient(self, file_obj, num_atoms, state_inds_gradient=None):
        state_inds_gradient = self._get_state_inds_gradient(state_inds_gradient)
        gradient = np.zeros((num_atoms, 3, len(state_inds_gradient)))
        flag_words_grad = ["Gradient of the state energy (including CIS Excitation Energy)", 
                           "Gradient of SCF Energy"
                        ]
        flag_words_out_grad = ["Gradient time", "Max gradient component"]
        for state_ind in range(len(state_inds_gradient)):
            flag_in = next(flag for flag in flag_words_grad if flag in file_obj[state_ind])
            flag_out = next(flag for flag in flag_words_out_grad if flag in file_obj[state_ind])
            gradient_matrix = file_obj[state_ind].split(flag_in)[1]
            gradient_matrix = gradient_matrix.split(flag_out)[0]
            # Ensure that the gradient matrix is clean
            for f in flag_words_out_grad:
                if gradient_matrix.count(f) != 0:
                    gradient_matrix = gradient_matrix.split(f)[0]
            stop_ind = 0
            i_count = 1
            gradient_matrix = gradient_matrix.splitlines()
            while stop_ind < num_atoms:
                jndx = np.array(gradient_matrix[i_count].split(), dtype=int) - 1
                gradient[jndx, 0, state_ind] = np.array(
                    gradient_matrix[i_count + 1].split()[1:], dtype=float
                )
                gradient[jndx, 1, state_ind] = np.array(
                    gradient_matrix[i_count + 2].split()[1:], dtype=float
                )
                gradient[jndx, 2, state_ind] = np.array(
                    gradient_matrix[i_count + 3].split()[1:], dtype=float
                )
                i_count += 4
                stop_ind = jndx.max() + 1
        self.results["gradient"] = gradient

    def _pull_derivative_coupling(self, file_obj, num_atoms, etf=True):
        l_states = []  # Line that identifies the states involved in derivative coupling
        l_etf = (
            []
        )  # Line that identifies the location of derivative coupling's values with ETF corrections.
        l_no_etf = (
            []
        )  # Line that identifies the location of derivative coupling's values without ETF corrections.
        for i, line in enumerate(file_obj):
            if "between states" in line:
                l_states.append(i)
            elif "with ETF" in line:
                l_etf.append(i)
            elif "without ETF" in line:
                l_no_etf.append(i)

        # Gather derivative coupling values.
        derivative_coupling_matrix = np.zeros((num_atoms, 3))
        derivative_coupling_dictionary = {}
        for i, states in enumerate(l_states):
            temp = file_obj[states].split()
            # States involved in derivative coupling.
            i_st = int(temp[-3])
            j_st = int(temp[-1])
            # Base line for derivative coupling values.
            base = l_etf[i] if etf else l_no_etf[i]
            for j in range(num_atoms):
                derivative_coupling_matrix[j, :] = np.array(
                    file_obj[base + 3 + j].split()[1:], dtype=float
                )
            derivative_coupling_dictionary[(i_st, j_st)] = (
                derivative_coupling_matrix.copy()
            )
        self.results["derivative_coupling"] = derivative_coupling_dictionary

    def _pull_overlaps(self, amplitudes_previous=None, amplitudes_current=None):
        if amplitudes_previous is None:
            raise ValueError(
                "Previous excited state amplitudes must be provided \n"
                "to compute wavefunction overlaps."
            )
        if amplitudes_current is None:
            raise ValueError(
                "Current excited state amplitudes must be provided \n"
                "to compute wavefunction overlaps."
            )
        # Create global variables needed for computing overlaps.
        self.num_basis_functions = amplitudes_current["num_basis_functions"]
        self.num_alpha_electrons = amplitudes_current["num_alpha_electrons"]
        self.num_excited_states = amplitudes_current["num_excited_states"]
        # Extract molecular orbitals overlaps between two geometries.
        mo_overlaps = self._pull_mo_overlaps()
        # Compute overlaps matrices.
        if self.method_ex.lower() == "tddft":
            self._get_overlaps_tddft(
                amplitudes_previous["x"],
                amplitudes_previous["y"],
                amplitudes_current["x"],
                amplitudes_current["y"],
                mo_overlaps,
            )
        elif self.method_ex.lower() == "cis":
            self._get_overlaps_cis(
                amplitudes_previous["x"],
                amplitudes_current["x"],
                mo_overlaps,
            )
        else: 
            raise ValueError(f"Method {self.method_ex} not recognized. Supported methods are 'tddft' and 'cis'.")

    def _get_overlaps_cis(self, alpha_prev, alpha_curr, mo_overlaps):
        s_oo = mo_overlaps[0 : self.num_alpha_electrons, 0 : self.num_alpha_electrons]
        s_ov = mo_overlaps[0 : self.num_alpha_electrons, self.num_alpha_electrons :]
        s_vo = mo_overlaps[self.num_alpha_electrons :, 0 : self.num_alpha_electrons]
        s_vv = mo_overlaps[self.num_alpha_electrons :, self.num_alpha_electrons :]
        # Compute <GS^(1)|GS^(2)>.
        sing, log_determinant = np.linalg.slogdet(s_oo)
        gs_overlap = sing * np.exp(log_determinant)
        # Compute <GS^(1)|ES_i^(2)>.
        overlaps_gs_ex= self._compute_overlap_gs_ex(alpha_curr, s_oo, s_ov)
        # Compute <ES_i^(1)|GS^(2)>.
        overlaps_ex_gs = self._compute_overlap_ex_gs(alpha_prev, s_oo, s_vo)
        # Compute <ES_i^(1)|ES_j^(2)>.
        overlaps_ex_ex = self._compute_overlap_ex_ex(
            alpha_prev, alpha_curr, s_oo, s_ov, s_vo, s_vv
        )
        self.results["wf_overlaps"] = np.zeros(
            (self.num_excited_states + 1, self.num_excited_states + 1)
        )
        self.results["wf_overlaps"][0, 0] = gs_overlap
        self.results["wf_overlaps"][0, 1:] = 2.0 * overlaps_gs_ex
        self.results["wf_overlaps"][1:, 0] = 2.0 * overlaps_ex_gs
        self.results["wf_overlaps"][1:, 1:] = 2.0 * overlaps_ex_ex

    def _get_overlaps_tddft(self, x_prev, y_prev, x_curr, y_curr, mo_overlaps):
        s_oo = mo_overlaps[0 : self.num_alpha_electrons, 0 : self.num_alpha_electrons]
        s_ov = mo_overlaps[0 : self.num_alpha_electrons, self.num_alpha_electrons :]
        s_vo = mo_overlaps[self.num_alpha_electrons :, 0 : self.num_alpha_electrons]
        s_vv = mo_overlaps[self.num_alpha_electrons :, self.num_alpha_electrons :]
        # Compute <GS^(1)|GS^(2)>.
        sing, log_determinant = np.linalg.slogdet(s_oo)
        gs_overlap = sing * np.exp(log_determinant)
        # Compute <GS^(1)|ES_i^(2)>.
        overlaps_gs_ex_x = self._compute_overlap_gs_ex(x_curr, s_oo, s_ov)
        overlaps_gs_ex_y = self._compute_overlap_gs_ex(y_curr, s_oo, s_ov)
        overlaps_gs_ex = overlaps_gs_ex_x - overlaps_gs_ex_y
        # Compute <ES_i^(1)|GS^(2)>.
        overlaps_ex_gs_x = self._compute_overlap_ex_gs(x_prev, s_oo, s_vo)
        overlaps_ex_gs_y = self._compute_overlap_ex_gs(y_prev, s_oo, s_vo)
        overlaps_ex_gs = overlaps_ex_gs_x - overlaps_ex_gs_y
        # Compute <ES_i^(1)|ES_j^(2)>.
        overlaps_ex_ex_x_x = self._compute_overlap_ex_ex(
            x_prev, x_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_x_y = self._compute_overlap_ex_ex(
            x_prev, y_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_y_x = self._compute_overlap_ex_ex(
            y_prev, x_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex_y_y = self._compute_overlap_ex_ex(
            y_prev, y_curr, s_oo, s_ov, s_vo, s_vv
        )
        overlaps_ex_ex = (
            overlaps_ex_ex_x_x
            - overlaps_ex_ex_x_y
            - overlaps_ex_ex_y_x
            + overlaps_ex_ex_y_y
        )
        self.results["wf_overlaps"] = np.zeros(
            (self.num_excited_states + 1, self.num_excited_states + 1)
        )
        self.results["wf_overlaps"][0, 0] = gs_overlap
        self.results["wf_overlaps"][0, 1:] = 2.0 * overlaps_gs_ex
        self.results["wf_overlaps"][1:, 0] = 2.0 * overlaps_ex_gs
        self.results["wf_overlaps"][1:, 1:] = 2.0 * overlaps_ex_ex

    def _compute_overlap_gs_ex(self, excited_amplitudes, s_oo, s_ov):
        a_matrix = s_oo
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        w_matrix = np.linalg.solve(a_matrix, s_ov)
        x = np.asarray(excited_amplitudes)
        overlaps_gs_ex = determinant_a_matrix * np.einsum("eia,ia->e", x, w_matrix)
        return overlaps_gs_ex

    def _compute_overlap_ex_gs(self, excited_amplitudes, s_oo, s_vo):
        a_matrix = s_oo
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinant_a_matrix = sign * np.exp(log_determinant)
        w_matrix = np.linalg.solve(a_matrix.T, s_vo.T).T
        x = np.asarray(excited_amplitudes)
        overlaps_ex_gs = determinant_a_matrix * np.einsum("eia,ai->e", x, w_matrix)
        return overlaps_ex_gs

    def _compute_overlap_ex_ex(
        self, geometry_1_amplitudes, geometry_2_amplitudes, s_oo, s_ov, s_vo, s_vv
    ):
        a_matrix = s_oo
        num_occpied_orbitals = a_matrix.shape[0]
        x = np.asarray(geometry_1_amplitudes)
        x2 = np.asarray(geometry_2_amplitudes)
        sign, log_determinant = np.linalg.slogdet(a_matrix)
        determinat_a_matrix = sign * np.exp(log_determinant)
        inverse_a_matrix = np.linalg.solve(a_matrix, np.eye(num_occpied_orbitals))
        g_matrix = inverse_a_matrix @ s_ov
        overlap = np.zeros((x.shape[0], x2.shape[0]))
        for j in range(num_occpied_orbitals):
            aj = a_matrix[j, :]
            column_j = inverse_a_matrix[:, j]
            delta = s_vo - aj[None, :]
            delta_dot_column_j = delta @ column_j
            delta_dot_g = delta @ g_matrix
            for i in range(num_occpied_orbitals):
                q = inverse_a_matrix[i, j]
                gi = g_matrix[i, :]
                aji = a_matrix[j, i]
                delta_i = delta[:, i]
                alpha = s_vv - s_vo[:, i][:, None] - s_ov[j, :][None, :] + aji
                b21 = (delta_dot_g - delta_i[:, None]) + alpha * (gi[None, :] - 1.0)
                b22 = 1.0 + delta_dot_column_j[:, None] + alpha * q
                determinant_m = determinat_a_matrix * (gi[None, :] * b22 - q * b21)
                conjx = np.conj(x[:, j, :])
                x2ia = x2[:, i, :]
                overlap += np.einsum(
                    "eb,fa,ba->ef", conjx, x2ia, determinant_m, optimize=True
                )
        return overlap

    def _pull_mo_overlaps(self):
        qcscratch = os.environ["QCSCRATCH"]  # Q-Chem scratch folder.
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
        mo_overlaps = np.zeros((self.num_basis_functions, self.num_basis_functions))
        num_blocks = int(len(overlaps_init) / self.num_basis_functions)
        for i in range(self.num_basis_functions):
            temp_values = np.concatenate(
                [
                    overlaps_init[self.num_basis_functions * j + i]
                    for j in range(num_blocks)
                ]
            )
            mo_overlaps[i, :] = temp_values
        mo_overlaps = mo_overlaps.T
        return mo_overlaps

    def _pull_excited_amplitudes(self):
        """
        This function reads the FCHK file generated by Q-Chem
        to extract the excited state amplitudes for TDDFT and CIS calculations.
        Alpha X and Alpha Y amplitudes are extracted for linear TDDFT calculations,
        while only Alpha  and Beta amplitudes are extracted for TDA-TDDFT and CIS calculations.
        Only alpha amplitudes are extracted because the software only supports closed-shell calculations.

        For CIS, X_alpha is taken as Alpha amplitudes
        """
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
        x_alpha = []
        y_alpha = []
        for i, line in enumerate(data[index_restart:]):
            if keywords_to_find[4] in line:
                dimension_amplitudes = int(line.split()[-1])
                j = i + 1 + index_restart
                while keyword_to_stop[0] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_alpha.append(temp_values)
                    j = j + 1
            if keywords_to_find[5] in line:
                j = i + 1 + index_restart
                while keyword_to_stop[1] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    y_alpha.append(temp_values)
                    j = j + 1
            if keywords_to_find[6] in line:
                dimension_amplitudes = int(line.split()[-1])
                j = i + 1 + index_restart
                while keyword_to_stop[2] not in data[j]:
                    temp_values = np.array(data[j].split(), dtype=float)
                    x_alpha.append(temp_values)
                    y_alpha.append(np.zeros_like(temp_values))
                    j = j + 1
        """"
         Check that the number of basis functions used for Q-Chem corresponds 
         to the number of basis functions determined from the basis set information.

         Q-Chem sometimes can automatically project out near-linear dependencies, 
         which reduces the number of MOs below the number of basis functions.
        """
        
        num_basis_functions_check = int((dimension_amplitudes/num_excited_states)/num_alpha_electrons)+num_alpha_electrons
        if num_basis_functions_check != num_basis_functions:
            num_basis_functions = num_basis_functions_check
            print("Warning: The number of basis functions used for Q-Chem does not correspond to "
                  "the number of basis functions determined from the basis set information.")
    
        x_alpha = np.concatenate(x_alpha)
        x_alpha = np.reshape(
            x_alpha,
            (
                num_excited_states,
                num_alpha_electrons,
                (num_basis_functions - num_alpha_electrons),
            ),
        )
        y_alpha = np.concatenate(y_alpha)
        y_alpha = np.reshape(
            y_alpha,
            (
                num_excited_states,
                num_alpha_electrons,
                (num_basis_functions - num_alpha_electrons),
            ),
        )
        return (
            x_alpha,
            y_alpha,
            num_basis_functions,
            num_alpha_electrons,
            num_excited_states,
        )

    def write_input(self, **kwargs):
        system_changes = None
        properties = kwargs.keys()
        FileIOCalculator.write_input(self, self.atoms, properties, system_changes)
        filename = self.label + ".inp"
        job_specs = self._build_job_specs(properties)
        with open(filename, "w") as file_obj:
            self._write_job(file_obj, job_specs, **kwargs)