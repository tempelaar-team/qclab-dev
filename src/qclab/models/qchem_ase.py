"""
This module contains the Q-Chem ASE Model class.
"""

import numpy as np
import re
from ase.calculators.calculator import FileIOCalculator, SCFError
import ase.units
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
from qclab.numerical_constants import HA_TO_300K, INVCM_TO_300K
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
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = True
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        mol = self.constants.ase_atoms_object
        atom_masses = mol.get_masses()
        num_atoms = len(atom_masses)
        self.constants.num_classical_coordinates = num_atoms * 3
        self.constants.classical_coordinate_mass = (
            atom_masses[np.newaxis] * np.ones((3, num_atoms))
        ).flatten()
        self.constants.classical_coordinate_weight = np.ones(
            self.constants.num_classical_coordinates
        )
        if not "energy_offset" in self.constants.__dict__:
            print('Calculating energy offset')
            mol.calc = QCLabQChemCalculator(
                **{
                    **self.constants.qchem_args,
                    **self.constants.qchem_tddft_args,
                    "seed": None,
                }
            )
            mol.calc.write_input(mol, properties=["energy"])
            mol.calc.execute()
            mol.calc.read_results()
            self.constants.energy_offset = mol.calc.results["energy"][0] * HA_TO_300K
        if not "harmonic_frequency" in self.constants.__dict__:
            print('Calculating harmonic frequencies and normal modes')
            mol.calc = QCLabQChemCalculator(**{**self.constants.qchem_args, "seed": None})
            mol.calc.write_input(mol, properties=["frequency"])
            mol.calc.execute()
            mol.calc.read_results()
            self.constants.harmonic_frequency = mol.calc.results["frequency"] * INVCM_TO_300K
            self.constants.mass_weighted_normal_modes = (
                mol.calc.results["normal_mode"]
                .reshape((len(self.constants.harmonic_frequency), num_atoms * 3))
                .T
            )

    def init_classical(self, parameters, **kwargs):
        # temporarily set the masses to 1 for mass-weighted normal mode sampling.
        normal_modes = self.constants.mass_weighted_normal_modes / np.sqrt(
            self.constants.classical_coordinate_mass[:, np.newaxis]
        )
        old_constants = copy.deepcopy(self.constants)
        self.constants = Constants()
        self.constants.num_classical_coordinates = len(old_constants.harmonic_frequency)# set to number of normal modes
        self.constants.classical_coordinate_mass = np.ones(
            self.constants.num_classical_coordinates
        )
        self.constants.harmonic_frequency = old_constants.harmonic_frequency
        self.constants.classical_coordinate_weight = self.constants.harmonic_frequency
        self.constants.kBT = old_constants.kBT
        z_mwnm = ingredients.init_classical_wigner_harmonic(self, parameters, **kwargs)
        # convert back to mass-weighted normal coordinates
        q_mwnm = z_to_q(
            z_mwnm, self.constants.classical_coordinate_mass[np.newaxis], self.constants.classical_coordinate_weight[np.newaxis]
        )
        p_mwnm = z_to_p(
            z_mwnm, self.constants.classical_coordinate_mass[np.newaxis], self.constants.classical_coordinate_weight[np.newaxis]
        )
        # convert back to Cartesian coordinates.
        q = np.einsum("tm, cm ->tc", q_mwnm, normal_modes)
        p = np.einsum("tm, cm ->tc", p_mwnm, normal_modes)
        self.constants = old_constants
        z = qp_to_z(
            q, p, self.constants.classical_coordinate_mass[np.newaxis], self.constants.classical_coordinate_weight[np.newaxis]
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
        mol = self.constants.ase_atoms_object
        qchem_args = self.constants.qchem_args
        qchem_tddft_args = self.constants.qchem_tddft_args
        q = z_to_q(
            z,
            self.constants.classical_coordinate_mass,
            self.constants.classical_coordinate_weight,
        )
        mol.set_positions(q.reshape((self.constants.num_classical_coordinates // 3, 3)))
        mol.calc = QCLabQChemCalculator(
            **{
                **qchem_args,
                **qchem_tddft_args,
                "seed": None,
            }
        )
        mol.calc.write_input(mol, properties=["energy"])
        mol.calc.execute()
        mol.calc.read_results()
        diag_h_qc = (
            mol.calc.results["energy"][:num_quantum_states] * HA_TO_300K
            - self.constants.energy_offset
        )
        assert (
            len(diag_h_qc) == num_quantum_states
        ), "Number of quantum states mismatch." + str(diag_h_qc)
        return np.diag(diag_h_qc)

    @make_ingredient_sparse
    @vectorize_ingredient
    def dh_qc_dzc(self, parameters, **kwargs):
        z = kwargs["z"]
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        mol = self.constants.ase_atoms_object
        qchem_args = self.constants.qchem_args
        qchem_tddft_args = self.constants.qchem_tddft_args
        out = np.zeros(
            (num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        q = z_to_q(
            z,
            self.constants.classical_coordinate_mass,
            self.constants.classical_coordinate_weight,
        )
        mol.set_positions(q.reshape((num_classical_coordinates // 3, 3)))
        for state_ind in range(num_quantum_states):
            if state_ind == 0:
                # For state_ind == 0 do a ground state calculation.
                calc = QCLabQChemCalculator(
                    **{**qchem_args, "seed": None}
                )
            else:
                # Otherwise do an excited state calculation.
                calc = QCLabQChemCalculator(
                    **{
                        **qchem_args,
                        **qchem_tddft_args,
                        "CIS_STATE_DERIV": str(state_ind),
                        "seed": None,
                    }
                )
            mol.calc = calc
            mol.calc.write_input(mol, properties=["gradient"])
            mol.calc.execute()
            mol.calc.read_results()
            # Convert to derivative w.r.t. zc.
            out[:, state_ind, state_ind] = dqdp_to_dzc(
                mol.calc.results["gradient"].flatten() * HA_TO_300K, None, m, h
            )
        return out

    @vectorize_ingredient
    def derivative_coupling_dzc(self, parameters, **kwargs):
        z = kwargs["z"]
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        m = self.constants.classical_coordinate_mass
        h = self.constants.classical_coordinate_weight
        mol = self.constants.ase_atoms_object
        qchem_args = self.constants.qchem_args
        qchem_tddft_args = self.constants.qchem_tddft_args
        q = z_to_q(z, m, h)
        mol.set_positions(q.reshape((num_classical_coordinates // 3, 3)))
        out = np.zeros(
            (num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        if num_quantum_states < 2:
            return out
        mol.calc = QCLabQChemCalculator(
            **{
                **qchem_args,
                **qchem_tddft_args,
                "CALC_NAC": "True",
                "CIS_DER_NUMSTATE": str(num_quantum_states),
                "seed": None,
            }
        )
        mol.calc.write_input(
            mol, properties=["derivative_coupling"], num_states=num_quantum_states
        )
        mol.calc.execute()
        mol.calc.read_results()
        derivative_coupling_dq = mol.calc.results["derivative_coupling"]
        for key, val in derivative_coupling_dq.items():
            out[:, key[0], key[1]] = dqdp_to_dzc(val.flatten(), None, m, h)
            out[:, key[1], key[0]] = -np.conj(out[:, key[0], key[1]])
        return out

    ingredients = [
        ("h_q", h_q),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_free),
        ("dh_c_dzc", ingredients.dh_c_dzc_free),
        ("dh_qc_dzc", dh_qc_dzc),
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

    name = "QChem"
    implemented_properties = ["energy", "gradient", "derivative_coupling", "frequency"]
    _legacy_default_command = "qchem PREFIX.inp PREFIX.out"
    default_parameters = {"jobtype": None}

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=FileIOCalculator._deprecated,
        label="qchem",
        scratch=None,
        np=1,
        nt=1,
        pbs=False,
        basisfile=None,
        ecpfile=None,
        atoms=None,
        silent=True,
        seed=None,
        **kwargs,
    ):
        """
        The scratch directory, number of processor and threads as well as a few
        other command line options can be set using the arguments explained
        below. The remaining kwargs are copied as options to the input file.
        The calculator will convert these options to upper case
        (Q-Chem standard) when writing the input file.

        scratch: str
            path of the scratch directory
        np: int
            number of processors for the -np command line flag
        nt: int
            number of threads for the -nt command line flag
        pbs: boolean
            command line flag for pbs scheduler (see Q-Chem manual)
        basisfile: str
            path to file containing the basis. Use in combination with
            basis='gen' keyword argument.
        ecpfile: str
            path to file containing the effective core potential. Use in
            combination with ecp='gen' keyword argument.
        """

        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, **kwargs
        )

        # Augment the command by various flags
        if pbs:
            self.command = "qchem -pbs "
        else:
            self.command = "qchem "
        if np != 1:
            self.command += "-np %d " % np
        if nt != 1:
            self.command += "-nt %d " % nt
        self.command += "PREFIX.inp PREFIX.out"
        if scratch is not None:
            self.command += f" {scratch}"
        if silent:
            self.command += " > /dev/null 2>&1"

        self.basisfile = basisfile
        self.ecpfile = ecpfile
        if not (seed is None):
            self.label += f"_{seed}"

    def read(self, label):
        raise NotImplementedError

    def read_results(self):
        filename = self.label + ".out"
        derivative_coupling = {}
        use_etf = True
        target_str = "with ETF" if use_etf else "without ETF"

        # --- helper to parse vibrational analysis (freqs + modes) ---
        def _parse_vibrational_analysis(lines):
            """
            Parse Q-Chem VIBRATIONAL ANALYSIS section.

            Returns
            -------
            freqs : np.ndarray, shape (nmodes,)
                Harmonic frequencies in cm^-1.
            modes_mw : np.ndarray, shape (nmodes, natoms, 3)
                Mass-weighted normal mode displacements:
                modes_mw[k, A, :] = sqrt(m_A) * d_{A}^{(k)},
                where d are the Cartesian displacements printed by Q-Chem.
            """
            nlines = len(lines)

            # 1) Number of atoms from "Standard Nuclear Orientation"
            natoms = None
            for i, line in enumerate(lines):
                if "Standard Nuclear Orientation" in line:
                    j = i
                    # find dashed line
                    while j < nlines and not lines[j].strip().startswith("----"):
                        j += 1
                    j += 1  # first atom line
                    count = 0
                    while j < nlines and not lines[j].strip().startswith("----"):
                        if lines[j].strip():
                            count += 1
                        j += 1
                    natoms = count
                    break

            if natoms is None:
                return None, None

            # 2) Atomic masses from "Atom   1 Element C  Has Mass   12.00000"
            masses = [None] * natoms
            for line in lines:
                s = line.strip()
                if s.startswith("Atom") and "Has Mass" in s:
                    parts = s.split()
                    # "Atom", "1", "Element", "C", "Has", "Mass", "12.00000"
                    idx = int(parts[1]) - 1
                    mass = float(parts[-1])
                    masses[idx] = mass

            if any(m is None for m in masses):
                # No thermochemistry / masses => probably no vib analysis
                return None, None

            masses = np.array(masses, dtype=float)

            # 3) Find start of VIBRATIONAL ANALYSIS
            vib_start = None
            for i, line in enumerate(lines):
                if "VIBRATIONAL ANALYSIS" in line:
                    vib_start = i
                    break

            if vib_start is None:
                return None, None

            freqs = []
            modes = []  # list of (natoms, 3)
            i = vib_start + 1

            while i < nlines:
                line = lines[i]

                # End of vibrational section
                if line.strip().startswith("STANDARD THERMODYNAMIC QUANTITIES"):
                    break

                if line.strip().startswith("Mode:"):
                    # Example: " Mode:                 1                      2                      3"
                    after = line.split("Mode:")[1]
                    mode_nums = [int(tok) for tok in after.split() if tok.isdigit()]
                    n_block = len(mode_nums)

                    # Frequencies line is next
                    freq_line = lines[i + 1]
                    block_freqs = [float(x) for x in re.findall(r"[-+]?\d+\.\d*", freq_line)]
                    if len(block_freqs) != n_block:
                        raise RuntimeError(
                            f"Frequency count mismatch in mode block at line {i}."
                        )
                    freqs.extend(block_freqs)

                    # Find "X      Y      Z" header for the displacement block
                    header_idx = None
                    for j in range(i, min(i + 40, nlines)):
                        if "X      Y      Z" in lines[j]:
                            header_idx = j
                            break
                    if header_idx is None:
                        raise RuntimeError("Could not find X Y Z header for mode block.")

                    # Allocate [n_block, natoms, 3]
                    block_coords = [
                        [[0.0, 0.0, 0.0] for _ in range(natoms)]
                        for _ in range(n_block)
                    ]

                    # Each of the next natoms lines contains displacements
                    for atom_idx in range(natoms):
                        atom_line = lines[header_idx + 1 + atom_idx]
                        parts = atom_line.split()
                        # parts[0] is element label, remaining are 3*n_block floats
                        float_vals = list(map(float, parts[1:]))
                        if len(float_vals) != 3 * n_block:
                            raise RuntimeError(
                                f"Expected {3*n_block} displacement values, got {len(float_vals)}\n"
                                f"Line: {atom_line}"
                            )
                        for m in range(n_block):
                            x, y, z = float_vals[3*m:3*m+3]
                            block_coords[m][atom_idx] = [x, y, z]

                    modes.extend(block_coords)

                    # Skip this block: atom lines + TransDip line
                    i = header_idx + 1 + natoms + 1
                    continue

                i += 1

            if not freqs:
                return None, None

            freqs = np.array(freqs, dtype=float)
            modes = np.array(modes, dtype=float)  # (nmodes, natoms, 3)
            return freqs, modes

        # --- main parsing: energies, gradients, derivative couplings ---
        with open(filename) as fileobj:
            lines = fileobj.readlines()

        lineiter = iter(lines)
        for line in lineiter:
            if "SCF failed to converge" in line:
                raise SCFError()
            elif "ERROR: alpha_min" in line:
                raise SCFError()

            elif " Total energy in the final basis set =" in line:
                convert = ase.units.Hartree
                self.results["energy"] = np.array(
                    [float(line.split()[8]) * convert]
                )

            elif " Total energy for state  " in line:
                ind = int(line.split()[-3].split(":")[:-1][0])
                convert = ase.units.Hartree
                self.results["energy"] = np.append(
                    self.results["energy"], float(line.split()[-2]) * convert
                )

            elif " Gradient of the state energy" in line:
                # Read gradient as 3 by N array and transpose at the end
                gradient = [[] for _ in range(3)]
                next(lineiter)  # skip atom numbering line
                while True:
                    for i in range(3):
                        line = next(lineiter)[5:].rstrip()
                        gradient[i].extend(
                            list(
                                map(
                                    float,
                                    [line[j : j + 12] for j in range(0, len(line), 12)],
                                )
                            )
                        )
                    if "Gradient time" in next(lineiter):
                        self.results["gradient"] = np.array(gradient).T * (
                            ase.units.Hartree / ase.units.Bohr
                        )
                        break

            elif " Gradient of SCF Energy" in line:
                gradient = [[] for _ in range(3)]
                next(lineiter)  # skip atom numbering line
                while True:
                    for i in range(3):
                        line = next(lineiter)[5:].rstrip()
                        gradient[i].extend(
                            list(
                                map(
                                    float,
                                    [line[j : j + 12] for j in range(0, len(line), 12)],
                                )
                            )
                        )
                    if " Max gradient component" in next(lineiter):
                        self.results["gradient"] = np.array(gradient).T * (
                            ase.units.Hartree / ase.units.Bohr
                        )
                        break

            elif "CIS Derivative Couplings" in line:
                # now inside that section
                for line in lineiter:
                    if "between states" in line:
                        parts = line.split()
                        ind_i = int(parts[-3])
                        ind_j = int(parts[-1])

                        # search forward for "with ETF"/"without ETF"
                        for line in lineiter:
                            if "between states" in line:
                                # overshot into next block
                                break
                            if target_str in line:
                                # find Atom header
                                for line in lineiter:
                                    if "Atom" in line:
                                        break
                                next(lineiter)  # skip dashed line

                                rows = []
                                for line in lineiter:
                                    s = line.strip()
                                    if not s or set(s) == {"-"}:
                                        break
                                    cols = s.split()
                                    atom = int(cols[0])
                                    x, y, z = map(float, cols[1:4])
                                    rows.append([x, y, z])

                                derivative_coupling[(ind_i, ind_j)] = np.array(rows)
                                break

        # Store derivative couplings
        self.results["derivative_coupling"] = derivative_coupling

        # Parse vibrational frequencies + mass-weighted modes
        freqs, modes = _parse_vibrational_analysis(lines)
        self.results["frequency"] = freqs               # (nmodes,)
        self.results["normal_mode"] = modes              # (nmodes, natoms, 3)

    def write_input(self, atoms, properties=None, system_changes=None, num_states=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        filename = self.label + ".inp"

        with open(filename, "w") as fileobj:
            fileobj.write("$comment\n   ASE generated input file\n$end\n\n")

            fileobj.write("$rem\n")
            if self.parameters["jobtype"] is None:
                if "gradient" in properties:
                    fileobj.write("   %-25s   %s\n" % ("JOBTYPE", "FORCE"))
                elif "frequency" in properties:
                    fileobj.write("   %-25s   %s\n" % ("JOBTYPE", "FREQ"))
                else:
                    fileobj.write("   %-25s   %s\n" % ("JOBTYPE", "SP"))
            for prm in self.parameters:
                if prm not in ["charge", "multiplicity"]:
                    if self.parameters[prm] is not None:
                        fileobj.write(
                            "   %-25s   %s\n"
                            % (prm.upper(), self.parameters[prm].upper())
                        )

            # Not even a parameters as this is an absolute necessity
            fileobj.write("   %-25s   %s\n" % ("SYM_IGNORE", "TRUE"))
            fileobj.write("$end\n\n")

            if "derivative_coupling" in properties:
                fileobj.write("$derivative_coupling\n")
                fileobj.write("0 is the reference state\n")
                fileobj.write(
                    (("%d " * num_states) + "\n") % tuple(np.arange(num_states))
                )
                fileobj.write("$end\n\n")

            fileobj.write("$molecule\n")
            # Following the example set by the gaussian calculator
            if "multiplicity" not in self.parameters:
                tot_magmom = atoms.get_initial_magnetic_moments().sum()
                mult = tot_magmom + 1
            else:
                mult = self.parameters["multiplicity"]
            # Default charge of 0 is defined in default_parameters
            fileobj.write("   %d %d\n" % (self.parameters["charge"], mult))
            for a in atoms:
                fileobj.write(
                    "   {}  {:f}  {:f}  {:f}\n".format(a.symbol, a.x, a.y, a.z)
                )
            fileobj.write("$end\n\n")

            if self.basisfile is not None:
                with open(self.basisfile) as f_in:
                    basis = f_in.readlines()
                fileobj.write("$basis\n")
                fileobj.writelines(basis)
                fileobj.write("$end\n\n")

            if self.ecpfile is not None:
                with open(self.ecpfile) as f_in:
                    ecp = f_in.readlines()
                fileobj.write("$ecp\n")
                fileobj.writelines(ecp)
                fileobj.write("$end\n\n")
