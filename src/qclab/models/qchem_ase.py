"""
This module contains the Q-Chem ASE Model class.
"""

import numpy as np
from ase.calculators.calculator import FileIOCalculator, SCFError
import ase.units
from qclab.functions import (
    vectorize_ingredient,
    make_ingredient_sparse,
    z_to_q,
    dqdp_to_dzc,
)
from qclab.model import Model
from qclab import ingredients
from qclab.numerical_constants import HA_TO_300K


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
        self.constants.init_position = mol.get_positions().flatten()
        np.random.seed(10)
        self.constants.init_momentum = np.random.rand(num_atoms * 3)

    def h_q(self, parameters, **kwargs):
        batch_size = kwargs["batch_size"]
        num_quantum_states = self.constants.num_quantum_states
        out = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )
        out[:, range(num_quantum_states), range(num_quantum_states)] = 1
        return out

    @vectorize_ingredient
    def h_qc(self, parameters, **kwargs):
        z = kwargs["z"]
        num_quantum_states = self.constants.num_quantum_states
        out = np.zeros((num_quantum_states, num_quantum_states), dtype=complex)
        mol = self.constants.ase_atoms_object
        qchem_args = self.constants.qchem_args
        qchem_tddft_args = self.constants.qchem_tddft_args
        q = z_to_q(
            z,
            self.constants.classical_coordinate_mass,
            self.constants.classical_coordinate_weight,
        )
        mol.set_positions(q.reshape((self.constants.num_classical_coordinates // 3, 3)))
        mol.calc = QCLabQChemCalculator(**{**qchem_args, **qchem_tddft_args})
        mol.calc.write_input(mol, properties=["energy"])
        mol.calc.execute()
        mol.calc.read_results()
        return np.diag(mol.calc.results["energy"] * HA_TO_300K)

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
                # Just do a ground state calculation for this one
                calc = QCLabQChemCalculator(**{**qchem_args})
            else:
                # Now specify the excited state index.
                calc = QCLabQChemCalculator(
                    **{
                        **qchem_args,
                        **qchem_tddft_args,
                        "CIS_STATE_DERIV": str(state_ind),
                    }
                )
            mol.calc = calc
            mol.calc.write_input(mol, properties=["gradient"])
            mol.calc.execute()
            mol.calc.read_results()
            out[:, state_ind, state_ind] = dqdp_to_dzc(
                mol.calc.results["gradient"].flatten() * HA_TO_300K, None, m, h
            )
        return out

    @vectorize_ingredient
    def derivative_coupling(self, parameters, **kwargs):
        z = kwargs["z"]
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        mol = self.constants.ase_atoms_object
        qchem_args = self.constants.qchem_args
        qchem_tddft_args = self.constants.qchem_tddft_args
        q = z_to_q(
            z,
            self.constants.classical_coordinate_mass,
            self.constants.classical_coordinate_weight,
        )
        mol.set_positions(q.reshape((num_classical_coordinates // 3, 3)))
        out = np.zeros(
            (num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        mol.calc = QCLabQChemCalculator(
            **{
                **qchem_args,
                **qchem_tddft_args,
                "CALC_NAC": "True",
                "CIS_DER_NUMSTATE": str(num_quantum_states),
            }
        )
        mol.calc.write_input(
            mol, properties=["derivative_coupling"], num_states=num_quantum_states
        )
        mol.calc.execute()
        mol.calc.read_results()
        derivative_coupling = mol.calc.results["derivative_coupling"]
        for key, val in derivative_coupling.items():
            out[:, key[0], key[1]] = val.flatten()
            out[:, key[1], key[0]] = -np.conj(out[:, key[0], key[1]])
        return out

    ingredients = [
        ("h_q", h_q),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_free),
        ("dh_c_dzc", ingredients.dh_c_dzc_free),
        ("dh_qc_dzc", dh_qc_dzc),
        ("init_classical", ingredients.init_classical_definite_position_momentum),
        ("derivative_coupling", derivative_coupling),
        ("_init_model", _init_model),
    ]


class QCLabQChemCalculator(FileIOCalculator):
    """
    QChem calculator implemented for QC Lab calculations.
    """

    name = "QChem"

    implemented_properties = ["energy", "gradient", "derivative_coupling"]
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

    def read(self, label):
        raise NotImplementedError

    def read_results(self):
        filename = self.label + ".out"
        derivative_coupling = {}
        use_etf = True
        target_str = "with ETF" if use_etf else "without ETF"
        with open(filename) as fileobj:
            lineiter = iter(fileobj)
            for line in lineiter:
                if "SCF failed to converge" in line:
                    raise SCFError()
                elif "ERROR: alpha_min" in line:
                    # Even though it is not technically a SCFError:
                    raise SCFError()
                elif " Total energy in the final basis set =" in line:
                    convert = ase.units.Hartree
                    self.results["energy"] = float(line.split()[8]) * convert
                elif " Total energy for state  " in line:
                    ind = int(line.split()[-3].split(":")[:-1][0])
                    convert = ase.units.Hartree
                    self.results["energy"] = np.append(
                        self.results["energy"], float(line.split()[-2]) * convert
                    )

                elif " Gradient of the state energy" in line:
                    # Read gradient as 3 by N array and transpose at the end
                    gradient = [[] for _ in range(3)]
                    # Skip first line containing atom numbering
                    next(lineiter)
                    while True:
                        # Loop over the three Cartesian coordinates
                        for i in range(3):
                            # Cut off the component numbering and remove
                            # trailing characters ('\n' and stuff)
                            line = next(lineiter)[5:].rstrip()
                            # Cut in chunks of 12 symbols and convert into
                            # strings. This is preferred over string.split() as
                            # the fields may overlap for large gradients
                            gradient[i].extend(
                                list(
                                    map(
                                        float,
                                        [
                                            line[i : i + 12]
                                            for i in range(0, len(line), 12)
                                        ],
                                    )
                                )
                            )

                        # After three force components we expect either a
                        # separator line, which we want to skip, or the end of
                        # the gradient matrix which is characterized by the
                        # line ' Max gradient component'.
                        # Maybe change stopping criterion to be independent of
                        # next line. Eg. if not lineiter.next().startswith(' ')
                        if "Gradient time" in next(lineiter):
                            self.results["gradient"] = np.array(gradient).T * (
                                ase.units.Hartree / ase.units.Bohr
                            )
                            break
                elif " Gradient of SCF Energy" in line:
                    # Read gradient as 3 by N array and transpose at the end
                    gradient = [[] for _ in range(3)]
                    # Skip first line containing atom numbering
                    next(lineiter)
                    while True:
                        # Loop over the three Cartesian coordinates
                        for i in range(3):
                            # Cut off the component numbering and remove
                            # trailing characters ('\n' and stuff)
                            line = next(lineiter)[5:].rstrip()
                            # Cut in chunks of 12 symbols and convert into
                            # strings. This is preferred over string.split() as
                            # the fields may overlap for large gradients
                            gradient[i].extend(
                                list(
                                    map(
                                        float,
                                        [
                                            line[i : i + 12]
                                            for i in range(0, len(line), 12)
                                        ],
                                    )
                                )
                            )

                        # After three force components we expect either a
                        # separator line, which we want to skip, or the end of
                        # the gradient matrix which is characterized by the
                        # line ' Max gradient component'.
                        # Maybe change stopping criterion to be independent of
                        # next line. Eg. if not lineiter.next().startswith(' ')
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
                                    # push back is hard with iterator; easiest is: break and rely on outer loop
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
        self.results["derivative_coupling"] = derivative_coupling

    def write_input(self, atoms, properties=None, system_changes=None, num_states=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        filename = self.label + ".inp"

        with open(filename, "w") as fileobj:
            fileobj.write("$comment\n   ASE generated input file\n$end\n\n")

            fileobj.write("$rem\n")
            if self.parameters["jobtype"] is None:
                if "gradient" in properties:
                    fileobj.write("   %-25s   %s\n" % ("JOBTYPE", "FORCE"))
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
