"""
This module contains the DCMeanField algorithm class.

"""

from functools import partial
from qclab.algorithm import Algorithm
from qclab import model, tasks
from qclab import ingredients
import numpy as np 
from qclab import Simulation
from qclab import functions
from qclab import numerical_constants
from qclab.functions import qp_to_z
from qclab.tasks.update_tasks import update_dh_qc_dzc

def initialize_z_vac(
    sim: Simulation,
    state: dict,
    parameters: dict,
    seed_name: str = "seed",
    z_name: str = "z_vac",
):
    """
    Initializes the classical coordinate by using the init_classical function from the
    Model object.

        #TODO: update description
   
    Optional Keyword Arguments
    --------------------------
    seed_name:
        Name of seed array in State object.
    z_name:
        Name of classical coordinate in State object.

    Ingredients
    -----------
    init_classical:
        Classical coordinate initialization.

    Reads
    -----
    state[seed_name]: ndarray of shape (B,), dtype=int
        Seed in each trajectory.

    Writes
    ------
    state[z_name] : ndarray of shape (B, C), dtype=complex128
        Initialized classical coordinates.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    focused_sampling = getattr(sim.algorithm.settings, "focused_sampling", False)

    seed = state[seed_name]
    if focused_sampling:
        state[z_name] = init_classical_wigner_harmonic_focused(
            sim.model, parameters, seed=seed
        )
    else:
        state[z_name] = init_classical_wigner_harmonic_new(
            sim.model, parameters, seed=seed, kBT=0
        )
    return state, parameters

def init_classical_wigner_harmonic_new(model, parameters, **kwargs):
    """
    Initialize classical coordinates according to the Wigner distribution of the ground
    state of a harmonic oscillator.

    .. rubric:: Keyword Args
    seed : ndarray, int
        Random seeds for each trajectory.

    .. rubric:: Model Constants
    kBT : float
        Thermal quantum.
    harmonic_frequency : ndarray
        Harmonic frequency of each classical coordinate.

    .. rubric:: Returns
    z : ndarray
        complex classical coordinates.
    """
    seed = kwargs["seed"]
    w = model.constants.harmonic_frequency
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight
    kBT = model.constants.kBT if kwargs.get("kBT") is None else kwargs["kBT"]
    z = np.zeros(
        (len(seed), model.constants.num_classical_coordinates), dtype=complex
    )
    # Calculate the standard deviations for q and p.
    if kBT > 0:
        std_q = np.sqrt(0.5 / (w * m * np.tanh(0.5 * w / kBT)))
        std_p = np.sqrt(0.5 * m * w / np.tanh(0.5 * w / kBT))
    else:
        std_q = np.sqrt(0.5 / (w * m))
        std_p = np.sqrt(0.5 * m * w)
    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        # Generate random q and p values.
        q = np.random.normal(
            loc=0, scale=std_q, size=model.constants.num_classical_coordinates
        )
        p = np.random.normal(
            loc=0, scale=std_p, size=model.constants.num_classical_coordinates
        )
        # Calculate the complex-valued classical coordinates.
        z[s] = functions.qp_to_z(q, p, m, h)
    return z

def init_classical_wigner_harmonic_focused(model, parameters, **kwargs):
    """
    Initialize classical coordinates according to the Focused Sampling.

    .. rubric:: Keyword Args
    seed : ndarray, int
        Random seeds for each trajectory.

    .. rubric:: Model Constants
    kBT : float
        Thermal quantum.
    harmonic_frequency : ndarray
        Harmonic frequency of each classical coordinate.

    .. rubric:: Returns
    z : ndarray
        Complex classical coordinate.
    """
    seed = kwargs["seed"]
    w = model.constants.harmonic_frequency
    m = model.constants.classical_coordinate_mass
    h = model.constants.classical_coordinate_weight
    z = np.zeros(
        (len(seed), model.constants.num_classical_coordinates), dtype=complex
    )

    for s, seed_value in enumerate(seed):
        np.random.seed(seed_value)
        theta = np.random.uniform(0, 2 * np.pi, size=model.constants.num_classical_coordinates)
        q = np.cos(theta) / np.sqrt(w)
        p = np.sqrt(w) * np.sin(theta)
        z[s] = qp_to_z(q, p, m, h)
    return z

def update_mask_pop_adb_h_q(
    sim: Simulation,
    state: dict,
    parameters: dict,
    h_q_name: str = "h_q",
    mask_name: str = "mask",
):
    """
    Read the diagonal of h_q and store the per-trajectory diagonal ordering.

    state[order_name][b, i] gives the rank (0 = lowest energy) of diagonal element i
    in trajectory b.
    """
    h_q = state[h_q_name]
    diag_h_q = np.real(np.diagonal(h_q, axis1=-2, axis2=-1))

    # Compute rank per original index.
    order = np.argsort(np.argsort(diag_h_q, axis=1), axis=1)
    order_i = order[:, :, None] 
    order_j = order[:, None, :]
    mask = order_j >= order_i  # if True, use w_j. If False, use w_i.
    
    state[mask_name] = mask
    return state, parameters

def update_h_q_tot_dcmf(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    z_vac_name: str = "z_vac",
    h_q_name: str = "h_q",
    h_qc_name: str = "h_qc",
    h_q_tot_name: str = "h_q_tot",
    pop_adb_h_q_name: str = "pop_adb_h_q",
    eigvecs_h_q_name: str = "eigvecs_h_q",
    eigvals_h_q_name: str = "eigvals_h_q",
    wf_db_name: str = "wf_db",
    wf_adb_h_q_name: str = "wf_adb_h_q",
    h_qc_vac_name: str = "h_qc_vac",
    h_qc_vac_adb_h_q_name: str = "h_qc_vac_adb_h_q",
    mask_name: str = "mask",
):
    """
    Updates the total quantum Hamiltonian matrix.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in State object.
    h_q_name:
        Name of the quantum Hamiltonian in the State object.
    h_qc_name:
        Name of the quantum-classical coupling Hamiltonian in the State object.
    h_q_tot_name:
        Name of the total Hamiltonian of the quantum subsystem in the State object.
        (``h_q + h_qc``)

    Constants and Settings
    ----------------------
    sim.model.update_h_q: Bool
        Flag used to determine if the quantum Hamiltonian should be updated at each timestep.

    Ingredients
    -----------
    h_q:
        Quantum Hamiltonian.
    h_qc:
        Quantum-classical Hamiltonian.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinate.

    Writes
    ------
    state[h_q_name]: ndarray of shape (B, N, N), dtype=complex128
        Quantum Hamiltonian.
    state[h_qc_name]: ndarray of shape (B, N, N), dtype=complex128
        Quantum-classical Hamiltonian.
    state[h_q_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian (quantum plus quantum-classical).

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    * N = sim.model.constants.num_quantum_states
    """
    tasks.update_h_q_tot(sim, state, parameters)

    if eigvecs_h_q_name not in state or sim.model.update_h_q: 
        if sim.model.diagonal_h_q:
            update_mask_pop_adb_h_q(sim, state, parameters)
            state[eigvecs_h_q_name] = None
        else:
            tasks.diagonalize_matrix(sim, state, parameters, matrix_name=h_q_name, eigvals_name=eigvals_h_q_name, eigvecs_name=eigvecs_h_q_name)

    if sim.model.diagonal_h_q:
        pop_adb_h_q = np.abs(state[wf_db_name])**2
    else:
        tasks.update_vector_basis(sim, state, parameters, input_vec_name=wf_db_name, basis_name=eigvecs_h_q_name, output_vec_name=wf_adb_h_q_name)
        pop_adb_h_q = np.abs(state[wf_adb_h_q_name])**2

    state[pop_adb_h_q_name] = pop_adb_h_q

    h_qc, _ = sim.model.get("h_qc")
    h_qc_vac = h_qc(sim.model, parameters, z=state[z_vac_name])
    state[h_qc_vac_name] = h_qc_vac
    pop_adb_h_q_i = pop_adb_h_q[:, :, None] * np.ones_like(h_qc_vac)
    pop_adb_h_q_j = pop_adb_h_q[:, None, :] * np.ones_like(h_qc_vac)
    denominator = (pop_adb_h_q_i + pop_adb_h_q_j + numerical_constants.SMALL)

    if sim.model.diagonal_h_q:
        numerator = np.where(state[mask_name], pop_adb_h_q_j, pop_adb_h_q_i)
        h_qc_weighted = numerator / denominator * h_qc_vac
    else:
        eigvecs = state[eigvecs_h_q_name]
        h_qc_vac_eig = functions.transform_mat(h_qc_vac, eigvecs)
        state[h_qc_vac_adb_h_q_name] = h_qc_vac_eig
        numerator = np.triu(pop_adb_h_q_j) + np.tril(pop_adb_h_q_i, k=-1)
        h_qc_weighted_eig = numerator / denominator * h_qc_vac_eig
        h_qc_weighted = functions.transform_mat(h_qc_weighted_eig, eigvecs, adb_to_db=True)
    
    state[h_qc_name] += h_qc_weighted
    state[h_q_tot_name] = state[h_q_name] + state[h_qc_name]
    return state, parameters

def update_quantum_classical_force_dcmf(
    sim: Simulation,
    state: dict,
    parameters: dict,
    z_name: str = "z",
    z_vac_name: str = "z_vac",
    wf_db_name: str = "wf_db",
    dh_qc_dzc_name: str = "dh_qc_dzc",
    quantum_classical_force_name: str = "quantum_classical_force",
    quantum_classical_force_vac_name: str = "quantum_classical_force_vac",
    wf_changed: bool = True,
    h_q_tot_name: str = "h_q_tot",
    update_dh_qc_dzc_flag: bool = False,
    wf_adb_h_q_name: str = "wf_adb_h_q",
    pop_adb_h_q_name: str = "pop_adb_h_q",
    pop_adb_h_q_prev_name: str = "pop_adb_h_q_prev",
    eigvecs_h_q_name: str = "eigvecs_h_q",
    dh_qc_dzc_adb_h_q_name: str = "dh_qc_dzc_adb_h_q",
    h_qc_vac_adb_h_q_name: str = "h_qc_vac_adb_h_q",
    h_qc_vac_name: str = "h_qc_vac",
    mask_name: str = "mask",
    dh_qc_dzc_vac_name: str = "dh_qc_dzc_vac",
):
    """
    Updates the quantum-classical force w.r.t. the wavefunction defined by ``wf_db``.

    If the model has a ``gauge_field_force`` ingredient, this term will be added
    to the quantum-classical force.

    If the model has a ``derivative_coupling_dzc`` ingredient, this conribution will
    be added to the quantum-classical force.

    Optional Keyword Arguments
    --------------------------
    z_name:
        Name of classical coordinates in the State object.
    wf_db_name:
        Name of the wavefunction (in the diabatic basis) in the State object.
    dh_qc_dzc_name:
        Name of the gradient of the quantum-classical Hamiltonian in the State object.
    quantum_classical_force_name:
        Name under which to store the quantum-classical force in the State object.
    state_ind_name:
        Name in the State object of the state index for which to obtain the gauge field force.
        Required if ``algorithm.settings.use_gauge_field_force`` is ``True``.
    wf_changed:
        If ``True``, the wavefunction has changed since the last time the force were calculated.
    h_q_tot_name:
        Name under which to store the total Hamiltonian in the State object.

    Constants and Settings
    ----------------------
    sim.model.update_dh_qc_dzc: Bool, default: False
        Model flag indicating if the quantum-classical Hamiltonian is to be updated at each timestep.

    Ingredients
    -----------
    derivative_coupling_dzc:
        Derivative coupling tensor.

    Reads
    -----
    state[z_name]: ndarray of shape (B, C), dtype=complex128
        Complex-valued classical coordinates.
    state[wf_db_name]: ndarray of shape (B, N), dtype=complex128
        Wavefunction coefficients in the diabatic basis.
    state[dh_qc_dzc_name]: tuple
        Gradient of the quantum-classical Hamiltonian in sparse format.
    state[h_qc_tot_name]: ndarray of shape (B, N, N), dtype=complex128
        Total quantum Hamiltonian.

    Writes
    ------
    state[quantum_classical_force_name]: ndarray of shape (B, C), dtype=complex128
        Quantum-classical force.

    Notes
    -----
    * B = sim.settings.batch_size
    * C = sim.model.constants.num_classical_coordinates
    """
    z = state[z_name]
    z_vac = state[z_vac_name]

    tasks.update_quantum_classical_force(sim, state, parameters, z_name=z_name, wf_changed=wf_changed, update_dh_qc_dzc_flag=update_dh_qc_dzc_flag)

    if update_dh_qc_dzc_flag or sim.model.update_h_q:
        # Update the gradient of h_qc.
        state, parameters = update_dh_qc_dzc(
            sim, state, parameters, z_name=z_vac_name, dh_qc_dzc_name=dh_qc_dzc_vac_name
        )

        if not sim.model.diagonal_h_q:
            inds, mels, shape = state[dh_qc_dzc_vac_name] 
            dh_dense = np.zeros(shape, dtype=mels.dtype)
            dh_dense[inds] = mels

            U = state[eigvecs_h_q_name]
            dh_eig = np.einsum("bki,bcij,bjl->bckl", U.conj(), dh_dense, U, optimize="greedy")

            inds = np.where(dh_eig != 0)
            mels = dh_eig[inds]
            shape = np.shape(dh_eig)
            state[dh_qc_dzc_adb_h_q_name] = (inds, mels, shape)

    # Calculate the expectation value w.r.t. the wavefunction.
    # If not(wf_changed) and sim.model.update_dh_qc_dzc then recalculate.
    # If wf_changed then recalculate.
    # If quantum_classical_force_name not in state then recalculate.
    if (
        not (quantum_classical_force_vac_name in state)
        or wf_changed
        or (not (wf_changed) and sim.model.update_dh_qc_dzc)
    ):
        if not (quantum_classical_force_vac_name in state):
            state[quantum_classical_force_vac_name] = np.zeros_like(z_vac)

        pop_adb_h_q = state[pop_adb_h_q_name]

        if sim.model.diagonal_h_q:
            wf = state[wf_db_name]
            mask = state[mask_name]
            inds, mels, shape = state[dh_qc_dzc_vac_name]
            b, c, i, j = inds
            pop_adb_h_q_i = pop_adb_h_q[b, i]
            pop_adb_h_q_j = pop_adb_h_q[b, j]
            factor = 1 / (pop_adb_h_q_i + pop_adb_h_q_j + numerical_constants.SMALL)
            weighting = np.where(mask[b, i, j], pop_adb_h_q_j * factor, pop_adb_h_q_i * factor)
        else:
            wf = state[wf_adb_h_q_name]
            inds, mels, shape = state[dh_qc_dzc_adb_h_q_name]
            b, c, i, j = inds
            pop_adb_h_q_i = pop_adb_h_q[b, i]
            pop_adb_h_q_j = pop_adb_h_q[b, j]
            factor = 1 / (pop_adb_h_q_i + pop_adb_h_q_j + numerical_constants.SMALL)
            weighting = np.where(i <= j, pop_adb_h_q_j * factor, pop_adb_h_q_i * factor)
        
        dh_qc_dzc_weighted = (inds, mels * weighting, shape)
        state[quantum_classical_force_vac_name] = functions.calc_sparse_inner_product(
            *dh_qc_dzc_weighted,
            wf.conj(),
            wf,
            out=state[quantum_classical_force_vac_name].reshape(-1),
        ).reshape(np.shape(z_vac))

        # if getattr(sim.algorithm.settings, "use_energy_conserving_force", False):
        #     m = sim.model.constants.classical_coordinate_mass
        #     h = sim.model.constants.classical_coordinate_weight
        #     w = sim.model.constants.harmonic_frequency
        #     kBT = sim.model.constants.kBT

        #     rho_now = state[pop_adb_h_q_name]
        #     rho_prev = state[pop_adb_h_q_prev_name]
        #     shape_ref = np.shape(state[h_qc_vac_name])
        #     rho_now_row = rho_now[:, :, None] * np.ones(shape_ref)
        #     rho_now_col = rho_now[:, None, :] * np.ones(shape_ref)
        #     rho_prev_row = rho_prev[:, :, None] * np.ones(shape_ref)
        #     rho_prev_col = rho_prev[:, None, :] * np.ones(shape_ref)
        #     p = functions.z_to_p(z, m, h)
        #     p_vac = functions.z_to_p(z_vac, m, h)

        #     if sim.model.diagonal_h_q:
        #         h_qc_vac = state[h_qc_vac_name]
        #         gamma_now = np.where(state[mask_name], rho_now_col, rho_now_row)/(rho_now_col + rho_now_row + numerical_constants.SMALL)
        #         gamma_prev = np.where(state[mask_name], rho_prev_col, rho_prev_row)/(rho_prev_col + rho_prev_row + numerical_constants.SMALL)
        #         dgamma_dt = (gamma_now - gamma_prev) / sim.settings.dt_update
        #         drho_dq_hqc = dgamma_dt * h_qc_vac
        #         drho_dq_hqc_expectation_value = np.real(np.einsum('bi,bij,bj->b', wf.conj(), drho_dq_hqc, wf, optimize="greedy"))

        #     else:
        #         h_qc_vac_adb_h_q = state[h_qc_vac_adb_h_q_name]
        #         gamma_now = (np.triu(rho_now_col) + np.tril(rho_now_row, k=-1))/(rho_now_col + rho_now_row + numerical_constants.SMALL)
        #         gamma_prev = (np.triu(rho_prev_col) + np.tril(rho_prev_row, k=-1))/(rho_prev_col + rho_prev_row + numerical_constants.SMALL)
        #         dgamma_dt = (gamma_now - gamma_prev) / sim.settings.dt_update
        #         drho_dq_hqc_adb_h_q = dgamma_dt * h_qc_vac_adb_h_q
        #         drho_dq_hqc_expectation_value = np.real(np.einsum('bi,bij,bj->b', wf.conj(), drho_dq_hqc_adb_h_q, wf, optimize="greedy"))

        #     if kBT > 0:
        #         std_p = np.sqrt(0.5 * m * w / np.tanh(0.5 * w / kBT))
        #     else:
        #         std_p = np.sqrt(0.5 * m * w)
        #     std_p_vac = np.sqrt(0.5 * m * w)

        #     fraction = getattr(sim.algorithm.settings, "energy_conserving_force_fraction", 0.1)
        #     additional_force = drho_dq_hqc_expectation_value[:, np.newaxis] / (np.sign(p + numerical_constants.SMALL) * np.maximum(np.abs(p), fraction * std_p))
        #     additional_force_vac = drho_dq_hqc_expectation_value[:, np.newaxis] / (np.sign(p_vac + numerical_constants.SMALL) * np.maximum(np.abs(p_vac), fraction * std_p_vac))
        #     state[quantum_classical_force_name] += -additional_force
        #     state[quantum_classical_force_vac_name] += -additional_force_vac
    return state, parameters


class DecoupledMeanField(Algorithm):
    """
    Decoupled Mean-field dynamics algorithm class.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {
                "focused_sampling": False,
                "use_energy_conserving_force": False,
                "energy_conserving_force_fraction": 0.01,
            }
        self.default_settings = {}
        super().__init__(self.default_settings, settings)

    initialization_recipe = [
        tasks.initialize_variable_objects,
        tasks.initialize_norm_factor,
        tasks.initialize_z,
        initialize_z_vac,
        update_h_q_tot_dcmf,
        partial(tasks.copy_in_state, orig_name="pop_adb_h_q", copy_name="pop_adb_h_q_prev"),
        partial(update_quantum_classical_force_dcmf, wf_changed=False, update_dh_qc_dzc_flag=True),
    ]

    update_recipe = [
        # Begin RK4 integration steps.
        partial(tasks.update_classical_force, z_name="z"),
        partial(tasks.update_classical_force, z_name="z_vac", classical_force_name="classical_force_vac"),
        partial(update_quantum_classical_force_dcmf),
        tasks.update_z_rk4_k123,
        partial(
            tasks.update_z_rk4_k123, 
            z_name="z_vac", 
            z_k_name="z_1_vac", 
            k_name="z_rk4_k1_vac", 
            classical_force_name="classical_force_vac", 
            quantum_classical_force_name="quantum_classical_force_vac"
            ),
        
        partial(tasks.update_classical_force, z_name="z_1"),
        partial(tasks.update_classical_force, z_name="z_1_vac", classical_force_name="classical_force_vac"),
        partial(update_quantum_classical_force_dcmf, z_name="z_1", z_vac_name="z_1_vac", wf_changed=False),
        partial(tasks.update_z_rk4_k123, z_name="z", z_k_name="z_2", k_name="z_rk4_k2"),
        partial(
            tasks.update_z_rk4_k123, 
            z_name="z_vac", 
            z_k_name="z_2_vac", 
            k_name="z_rk4_k2_vac", 
            classical_force_name="classical_force_vac", 
            quantum_classical_force_name="quantum_classical_force_vac"
            ),

        partial(tasks.update_classical_force, z_name="z_2"),
        partial(tasks.update_classical_force, z_name="z_2_vac", classical_force_name="classical_force_vac"),
        partial(update_quantum_classical_force_dcmf, z_name="z_2", z_vac_name="z_2_vac", wf_changed=False),
        partial(
            tasks.update_z_rk4_k123,
            z_name="z",
            z_k_name="z_3",
            k_name="z_rk4_k3",
            dt_factor=1.0,
        ),
        partial(
            tasks.update_z_rk4_k123,
            z_name="z_vac",
            z_k_name="z_3_vac",
            k_name="z_rk4_k3_vac",
            classical_force_name="classical_force_vac", 
            quantum_classical_force_name="quantum_classical_force_vac",
            dt_factor=1.0,
        ),

        partial(tasks.update_classical_force, z_name="z_3"),
        partial(tasks.update_classical_force, z_name="z_3_vac", classical_force_name="classical_force_vac"),
        partial(update_quantum_classical_force_dcmf, z_name="z_3", z_vac_name="z_3_vac", wf_changed=False),
        tasks.update_z_rk4_k4,
        partial(tasks.update_z_rk4_k4,
            z_name="z_vac",
            k1_name = "z_rk4_k1_vac",
            k2_name = "z_rk4_k2_vac",
            k3_name = "z_rk4_k3_vac",
            classical_force_name = "classical_force_vac",
            quantum_classical_force_name = "quantum_classical_force_vac"
            ),
        # End RK4 integration steps.
        tasks.update_wf_db_rk4,
        partial(tasks.copy_in_state, orig_name="pop_adb_h_q", copy_name="pop_adb_h_q_prev"),
        update_h_q_tot_dcmf,
    ]

    collect_recipe = [
        tasks.update_t,
        tasks.update_dm_db_wf,
        tasks.update_quantum_energy_wf,
        tasks.update_classical_energy,
        partial(tasks.update_classical_energy, z_name="z_vac", classical_energy_name="classical_energy_vac"),
        tasks.collect_t,
        tasks.collect_dm_db,
        tasks.collect_classical_energy,
        partial(tasks.collect_classical_energy, classical_energy_name="classical_energy_vac", classical_energy_output_name="classical_energy_vac"),
        tasks.collect_quantum_energy,
    ]

