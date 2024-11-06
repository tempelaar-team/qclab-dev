from numba import njit
import numpy as np
import dill as pickle


############################################################
#                    CLASSICAL EVOLUTION                   #
############################################################


def rk4_c(state, model, params, z_coord, qfzc, dt):
    """ 4-th order Runge-Kutta integrator for the z_coord coordinate with force qfzc"""
    k1 = -1.0j * (model.dh_c_dzc(state, model, params, z_coord) + qfzc)
    k2 = -1.0j * (model.dh_c_dzc(state, model, params, z_coord + 0.5 * dt * k1) + qfzc)
    k3 = -1.0j * (model.dh_c_dzc(state, model, params, z_coord + 0.5 * dt * k2) + qfzc)
    k4 = -1.0j * (model.dh_c_dzc(state, model, params, z_coord + dt * k3) + qfzc)
    z_coord = z_coord + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z_coord

############################################################
#                   BASIS TRANSFORMATIONS                  #
############################################################

def vec_adb_to_db(vec_adb, eigvecs):
    return np.einsum('...ij,...j->...i', eigvecs, vec_adb, optimize='greedy')


def vec_db_to_adb(vec_db, eigvecs):
    return np.einsum('...ji,...j->...i', np.conj(eigvecs), vec_db, optimize='greedy')


def mat_adb_to_db(mat_adb, eigvecs):
    return np.einsum('...ni,...ij,...mj->...nm', eigvecs, mat_adb, np.conj(eigvecs), optimize='greedy')


def mat_db_to_adb(mat_db, eigvecs):
    return np.einsum('...ni,...nm,...mj->...ij', np.conj(eigvecs), mat_db, eigvecs, optimize='greedy')


############################################################
#                     QUANTUM EVOLUTION                    #
############################################################


def evolve_wf_eigs(wf_db, eigvals, eigvecs, dt):
    # construct eigenvalue exponential
    evals_exp_branch = np.exp(-1.0j * eigvals * dt)
    # transform wavefunctions to adiabatic basis
    # wf_adb = np.copy(psi_db_to_adb_branch(wf_db, eigvecs)) # TODO remove this comment
    wf_adb = np.copy(vec_db_to_adb(wf_db, eigvecs))
    # multiply by propagator
    wf_adb = np.copy(evals_exp_branch * wf_adb)
    # transform back to diabatic basis
    # wf_db = np.copy(psi_adb_to_db_branch(wf_adb, eigvecs)) # TODO remove this comment
    wf_db = np.copy(vec_adb_to_db(wf_adb, eigvecs))
    return wf_db, wf_adb


def rk4_q(h, psi, dt):
    # h and psi have to have shapes ...ij,...j
    k1 = (-1j * np.einsum('...ij,...j->...i', h, np.ascontiguousarray(psi), optimize='greedy'))
    k2 = (-1j * np.einsum('...ij,...j->...i', h, np.ascontiguousarray(psi) + 0.5 * dt * k1, optimize='greedy'))
    k3 = (-1j * np.einsum('...ij,...j->...i', h, np.ascontiguousarray(psi) + 0.5 * dt * k2, optimize='greedy'))
    k4 = (-1j * np.einsum('...ij,...j->...i', h, np.ascontiguousarray(psi) + dt * k3, optimize='greedy'))
    psi = psi + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi


############################################################
#                       GAUGE FIXING                      #
############################################################


def get_der_couple_phase(state, model, params, z_coord, evals, evecs): # TODO update usage of this function in ingredients
    """
    Computes the diagonal gauge transformation G such that (VG)^{dagger}\nabla(VG) is real-valued. :param evals:
    eigenvalues :param evecs: eigenvectors (V) :param diff_vars: sparse matrix variables of \nabla_{z} H and \nabla_{
    zc} H (stored in model.dq_vars) :return: der_couple_q_phase (diag(G^{dagger}) calculated using d_{ab}^{q}),
    der_couple_p_phase ( diag(G^{dagger}) calculated using d_{ab}^{p})
    """
    # TODO fix and validate this
    der_couple_q_phase = np.ones(len(evals), dtype=complex)
    der_couple_p_phase = np.ones(len(evals), dtype=complex)
    for i in range(len(evals) - 1):
        j = i + 1
        evec_i = evecs[:, i]
        evec_j = evecs[:, j]
        eval_i = evals[i]
        eval_j = evals[j]
        ev_diff = eval_j - eval_i
        plus = 0
        if np.abs(ev_diff) < 1e-14:
            plus = 1
            print('Warning: Degenerate eigenvalues')
        der_couple_z = model.dh_qc_dz(state, model, params, z_coord, evec_i, evec_j) / (ev_diff + plus)
        der_couple_zc = model.dh_qc_dzc(state, model, params, z_coord, evec_i, evec_j) / (ev_diff + plus)
        # convert to q/p nonadiabatic couplings
        der_couple_q = np.sqrt(model.pq_weight * model.mass / 2) * (der_couple_z + der_couple_zc)
        der_couple_p = np.sqrt(1 / (2 * model.pq_weight * model.mass)) * 1.0j * (der_couple_z - der_couple_zc)
        der_couple_q_angle = np.angle(der_couple_q[np.argmax(np.abs(der_couple_q))])
        der_couple_p_angle = np.angle(der_couple_p[np.argmax(np.abs(der_couple_p))])
        if np.max(np.abs(der_couple_q)) < 1e-14:
            der_couple_q_angle = 0
        if np.max(np.abs(der_couple_p)) < 1e-14:
            der_couple_p_angle = 0
        der_couple_q_phase[i + 1:] = np.exp(1.0j * der_couple_q_angle) * der_couple_q_phase[i + 1:]
        der_couple_p_phase[i + 1:] = np.exp(1.0j * der_couple_p_angle) * der_couple_p_phase[i + 1:]
    return der_couple_q_phase, der_couple_p_phase


@njit
def sign_adjust_branch_0(evecs_branch, evecs_branch_previous, phase_out):
    # signs = np.sign(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch))
    signs = np.sign(np.sum(np.conjugate(evecs_branch_previous) * evecs_branch, axis=-2))
    # evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,signs)
    #evecs_branch = evecs_branch * (signs.reshape(len(signs), 1, len(signs[0])))
    evecs_branch = evecs_branch * signs[..., np.newaxis, :]
    phase_out = phase_out * signs
    return evecs_branch, phase_out


@njit
def sign_adjust_branch_1(evecs_branch, evecs_branch_previous, phase_out):  # make name more descriptive
    # phases = np.exp(-1.0j*np.angle(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch)))
    phases = np.exp(-1.0j * np.angle(np.sum(np.conjugate(evecs_branch_previous) * evecs_branch, axis=-2)))
    # evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,phases)
    #evecs_branch = evecs_branch * phases.reshape(len(phases), 1, len(phases[0]))
    evecs_branch = evecs_branch * phases[..., np.newaxis, :]
    phase_out = phase_out * phases
    return evecs_branch, phase_out


def sign_adjust_branch(state, model, params, z_coord, evecs_branch, evecs_branch_previous, evals_branch):
    # commented out einsum terms found to be slower
    phase_out = np.ones(np.shape(evals_branch), dtype=complex) # TODO this is wrong!!
    if params.gauge_fix >= 1:
        evecs_branch, phase_out = sign_adjust_branch_1(evecs_branch, evecs_branch_previous, phase_out)
    if params.gauge_fix >= 2:
        der_couple_phase_mat = np.ones((len(evecs_branch), len(evecs_branch)), dtype=complex)
        for i in range(params.batch_size):
            der_couple_q_phase_list, der_couple_p_phase_list = (
                get_der_couple_phase(state, z_coord[i], evals_branch[i], evecs_branch[i]))
            der_couple_phase_list = np.conjugate(der_couple_q_phase_list)
            der_couple_phase_mat[i] = der_couple_phase_list
            phase_out[i] *= der_couple_phase_list
        #    evecs_branch[i] = np.einsum('jk,k->jk',evecs_branch[i],der_couple_phase_list)
        evecs_branch = np.einsum('ijk,ik->ijk', evecs_branch, der_couple_phase_mat, optimize='greedy')
    if params.gauge_fix >= 0:
        evecs_branch, phase_out = sign_adjust_branch_0(evecs_branch, evecs_branch_previous, phase_out)
    return evecs_branch, phase_out


def sign_adjust_branch_pair_eigs(state, model, params, z_coord, eigvecs_branch_pair, eigvals_branch_pair, eigvecs_branch_pair_previous):
    for i in range(params.num_branches):
        for j in range(i + 1, params.num_branches):
            z_coord_ij = np.array([(z_coord[i] + z_coord[j]) / 2])
            eigvecs_branch_pair[i, j], _ = sign_adjust_branch(state, model, params, z_coord_ij, eigvecs_branch_pair[i, j],
                                                            eigvecs_branch_pair_previous[i, j],
                                                            eigvals_branch_pair[i, j])
            eigvecs_branch_pair[j, i] = eigvecs_branch_pair[i, j]
    return eigvals_branch_pair, eigvecs_branch_pair


############################################################
#                       MISCELLANEOUS                      #
############################################################





def get_classical_overlap(state, model, params, z_coord):
    out_mat = np.zeros((len(z_coord), len(z_coord)))
    zc_coord = np.conjugate(z_coord)
    q_coord = (1 / np.sqrt(2 * model.mass * model.pq_weight)) * (z_coord + zc_coord)
    p_coord = -1.0j * np.sqrt(model.pq_weight * model.mass / 2) * (z_coord - zc_coord)
    for i in range(len(z_coord)):
        for j in range(len(z_coord)):
            out_mat[i, j] = np.exp(-(1 / 2) * np.sum(np.abs((p_coord[i] - p_coord[j]) * (q_coord[i] - q_coord[j]))))
    return out_mat


@njit
def nan_num(num):
    """
    converts nan to a large or small number using numba acceleration
    """
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num


# vectorized form of nan_num
nan_num_vec = np.vectorize(nan_num)


############################################################
#        DEFAULT FUNCTIONS FOR HARMONIC OSCILLATOR        #
############################################################


def harmonic_oscillator_hop(state, model, params, z_coord, delta_z_coord, ev_diff):
    # TODO change z variable name to z_coord
    """
    Carries out the hopping procedure for a harmonic oscillator Hamiltonian, defined on a single branch only. 
    :param z: z coordinate
    :param delta_z: rescaling direction
    :param ev_diff: change in quantum energy following a hop: e_{final} - e_{initial}
    :param model: model object
    :return z, hopped: updated z coordinate and boolean indicating if a hop has or has not occured
    """
    hopped = False
    delta_zc_coord = np.conj(delta_z_coord)
    zc = np.conj(z_coord)
    akj_z = np.real(np.sum(model.pq_weight * delta_zc_coord * delta_z_coord))
    bkj_z = np.real(np.sum(1j * model.pq_weight * (zc * delta_z_coord - z_coord * delta_zc_coord)))
    ckj_z = ev_diff
    disc = bkj_z ** 2 - 4 * akj_z * ckj_z
    if disc >= 0:
        if bkj_z < 0:
            gamma = bkj_z + np.sqrt(disc)
        else:
            gamma = bkj_z - np.sqrt(disc)
        if akj_z == 0:
            gamma = 0
        else:
            gamma = gamma / (2 * akj_z)
        # adjust classical coordinate
        z_coord = z_coord - 1.0j * np.real(gamma) * delta_z_coord
        hopped = True
    return z_coord, hopped


def harmonic_oscillator_boltzmann_init_classical(model, seed=None):
    """
    Initialize classical coordiantes according to Boltzmann statistics
    :param model: model object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(m*pq_weight/2)*(q + i*(p/((m*pq_weight))), z* = sqrt(m*pq_weight/2)*(q - i*(p/((m*pq_weight)))
    """
    np.random.seed(seed)
    q = np.random.normal(loc=0, scale=np.sqrt(model.temp / (model.mass * (model.pq_weight ** 2))),
                         size=model.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=np.sqrt(model.temp), size=model.num_classical_coordinates)
    z = np.sqrt(model.pq_weight * model.mass / 2) * (q + 1.0j * (p / (model.pq_weight * model.mass)))
    return z


def harmonic_oscillator_wigner_init_classical(model, seed=None):
    """
    Initialize classical coordiantes according to Wigner distribution of the ground state of a harmonic oscillator
    :param model: model object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(m*pq_weight/2)*(q + i*(p/((m*pq_weight))), z* = sqrt(m*pq_weight/2)*(q - i*(p/((m*pq_weight)))
    """
    np.random.seed(seed)
    q = np.random.normal(loc=0, scale=np.sqrt(1 / (2 * model.pq_weight * model.mass)),
                         size=model.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=np.sqrt((model.mass * model.pq_weight) / 2), size=model.num_classical_coordinates)
    z = np.sqrt(model.pq_weight * model.mass / 2) * (q + 1.0j * (p / (model.pq_weight * model.mass)))
    return z


def harmonic_oscillator_focused_init_classical(model, seed=None):
    """
    Initialize classical coordiantes according to focused sampling of the 
    Wigner distribution of the ground state of a harmonic oscillator
    :param model: model object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(m*pq_weight/2)*(q + i*(p/((m*pq_weight))), z* = sqrt(m*pq_weight/2)*(q - i*(p/((m*pq_weight)))
    """
    np.random.seed()
    phase = np.random.rand(model.num_classical_coordinates) * 2 * np.pi
    q = np.sqrt(1 / (model.pq_weight * model.mass)) * np.cos(phase)
    p = np.sqrt((model.mass * model.pq_weight)) * np.sin(phase)
    z = np.sqrt(model.pq_weight * model.mass / 2) * (q + 1.0j * (p / (model.pq_weight * model.mass)))
    return z


def harmonic_oscillator_h_c(state, model, params, z_coord):
    return np.real(np.sum(model.w[..., :] * np.conj(state.z_coord) * state.z_coord, axis=(-1)))


def harmonic_oscillator_dh_c_dz(state, model, params, z_coord):
    """
    Gradient of harmonic oscillator hamiltonian wrt z_coord
    :param z_coord: z coordinate in each branch
    :return:
    """
    return model.w[np.newaxis, np.newaxis, :] * np.conj(z_coord)


def harmonic_oscillator_dh_c_dzc(state, model, params, z_coord):
    """
    Gradient of harmonic oscillator hamiltonian wrt zc_branch
    :param z_coord: z coordinate in each branch
    :return:
    """
    return model.w[np.newaxis, np.newaxis, :] * z_coord



############################################################
#                      FILE MANAGEMENT                     #
############################################################


def save_pickle(obj, filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()
    return


def load_pickle(filename):
    file = open(filename, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


############################################################
#                    DYNAMICS FUNCTIONS                   #
############################################################


def initialize_timesteps(params):
    params.tmax_n = np.round(params.tmax / params.dt, 1).astype(int)
    params.dt_output_n = np.round(params.dt_output / params.dt, 1).astype(int)
    params.tdat = np.arange(0, params.tmax_n + 1, 1) * params.dt
    params.tdat_n = np.arange(0, params.tmax_n + 1, 1)
    params.tdat_output = np.arange(0, params.tmax_n + 1, params.dt_output_n) * params.dt
    params.tdat_ouput_n = np.arange(0, params.tmax_n + 1, params.dt_output_n)
    return params


def evaluate_observables_t(recipe):
    observables_dic = dict()
    state_dic = vars(recipe.state)
    for key in recipe.output_names:
        observables_dic[key] = state_dic[key]
    return observables_dic


def generate_seeds(params, data):
    if len(data.seed_list) > 0:
        new_seeds = np.max(data.seed_list) + np.arange(params.num_trajs, dtype=int) + 1
    else:
        new_seeds = np.arange(params.num_trajs, dtype=int)
    return new_seeds



class Trajectory:
    def __init__(self):
        self.data_dic = {}  # dictionary to store data
    def add_to_dic(self, name, data):
        self.data_dic.__setitem__(name, data)
        return
    def new_observable(self, name, shape, type):
        self.data_dic[name] = np.zeros(shape, dtype=type)
        return
    def add_observable_dict(self, ind, dic):
        for key in dic.keys():
            if key in self.data_dic.keys():
                self.data_dic[key][ind] = self.data_dic[key][ind] + dic[key]
        return
    
class Data:
    def __init__(self):
        self.data_dic = {}
    def add_data(self, traj_obj):  # adds data from a traj_obj
        for key, val in traj_obj.data_dic.items():
            if key in self.data_dic:
                self.data_dic[key] = self.data_dic[key] + val
            else:
                self.data_dic[key] = val
        self.data_dic['seeds'] = np.append(self.data_dic['seeds'], traj_obj.seeds)
        return
    def sum_data(self, data_obj):  # adds data from a data_obj
        for key, val in data_obj.data_dic.items():
            if key in self.data_dic:
                self.data_dic[key] = self.data_dic[key] + val
            else:
                self.data_dic[key] = val
        self.data_dic['seeds'] = np.concatenate((self.data_dic['seeds'], data_obj.data_dic['seeds']))
        return