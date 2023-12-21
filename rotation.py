import numpy as np
from numba import jit

def rotate(sim):
    """
    Executes a rotation over the classical and quantum dimensions for a system that is linear
    in the classical coordinates.
    :param sim: simulation object
    :return: simulation object in rotated basis
    """
    U_q = sim.U_q() # quantum rotation matrix
    U_c = sim.U_c() # classical rotation matrix

    (dz_shape, dz_ind, dz_mels, dzc_shape, dzc_ind, dzc_mels) = sim.diff_vars

    # rotate h_q()
    h_q_mat = np.copy(np.matmul(np.conjugate(np.transpose(U_q)),np.matmul(sim.h_q(), U_q)))
    def h_q():
        return h_q_mat

    # reconstruct \nabla_{z}H and \nabla_{zc}H tensors
    dz_mat = np.zeros(dz_shape, dtype=complex)
    dzc_mat = np.zeros(dzc_shape, dtype=complex)
    dz_mat[dz_ind] = dz_mels
    dzc_mat[dzc_ind] = dzc_mels
    # transform according to U_c
    dz_mat_trans = np.einsum('ijk,li->ljk',dz_mat, U_c)
    dzc_mat_trans = np.einsum('ijk,li->ljk', dzc_mat, np.conjugate(U_c))
    # transform according to U_q
    dz_mat_trans = np.einsum('lj,ijk->ilk',np.conjugate(np.transpose(U_q)),
                             np.einsum('ijk,kl->ijl',dz_mat_trans, U_q))
    dzc_mat_trans = np.einsum('lj,ijk->ilk',np.conjugate(np.transpose(U_q)),
                             np.einsum('ijk,kl->ijl',dzc_mat_trans, U_q))
    # copy over new tensors
    dz_mat = np.copy(dz_mat_trans)
    dzc_mat = np.copy(dzc_mat_trans)
    # regenerate sparse indices
    dz_shape = np.shape(dz_mat)
    dzc_shape = np.shape(dzc_mat)
    # position of nonzero matrix elements
    dz_ind = np.where(np.abs(dz_mat) > 1e-12)
    dzc_ind = np.where(np.abs(dzc_mat) > 1e-12)
    # nonzero matrix elements
    dz_mels = dz_mat[dz_ind]
    dzc_mels = dzc_mat[dzc_ind]
    # necessary variables for computing expectation values
    diff_vars = (dz_shape, dz_ind, dz_mels, dzc_shape, dzc_ind, dzc_mels)


    # now we can reconstruct h_qc in the rotated basis
    @jit(nopython=True)
    def h_qc(z,zc):
        h_dz = np.ascontiguousarray(np.zeros((dz_shape[1],dz_shape[2]))) + 0.0j
        h_dzc = np.ascontiguousarray(np.zeros((dzc_shape[1],dzc_shape[2]))) + 0.0j
        for i in range(len(dz_mels)):
            h_dz[dz_ind[1][i],dz_ind[2][i]] += dz_mels[i]*z[dz_ind[0][i]]
            h_dzc[dzc_ind[1][i],dzc_ind[2][i]] += dzc_mels[i]*zc[dz_ind[0][i]]
        return h_dz + h_dzc

    def h_c(z,zc):
        return np.real(np.sum(sim.w*z*zc))
    init_classical_old = sim.init_classical
    def init_classical():
        z,zc = init_classical_old()
        zc_out = np.matmul(np.conjugate(U_c),zc)
        z_out = np.matmul(U_c,z)
        return z_out, zc_out

    # equip simulation object with necessary functions
    sim.init_classical = init_classical
    sim.w_c = sim.w
    sim.h_q = h_q
    sim.h_qc = h_qc
    sim.h_c = h_c
    sim.U_c = U_c
    sim.U_q = U_q
    sim.diff_vars = diff_vars
    sim.calc_dir = 'holstein_lattice_g_' + str(sim.g) + '_j_' + str(sim.j) + '_w_' + str(sim.w) + \
                   '_temp_' + str(sim.temp) + '_nstates_' + str(sim.num_states)
    # rotate initial diabatic wavefunction
    sim.psi_db_0 = np.matmul(np.conjugate(np.transpose(U_q)), sim.psi_db_0)

    return sim


