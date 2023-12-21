import numpy as np

def rotate(sim):
    """
    Executes a rotation over the classical and quantum dimensions for a system that is linear
    in the classical coordinates.
    :param sim: simulation object
    :return: simulation object in rotated basis
    """
    U_q = sim.U_q() # quantum rotation matrix
    U_c = sim.U_c() # classical rotation matrix

    (dq_shape, dq_ind, dq_mels, dp_shape, dp_ind, dp_mels) = sim.dq_vars

    # rotate h_q()
    h_q_mat = np.copy(np.matmul(np.conjugate(np.transpose(U_q)),np.matmul(sim.h_q(), U_q)))
    def h_q():
        return h_q_mat


    # reconstruct \nabla_{q} H and \nabla_{p} H tensor


