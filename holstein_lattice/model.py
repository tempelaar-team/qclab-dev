import numpy as np

def initialize(sim):

    def init_classical():
        q = np.random.normal(loc = 0, scale = np.sqrt(sim.T),size = sim.num_states)
        p = np.random.normal(loc = 0, scale = np.sqrt(sim.T/(sim.w)), size=sim.num_states)
        return q, p
    def h_q():
        out = np.zeros((sim.num_states, sim.num_states), dtype=complex)
        for i in range(sim.num_states-1):
            out[i,i+1] = -sim.j
            out[i+1,i] = -sim.j
        out[0,-1] = -sim.j
        out[-1,0] = -sim.j
        return out

    def h_qc(q,p):
        out = np.zeros((sim.num_states, sim.num_states), dtype=complex)
        out[range(sim.num_states),range(sim.num_states)] = sim.g*np.sqrt(2*(sim.w**3))*q
        return out

    # initialize derivatives of h wrt q and p
    # tensors have dimension # classical osc \times # quantum states \times # quantum states
    dq_mat = np.zeros((sim.num_states, sim.num_states, sim.num_states),dtype=complex)
    dp_mat = np.zeros((sim.num_states, sim.num_states, sim.num_states),dtype=complex)
    for i in range(sim.num_states):
        dq_mat[i,i,i] = sim.g*np.sqrt(2*(sim.w**3))
    dq_shape = np.shape(dq_mat)
    dp_shape = np.shape(dp_mat)
    # position of nonzero matrix elements
    dq_ind = np.where(np.abs(dq_mat) > 1e-18)
    dp_ind = np.where(np.abs(dp_mat) > 1e-18)
    # nonzero matrix elements
    dq_mels = dq_mat[dq_ind]
    dp_mels = dp_mat[dp_ind]
    dq_vars = (dq_shape, dq_ind, dq_mels, dp_shape, dp_ind, dp_mels)

    # equip sim with necessary functions
    sim.init_classical = init_classical
    sim.w_c = sim.w
    sim.h_q = h_q
    sim.h_qc = h_qc
    sim.dq_vars = dq_vars

    return sim




