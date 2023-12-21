import numpy as np

def initialize(sim):

    def init_classical():
        q = np.random.normal(loc = 0, scale = np.sqrt(sim.w/sim.T),size = sim.num_states)
        p = np.random.normal(loc = 0, scale = np.sqrt(1/(sim.w*sim.T)), size=sim.num_states)
        return
    def H_q():
        out = np.zeros((sim.num_states, sim.num_states), dtype=complex)
        for i in range(sim.num_states-1):
            out[i,i+1] = -sim.j
            out[i+1,i] = -sim.j
        out[0,-1] = -sim.j
        out[-1,0] = -sim.j
        return out

