import numpy as np
import qclab.auxilliary as auxilliary


class MeanFieldDynamics:
    def __init__(self, sim):
        self.dm_db_branch = None
        self.dm_db = None
        self.observables_t = None
        self.t_ind = None
        self.quantum_force_zc = None
        self.wf_db = None
        self.h_tot = None
        self.h_q = None
        self.dm_adb_0_traj = None
        self.wf_adb = None
        self.z_coord = None
        self.tdat = None
        self.tdat_output = None
        var_names = list(sim.__dict__.keys())
        defaults = {
            'init_classical': auxilliary.harmonic_oscillator_bolztmann_init_classical,
            'h_c_branch': auxilliary.harmonic_oscillator_h_c_branch,
            'dh_c_dz_branch': auxilliary.harmonic_oscillator_dh_c_dz_branch,
            'dh_c_dzc_branch': auxilliary.harmonic_oscillator_dh_c_dzc_branch,
            'h_c_params': sim.h,
            'h_qc_params': None,
            'h_q_params': None,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': 1,
            'gauge_fix': 0,
            'dmat_const': 0,
            'observables': auxilliary.no_observables,
            'num_classical_coordinates': None
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                sim.__dict__[name] = defaults[name]
        assert sim.num_branches == 1
        return

    def initialize_dynamics(self, sim):
        # initialize time axes 
        self.tdat_output = np.arange(0, sim.tmax + sim.dt_output, sim.dt_output)
        self.tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
        # initialize variables describing the state of the system
        self.z_coord = np.zeros((sim.num_trajs, sim.num_branches, sim.num_classical_coordinates), dtype=complex)
        # load initial values of the z coordinate 
        for traj_n in range(sim.num_trajs):
            self.z_coord[traj_n, :, :] = sim.init_classical(sim, sim.seeds[
                traj_n])  # init_classical could arguablty be in init_state
        self.z_coord = self.z_coord.reshape(sim.num_trajs * sim.num_branches, sim.num_classical_coordinates)
        # load initial values of the wavefunction
        self.wf_db = (np.zeros((sim.num_trajs * sim.num_branches, sim.num_states), dtype=complex)
                      + sim.wf_db[np.newaxis, :])
        # initialize gradients (Hamiltonian and quantum forces)
        self.h_tot = sim.h_q(sim.h_q_params)[np.newaxis, :, :] + sim.h_qc_branch(sim.h_qc_params, self.z_coord)
        self.quantum_force_zc = auxilliary.quantum_force_branch(self.wf_db, None, self.z_coord, sim)
        return

    def propagate_classical_subsystem(self, sim):
        self.z_coord = auxilliary.rk4_c(self.z_coord, self.quantum_force_zc, sim.dt, sim)
        return

    def propagate_quantum_subsystem(self, sim):
        self.wf_db = auxilliary.rk4_q_branch(self.h_tot, self.wf_db, sim.dt)
        return

    def update_state(self, sim):
        self.quantum_force_zc = auxilliary.quantum_force_branch(self.wf_db, None, self.z_coord, sim)
        self.h_tot = sim.h_q(sim.h_q_params)[np.newaxis, :, :] + sim.h_qc_branch(sim.h_qc_params, self.z_coord)
        return

    def calculate_observables(self, sim):
        self.dm_db = np.einsum('ni,nk->ik', self.wf_db, np.conj(self.wf_db)) / sim.num_branches
        self.observables_t = sim.observables(sim, self)
        self.observables_t['e_q'] = np.real(
            np.einsum('ni,nij,nj', np.conjugate(self.wf_db), self.h_tot, self.wf_db)) / sim.num_branches
        self.observables_t['e_c'] = np.sum(sim.h_c_branch(sim.h_c_params, self.z_coord)) / sim.num_branches
        self.observables_t['dm_db'] = self.dm_db
        return
