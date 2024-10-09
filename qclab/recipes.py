import qclab.ingredients as ingredients
import qclab.auxiliary as auxiliary
import argparse


class MeanFieldDynamicsRecipe:
    def __init__(self, sim):
        self.sim = sim
        self.initialize = [
            ingredients.initialize_wf_db,
            ingredients.initialize_z_coord,
            ingredients.update_h_quantum,
            ingredients.update_quantum_force_wf_db,
        ]
        self.update = [ingredients.update_z_coord_rk4,
                       ingredients.update_wf_db_rk4,
                       ingredients.update_h_quantum,
                       ingredients.update_quantum_force_wf_db,
                       ]
        self.output = [ingredients.update_dm_db_mf,
                       ingredients.update_e_c,
                       ingredients.update_e_q_mf,
                       ]
        self.output_names = ['dm_db',
                             'e_c',
                             'e_q',
                             ]
        self.state = argparse.Namespace()

        return

    @staticmethod
    def defaults(sim):
        var_names = list(sim.__dict__.keys())
        defaults = {
            'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
            'h_c': auxiliary.harmonic_oscillator_h_c,
            'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
            'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
            'h_c_params': sim.h,
            'h_qc_params': None,
            'h_q_params': None,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': 1,
            'num_classical_coordinates': None
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                sim.__dict__[name] = defaults[name]
        assert sim.num_branches == 1
        return sim


class FewestSwitchesSurfaceHoppingDynamicsRecipe:
    def __init__(self, sim):
        self.sim = sim
        self.initialize = [
            ingredients.initialize_random_values,
            ingredients.initialize_wf_db,
            ingredients.initialize_z_coord,
            ingredients.update_h_quantum,
            ingredients.update_eigs,
            ingredients.analytic_gauge_fix_eigs,
            ingredients.update_eigs_previous,
            ingredients.initialize_wf_adb,
            ingredients.initialize_active_surface,
            ingredients.update_quantum_force_act_surf,
            ingredients.initialize_dm_adb_0_fssh,
        ]

        self.update = [ingredients.update_eigs_previous,
                       ingredients.update_z_coord_rk4,
                       ingredients.update_wf_db_eigs,
                       ingredients.update_h_quantum,
                       ingredients.update_eigs,
                       ingredients.gauge_fix_eigs,
                       ingredients.update_active_surface_fssh,
                       ingredients.update_quantum_force_act_surf,
                       ]

        self.output = [ingredients.update_dm_adb_fssh,
                       ingredients.update_dm_db_fssh,
                       ingredients.update_e_c,
                       ingredients.update_e_q_fssh,
                       ]
        self.output_names = ['dm_db',
                             'e_q',
                             'e_c',
                             ]

        self.state = argparse.Namespace()
        return

    @staticmethod
    def defaults(sim):
        var_names = list(sim.__dict__.keys())
        defaults = {
            'hop': auxiliary.harmonic_oscillator_hop,
            'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
            'h_c': auxiliary.harmonic_oscillator_h_c,
            'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
            'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
            'h_c_params': sim.h,
            'h_qc_params': None,
            'h_q_params': None,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': sim.num_states,
            'sh_deterministic': True,
            'gauge_fix': 0,
            'num_classical_coordinates': None
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                sim.__dict__[name] = defaults[name]
        if sim.sh_deterministic:
            assert sim.num_branches == sim.num_states
        return sim


class CoherentFewestSwitchesSurfaceHoppingDynamicsRecipe:
    def __init__(self, sim):
        self.sim = sim
        self.initialize = [
            ingredients.initialize_random_values,
            ingredients.initialize_wf_db,
            ingredients.initialize_z_coord,
            ingredients.update_h_quantum,
            ingredients.update_eigs,  # update_eigs
            ingredients.analytic_gauge_fix_eigs,
            ingredients.update_branch_pair_eigs,
            ingredients.analytic_gauge_fix_branch_pair_eigs,
            ingredients.update_eigs_previous,
            ingredients.update_branch_pair_eigs_previous,
            ingredients.initialize_wf_adb,
            ingredients.initialize_active_surface,
            ingredients.initialize_wf_adb_delta,
            ingredients.update_quantum_force_act_surf,
            ingredients.initialize_dm_adb_0_fssh,
            ingredients.initialize_branch_phase,
        ]

        self.update = [ingredients.update_eigs_previous,
                       ingredients.update_z_coord_rk4,
                       ingredients.update_wf_db_eigs,
                       ingredients.update_wf_db_delta_eigs,
                       ingredients.update_branch_phase,
                       ingredients.update_h_quantum,
                       ingredients.update_eigs,
                       ingredients.gauge_fix_eigs,
                       ingredients.update_active_surface_cfssh,
                       ingredients.update_quantum_force_act_surf,
                       ]

        self.output = [ingredients.update_branch_pair_eigs_previous,
                       ingredients.update_branch_pair_eigs,
                       ingredients.gauge_fix_branch_pair_eigs,
                       # ingredients.update_dm_adb_cfssh,
                       # ingredients.update_dm_db_cfssh,
                       # ingredients.update_e_c,
                       # ingredients.update_e_q,
                       ]
        self.output_names = [  # 'dm_db',
            # 'e_q',
            # 'e_c',
        ]

        self.state = argparse.Namespace()
        return

    def observables_t(self):
        observables_dic = dict()
        state_dic = vars(self.state)
        for key in self.output_names:
            observables_dic[key] = state_dic[key]
        return observables_dic

    @staticmethod
    def defaults(sim):
        var_names = list(sim.__dict__.keys())
        defaults = {
            'hop': auxiliary.harmonic_oscillator_hop,
            'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
            'h_c': auxiliary.harmonic_oscillator_h_c,
            'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
            'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
            'h_c_params': sim.h,
            'h_qc_params': None,
            'h_q_params': None,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': sim.num_states,
            'sh_deterministic': True,
            'gauge_fix': 0,
            'observables': auxiliary.no_observables,
            'num_classical_coordinates': None
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                sim.__dict__[name] = defaults[name]
        if sim.sh_deterministic:
            assert sim.num_branches == sim.num_states
        return sim


class ManyBodyMeanFieldDynamicsRecipe:
    def __init__(self, sim):
        self.sim = sim 
        self.initialize = [
                           ingredients.initialize_wf_db_mb,
                           ingredients.initialize_wf_db_mb_coeffs,
                           ingredients.initialize_z_coord,
                           ingredients.update_h_quantum,
                           ingredients.update_quantum_force_wf_db_mbmf,
                           ]
        self.update = [ingredients.update_z_coord_rk4, 
                       ingredients.update_wf_db_mb_rk4,
                       ingredients.update_h_quantum,
                       ingredients.update_quantum_force_wf_db_mbmf,
                       ]
        self.output = [
                       ingredients.update_e_c, 
                       ingredients.update_e_q_mbmf,
                       ]
        self.output_names = [
                             'e_c', 
                             'e_q',
                             'wf_db_MB',
                             'z_coord',
                             ]
        self.state = argparse.Namespace()
        
        return
    
    def defaults(self, sim):
        var_names = list(sim.__dict__.keys())
        defaults = {
            'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
            'h_c': auxiliary.harmonic_oscillator_h_c,
            'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
            'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
            'h_c_params': sim.h,
            'h_qc_params': None,
            'h_q_params': None,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': 1,
            'num_classical_coordinates': None,
            'delay_time_ind':100
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                sim.__dict__[name] = defaults[name]
        assert sim.num_branches == 1
        return sim


class ManyBodyMeanFieldDynamicsARPESRecipe:
    def __init__(self, sim):
        self.sim = sim 
        self.initialize = [
                           ingredients.initialize_wf_db_mb,
                           ingredients.initialize_wf_db_mb_coeffs,
                           ingredients.initialize_z_coord,
                           ingredients.update_h_quantum,
                           ingredients.update_quantum_force_wf_db_mbmf_arpes,
                           ]
        self.update = [ingredients.update_z_coord_rk4, 
                       ingredients.update_wf_db_mb_rk4,
                       ingredients.update_h_quantum,
                       ingredients.update_quantum_force_wf_db_mbmf_arpes,
                       ]
        self.output = [
                       ingredients.update_e_c, 
                       ingredients.update_e_q_mbmf_arpes,
                       ]
        self.output_names = [
                             'e_c', 
                             'e_q_branch',
                             'e_q'
                             ]
        self.state = argparse.Namespace()
        
        return
    
    def defaults(self, sim):
        var_names = list(sim.__dict__.keys())
        defaults = {
            'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
            'h_c': auxiliary.harmonic_oscillator_h_c,
            'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
            'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
            'h_c_params': sim.h,
            'h_qc_params': None,
            'h_q_params': None,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': 1,
            'num_classical_coordinates': None,
            'delay_time_ind':100
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                sim.__dict__[name] = defaults[name]
        assert sim.num_branches == 1
        return sim