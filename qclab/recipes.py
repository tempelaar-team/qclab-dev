import qclab.ingredients as ingredients
import qclab.auxiliary as auxiliary
import argparse


class MeanFieldDynamicsRecipe:
    def __init__(self, model):
        self.model = model
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
    def defaults(model):
        var_names = list(model.__dict__.keys())
        defaults = {
            'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
            'h_c': auxiliary.harmonic_oscillator_h_c,
            'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
            'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
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
                model.__dict__[name] = defaults[name]
        assert model.num_branches == 1
        return model


class FewestSwitchesSurfaceHoppingDynamicsRecipe:
    def __init__(self, model):
        self.model = model
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

        self.output = [  # ingredients.update_dm_adb_fssh,
            ingredients.update_dm_db_fssh,
            ingredients.update_e_q_fssh,
            ingredients.update_e_c,
        ]
        self.output_names = ['dm_db',
                             'e_q',
                             'e_c',
                             ]

        self.state = argparse.Namespace()
        return

    @staticmethod
    def defaults(model):
        var_names = list(model.__dict__.keys())
        defaults = {
            'hop': auxiliary.harmonic_oscillator_hop,
            'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
            'h_c': auxiliary.harmonic_oscillator_h_c,
            'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
            'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': model.num_states,
            'sh_deterministic': True,
            'gauge_fix': 0,
            'num_classical_coordinates': None
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                model.__dict__[name] = defaults[name]
        if model.sh_deterministic:
            assert model.num_branches == model.num_states
        return model


class CoherentFewestSwitchesSurfaceHoppingDynamicsRecipe:
    def __init__(self, model):
        self.model = model
        self.initialize = [
            ingredients.initialize_random_values,
            ingredients.initialize_wf_db,
            ingredients.initialize_z_coord,
            ingredients.update_h_quantum,
            ingredients.update_eigs,
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
                       ingredients.update_classical_overlap,
                       ingredients.update_dm_db_cfssh,
                       ingredients.update_e_c,
                       ingredients.update_e_q_fssh,
                       ]
        self.output_names = [
            'dm_db',
            'e_q',
            'e_c',
        ]

        self.state = argparse.Namespace()
        return

    @staticmethod
    def defaults(model):
        var_names = list(model.__dict__.keys())
        defaults = {
            'hop': auxiliary.harmonic_oscillator_hop,
            'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
            'h_c': auxiliary.harmonic_oscillator_h_c,
            'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
            'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': model.num_states,
            'sh_deterministic': True,
            'gauge_fix': 0,
            'num_classical_coordinates': None
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                model.__dict__[name] = defaults[name]
        if model.sh_deterministic:
            assert model.num_branches == model.num_states
        return model


class ManyBodyMeanFieldDynamicsRecipe:
    def __init__(self, sim):
        self.sim = sim 
        self.initialize = [
                           ingredients.initialize_wf_db_mb,
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
                       ingredients.update_rdm1,
                       ingredients.update_rdm2,
                       ]
        self.output_names = [
                             'e_c', 
                             'e_q',
                             'rdm1',
                             'rdm2',
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
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp': 1,
            'num_states': 2,
            'num_branches': 1,
            'num_classical_coordinates': None,
        }
        for name in defaults.keys():
            if not (name in list(var_names)):
                sim.__dict__[name] = defaults[name]
        assert sim.num_branches == 1
        return sim