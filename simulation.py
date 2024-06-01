import numpy as np
from argparse import Namespace


class Simulation:
    def __init__(self):

        self.params = dict(\
            dynamics_method = None,
            tmax = None,
            dt = None,
            dt_bath = None,
            calc_mf_obs = None,
            calc_fssh_obs = None,
            calc_cfssh_obs = None,
            state_vars_list = None,
            num_branches = None,
            dmat_const = None,
            pab_cohere = None,
            gauge_fix = None,
            cfssh_branch_pair_update = None,
            sh_deterministic = None,
            # z coordinate
            m = None,
            h = None,
            init_classical_branch = None,
            # Hamiltonian
            h_q_branch = None,
            h_qc_branch = None,
            h_c_branch = None,
            # Gradients
            dh_qc_dz_branch = None,
            dh_qc_dzc_branch = None,
            dh_c_dz_branch = None,
            dh_c_dzc_branch = None,
            # Observables
            mf_observables = None,
            fssh_observables = None,
            cfssh_observables = None,
            # initial wavefunction
            psi_db_0 = None,
            )
        
        self.ns = Namespace(**self.params)

    def load_params(self,input_params:dict):
        for key in input_params.keys():
            self.params[key] = input_params[key]
        self.ns = Namespace(**self.params)
    


class Trajectory:
    def __init__(self, seed):
        self.seed = seed  # seed used to initialize random variables
        self.data_dic = {}  # dictionary to store data
    def new_observable(self, name, shape, type):
        self.data_dic[name] = np.zeros(shape, dtype=type)
        return
    def add_observable_dict(self, ind, dic):
        for key in dic.keys():
            if key in self.data_dic.keys():
                self.data_dic[key][ind] += dic[key]
        return