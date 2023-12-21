import numpy as np
class Simulation:
    def __init__(self, input_file):
        # Define default values
        defaults = {
            "dynamics_method": "MF",  # which dynamics method, "MF", "FSSH", "CFSSH"
            "num_procs": 4,  # number of processors to use
            "num_trajs": 4,  # number of trajectories to run
            "tmax": 10, # maximum simulation time
            "dt": 0.1, # timestep of output
            "dt_bath":0.01, # bath timestep
            "model_dir":"./", # model directory
            "pab_cohere":True, # Uses full adiabatic wavefunction to compute hopping probabilities
            "gauge_fix":1 # gauge fixing level 0, 1, 2
        }
        # Read input values from input_file
        input_params = {}  # store them in input_params
        with open(input_file) as file:
            for line in file:
                exec(str(line), input_params)
        inputs = list(input_params)  # inputs is list of keys in input_params
        for key in inputs:  # copy input values into defaults
            defaults[key] = input_params[key]
        # read modified defaults into object
        self.dynamics_method = defaults['dynamics_method']
        self.num_procs = defaults['num_procs']
        self.num_trajs = defaults['num_trajs']
        self.tmax = defaults['tmax']
        self.dt = defaults['dt']
        self.dt_bath = defaults['dt_bath']
        self.model_dir = defaults['model_dir']
        self.pab_cohere = defaults['pab_cohere']
        self.gauge_fix = defaults['gauge_fix']
        self.input_file = input_file
        self.input_params = defaults
class Trajectory:
    def __init__(self, seed, index):
        self.seed = seed  # seed used to initialize random variables
        self.index = index  # index of trajectory
        self.data_dic = {}  # dictionary to store data
    def add_to_dic(self, name, data):
        self.data_dic.__setitem__(name, data)
        return

class Data:
    def __init__(self, filename):
        self.filename = filename
        self.data_dic = {}
        self.index_list = np.array([], dtype=int)
        self.seed_list = np.array([], dtype=int)

    def add_data(self, traj_obj):  # adds data from a traj_obj
        for key, val in traj_obj.data_dic.items():
            if key in self.data_dic:
                self.data_dic[key] = self.data_dic[key] + val
            else:
                self.data_dic[key] = val
        self.seed_list = np.append(self.seed_list, traj_obj.seed)
        self.index_list = np.append(self.index_list, traj_obj.index)
        return

    def sum_data(self, data_obj):  # adds data from a data_obj
        for key, val in data_obj.data_dic.items():
            if key in self.data_dic:
                self.data_dic[key] = self.data_dic[key] + val
            else:
                self.data_dic[key] = val
        self.seed_list = np.concatenate((self.seed_list, data_obj.seed_list))
        self.index_list = np.concatenate((self.index_list, data_obj.index_list))
        return