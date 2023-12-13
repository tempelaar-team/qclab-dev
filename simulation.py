class Simulation:
    def __init__(self, input_file):
        # Define default values
        defaults = {
            "dynamics_method": "MF",  # which dynamics method, "MF", "FSSH", "CFSSH"
            "num_procs": 4,  # number of processors to use
            "num_trajs": 4,  # number of trajectories to run
            "sys_hamil": "holstein",  # system hamiltonian
            "hsys_rot": False,  # rotation of system hamiltonian. Default: no rotation
            "phonon_rot": False,  # rotation of phonon coordinates. Default: no rotation
            "qp_dist": "boltz"  # phonon coordinate sampling function. Default: thermal Boltzmann distribution
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
        self.sys_hamil = defaults['sys_hamil']
        self.hsys_rot = defaults['hsys_rot']
        self.phonon_rot = defaults['phonon_rot']
        self.qp_dist = defaults['qp_dist']
