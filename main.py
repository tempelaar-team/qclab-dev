import sys
import time
import numpy as np
import dynamics
import simulation
import initialize

if __name__ == '__main__':
    start_time = time.time()
    args = sys.argv[-1]
    if not args:
        print('Usage: python main.py [opts] input_file cluster_args')
        sys.exit()
    input_file = args[-2]
    cluster_args = args[-1]
    opt = args[0]
    # initialize simulation object
    sim = simulation.Simulation(input_file)
    # initialize simulation functions
    sim = initialize.initialize(sim)
    # run dynamics
    sim = dynamics.run_dynamics(sim)

    end_time = time.time()
    print('took ', np.round(end_time - start_time, 3), ' seconds.')
    # comment
