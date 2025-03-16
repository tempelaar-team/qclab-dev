import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from qc_lab import Simulation 
from qc_lab.models import SpinBoson
from qc_lab.algorithms import MeanField 
from qc_lab.dynamics import parallel_driver_mpi

# instantiate a simulation
sim = Simulation()

# change settings to customize simulation
sim.settings.num_trajs = 400
sim.settings.batch_size = 100
sim.settings.tmax = 10
sim.settings.dt = 0.001

# instantiate a model 
sim.model = SpinBoson()
# instantiate an algorithm 
sim.algorithm = MeanField()
# define an initial diabatic wavefunction 
sim.state.wf_db = np.array([1, 0], dtype=complex)

data = parallel_driver_mpi(sim)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    print(data.data_dic['seed'])
