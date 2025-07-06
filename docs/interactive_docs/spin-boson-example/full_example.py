
import numpy as np
import matplotlib.pyplot as plt
from qc_lab import Simulation
from qc_lab.models import SpinBoson
from qc_lab.algorithms import MeanField
from qc_lab.dynamics import serial_driver


# Initialize the simulation object.
sim = Simulation()
# Equip it with a SpinBoson model object.
sim.model = SpinBoson()
# Attach the MeanField algorithm.
sim.algorithm = MeanField()
# Initialize the diabatic wavefunction. 
# Here, the first state is the upper state and the second is the lower state.
sim.state.wf_db = np.array([1, 0], dtype=complex)
# Run the simulation.
data = serial_driver(sim)

# Pull out the time.
t = data.data_dict["t"]
# Get populations from the diagonal of the density matrix.
populations = np.real(np.einsum("tii->ti", data.data_dict["dm_db"]))
# Get the classical and quantum energy.
classical_energy = data.data_dict["classical_energy"]
quantum_energy = data.data_dict["quantum_energy"]

plt.plot(t, populations[:, 0], label="upper state")
plt.plot(t, populations[:, 1], label="lower state")
plt.xlabel("time")
plt.ylabel("population")
plt.legend()
#plt.savefig("populations.png", dpi=150)
plt.close()

plt.plot(t, classical_energy - classical_energy[0], label="classical energy")
plt.plot(t, quantum_energy - quantum_energy[0], label="quantum energy")
plt.plot(
    t,
    classical_energy + quantum_energy - classical_energy[0] - quantum_energy[0],
    label="total energy",
)
plt.xlabel("time")
plt.ylabel("energy")
plt.legend()
#plt.savefig("energies.png", dpi=150)
plt.close()

from qc_lab.algorithms import FewestSwitchesSurfaceHopping

sim.algorithm = FewestSwitchesSurfaceHopping()
data_fssh = serial_driver(sim)

plt.plot(data.data_dict["t"], np.real(data.data_dict["dm_db"][:,0,0]), label='MF')
plt.plot(data_fssh.data_dict["t"], np.real(data_fssh.data_dict["dm_db"][:,0,0]), label='FSSH')
plt.xlabel('Time')
plt.ylabel('Excited state population')
plt.legend()
plt.show()

sim.settings.num_trajs = 1000
# you can also change the batch size if you want to run more trajectories in each batch.
sim.settings.batch_size = 1000
data_fssh_1000 = serial_driver(sim)


plt.plot(data.data_dict["t"], np.real(data.data_dict["dm_db"][:,0,0]), label='MF')
plt.plot(data_fssh.data_dict["t"], np.real(data_fssh.data_dict["dm_db"][:,0,0]), label='FSSH')
plt.plot(data_fssh_1000.data_dict["t"], np.real(data_fssh_1000.data_dict["dm_db"][:,0,0]), label='FSSH, num_trajs=1000')
plt.xlabel('Time')
plt.ylabel('Excited state population')
plt.legend()
#plt.savefig('fssh_numtrajs.png')
plt.show()


from qc_lab.dynamics import serial_driver, parallel_driver_multiprocessing
import time

sim.settings.num_trajs = 1000
sim.settings.batch_size = 125 # split them up into batches of 125

st = time.time()
data_parallel = parallel_driver_multiprocessing(sim)
et = time.time()
print(f"Parallel driver took {et-st:.2f} seconds to run.")
st = time.time()
data_serial = serial_driver(sim)
et = time.time()
print(f"Serial driver took {et-st:.2f} seconds to run.")



sim.model.constants.l_reorg = 0.05

# Now let's run the simulation again
data_fssh_1000_05 = parallel_driver_multiprocessing(sim)

plt.plot(data_fssh_1000.data_dict["t"], np.real(data_fssh_1000.data_dict["dm_db"][:,0,0]), label=r'$\lambda = 0.005$')
plt.plot(data_fssh_1000_05.data_dict["t"], np.real(data_fssh_1000_05.data_dict["dm_db"][:,0,0]), label=r'$\lambda = 0.05$')
plt.xlabel('Time')
plt.ylabel('Excited state population')
#plt.savefig('fssh_lreorg.png')
plt.legend()
plt.show()


def update_z_reverse_frustrated_fssh(algorithm, sim, parameters, state):
    """
    Reverse the velocities of frustrated trajectories in the FSSH algorithm.
    """
    # get the indices of trajectories that were frustrated
    # (i.e., did not successfully hop but were eligible to hop)
    frustrated_indices = state.hop_ind[~state.hop_successful]
    # reverse the velocities for these indices, in the complex calssical coordinate
    # formalism, this means conjugating the z coordiante.
    state.z[frustrated_indices] = state.z[frustrated_indices].conj()
    return parameters, state

from qc_lab.algorithms import FewestSwitchesSurfaceHopping

# Create an instance of the FSSH algorithm
fssh_algorithm = FewestSwitchesSurfaceHopping()

# Print the update recipe to see where to insert our task
for ind, task in enumerate(fssh_algorithm.update_recipe):
    print(f"Task #{ind}", task)

# Insert the new task into the update recipe
fssh_algorithm.update_recipe.insert(10, update_z_reverse_frustrated_fssh)
# Now we can verify we put it in the right place by printing the update recipe again
for ind, task in enumerate(fssh_algorithm.update_recipe):
    print(f"Task #{ind}", task)


from qc_lab import Simulation # import simulation class
from qc_lab.models import SpinBoson # import model class
from qc_lab.dynamics import parallel_driver_multiprocessing

# Create an instance of the original FSSH algorithm
original_fssh_algorithm = FewestSwitchesSurfaceHopping()


sim = Simulation()

sim.settings.num_trajs = 4000
sim.settings.batch_size = 1000
sim.settings.tmax = 30
sim.settings.dt_update = 0.01

sim.model = SpinBoson({
    'V':0.5,
    'E':0.5,
    'A':100,
    'W':0.1,
    'l_reorg':0.1,
    'boson_mass':1.0,
    'kBT':1.0,

})
sim.state.wf_db= np.array([1,0], dtype=complex)
# Run the simulation with the original FSSH algorithm
sim.algorithm = original_fssh_algorithm
data_original = parallel_driver_multiprocessing(sim)

# Now run the simulation with the modified FSSH algorithm
sim.algorithm = fssh_algorithm
data_modified = parallel_driver_multiprocessing(sim)

t_original = data_original.data_dict['t']
pops_original = np.real(np.einsum('tii->ti',data_original.data_dict['dm_db']))
t_modified = data_modified.data_dict['t']
pops_modified = np.real(np.einsum('tii->ti',data_modified.data_dict['dm_db']))
plt.plot(t_original, pops_original, label='Original FSSH')
plt.plot(t_modified, pops_modified, label='Modified FSSH')
plt.xlabel('Time')
plt.ylabel('Diabatic populations')
plt.legend()
#plt.savefig('modified_fssh_populations.png')
plt.show()