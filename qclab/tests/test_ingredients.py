import pytest 
import qclab.ingredients as ingredients
from qclab import Simulation, VectorObject, Constants, Model, Algorithm
import numpy as np
from tqdm import tqdm

def test_numerical_boltzmann_init_classical():
    sim = Simulation()
    sim.model = Model()
    batch_size = 10000
    sim.settings.batch_size = batch_size
    parameters = VectorObject(size=batch_size, vectorized=True)
    parameters.index = np.arange(batch_size)
    parameters.make_consistent()
    seeds = np.arange(batch_size)
    parameters.z_coord = np.zeros((batch_size))
    potential = 0
    kinetic = 0
    kinetic_list = np.zeros((len(seeds)))
    for seed, n in tqdm(enumerate(seeds)):
        z_coord = ingredients.harmonic_oscillator_boltzmann_init_classical(sim.model, sim.model.constants, parameters, seed=seed)
        kinetic += sim.model.h_c(sim.model.constants, parameters, z_coord = 1.0j*np.imag(z_coord))
        potential += sim.model.h_c(sim.model.constants, parameters, z_coord = np.real(z_coord))
        kinetic_list[n] = sim.model.h_c(sim.model.constants, parameters, z_coord = 1.0j*np.imag(z_coord))

    print('kinetic: ', kinetic/batch_size)
    print('potential: ', potential/batch_size)
    import matplotlib.pyplot as plt
    plt.hist(kinetic_list,bins=100)


    sim = Simulation()
    sim.model = Model()
    batch_size = 10000
    sim.settings.batch_size = batch_size
    parameters = VectorObject(batch_size)
    parameters = VectorObject(size=batch_size, vectorized=True)
    parameters.index = np.arange(batch_size)
    parameters.make_consistent()
    seeds = np.arange(batch_size)
    potential = 0
    kinetic = 0
    kinetic_list = np.zeros((len(seeds)))
    for seed, n in tqdm(enumerate(seeds)):
        z_coord = ingredients.numerical_boltzmann_init_classical(sim.model, sim.model.constants, parameters, seed=seed)
        kinetic += sim.model.h_c(sim.model.constants, parameters, z_coord = 1.0j*np.imag(z_coord))
        potential += sim.model.h_c(sim.model.constants, parameters, z_coord = np.real(z_coord))
        kinetic_list[n] = sim.model.h_c(sim.model.constants, parameters, z_coord = 1.0j*np.imag(z_coord))


    print('kinetic: ', kinetic/batch_size)
    print('potential: ', potential/batch_size)
    plt.hist(kinetic_list,bins=100)
    plt.show()

    



if __name__ == "__main__":
    #pytest.main()
    test_numerical_boltzmann_init_classical()