import simulation
import numpy as np

def boltzmann_harmonic():
    """
    Sample harmonic nuclear phase space variables from the Boltzmann distribution
    """
    # position
    q = np.random.normal(loc=0, scale=np.sqrt(sim.temp / (sim.w ** 2)), size=sim.ndyn_phsets)
    # momentum
    p = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=sim.ndyn_phsets)
    # z coordinate
    z = np.sqrt(sim.w / 2.0) * (q + 1.0j * p / sim.w)
    zc = np.conj(z)
    return z, zc

def wigner_harmonic():
    """
    Sample harmonic nuclear phase space variables from Wigner distribution under
    harmonic approximation
    """
    # position
    q = np.random.normal(loc=0, scale=np.sqrt(1.0 / (2.0 * sim.w * np.tanh(sim.w / (2.0 * sim.temp)))), size=sim.ndyn_phsets)
    # momentum
    p = np.random.normal(loc=0, scale=np.sqrt(sim.w / (2.0 * np.tanh(sim.w / (2.0 * sim.temp)))), size=sim.ndyn_phsets)
    # z coordinate
    z = np.sqrt(sim.w/2.0) * (q + 1.0j*p/sim.w)
    zc = np.conj(z)
    return z, zc

def boltzmann_quartic(a, b):
    """
    Sample nuclear phase space variables from the Boltzmann distribution of double-well (quartic) potential
    a: coefficient of 4th order term
    b: coefficient of 2nd order term
    """
    'Here going to write Monte Carlo simulator to sample p & q from quartic potential'

    return z, zc

def boltzmann_morse(Ediss):
    """
    Sample nuclear phase space variables from the Boltzmann distribution of double-well (quartic) potential
    Ediss: dissociation energy
    """
    'Here going to write Monte Carlo simulator to sample p & q from Morse potential'

    return z, zc


if sim.potential == 'harmonic':
    if sim.qp_dist == 'boltzmann':
        sim.init_classical = boltzmann_harmonic
    elif sim.qp_dist == 'wigner':
        sim.init_classical = wigner_harmonic
elif sim.potential == 'morse':
    if sim.qp_dist == 'boltzmann':
        sim.init_classical = boltzmann_morse
elif sim.potential == 'double-well':
    if sim.qp_dist == 'boltzmann':
        sim.init_classical = boltzmann_quartic
# there could be more options for potential energy shapes

