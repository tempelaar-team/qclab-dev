import numpy as np
import constants


def harmonic_oscillator(z, zc, sim):
    """
    Harmonic oscillator Hamiltonian
    :param z: z(t)
    :param zc: conjugate z(t)
    :return: h_c(z,zc) Hamiltonian
    """
    return np.real(np.sum(sim.h * zc * z))

def harmonic_oscillator_dh_c_dz(z, zc, sim):
    """
    Gradient of harmonic oscillator hamiltonian w.r.t z
    :param z: z coordinate
    :param zc: z* coordinate
    :param sim: simulation object
    :return:
    """
    return sim.h*zc

def harmonic_oscillator_dh_c_dzc(z, zc, sim):
    """
    Gradient of harmonic oscillator hamiltonian wrt z*
    :param z: z coordinate
    :param zc: z* coordinate
    :param sim: simulation object
    :return:
    """
    return sim.h*z

def harmonic_oscillator_boltzmann(sim):
    """
    Initialize classical coordiantes according to Boltzmann statistics
    :param sim: simulation object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(w*h/2)*(q + i*(p/((w*h))), z* = sqrt(w*h/2)*(q - i*(p/((w*h)))
    """
    q = np.random.normal(loc=0, scale=np.sqrt(constants.kB_therm*sim.temp/(sim.m*(sim.h**2))), size=sim.num_states)
    p = np.random.normal(loc=0, scale=np.sqrt(sim.m*constants.kB_therm*sim.temp), size=sim.num_states)
    z = np.sqrt(sim.h * sim.m / 2) * (q + 1.0j * (p / (sim.h*sim.m)))
    zc = np.conjugate(z)
    return z, zc