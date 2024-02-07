import numpy as np

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