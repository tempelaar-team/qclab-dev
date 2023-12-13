import numpy as np
import hamiltonian
from rotation import strot, strot_dag, phrot, phrot_dag


def sample_qp_wigner(q0, p0, beta, freq):
    """
    Sample nuclear phase space variables from Wigner distribution under
    harmonic approximation
    """
    # position
    q = np.random.normal(loc=q0, scale=np.sqrt(1.0/(2.0*freq*np.tanh(beta*freq/2.0))))
    # momentum
    p = np.random.normal(loc=p0, scale=np.sqrt(freq/(2.0*np.tanh(beta*freq/2.0))))
    return q, p


def sample_qp_boltzmann(q0, p0, beta, freq):
    """
    Sample nuclear phase space variables from Boltzmann distribution
    """
    # position
    q = np.random.normal(loc=q0, scale=np.sqrt(1.0/(beta*freq**2)))
    # momentum
    p = np.random.normal(loc=p0, scale=np.sqrt(1.0/beta))
    return q, p


def initialize(sim):  # here we compute Hq, Hqc(q,p), generator of q,p and gradient of Hqc
    """Define system hamiltonian in a general rotated basis"""
    hq = hamiltonian.hamil(sim.sys_hamil)
    if sim.Hsys_rot:
        hq = np.matmul(strot, np.matmul(hq, strot_dag))
    sim.hq = hq

    """Sample q & p and rotate into a desirable basis"""
    ...
    return sim
