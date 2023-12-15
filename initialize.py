import numpy as np
import system
from system import *


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
    sim.hsys = Hsys

    """Sample q & p and rotate into a desirable basis"""
    if sim.qp_dist == 'boltz':
        sample_qp = sample_qp_boltzmann
    elif sim.qp_dist == 'wigner':
        sample_qp = sample_qp_wigner
    qp = np.zeros((ntraj, 2, ndyn_phset, nph_per_set))
    for itraj in range(ntraj):
        qp_temp = np.zeros((2, nstate, nph_per_set))
        zn = np.zeros((nstate, nph_per_set), dtype=complex)
        for n in range(nstate):
            for i in range(nph_per_set):
                qp_temp[0, n, i], qp_temp[1, n, i] = sample_qp(q0[i], frq[i])
                zn[n, i] = np.sqrt(frq[i] / 2.0) * (qp_temp[0, n, i] + 1j * qp_temp[1, n, i] / frq[i])

        # Rotate phonon modes
        zalp = np.zeros((nstate, nph_per_set), dtype=complex)
        for i in range(nph_per_set):
            zalp[:, i] = np.matmul(phrot, zn[:, i])

        # Positions and momenta of rotated phonon modes:
        for alp in range(nstate):
            for i in range(nph_per_set):
                qp_temp[0, alp, i] = np.real(zalp[alp, i]) * np.sqrt(2.0 / frq[i])
                qp_temp[1, alp, i] = np.imag(zalp[alp, i]) * np.sqrt(2.0 * frq[i])
        qp[itraj] = np.delete(qp_temp, trunc_modes, axis=1)  # only include selected modes
    sim.qp = qp
    return sim
