from system import *


def sample_qp_wigner(q0, p0, bet, freq):
    """
    Sample nuclear phase space variables from Wigner distribution under
    harmonic approximation
    """
    # position
    q = np.random.normal(loc=q0, scale=np.sqrt(1.0/(2.0*freq*np.tanh(bet*freq/2.0))))
    # momentum
    p = np.random.normal(loc=p0, scale=np.sqrt(freq/(2.0*np.tanh(bet*freq/2.0))))
    return q, p


def sample_qp_boltzmann(q0, p0, bet, freq):
    """
    Sample nuclear phase space variables from the Boltzmann distribution
    """
    # position
    q = np.random.normal(loc=q0, scale=np.sqrt(1.0/(bet*freq**2)))
    # momentum
    p = np.random.normal(loc=p0, scale=np.sqrt(1.0/bet))
    return q, p


sample_qp = {'boltz': sample_qp_boltzmann, 'wigner': sample_qp_wigner}


def initialize(sim):  # here we compute Hq, Hqc(q,p), generator of q,p and gradient of Hqc
    """Define system hamiltonian in a general rotated basis"""
    sim.hsys = Hsys

    """Sample q & p and rotate into a desirable basis"""
    qp = np.zeros((ntraj, 2, ndyn_phset, nph_per_set))
    for itraj in range(ntraj):
        qp_temp = np.zeros((2, nstate, nph_per_set))
        zn = np.zeros((nstate, nph_per_set), dtype=complex)
        for ist in range(nstate):
            for i in range(nph_per_set):
                qp_temp[0, ist, i], qp_temp[1, ist, i] = sample_qp[sim.qp_dist](q0[i], p0[i], beta, frq[i])
                zn[ist, i] = np.sqrt(frq[i] / 2.0) * (qp_temp[0, ist, i] + 1j * qp_temp[1, ist, i] / frq[i])

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
