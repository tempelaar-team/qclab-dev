from system import *
import spin_boson as sb


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


def sample_debye(wc, lam):
    """
    Debye spectral density sampling function
    'wc': cutoff frequency
    'lam': reorganization energy
    """
    frq, g = [], []
    for j in range(nph_per_set):
        frq.append(wc * np.tan((pi/(2*nph_per_set)) * (j + 1 - 0.5)))
    frq = np.array(frq)
    for j in range(nph_per_set):
        g.append(frq[j] * np.sqrt(2*lam / nph_per_set))
    g = np.array(g)
    return frq, g


sample_qp = {'boltz': sample_qp_boltzmann, 'wigner': sample_qp_wigner}
sample_w_g = {'debye': sample_debye}


def truncate_phrot(u):
    """
    Truncate phonon rotation matrix according to the specified modes to be retained in dynamics.
    "u": time-independent phonon-mode rotation matrix
    """
    uminor = np.delete(u, trunc_modes, axis=0)
    uminor_d = np.conj(uminor.T)
    return uminor, uminor_d


def truncate_strot(v):
    """
    Truncate site/state rotation matrix according to the specified number
    of quantum sites/states to be retained in dynamics.
    "Vel": time-independent electronic Hamiltonian rotation matrix
    """
    vminor = np.delete(v, trunc_states, axis=0)
    vminor_d = np.conj(vminor.T)
    return vminor, vminor_d


def truncate_coef(vec):
    """
    Truncate quantum coefficients according to the specified number
    of sites/states to be retained in dynamics
    "vec": Initial state expansion coefs in a general rotated basis
    """
    vecminor = np.delete(vec, trunc_states)
    return vecminor


def initialize(sim):  # here we compute Hq, Hqc(q,p), generator of q,p and gradient of Hqc
    """Sample phonon frequencies and couplings"""
    if sim.specden != 'single':
        sim.frq, sim.g = sample_w_g[sim.specden](sim.w_cutoff, sim.reorg_en)
    frq, g = sim.frq, sim.g
    frq_ext, g_ext = frq, g  # extended array of frequencies and couplings
    for m in range(ndyn_phset - 1):
        frq_ext = np.concatenate((frq_ext, frq))
        g_ext = np.concatenate((g_ext, g))
    frq_ext, g_ext = np.diag(frq_ext), np.diag(g_ext)
    sim.frq_ext, sim.g_ext = frq_ext, g_ext

    """Sample q & p and rotate into a desirable basis"""
    qp = np.zeros((sim.ntraj, 2, ndyn_phset, nph_per_set))
    for itraj in range(sim.ntraj):
        qp_temp = np.zeros((2, nstate, nph_per_set))
        zn = np.zeros((nstate, nph_per_set), dtype=complex)
        for ist in range(nstate):
            for i in range(nph_per_set):
                qp_temp[0, ist, i], qp_temp[1, ist, i] = sample_qp[sim.qp_dist](sim.q0[i], sim.p0[i], sim.beta, frq[i])
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

    """Perform truncation of matrices"""
    hsys = np.delete(Hsys, trunc_states, axis=0)
    hsys = np.delete(hsys, trunc_states, axis=1)
    sim.Hsys = hsys
    sim.strot, sim.strot_d = truncate_strot(strot)
    sim.strot_s = np.conj(strot)
    sim.phrot, sim.phrot_d = truncate_phrot(phrot)
    sim.phrot_s = np.conj(phrot)
    sim.coef = truncate_coef(coef)
    sim.grr, sim.gri, sim.gir, sim.gii = sb.get_gmat(sim.strot)

    return sim
