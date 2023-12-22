import numpy as np
pi = np.pi

def initialize(sim):
    defaults = {
        'temp': 0.25826,  # Temperature. 77K in thermal energy unit (298.15K = 1 teu)
        'specden': 'debye',  # choice of spectral density
        'wcutoff': 0.509065,  # Cutoff frequency. 106.14 cm-1 in thermal energy unit
        'lam': 0.168148,  # reorganization energy. 35 cm-1 in thermal energy unit
        'num_states': 8,  # number of states. Can be 7 or 8.
        'ndyn_states': 8,  # number of states in dynamics
        'ndyn_phsets': 8,  # number of sets of phonon modes in dynamics
        'nph_per_set': 200,  # number of phonon modes per state
        'quantum_rotation': None,  # rotation of quantum subspace
        'classical_rotation': None  # rotation of classical subspace
    }

    inputs = list(sim.input_params)
    for key in inputs:
        defaults[key] = sim.input_params[key]

    # Load model specific parameters
    sim.temp = defaults['temp']
    sim.specden = defaults['specden']
    sim.num_states = defaults['num_states']
    sim.ndyn_states = defaults['ndyn_states']
    sim.ndyn_phsets = defaults['ndyn_phsets']
    sim.nph_per_state = defaults['nph_per_set']
    sim.quantum_rotation = defaults['quantum_rotation']
    sim.classical_rotation = defaults['classical_rotation']


    def sample_qp_wigner(frq):
        """
        Sample nuclear phase space variables from Wigner distribution under
        harmonic approximation
        """
        # position
        q = np.random.normal(loc=0, scale=np.sqrt(1.0 / (2.0 * frq * np.tanh(frq / (2.0 * sim.temp)))), size=sim.ndyn_phsets)
        # momentum
        p = np.random.normal(loc=0, scale=np.sqrt(frq / (2.0 * np.tanh(frq / (2.0 * sim.temp)))), size=sim.ndyn_phsets)
        # z coordinate
        #z = np.sqrt(frq/2.0) * (q + 1.0j*p/frq)
        #zc = np.conj(z)
        return q, p

    def sample_qp_boltzmann(frq):
        """
        Sample nuclear phase space variables from the Boltzmann distribution
        """
        # position
        q = np.random.normal(loc=0, scale=np.sqrt(sim.temp / (frq ** 2)), size=sim.ndyn_phsets)
        # momentum
        p = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=sim.ndyn_phsets)
        # z coordinate
        #z = np.sqrt(frq / 2.0) * (q + 1.0j * p / frq)
        #zc = np.conj(z)
        return q, p

    def sample_debye(wc, lam, nph, nstate):
        """
        Debye spectral density sampling function
        'wc': cutoff frequency
        'lam': reorganization energy
        'nph': number of phonon modes per state
        'nstate': number of states in dynamics
        :return: 1D array of frequencies and couplings (w & g) as well as their diagonal matrices
                 extended for all modes in each state (w_ext & g_ext)
        """
        w, g = [], []
        for j in range(nph):
            w.append(wc * np.tan((pi / (2 * nph)) * (j + 1 - 0.5)))
        w = np.array(w)
        for j in range(nph):
            g.append(w[j] * np.sqrt(2 * lam / nph))
        g = np.array(g)

        w_ext, g_ext = w, g  # extended array of frequencies and couplings
        for m in range(nstate - 1):
            w_ext = np.concatenate((w_ext, w))
            g_ext = np.concatenate((g_ext, g))
        w_ext, g_ext = np.diag(w_ext), np.diag(g_ext)
        return w, w_ext, g, g_ext

    sample_qp = {'boltz': sample_qp_boltzmann, 'wigner': sample_qp_wigner}
    sample_w_g = {'debye': sample_debye}

    if sim.num_states == 8:
        def h_q():
            """
            8-site FMO model Hamiltonian (originally in cm-1).
            Cited from Marcel Schmidt am Busch et al. J. Phys. Chem. Lett. 2011
            :return:
            """
            ham = np.array([[12505.0,    94.8,     5.5,    -5.9,     7.1,   -15.1,   -12.2,    39.5],
                            [   94.8, 12425.0,    29.8,     7.6,     1.6,    13.1,     5.7,     7.9],
                            [    5.5,    29.8, 12195.0,   -58.9,    -1.2,    -9.3,     3.4,     1.4],
                            [   -5.9,     7.6,   -58.9, 12375.0,   -64.1,   -17.4,   -62.3,    -1.6],
                            [    7.1,     1.6,    -1.2,   -64.1, 12600.0,    89.5,    -4.6,     4.4],
                            [  -15.1,    13.1,    -9.3,   -17.4,    89.5, 12515.0,    35.1,    -9.1],
                            [  -12.2,     5.7,     3.4,   -62.3,    -4.6,    35.1, 12465.0,   -11.1],
                            [   39.5,     7.9,     1.4,    -1.6,     4.4,    -9.1,   -11.1, 12700.0]])
            #ham = ham * (clight * 10 ** 2) * planck / eh2j  # conversion to a.u.
            ham /= 218.5  # conversion to thermal energy unit
            return ham
    elif sim.num_states == 7:
        def h_q():
            """
            7-site FMO model Hamiltonian (originally in cm-1).
            Cited from E. Mulvihill, E. Geva et al., J. Chem. Phys. 2021
            :return:
            """
            ham = np.array([[12410, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
                            [-87.7, 12530,  30.8,   8.2,   0.7,  11.8,   4.3],
                            [  5.5,  30.8, 12210, -53.5,  -2.2,  -9.6,   6.0],
                            [ -5.9,   8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                            [  6.7,   0.7,  -2.2, -70.7, 12480,  81.1,  -1.3],
                            [-13.7,  11.8,  -9.6, -17.0,  81.1, 12630,  39.7],
                            [ -9.9,   4.3,   6.0, -63.3,  -1.3,  39.7, 12440]])
            #ham = ham * (clight * 10 ** 2) * planck / eh2j  # conversion to a.u.
            ham /= 218.5  # conversion to thermal energy unit
            return ham

    if sim.dynamics_method == "FSSH":
        def h_qc(q, p, w, g, Uc_d, Uq, Uq_d):
            """
            Compute state-phonon interaction hamiltonian
            q & p are 2d array: 1st dimension is for branches
            """
            out = np.zeros((sim.ndyn_states, sim.ndyn_states, sim.ndyn_states),
                           dtype=complex)  # The 1st index is for branches
            for j in range(sim.ndyn_states):
                # Define q in terms of rotated q&p
                qmat = q[j].reshape(sim.ndyn_phsets, sim.nph_per_set)  # reshaped into alp x i matrix
                pmat = p[j].reshape(sim.ndyn_phsets, sim.nph_per_set)  # reshaped into alp x i matrix
                qn = (np.real(Uc_d).dot(qmat)
                      - np.imag(Uc_d).dot(pmat / w)).dot(g)  # vector with index running over unrotated basis n
                # Phonon-site/state coupling
                hqc = (Uq * qn).dot(Uq_d)
                out[j] = hqc.T
            return out
        def h_c(q, p, w_ext):
            """
            Compute classical hamiltonian
            """
            out = np.zeros((sim.ndyn_states, sim.ndyn_states, sim.ndyn_states),
                           dtype=complex)  # The 1st index is for branches
            for j in range(sim.ndyn_states):
                temp = 0.5 * (np.dot(p[j], p[j]) + q[j].dot(w_ext ** 2).dot(q[j]))
                out[j] = np.identity(sim.ndyn_states, dtype=complex) * temp
            return out
    elif sim.dynamics_method == "MF":
        def h_qc(q, p, w, g, Uc_d, Uq, Uq_d):
            """
            Compute state-phonon interaction hamiltonian
            """
            # Define q in terms of rotated q&p
            qmat = q.reshape(sim.ndyn_phsets, sim.nph_per_set)
            pmat = p.reshape(sim.ndyn_phsets, sim.nph_per_set)
            qn = (np.real(Uc_d).dot(qmat) - np.imag(Uc_d).dot(pmat / w)).dot(g)
            # Phonon-site/state coupling
            out = (Uq * qn).dot(Uq_d)
            out = out.T
            return out
        def h_c(q, p, w_ext):
            """
            Compute classical Hamiltonian
            """
            out = 0.5 * (np.dot(p, p) + q.dot(w_ext ** 2).dot(q))
            out = np.identity(sim.ndyn_state, dtype=complex) * out
            return out

    """
    Initialize the rotation matrices of quantum and classical subsystems
    """
    if sim.quantum_rotation is None:
        def U_q():
            return np.identity(sim.num_state)
    elif sim.quantum_rotation == "excitonic":
        def U_q():
            eigval, eigvec = np.linalg.eigh(h_q())
            return eigvec.T
    if sim.classical_rotation == "manual":
        def U_c():
            """Example case: initial state as 5th lowest-energy excitonic state, remove 2 phonon sets"""
            out = [[0.01163780977, 0.006501060214, 0.0031789684, -0.02342191424, -0.9415700536, 0.1081508025, 0.3178129986, -0.0027823778],
                   [0.0003950231801, -0.01166037954, 0.9998312494, -0.01227454971, 0.004333880263, 0.005636216649, 0.0002427718267, 0.0002815208143],
                   [-0.07715118173, -0.1028953023, -0.001014611717, -0.01156381834, -0.001724775688, -0.001411743314, -0.009222701997, -0.9915823085],
                   [-0.01948147454, -0.01710128539, -0.01190598047, -0.9924728442, -0.0156993118, -0.06799250317, -0.09519504722, 0.01588626331],
                   [-0.4025418975, -0.9013041051, -0.009848391472, 0.03345130572, -0.03942903466, 0.011131001, -0.08376452683, 0.1252992668],
                   [0.02442337814, 0.0206058886, 0.007589480246, 0.09616026397, -0.2159539947, -0.916027307, -0.3223850259, -0.0004894128992],
                   [-0.08877979561, -0.05944360194, -0.0005641994926, -0.05963560199, 0.25372173, -0.3798756889, 0.8810876256, 0.005676570149],
                   [-0.9072013227, 0.4154932493, 0.005447542487, 0.01557207953, -0.02474096694, 0.01054505433, -0.05082990138, 0.0277842044]]
            return np.array(out)
    else:
        def U_q():
            return np.identity(sim.num_state)

    # equip simulation object with necessary functions
    sim.init_classical = sample_qp[sim.qp_dist]
    sim.specden = sample_w_g[sim.specden]
    sim.h_q = h_q
    sim.h_qc = h_qc
    sim.h_c = h_c
    sim.U_c = U_c
    sim.U_q = U_q
    sim.calc_dir = 'fmo' + str(sim.num_states) + '_' + str(sim.specden) + '_wc_' + str(sim.wcutoff) + \
                   '_lam_' + str(sim.lam) + '_temp_' + str(sim.temp * 298.15) + 'K' + '_ndynstates_' + \
                   str(sim.ndyn_states) + '_ndynphsets_' + str(sim.ndyn_phsets)

    dist0 = np.zeros(sim.num_state, dtype=complex)
    for j in sim.init_state:
        dist0[j-1] = 1.0
    sim.psi_db_0 = dist0 / np.sqrt(len(sim.init_state))
    return sim