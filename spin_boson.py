"""
Repository of functions to compute Hq, Hqc, dHqc/dq, dHqc/dp
for spin-boson type hamiltonians
"""

from system import *
strot_s = np.conj(strot)
phrot_s = np.conj(phrot)


def update_hq(q, p, frq, g):
    """
    Update quantum hamiltonian (for SH)
    q & p are 2d array: 1st dimension is for branches
    """
    hq = np.zeros((ndyn_state, ndyn_state, ndyn_state), dtype=complex)  # The 1st index is for branches
    for j in range(ndyn_state):
        # Define q in terms of rotated q&p
        qmat = q[j].reshape(ndyn_phset, nph_per_set)  # reshaped into alp x i matrix
        pmat = p[j].reshape(ndyn_phset, nph_per_set)  # reshaped into alp x i matrix
        qn = (np.real(phrot_d).dot(qmat)
              - np.imag(phrot_d).dot(pmat / frq)).dot(g)  # vector with index running over unrotated basis n
        # Phonon-site/state coupling
        hqc = (strot * qn).dot(strot_d)
        hq[j] = Hsys + hqc.T
    return hq


def update_htot(q, p, frq, g, frq_ext):
    """
    Update the total hamiltonian (for MF)
    """
    # Phonon Hamiltonian
    h_ph = 0.5 * (np.dot(p, p) + q.dot(frq_ext ** 2).dot(q))
    h_ph = np.identity(ndyn_state, dtype=complex) * h_ph
    # Define q in terms of rotated q&p
    qmat = q.reshape(ndyn_phset, nph_per_set)
    pmat = p.reshape(ndyn_phset, nph_per_set)
    qn = (np.real(phrot_d).dot(qmat) - np.imag(phrot_d).dot(pmat / frq)).dot(g)
    # Phonon-site/state coupling
    h_qc = (strot * qn).dot(strot_d)
    h_qc = h_qc.T
    return Hsys + h_qc + h_ph


def get_fssh_phonon_force(active_vec, frq, g, grr, gri, gir, gii):
    """
    Compute the quantum forces acting on classical dofs (for SH)
    "active_vec": active adiabatic state expansion coefficients vector
    """
    csvs = np.dot(np.conj(active_vec), strot_s)  # vector with index running over unrotated basis n
    cv = np.dot(active_vec, strot)  # vector with index running over unrotated basis n
    cvcv = csvs * cv  # vector with index running over unrotated basis n
    re_cvcv = np.dot(np.real(phrot_s), cvcv)  # vector with index running over rotated basis alp
    im_cvcv = np.dot(np.imag(phrot_s), cvcv)  # vector with index running over rotated basis alp

    # Compute dHint/dQ and dHint/dP
    dhdq = np.outer(re_cvcv, g)  # matrix with the order of indices being alp x i
    dhdp = np.outer(im_cvcv, -(g / frq))  # matrix with the order of indices being alp x i

    # Compute the whole force terms
    qforce = (dhdq / frq).T.dot(gri) + dhdp.T.dot(gii)  # i x bet
    pforce = -dhdq.T.dot(grr) - (frq * dhdp).T.dot(gir)  # i x bet

    qforce = qforce.T.flatten()  # vector with index alp x i
    pforce = pforce.T.flatten()  # vector with index alp x i
    return qforce, pforce


def get_mf_phonon_force(coef, frq, g, grr, gri, gir, gii):
    """
    Compute the quantum forces acting on classical dofs (for MF)
    "coef": state expansion coefficients in a general rotated basis
    """
    ccvv = np.dot(np.conj(coef), strot_s) * np.dot(coef, strot)  # vector with index running over unrotated basis n
    re_ccvv = np.dot(np.real(phrot_s), ccvv)  # vector with index running over rotated electronic basis alp
    im_ccvv = np.dot(np.imag(phrot_s), ccvv)  # vector with index running over rotated electronic basis alp

    dhdq = np.outer(re_ccvv, g)  # matrix with the order of indices being alp x i
    dhdp = np.outer(im_ccvv, -(g / frq))  # matrix with the order of indices being alp x i

    qforce = (dhdq / frq).T.dot(gri) + dhdp.T.dot(gii)
    pforce = -dhdq.T.dot(grr) - (frq * dhdp).T.dot(gir)

    qforce = qforce.T.flatten()
    pforce = pforce.T.flatten()
    return qforce, pforce


def get_nac_numer(eigvec, frq, g):
    """
    Compute the numerator of Hellman-Feynman force: <mu|dHq/dq|mu'> & <mu|dHq/dp|mu'>
    The full NAC with adiabatic energy gap in the denominator will be computed
    in the subsequent routine in dynamics.
    'eigvec': coordinate-dependent eigenvector matrix of quantum Hamiltonian
    """
    crot = eigvec.T
    crot_s = np.conj(crot)

    # compute the numerator of Hellman-Feynman formula
    re_cvvc = np.zeros((ndyn_state, ndyn_state, ndyn_state), dtype=complex)
    im_cvvc = np.zeros((ndyn_state, ndyn_state, ndyn_state), dtype=complex)
    for gam in range(ndyn_state):
        re_cvvc[gam] = np.dot((np.dot(crot_s, strot_s) * np.real(phrot_s[gam])), np.dot(crot, strot).T)
        im_cvvc[gam] = np.dot((np.dot(crot_s, strot_s) * np.imag(phrot_s[gam])), np.dot(crot, strot).T)
    nacq_num = np.zeros((ndyn_phset * nph_per_set, ndyn_state, ndyn_state), dtype=complex)
    nacp_num = np.zeros((ndyn_phset * nph_per_set, ndyn_state, ndyn_state), dtype=complex)
    for gam in range(ndyn_state):
        for i in range(nph_per_set):
            nacq_num[nph_per_set * gam + i] = re_cvvc[gam] * g[i]
            nacp_num[nph_per_set * gam + i] = im_cvvc[gam] * (-g[i] / frq[i])
    return nacq_num, nacp_num


def get_gmatrices(trunc_phrot):
    """Compute G matrices"""
    grr = np.zeros((ndyn_phset, ndyn_phset))
    gri = np.zeros((ndyn_phset, ndyn_phset))
    gir = np.zeros((ndyn_phset, ndyn_phset))
    gii = np.zeros((ndyn_phset, ndyn_phset))
    for bet in range(ndyn_phset):
        for alp in range(ndyn_phset):
            for m in range(nstate):
                grr[bet, alp] += np.real(trunc_phrot[bet, m]) * np.real(trunc_phrot[alp, m])
                gri[bet, alp] += np.real(trunc_phrot[bet, m]) * np.imag(trunc_phrot[alp, m])
                gir[bet, alp] += np.imag(trunc_phrot[bet, m]) * np.real(trunc_phrot[alp, m])
                gii[bet, alp] += np.imag(trunc_phrot[bet, m]) * np.imag(trunc_phrot[alp, m])
    return grr, gri, gir, gii
