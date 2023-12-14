"""
– User-specified spin boson type hamiltonian
– User-specified state & phonon coordinates rotation matrices
"""

import numpy as np
pi = np.pi

"""
Parameters 
Temporarily specified here.
"""
nstate = 30  # number of states/sites
J = 1.0  # nearest-neighbor interaction strength
ndyn_state = 30  # number of states/sites after truncation


# As an example holstein model is used here
"""
Holstein model Hamiltonian
"""
Hsys = np.zeros((nstate, nstate), dtype=complex)
for n in range(nstate):
    np1 = n + 1
    nm1 = n - 1
    if np1 == nstate:
        np1 = 0
    if nm1 == -1:
        nm1 = nstate-1
    Hsys[n, np1] = 1.0
    Hsys[n, nm1] = 1.0
    Hsys[np1, n] = Hsys[n, np1]
    Hsys[nm1, n] = Hsys[n, nm1]
Hsys *= -J


"""
State rotation matrix (rotation to reciprocal space as an example)
"""
strot = np.zeros((nstate,nstate), dtype=complex)
for k in range(0, nstate):
    for n in range(0, nstate):
        strot[k, n] = np.exp(1j*(2*pi*k/nstate - pi)*n)/np.sqrt(nstate)
strot_d = np.conj(strot.T)
Hsys = np.matmul(strot, np.matmul(Hsys, strot_d))


"""
Rotation from dynamical basis to basis in which population is to be computed
"""
pop_rot = np.identity(ndyn_state, dtype=complex)


"""
Phonon coordinates rotation matrix (same as state rotation, as an example)
"""
phrot = strot
phrot_d = strot_d
