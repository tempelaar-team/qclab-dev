"""
FMO complex module file
"""

import numpy as np
pi = np.pi
planck = 6.62607015*10**-34  # Planck's constant in SI
hbar = planck/(2*pi)         # Reduced Planck's constant in SI
clight = 2.99792458*10**8    # Speed of light in SI
eh2j = 4.359744650*10**-18   # Hartree to Joule



def fmo7():
    """
    Hamiltonian of 7-site FMO complex of BChl (originally in cm-1)
    Cited from E. Mulvihill ... E. Geva, J. Chem. Phys. 2021
    """
    ham = np.array([[12410, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
                    [-87.7, 12530,  30.8,   8.2,   0.7,  11.8,   4.3],
                    [5.5,  30.8, 12210, -53.5,  -2.2,  -9.6,   6.0],
                    [-5.9,   8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                    [6.7,   0.7,  -2.2, -70.7, 12480,  81.1,  -1.3],
                    [-13.7,  11.8,  -9.6, -17.0,  81.1, 12630,  39.7],
                    [-9.9,   4.3,   6.0, -63.3,  -1.3,  39.7, 12440]])
    ham = ham * (clight*10**2) * planck/eh2j  # conversion to a.u.
    return ham