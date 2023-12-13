import numpy as np

pi = np.pi
planck = 6.62607015*10**-34  # Planck's constant in SI
hbar = planck/(2*pi)         # Reduced Planck's constant in SI
clight = 2.99792458*10**8    # Speed of light in SI
eh2j = 4.359744650*10**-18   # Hartree to Joule


"""
Hamiltonian of 8-site FMO complex of BChl (in cm-1)
Cited from Marcel Schmidt am Busch et al. J. Phys. Chem. Lett. 2011.
"""
fmo8 = np.array([[12505.0, 94.8, 5.5, -5.9, 7.1, -15.1, -12.2, 39.5],
                 [94.8, 12425.0, 29.8, 7.6, 1.6, 13.1, 5.7, 7.9],
                 [5.5, 29.8, 12195.0, -58.9, -1.2, -9.3, 3.4, 1.4],
                 [-5.9, 7.6, -58.9, 12375.0, -64.1, -17.4, -62.3, -1.6],
                 [7.1, 1.6, -1.2, -64.1, 12600.0, 89.5, -4.6, 4.4],
                 [-15.1, 13.1, -9.3, -17.4, 89.5, 12515.0, 35.1, -9.1],
                 [-12.2, 5.7, 3.4, -62.3, -4.6, 35.1, 12465.0, -11.1],
                 [39.5, 7.9, 1.4, -1.6, 4.4, -9.1, -11.1, 12700.0]])
fmo8 = fmo8 * (clight*10**2) * planck/eh2j  # conversion to a.u.

"""
Hamiltonian of 7-site FMO complex of BChl (in cm-1)
Cited from E. Mulvihill ... E. Geva, J. Chem. Phys. 2021
"""
fmo7 = np.array([[12410, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
                 [-87.7, 12530,  30.8,   8.2,   0.7,  11.8,   4.3],
                 [5.5,  30.8, 12210, -53.5,  -2.2,  -9.6,   6.0],
                 [-5.9,   8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                 [6.7,   0.7,  -2.2, -70.7, 12480,  81.1,  -1.3],
                 [-13.7,  11.8,  -9.6, -17.0,  81.1, 12630,  39.7],
                 [-9.9,   4.3,   6.0, -63.3,  -1.3,  39.7, 12440]])
fmo7 = fmo7 * (clight*10**2) * planck/eh2j  # conversion to a.u.

"""
Holstein model
"""

"""
Peierl model
"""

"""
TMD Hamiltonian
"""
