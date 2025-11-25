"""
This module contains numerical constants.
"""

# Numerical threshold for near zero values.
SMALL = 1e-10

# Numerical threshold for nonadiabatic coupling gauge fixing.
# The misalignment is allowed to be 1e-3 of the magnitude of the coupling.
GAUGE_FIX_THRESHOLD = 1e-3

# Conversion factor from electronvolts to wavenumbers.
EV_TO_INVCM = 8065.610420

# Conversion factor from Hartrees to electronvolts.
HA_TO_EV = 27.21138625

# Conversion factor from wavenumbers to kBT at 300 K.
INVCM_TO_300K = 1 / 208.521

# Conversion factor from electronvolts to kBT at 300 K.
EV_TO_300K = 0.025852

# Conversion factor from Hartrees to kBT at 300 K.
HA_TO_300K = EV_TO_300K / HA_TO_EV

"""
For ab initio calculations we use the following unit convention (referred to as ABU):
\hbar = 1
mass: atomic mass units
length: Angstroms

"""

# Conversion factor from eV to Hartrees.
EV_TO_HA = 1 / HA_TO_EV

# Conversion factor from wavenumbers to Hartrees.
INVCM_TO_HA =  (1 / EV_TO_INVCM) * EV_TO_HA

# Conversion factor from Angstroms to Bohr.
ANGSTROM_TO_BOHR = 1.8897259886

# Conversion factor from atomic mass units to electron mass.
AMU_TO_EMASS = 1822.89

# Conversion factor from atomic units of time to femtoseconds.
AU_TIME_TO_FS = 0.02419



# Finite difference step size.
FINITE_DIFFERENCE_DELTA = 1e-6

