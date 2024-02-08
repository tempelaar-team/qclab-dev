import simulation
import numpy as np

def boltzmann_harmonic():
    """
    Sample harmonic nuclear phase space variables from the Boltzmann distribution
    """
    # position
    q = np.random.normal(loc=0, scale=np.sqrt(sim.temp / (sim.w ** 2)), size=sim.ndyn_phsets)
    # momentum
    p = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=sim.ndyn_phsets)
    # z coordinate
    z = np.sqrt(sim.w / 2.0) * (q + 1.0j * p / sim.w)
    zc = np.conj(z)
    return z, zc

def wigner_harmonic():
    """
    Sample harmonic nuclear phase space variables from Wigner distribution under
    harmonic approximation
    """
    # position
    q = np.random.normal(loc=0, scale=np.sqrt(1.0 / (2.0 * sim.w * np.tanh(sim.w / (2.0 * sim.temp)))), size=sim.ndyn_phsets)
    # momentum
    p = np.random.normal(loc=0, scale=np.sqrt(sim.w / (2.0 * np.tanh(sim.w / (2.0 * sim.temp)))), size=sim.ndyn_phsets)
    # z coordinate
    z = np.sqrt(sim.w/2.0) * (q + 1.0j*p/sim.w)
    zc = np.conj(z)
    return z, zc

def boltzmann_quartic(a, b, qmax):
    """
    Sample nuclear phase space variables from the Boltzmann distribution of double-well (quartic) potential
    a: coefficient of 4th order term [energy/length^4]
    b: coefficient of 2nd order term [energy/length^2]
    qmax: Positive end of the range of position coordinate to cover the important area of Boltzmann distribution
    """
    # Potential energy function
    dwell = lambda c4, c2, x: c4*x**4 + c2*x**2
    step = 2*qmax/2000
    x = np.arange(-qmax, qmax, step)

    # Boltzmann distribution
    dwell_boltz = lambda x: np.exp(-(1.0/sim.temp) * dwell(a, b, x))
    y_boltz = [dwell_boltz(i) for i in x]

    # Monte Carlo sampling for position
    q = []
    ymin, ymax = 0.0, max(y_boltz)
    nsamp = sim.ndyn_phsets
    while len(q) < nsamp:
        qran = np.random.uniform(-qmax, qmax)
        yran = np.random.uniform(ymin, ymax)
        if yran <= dwell_boltz(qran):
            q.append(qran)
    # Momentum (normal distribution)
    p = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=sim.ndyn_phsets)
    # z coordinate
    z = np.sqrt(sim.w / 2.0) * (q + 1.0j * p / sim.w)
    zc = np.conj(z)
    return z, zc

def boltzmann_morse(Ediss, a, qmin, qmax):
    """
    Sample nuclear phase space variables from the Boltzmann distribution of double-well (quartic) potential
    Ediss: Dissociation energy relative to the minimum of energy
    a: Parameter for the width of the curve [length^-1]
    qmin, qmax: Negative & positive end of the range of position coordinate to cover the important area of Boltzmann distribution
    """
    # Potential energy function
    morse = lambda x: Ediss * (1 - np.exp(-a*x))**2
    step = (qmax - qmin) / 2000
    x = np.arange(qmin, qmax, step)

    # Boltzmann distribution
    morse_boltz = lambda x: np.exp(-(1.0/sim.temp) * morse(x))
    y_boltz = [morse_boltz(i) for i in x]

    # Monte Carlo sampling for position
    q = []
    ymin, ymax = y_boltz[-1], 1.0  # Minimum is set at dissociation asymptote
    nsamp = sim.ndyn_phsets
    while len(q) < nsamp:
        qran = np.random.uniform(qmin, qmax)
        yran = np.random.uniform(ymin, ymax)
        if yran <= dwell_boltz(qran):
            q.append(qran)
    # Momentum (normal distribution)
    p = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=sim.ndyn_phsets)
    # z coordinate
    z = np.sqrt(sim.w / 2.0) * (q + 1.0j * p / sim.w)
    zc = np.conj(z)

    return z, zc


if sim.potential == 'harmonic':
    if sim.qp_dist == 'boltzmann':
        sim.init_classical = boltzmann_harmonic
    elif sim.qp_dist == 'wigner':
        sim.init_classical = wigner_harmonic
elif sim.potential == 'morse':
    if sim.qp_dist == 'boltzmann':
        sim.init_classical = boltzmann_morse
elif sim.potential == 'double-well':
    if sim.qp_dist == 'boltzmann':
        sim.init_classical = boltzmann_quartic
# there could be more options for potential energy shapes

