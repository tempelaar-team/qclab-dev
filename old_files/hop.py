import numpy as np

def harmonic_oscillator_hop(z, zc, delta_z, delta_zc, ev_diff, sim): # this is the hoping function for a harmonic oscillator
        hopped = False
        akj_z = np.real(np.sum(sim.h * delta_zc * delta_z))
        bkj_z = np.real(np.sum(1j * sim.h * (zc * delta_z - z * delta_zc)))
        ckj_z = ev_diff
        disc = bkj_z ** 2 - 4 * akj_z * ckj_z
        if disc >= 0:
            if bkj_z < 0:
                gamma = bkj_z + np.sqrt(disc)
            else:
                gamma = bkj_z - np.sqrt(disc)
            if akj_z == 0:
                gamma = 0
            else:
                gamma = gamma / (2 * akj_z)
            # adjust classical coordinates
            z = z - 1.0j * np.real(gamma) * delta_z
            zc = zc + 1.0j * np.real(gamma) * delta_zc
            hopped = True
        return z, zc, hopped