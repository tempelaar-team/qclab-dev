def run_dynamics(sim):
    if sim.Hstruc == "spin_boson":
        import spin_boson as sb
    if sim.dynamics_method == "MF":
        sim.update_htot = sb.update_htot
        sim.get_phonon_force = sb.get_mf_phonon_force
    elif sim.dynamics_method == "FSSH":
        sim.update_hq = sb.update_hq
        sim.get_phonon_force = sb.get_fssh_phonon_force
        sim.get_nac_numer = sb.get_nac_numer
    elif sim.dynamics_method == "CFSSH":
        ...
    return sim


def cfssh_dynamics(traj, sim):
    return traj


def fssh_dynamics(traj, sim):
    return traj


def mf_dynamics(traj, sim):
    return traj
