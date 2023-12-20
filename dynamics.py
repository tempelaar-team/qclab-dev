def run_dynamics(sim):
    if sim.dynamics_method == "MF":
        return mf_dynamics(traj, sim)
    elif sim.dynamics_method == "FSSH":
        return fssh_dynamics(traj,sim)
    elif sim.dynamics_method == "CFSSH":
        return cfssh_dynamics(traj, sim)
    return sim


def cfssh_dynamics(traj, sim):
    return traj


def fssh_dynamics(traj, sim):
    return traj


def mf_dynamics(traj, sim):
    if sim.Hstruc == "spin_boson":
        import spin_boson as sb
    elif sim.Hstruc == "...":
        ...
    return traj
