"""
This module contains the tasks that initialize quantities in the state and parameters
objects.

These are typically used in the initialization recipe of the algorithm object.
"""

import logging
import numpy as np
from qc_lab import functions

logger = logging.getLogger(__name__)


def initialize_norm_factor(algorithm, sim, parameters, state, **kwargs):
    """
    Assign the normalization factor to the state object.

    Required constants:
        - None.
    """
    state.norm_factor = sim.settings.batch_size
    return parameters, state


def initialize_branch_seeds(algorithm, sim, parameters, state, **kwargs):
    """
    Convert seeds into branch seeds for deterministic surface hopping. This is done by
    first assuming that the number of branches is equal to the number of quantum states.
    Then, a branch index (state.branch_ind) is created which gives the branch index of
    each seed in the batch. Then a new set of seeds is created by floor dividing the
    original seeds by the number of branches so that the seeds corresponding to
    different branches within the same trajectory are the same.

    Notably, this leads to the number of unique classical initial conditions
    being equal to the number of trajectories divided by the number of
    branches in deterministic surface hopping.

    Required constants:
        - num_quantum_states (int): Number of quantum states. Default: None.
    """
    # First ensure that the number of branches is correct.
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size

    if batch_size % num_branches != 0:
        logger.error("Batch size must be divisible by number of branches.")
        raise ValueError("Batch size must be divisible by number of branches.")

    # Next, determine the number of trajectories that have been run by assuming that
    # the minimum seed in the current batch of seeds is the number of trajectories
    # that have been run modulo num_branches.
    orig_seeds = state.seed
    # Now construct a branch index for each trajectory in the expanded batch.
    state.branch_ind = (
        np.zeros((batch_size // num_branches, num_branches), dtype=int)
        + np.arange(num_branches)[np.newaxis, :]
    ).flatten()
    # Now generate the new seeds for each trajectory in the expanded batch.
    new_seeds = orig_seeds // num_branches
    parameters.seed = new_seeds
    state.seed = new_seeds
    return parameters, state


def initialize_z_mcmc(algorithm, sim, parameters, state, **kwargs):
    """
    Initialize classical coordinates according to Boltzmann statistics using Markov-
    Chain Monte Carlo with a Metropolis-Hastings algorithm.

    Required constants:
        - num_classical_coordinates (int): Number of classical coordinates.
          Default: None.
        - mcmc_burn_in_size (int): Number of burn-in steps. Default: 5000.
        - mcmc_std (float): Standard deviation for sampling. Default: 1.
        - mcmc_h_c_separable (bool): If the classical Hamiltonian is
          separable. Default: True.
        - mcmc_init_z (np.ndarray): Initial sample. Default: None.
        - kBT (float): Thermal quantum. Default: None.
    """
    seed = getattr(state, kwargs["seed"])
    name = kwargs["name"]
    burn_in_size = sim.model.constants.get("mcmc_burn_in_size", 1000)
    sample_size = sim.model.constants.get("mcmc_sample_size", 10000)
    mcmc_h_c_separable = sim.model.constants.get("mcmc_h_c_separable", True)
    burn_in_seeds = np.arange(burn_in_size)
    sample_seeds = np.arange(sample_size)
    save_inds = np.zeros(len(seed), dtype=int)
    out_tmp = np.zeros(
        (sample_size, sim.model.constants.num_classical_coordinates), dtype=complex
    )
    for s, seed_s in enumerate(seed):
        np.random.seed(seed_s)
        save_inds[s] = np.random.randint(0, sample_size)
    mcmc_init_z, _ = functions.gen_sample_gaussian(
        sim.model.constants, z0=None, seed=0, separable=False
    )
    sample = sim.model.constants.get("mcmc_init_z", mcmc_init_z)
    h_c, _ = sim.model.get("h_c")
    if mcmc_h_c_separable:
        for s, seed_s in enumerate(burn_in_seeds):
            last_sample = np.copy(sample)
            last_z = np.diag(last_sample)
            last_e = h_c(sim.model, parameters, z=last_z, batch_size=len(last_z))
            proposed_sample, rand = functions.gen_sample_gaussian(
                sim.model.constants, z0=last_sample, seed=seed_s, separable=True
            )
            new_z = np.diag(proposed_sample)
            new_e = h_c(sim.model, parameters, z=new_z, batch_size=len(new_z))
            thresh = np.minimum(
                np.ones(sim.model.constants.num_classical_coordinates),
                np.exp(-(new_e - last_e) / sim.model.constants.kBT),
            )
            sample[rand < thresh] = proposed_sample[rand < thresh]
        for s, seed_s in enumerate(sample_seeds):
            last_sample = np.copy(sample)
            last_z = np.diag(last_sample)
            last_e = h_c(sim.model, parameters, z=last_z, batch_size=len(last_z))
            proposed_sample, rand = functions.gen_sample_gaussian(
                sim.model.constants, z0=last_sample, seed=seed_s, separable=True
            )
            new_z = np.diag(proposed_sample)
            new_e = h_c(sim.model, parameters, z=new_z, batch_size=len(new_z))
            thresh = np.minimum(
                np.ones(sim.model.constants.num_classical_coordinates),
                np.exp(-(new_e - last_e) / sim.model.constants.kBT),
            )
            sample[rand < thresh] = proposed_sample[rand < thresh]
            out_tmp[s] = sample
            setattr(state, name, out_tmp[save_inds])
        return parameters, state
    # If not separable, do the full MCMC.
    for s, seed_s in enumerate(burn_in_seeds):
        last_sample = np.copy(sample)
        last_e = h_c(sim.model, parameters, z=last_sample, batch_size=len(last_sample))
        proposed_sample, rand = functions.gen_sample_gaussian(
            sim.model.constants, z0=last_sample, seed=seed_s, separable=False
        )
        new_e = h_c(
            sim.model,
            parameters,
            z=proposed_sample,
            batch_size=len(proposed_sample),
        )
        thresh = min(1, np.exp(-(new_e - last_e) / sim.model.constants.kBT))
        if rand < thresh:
            sample = proposed_sample
    for s, seed_s in enumerate(sample_seeds):
        last_sample = np.copy(sample)
        last_e = h_c(sim.model, parameters, z=last_sample, batch_size=len(last_sample))
        proposed_sample, rand = functions.gen_sample_gaussian(
            sim.model.constants, z0=last_sample, seed=seed_s, separable=False
        )
        new_e = h_c(
            sim.model,
            parameters,
            z=proposed_sample,
            batch_size=len(proposed_sample),
        )
        thresh = min(1, np.exp(-(new_e - last_e) / sim.model.constants.kBT))
        if rand < thresh:
            sample = proposed_sample
        out_tmp[s] = sample
    setattr(state, name, out_tmp[save_inds])
    return parameters, state


def initialize_z(algorithm, sim, parameters, state, **kwargs):
    """
    Initialize the classical coordinate by using the init_classical function from the
    model object.

    Required constants:
        - None.
    """
    seed = getattr(state, kwargs["seed"])
    name = kwargs["name"]
    init_classical, has_init_classical = sim.model.get("init_classical")
    if has_init_classical:
        setattr(state, name, init_classical(sim.model, parameters, seed=seed))
        return parameters, state
    parameters, state = initialize_z_mcmc(
        algorithm, sim, parameters, state, seed=kwargs["seed"], name=name
    )
    return parameters, state


def state_to_parameters(algorithm, sim, parameters, state, **kwargs):
    """
    Set parameters.parameters_name to state.state_name.

    Required constants:
        - None.
    """
    setattr(parameters, kwargs["parameters_name"], getattr(state, kwargs["state_name"]))
    return parameters, state


def copy_in_state(algorithm, sim, parameters, state, **kwargs):
    """
    Set state.dest_name to state.orig_name.

    Required constants:
        - None.
    """
    setattr(state, kwargs["dest_name"], np.copy(getattr(state, kwargs["orig_name"])))
    return parameters, state


def initialize_active_surface(algorithm, sim, parameters, state, **kwargs):
    """
    Initializes the active surface (act_surf), active surface index (act_surf_ind) and
    initial active surface index (act_surf_ind_0) for FSSH.

    If fssh_deterministic is true it will set act_surf_ind_0 to be the same as
    the branch index and assert that the number of branches (num_branches)
    is equal to the number of quantum states (num_states).

    If fssh_deterministic is false it will stochastically sample the active
    surface from the density specified by the initial quantum wavefunction in the
    adiabatic basis.

    Required constants:
        - num_quantum_states (int): Number of quantum states. Default: None.
    """
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_states = sim.model.constants.num_quantum_states
    num_trajs = sim.settings.batch_size // num_branches
    if sim.algorithm.settings.fssh_deterministic:
        act_surf_ind_0 = np.tile(np.arange(num_branches, dtype=int), (num_trajs, 1))
    else:
        intervals = np.cumsum(
            np.real(
                np.abs(state.wf_adb.reshape((num_trajs, num_branches, num_states))) ** 2
            ),
            axis=-1,
        )
        bool_mat = intervals > state.stochastic_sh_rand_vals[:, :, np.newaxis]
        act_surf_ind_0 = np.argmax(bool_mat, axis=-1).astype(int)
    state.act_surf_ind_0 = np.copy(act_surf_ind_0)
    state.act_surf_ind = np.copy(act_surf_ind_0)
    act_surf = np.zeros((num_trajs, num_branches, num_states), dtype=int)
    traj_inds = np.repeat(np.arange(num_trajs), num_branches)
    branch_inds = np.tile(np.arange(num_branches), num_trajs)
    act_surf[traj_inds, branch_inds, act_surf_ind_0.flatten()] = 1
    state.act_surf = act_surf.reshape((num_trajs * num_branches, num_states))
    parameters.act_surf_ind = state.act_surf_ind
    return parameters, state


def initialize_random_values_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Initialize a set of random variables using the trajectory seeds for FSSH.

    Required constants:
        - None.
    """
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    state.hopping_probs_rand_vals = np.zeros((batch_size, len(sim.settings.t_update)))
    state.stochastic_sh_rand_vals = np.zeros((batch_size, num_branches))
    for nt in range(batch_size):
        np.random.seed(state.seed[int(nt * num_branches)])
        state.hopping_probs_rand_vals[nt] = np.random.rand(len(sim.settings.t_update))
        state.stochastic_sh_rand_vals[nt] = np.random.rand(num_branches)
    return parameters, state


def initialize_dm_adb_0_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Initialize the initial adiabatic density matrix for FSSH.

    Required constants:
        - None.
    """
    state.dm_adb_0 = np.einsum(
        "...i,...j->...ij",
        state.wf_adb,
        np.conj(state.wf_adb),
        optimize="greedy",
    )
    return parameters, state
