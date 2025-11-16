"""
This module contains the parallel MPI driver.
"""

import logging
import copy
import numpy as np
import qclab.dynamics as dynamics
from qclab.utils import get_log_output, reset_log_output
from qclab import Data

logger = logging.getLogger(__name__)


"""
Parallel driver using mpi4py.
Rank 0: master, owns the progress bars and collects results.
Ranks 1..N-1: workers, run dynamics and send progress/results to rank 0.
"""

from mpi4py import MPI
import logging
import copy
import numpy as np

import qclab.dynamics as dynamics
from qclab.utils import get_log_output, reset_log_output
from qclab import Data
from .progressbar_utils import ProgressAggregator

logger = logging.getLogger(__name__)

TAG_TASKS = 10
TAG_PROGRESS = 11
TAG_RESULT = 12


def parallel_driver_mpi(sim, seeds=None, data=None, comm=None):
    """
    Parallel driver for the dynamics core using mpi4py.

    This function must be called on all MPI ranks in the communicator.

    Rank 0 acts as master:
        - splits total trajectories into batches
        - assigns batches to worker ranks 1..size-1
        - owns the ProgressAggregator and tqdm bars
        - collects all Data results and returns a merged Data object

    Worker ranks (>=1):
        - receive a list of tasks: [(task_id, batch_seeds), ...]
        - for each task, run dynamics.run_dynamics
        - send chunked progress updates to rank 0
        - send back (task_id, Data) to rank 0

    Parameters
    ----------
    sim : Simulation
        Simulation object.
    seeds : ndarray, optional
        Integer seeds for trajectories. If None, generated on rank 0.
    data : Data, optional
        Data object to append results to on rank 0. Ignored on workers.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator. Defaults to MPI.COMM_WORLD.

    Returns
    -------
    data : Data or None
        On rank 0: merged Data object with results and log.
        On worker ranks: None.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    if size < 2:
        # No point in doing MPI with a single rank; fall back to serial style.
        if rank == root:
            logger.warning("MPI size=1. Consider using serial_driver instead.")
            from .serial_driver import serial_driver  # adjust import if needed

            return serial_driver(sim, seeds=seeds, data=data)
        else:
            return None

    # Clear any in-memory log output from previous runs.
    reset_log_output()

    # Initialize model constants on all ranks
    sim.model.initialize_constants()

    # ---------- Rank 0: master ----------
    if rank == root:
        if data is None:
            data = Data()

        # Handle seeds and num_trajs like serial driver
        if seeds is None:
            if len(data.data_dict["seed"]) > 0:
                offset = np.max(data.data_dict["seed"]) + 1
            else:
                offset = 0
            seeds = offset + np.arange(sim.settings.num_trajs, dtype=int)
            num_trajs = sim.settings.num_trajs
        else:
            num_trajs = len(seeds)
            logger.warning(
                "Setting sim.settings.num_trajs to the number of provided seeds: %s",
                num_trajs,
            )
            sim.settings.num_trajs = num_trajs

        # Determine number of batches
        if num_trajs % sim.settings.batch_size == 0:
            num_batches = num_trajs // sim.settings.batch_size
        else:
            num_batches = num_trajs // sim.settings.batch_size + 1

        logger.info(
            "Running %s batches with %s seeds in each batch.",
            num_batches,
            sim.settings.batch_size,
        )

        # Build batch_seeds_list
        batch_seeds_list = (
            np.zeros((num_batches * sim.settings.batch_size), dtype=int) + np.nan
        )
        batch_seeds_list[:num_trajs] = seeds
        batch_seeds_list = batch_seeds_list.reshape(
            (num_batches, sim.settings.batch_size)
        )

        # Create jobs (one per non-empty batch), assign task_id = batch index
        sim.initialize_timesteps()
        jobs = []
        for n in range(num_batches):
            batch_seeds = batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(
                int
            )
            if len(batch_seeds) == 0:
                continue
            jobs.append((n, batch_seeds))

        num_tasks_actual = len(jobs)
        logger.info("Actual number of non-empty batches: %s", num_tasks_actual)

        if num_tasks_actual == 0:
            logger.info("No batches to run; returning early.")
            data.log = get_log_output()
            return data

        # Assign jobs to worker ranks 1..size-1 in round-robin fashion
        n_workers = size - 1
        tasks_for_rank = {r: [] for r in range(1, size)}
        for i, (task_id, batch_seeds) in enumerate(jobs):
            worker_rank = 1 + (i % n_workers)
            tasks_for_rank[worker_rank].append((task_id, batch_seeds))

        # Send task lists to each worker
        for r in range(1, size):
            comm.send(tasks_for_rank[r], dest=r, tag=TAG_TASKS)

        # Setup progress aggregator on master
        steps_per_task = [len(sim.settings.t_update_n)] * num_tasks_actual
        agg = ProgressAggregator(steps_per_task)

        logger.info("Starting MPI dynamics calculation. Master waiting for results.")
        # print("MPI master: starting progress + result collection")

        # Collect progress and results
        results = [None] * num_tasks_actual
        tasks_done = 0

        status = MPI.Status()

        while tasks_done < num_tasks_actual:
            # Receive either progress or result from any worker
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == TAG_PROGRESS:
                task_id, inc = msg
                agg.handle(task_id, inc)
            elif tag == TAG_RESULT:
                task_id, result_data = msg
                results[task_id] = result_data
                tasks_done += 1
                logger.info(
                    "Master received result for task %s (%s/%s).",
                    task_id,
                    tasks_done,
                    num_tasks_actual,
                )
            else:
                logger.warning("Master received unknown tag %s", tag)

        # All tasks accounted for; close progress bars
        agg.close()
        logger.info("Dynamics calculation completed on all MPI workers.")

        # Merge results into the Data object
        logger.info("Collecting results from all MPI tasks.")
        for result in results:
            if result is not None:
                data.add_data(result)

        logger.info("Simulation complete (MPI).")
        data.log = get_log_output()
        return data

    # ---------- Worker ranks: 1..size-1 ----------
    else:
        # Receive list of tasks assigned to this rank: [(task_id, batch_seeds), ...]
        tasks = comm.recv(source=root, tag=TAG_TASKS)
        logger.info("Rank %s received %s tasks.", rank, len(tasks))

        # If no tasks, just return
        if not tasks:
            return None

        # Initialize timesteps on each worker's sim
        sim.initialize_timesteps()

        for task_id, batch_seeds in tasks:
            logger.info(
                "Rank %s starting task %s with seeds %s.", rank, task_id, batch_seeds
            )
            # Create a local copy of sim for this task (to be safe)
            sim_copy = copy.deepcopy(sim)
            sim_copy.settings.batch_size = len(batch_seeds)

            state = {"seed": batch_seeds}
            parameters = {}
            new_data = Data(batch_seeds)

            # Chunked progress reporting (similar to multiprocessing version)
            total_steps = len(getattr(sim_copy.settings, "t_update_n", [])) or 1
            chunk = max(1, total_steps // 100)  # ~100 updates per task
            pending = 0

            def report_progress(_task_id, inc=1, _task_id_fixed=task_id):
                nonlocal pending
                pending += inc
                if pending >= chunk:
                    comm.send((_task_id_fixed, pending), dest=root, tag=TAG_PROGRESS)
                    pending = 0

            # Run dynamics
            result_data = dynamics.run_dynamics(
                sim_copy, state, parameters, new_data, report_progress, task_id
            )

            # Flush any remaining progress
            if pending > 0:
                comm.send((task_id, pending), dest=root, tag=TAG_PROGRESS)

            # Send result back to root
            comm.send((task_id, result_data), dest=root, tag=TAG_RESULT)
            logger.info("Rank %s completed task %s.", rank, task_id)

        # Workers return None; only rank 0 returns the merged Data
        return None
