"""
This module contains the parallel driver using the multiprocessing library.
"""

# at top of the module
import multiprocessing as mp
import threading
import logging
import copy
import queue
import numpy as np

import qclab.dynamics as dynamics
from qclab.utils import get_log_output, reset_log_output
from qclab import Data
from .progressbar_utils import ProgressAggregator

logger = logging.getLogger(__name__)

_PROGRESS_QUEUE = None


def _init_worker_progress(q):
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = q


def _run_dynamics_worker(sim, state, parameters, data, task_id):
    """
    Runs inside a Pool worker.

    Batches progress updates so we don't spam the queue every timestep.
    """

    # target ~100 progress updates per task
    total_steps = len(getattr(sim.settings, "t_update_n", [])) or 1
    chunk = max(1, total_steps // 100)

    pending = 0

    def report_progress(_task_id, inc=1):
        nonlocal pending
        pending += inc
        if pending >= chunk and _PROGRESS_QUEUE is not None:
            _PROGRESS_QUEUE.put((task_id, pending))
            pending = 0

    logger.info("Worker starting task %s", task_id)
    result = dynamics.run_dynamics(
        sim, state, parameters, data, report_progress, task_id
    )

    # flush any remaining increments
    if pending > 0 and _PROGRESS_QUEUE is not None:
        _PROGRESS_QUEUE.put((task_id, pending))

    return task_id, result


def parallel_driver_multiprocessing(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the python library multiprocessing.

    .. rubric:: Args
    sim: Simulation
        The simulation object containing the model, algorithm, initial state, and settings.
    seeds: ndarray, optional
        An array of integer seeds for the trajectories. If None, seeds will be
        generated automatically.
    data: Data, optional
        A Data object for collecting output data. If None, a new Data object
        will be created.
    num_tasks: int, optional
        The number of worker processes to use for parallel processing. If None, the
        number of available CPU cores will be used.

    .. rubric:: Returns
    data: Data
        The updated Data object containing collected output data.
    """
    # Clear any in-memory log output from previous runs.
    reset_log_output()

    # First initialize the model constants.
    sim.model.initialize_constants()

    if data is None:
        data = Data()

    # Handle seeds and num_trajs (same logic as serial_driver)
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

    # Number of worker processes
    if num_tasks is None:
        size = mp.cpu_count()
    else:
        size = num_tasks
    logger.info("Using %s worker processes for parallel processing.", size)

    # Determine the number of batches required to execute the total number of trajectories.
    if num_trajs % sim.settings.batch_size == 0:
        num_batches = num_trajs // sim.settings.batch_size
    else:
        num_batches = num_trajs // sim.settings.batch_size + 1

    logger.info(
        "Running %s batches with %s seeds in each batch.",
        num_batches,
        sim.settings.batch_size,
    )

    batch_seeds_list = (
        np.zeros((num_batches * sim.settings.batch_size), dtype=int) + np.nan
    )
    batch_seeds_list[:num_trajs] = seeds
    batch_seeds_list = batch_seeds_list.reshape((num_batches, sim.settings.batch_size))

    sim.initialize_timesteps()
    jobs = []
    for n in range(num_batches):
        batch_seeds = batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)
        if len(batch_seeds) == 0:
            continue
        sim_copy = copy.deepcopy(sim)
        sim_copy.settings.batch_size = len(batch_seeds)
        state = {"seed": batch_seeds}
        parameters = {}
        new_data = Data(batch_seeds)
        logger.info("Preparing batch %s with seeds %s.", n + 1, batch_seeds)
        jobs.append((n, sim_copy, state, parameters, new_data))

    num_batches = len(jobs)
    logger.info("Number of batches: %s", num_batches)

    if num_batches == 0:
        logger.info("No batches to run; returning early.")
        data.log = get_log_output()
        return data

    # Progress aggregation: one task per batch, each with len(t_update_n) steps
    steps_per_batch = [len(sim.settings.t_update_n)] * num_batches
    agg = ProgressAggregator(steps_per_batch)

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    worker_args = [
        (sim_copy, state, parameters, new_data, task_id)
        for (task_id, sim_copy, state, parameters, new_data) in jobs
    ]
    stop_event = threading.Event()

    def progress_loop():
        """
        Runs in a separate thread in the main process.
        Reads (task_id, inc) from progress_queue and updates ProgressAggregator.
        """
        while not stop_event.is_set() or not progress_queue.empty():
            try:
                task_id, inc = progress_queue.get(timeout=0.1)
                agg.handle(task_id, inc)
            except queue.Empty:
                # No progress message currently; loop again
                continue

        # Final drain to be extra sure we didn't miss anything
        while True:
            try:
                task_id, inc = progress_queue.get_nowait()
                agg.handle(task_id, inc)
            except queue.Empty:
                break

        agg.close()

    logger.info("Starting dynamics calculation (multiprocessing with Pool).")
    print("starting calculation (Pool + progress thread)")

    # Start the progress thread
    progress_thread = threading.Thread(target=progress_loop, daemon=True)
    progress_thread.start()

    # Run jobs in a Pool; workers are reused, so numba JIT happens once per worker
    with ctx.Pool(
        processes=size,
        initializer=_init_worker_progress,
        initargs=(progress_queue,),
    ) as pool:
        # starmap is blocking, but progress_thread keeps the bars updating
        results = pool.starmap(_run_dynamics_worker, worker_args)

    # Signal the progress thread to finish after the queue is drained
    stop_event.set()
    progress_thread.join()

    logger.info("Dynamics calculation completed.")
    logger.info("Collecting results from all tasks.")

    # results is a list of (task_id, Data)
    results_sorted = [None] * num_batches
    for task_id, result_data in results:
        results_sorted[task_id] = result_data

    for result in results_sorted:
        if result is not None:
            data.add_data(result)

    logger.info("Simulation complete.")
    # Attach collected log output.
    data.log = get_log_output()
    return data
