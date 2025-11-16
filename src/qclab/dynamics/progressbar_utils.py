from dataclasses import dataclass, field
from typing import List, Optional
from tqdm.auto import tqdm  # or from tqdm import tqdm if you prefer


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


IN_NOTEBOOK = _in_notebook()


@dataclass
class ProgressAggregator:
    steps_per_task: List[int]
    desc_total: str = "Tasks"
    desc_task: str = "Task"

    # smoothing knobs
    min_fraction_gap: float = 0.10
    min_updates_between_switches: int = 50

    # internal state
    tasks_done: int = 0
    per_task: List[int] = field(init=False)
    task_completed: List[bool] = field(init=False)
    active_task: Optional[int] = field(init=False, default=None)
    updates_since_switch: int = field(init=False, default=0)

    def __post_init__(self):
        self.per_task = [0] * len(self.steps_per_task)
        self.task_completed = [False] * len(self.steps_per_task)
        self.num_tasks = len(self.steps_per_task)
        self.max_task_steps = max(self.steps_per_task) if self.steps_per_task else 1

        common_kwargs = dict(leave=True, dynamic_ncols=True)

        if IN_NOTEBOOK:
            self.pbar_total = tqdm(
                total=self.num_tasks,
                desc=self.desc_total,
                **common_kwargs,
            )
            self.pbar_task = tqdm(
                total=self.max_task_steps,
                desc=f"{self.desc_task} (none)",
                **common_kwargs,
            )
        else:
            self.pbar_total = tqdm(
                total=self.num_tasks,
                desc=self.desc_total,
                position=0,
                **common_kwargs,
            )
            self.pbar_task = tqdm(
                total=self.max_task_steps,
                desc=f"{self.desc_task} (none)",
                position=1,
                **common_kwargs,
            )

    # ---- internal helpers ----

    def _set_active_task(self, task_idx: int):
        self.active_task = task_idx
        self.updates_since_switch = 0
        total = self.steps_per_task[task_idx]
        done = min(self.per_task[task_idx], total)  # clamp here

        self.pbar_task.total = total
        self.pbar_task.set_description(f"{self.desc_task} {task_idx}")
        self.pbar_task.n = done
        self.pbar_task.refresh()

    def _pick_slowest_started_unfinished(self) -> Optional[int]:
        slowest_idx = None
        slowest_frac = float("inf")

        for idx, (done, total) in enumerate(zip(self.per_task, self.steps_per_task)):
            if done > 0 and not self.task_completed[idx]:
                frac = done / total if total > 0 else 1.0
                if frac < slowest_frac:
                    slowest_frac = frac
                    slowest_idx = idx

        return slowest_idx

    # ---- public API ----

    def handle(self, task_idx: int, inc: int = 1):
        """
        Called whenever any task makes progress by `inc` steps.
        """
        self.updates_since_switch += 1

        # update this task's step count
        self.per_task[task_idx] += inc
        # clamp to avoid n > total
        if self.per_task[task_idx] > self.steps_per_task[task_idx]:
            self.per_task[task_idx] = self.steps_per_task[task_idx]

        # check for completion of this task (first time only)
        if (not self.task_completed[task_idx] and
                self.per_task[task_idx] >= self.steps_per_task[task_idx]):
            self.task_completed[task_idx] = True
            self.tasks_done += 1
            self.pbar_total.update(1)

        # if we don't have an active task yet, pick one (slowest started)
        if self.active_task is None:
            slowest_idx = self._pick_slowest_started_unfinished()
            if slowest_idx is not None:
                self._set_active_task(slowest_idx)
                return

        # if current active task just finished, force switch to new slowest
        if self.active_task is not None and self.task_completed[self.active_task]:
            slowest_idx = self._pick_slowest_started_unfinished()
            if slowest_idx is not None:
                self._set_active_task(slowest_idx)
            return

        # otherwise consider switching based on "slowest" with hysteresis
        slowest_idx = self._pick_slowest_started_unfinished()
        if slowest_idx is None:
            return

        # current active's fraction
        active = self.active_task
        if active is not None:
            active_total = self.steps_per_task[active]
            active_done = self.per_task[active]
            active_frac = active_done / active_total if active_total > 0 else 1.0
        else:
            active_frac = 1.0  # force picking slowest

        # slowest fraction
        slow_total = self.steps_per_task[slowest_idx]
        slow_done = self.per_task[slowest_idx]
        slow_frac = slow_done / slow_total if slow_total > 0 else 1.0

        should_switch = (
            (self.active_task is None or
             slow_frac + self.min_fraction_gap < active_frac)
            and self.updates_since_switch >= self.min_updates_between_switches
        )

        if should_switch:
            self._set_active_task(slowest_idx)
        elif self.active_task is not None:
            # stay on current task; just update its n (clamped)
            total = self.steps_per_task[self.active_task]
            done = min(self.per_task[self.active_task], total)
            self.pbar_task.n = done
            self.pbar_task.refresh()

    def close(self):
        self.pbar_total.close()

# from dataclasses import dataclass, field
# from typing import List, Optional
# from tqdm.auto import tqdm  # auto-chooses notebook vs terminal


# def _in_notebook() -> bool:
#     """Best-effort check for Jupyter/IPython notebook."""
#     try:
#         from IPython import get_ipython
#         shell = get_ipython().__class__.__name__
#         return shell == "ZMQInteractiveShell"  # Jupyter
#     except Exception:
#         return False


# IN_NOTEBOOK = _in_notebook()


# @dataclass
# class ProgressAggregator:
#     steps_per_task: List[int]
#     desc_total: str = "Tasks"
#     desc_task: str = "Task"

#     # now track completed *tasks*, not total steps
#     tasks_done: int = 0
#     per_task: List[int] = field(init=False)
#     task_completed: List[bool] = field(init=False)
#     active_task: Optional[int] = field(init=False, default=None)

#     def __post_init__(self):
#         self.per_task = [0] * len(self.steps_per_task)
#         self.task_completed = [False] * len(self.steps_per_task)
#         self.num_tasks = len(self.steps_per_task)
#         # still useful to keep around if the driver wants it
#         self.total_steps = sum(self.steps_per_task)
#         self.max_task_steps = max(self.steps_per_task)

#         common_kwargs = dict(leave=True, dynamic_ncols=True)

#         if IN_NOTEBOOK:
#             # Jupyter: no position, they’ll stack vertically as separate widgets
#             self.pbar_total = tqdm(
#                 total=self.num_tasks,
#                 desc=self.desc_total,
#                 **common_kwargs,
#             )
#             self.pbar_task = tqdm(
#                 total=self.max_task_steps,
#                 desc=f"{self.desc_task} 0",
#                 **common_kwargs,
#             )
#         else:
#             # Terminal: use position to stack lines
#             self.pbar_total = tqdm(
#                 total=self.num_tasks,
#                 desc=self.desc_total,
#                 position=0,
#                 **common_kwargs,
#             )
#             self.pbar_task = tqdm(
#                 total=self.max_task_steps,
#                 desc=f"{self.desc_task} 0",
#                 position=1,
#                 **common_kwargs,
#             )

#     # ----- task switching logic -----

#     def _set_active_task(self, task_idx: int):
#         self.active_task = task_idx
#         total = self.steps_per_task[task_idx]
#         done = self.per_task[task_idx]

#         self.pbar_task.reset(total=total)
#         self.pbar_task.set_description(f"{self.desc_task} {task_idx}")
#         self.pbar_task.n = done
#         self.pbar_task.refresh()

#     def _maybe_pick_initial_task(self, task_idx: int):
#         if self.active_task is None:
#             self._set_active_task(task_idx)

#     def _maybe_switch_task_on_completion(self, just_finished_idx: int):
#         if self.active_task != just_finished_idx:
#             return

#         # find any unfinished task
#         for idx, (done, total) in enumerate(zip(self.per_task, self.steps_per_task)):
#             if done < total:
#                 self._set_active_task(idx)
#                 return
#         # all done → leave last bar at 100%

#     # ----- public API -----

#     def handle(self, task_idx: int, inc: int = 1):
#         # update per-task step count
#         self.per_task[task_idx] += inc

#         # pick first active task if needed
#         self._maybe_pick_initial_task(task_idx)

#         # update active task bar in steps
#         if self.active_task == task_idx:
#             self.pbar_task.update(inc)

#         # if this task just finished (first time), bump the *task* bar by 1
#         if (not self.task_completed[task_idx] and
#                 self.per_task[task_idx] >= self.steps_per_task[task_idx]):
#             self.task_completed[task_idx] = True
#             self.tasks_done += 1
#             self.pbar_total.update(1)
#             # and maybe switch the second bar to another unfinished task
#             self._maybe_switch_task_on_completion(task_idx)

#     def close(self):
#         self.pbar_total.close()
#         self.pbar_task.close()
# from dataclasses import dataclass, field
# from typing import List, Optional
# from tqdm.auto import tqdm  # auto-chooses notebook vs terminal


# def _in_notebook() -> bool:
#     """Best-effort check for Jupyter/IPython notebook."""
#     try:
#         from IPython import get_ipython
#         shell = get_ipython().__class__.__name__
#         return shell == "ZMQInteractiveShell"  # Jupyter
#     except Exception:
#         return False


# IN_NOTEBOOK = _in_notebook()


# @dataclass
# class ProgressAggregator:
#     steps_per_task: List[int]
#     desc_total: str = "Total"
#     desc_task: str = "Task"

#     total_done: int = 0
#     per_task: List[int] = field(init=False)
#     active_task: Optional[int] = field(init=False, default=None)

#     def __post_init__(self):
#         self.per_task = [0] * len(self.steps_per_task)
#         self.total_steps = sum(self.steps_per_task)
#         self.max_task_steps = max(self.steps_per_task)

#         common_kwargs = dict(leave=True, dynamic_ncols=True)

#         if IN_NOTEBOOK:
#             # Jupyter: no position, they’ll stack vertically as separate widgets
#             self.pbar_total = tqdm(
#                 total=self.total_steps,
#                 desc=self.desc_total,
#                 **common_kwargs,
#             )
#             self.pbar_task = tqdm(
#                 total=self.max_task_steps,
#                 desc=f"{self.desc_task} 0",
#                 **common_kwargs,
#             )
#         else:
#             # Terminal: use position to stack lines
#             self.pbar_total = tqdm(
#                 total=self.total_steps,
#                 desc=self.desc_total,
#                 position=0,
#                 **common_kwargs,
#             )
#             self.pbar_task = tqdm(
#                 total=self.max_task_steps,
#                 desc=f"{self.desc_task} 0",
#                 position=1,
#                 **common_kwargs,
#             )

#     # ----- task switching logic (same as before) -----

#     def _set_active_task(self, task_idx: int):
#         self.active_task = task_idx
#         total = self.steps_per_task[task_idx]
#         done = self.per_task[task_idx]

#         self.pbar_task.reset(total=total)
#         self.pbar_task.set_description(f"{self.desc_task} {task_idx}")
#         self.pbar_task.n = done
#         self.pbar_task.refresh()

#     def _maybe_pick_initial_task(self, task_idx: int):
#         if self.active_task is None:
#             self._set_active_task(task_idx)

#     def _maybe_switch_task_on_completion(self, just_finished_idx: int):
#         if self.active_task != just_finished_idx:
#             return

#         # find any unfinished task
#         for idx, (done, total) in enumerate(zip(self.per_task, self.steps_per_task)):
#             if done < total:
#                 self._set_active_task(idx)
#                 return
#         # all done → leave last bar at 100%

#     # ----- public API -----

#     def handle(self, task_idx: int, inc: int = 1):
#         self.per_task[task_idx] += inc
#         self.total_done += inc

#         # update total bar
#         self.pbar_total.update(inc)

#         # pick first active task if needed
#         self._maybe_pick_initial_task(task_idx)

#         # update active task bar
#         if self.active_task == task_idx:
#             self.pbar_task.update(inc)

#         # maybe switch task on completion
#         if self.per_task[task_idx] == self.steps_per_task[task_idx]:
#             self._maybe_switch_task_on_completion(task_idx)

#     def close(self):
#         self.pbar_total.close()
#         self.pbar_task.close()


# from dataclasses import dataclass, field
# from typing import List
# from tqdm import tqdm

# from dataclasses import dataclass, field
# from typing import List, Optional
# from tqdm import tqdm


# @dataclass
# class ProgressAggregator:
#     steps_per_task: List[int]
#     desc_total: str = "Total"
#     desc_task: str = "Task"

#     total_done: int = 0
#     per_task: List[int] = field(init=False)
#     active_task: Optional[int] = field(init=False, default=None)

#     def __post_init__(self):
#         self.per_task = [0] * len(self.steps_per_task)
#         self.total_steps = sum(self.steps_per_task)

#         # global progress bar
#         self.pbar_total = tqdm(
#             total=self.total_steps, position=0, desc=self.desc_total, leave=True
#         )

#         # per-task bar (will be configured when we pick an active task)
#         self.pbar_task = tqdm(
#             total=1, position=1, desc=f"{self.desc_task} (none)", leave=True
#         )

#     # ----- internal helpers -----

#     def _set_active_task(self, task_idx: int):
#         """Switch the second bar to show a different task."""
#         self.active_task = task_idx
#         total = self.steps_per_task[task_idx]
#         done = self.per_task[task_idx]

#         # reset() clears n and lets us set a new total
#         self.pbar_task.reset(total=total)
#         self.pbar_task.set_description(f"{self.desc_task} {task_idx}")
#         # restore current progress
#         self.pbar_task.n = done
#         self.pbar_task.refresh()

#     def _maybe_pick_initial_task(self, task_idx: int):
#         """If no active task yet, choose this one as the first."""
#         if self.active_task is None:
#             self._set_active_task(task_idx)

#     def _maybe_switch_task_on_completion(self, just_finished_idx: int):
#         """If the currently displayed task just finished, switch to another unfinished task."""
#         if self.active_task != just_finished_idx:
#             return

#         # find any unfinished task
#         for idx, (done, total) in enumerate(zip(self.per_task, self.steps_per_task)):
#             if done < total:
#                 self._set_active_task(idx)
#                 return

#         # if we reach here, all tasks are done; leave the bar at 100%

#     # ----- public API -----

#     def handle(self, task_idx: int, inc: int = 1):
#         """
#         Handle a single (task_idx, inc) message.
#         Can be called from any backend (serial/MP/MPI) via the root.
#         """
#         # bookkeeping
#         self.per_task[task_idx] += inc
#         self.total_done += inc

#         # update total bar
#         self.pbar_total.update(inc)

#         # if no active task yet, pick one
#         self._maybe_pick_initial_task(task_idx)

#         # update task bar only for the active task
#         if self.active_task == task_idx:
#             self.pbar_task.update(inc)

#         # if this task just finished, maybe switch to another unfinished task
#         if self.per_task[task_idx] == self.steps_per_task[task_idx]:
#             self._maybe_switch_task_on_completion(task_idx)

#     def close(self):
#         self.pbar_total.close()
#         self.pbar_task.close()

# @dataclass
# class ProgressAggregator:
#     steps_per_task: List[int]
#     desc_total: str = "Total"
#     desc_fastest: str = "Fastest"

#     total_done: int = 0
#     per_task: List[int] = field(init=False)

#     def __post_init__(self):
#         self.per_task = [0] * len(self.steps_per_task)
#         self.total_steps = sum(self.steps_per_task)
#         self.max_task_steps = max(self.steps_per_task)

#         self.pbar_total = tqdm(
#             total=self.total_steps, position=0, desc=self.desc_total
#         )
#         self.pbar_fastest = tqdm(
#             total=self.max_task_steps, position=1, desc=self.desc_fastest
#         )

#     def handle(self, task_id: int, inc: int = 1):
#         """Handle a single (task_id, inc) message."""
#         self.per_task[task_id] += inc
#         self.total_done += inc

#         # update total
#         self.pbar_total.update(inc)

#         # update fastest
#         fastest_done = max(self.per_task)
#         self.pbar_fastest.n = fastest_done
#         self.pbar_fastest.refresh()

#     def close(self):
#         self.pbar_total.close()
#         self.pbar_fastest.close()