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
    steps_per_batch: List[int]
    desc_total: str = "Batches"
    desc_batch: str = "Batch"

    # smoothing knobs
    min_fraction_gap: float = 0.10
    min_updates_between_switches: int = 50

    # internal state
    batches_done: int = 0
    per_batch: List[int] = field(init=False)
    batch_completed: List[bool] = field(init=False)
    active_batch: Optional[int] = field(init=False, default=None)
    updates_since_switch: int = field(init=False, default=0)

    def __post_init__(self):
        self.per_batch = [0] * len(self.steps_per_batch)
        self.batch_completed = [False] * len(self.steps_per_batch)
        self.num_batches = len(self.steps_per_batch)
        self.max_batch_steps = max(self.steps_per_batch) if self.steps_per_batch else 1

        common_kwargs = dict(leave=True, dynamic_ncols=True)

        if IN_NOTEBOOK:
            self.pbar_total = tqdm(
                total=self.num_batches,
                desc=self.desc_total,
                **common_kwargs,
            )
            self.pbar_batch = tqdm(
                total=self.max_batch_steps,
                desc=f"{self.desc_batch} (none)",
                **common_kwargs,
            )
        else:
            self.pbar_total = tqdm(
                total=self.num_batches,
                desc=self.desc_total,
                position=0,
                **common_kwargs,
            )
            self.pbar_batch = tqdm(
                total=self.max_batch_steps,
                desc=f"{self.desc_batch} (none)",
                position=1,
                **common_kwargs,
            )

    # ---- internal helpers ----

    def _set_active_batch(self, task_idx: int):
        self.active_batch = task_idx
        self.updates_since_switch = 0
        total = self.steps_per_batch[task_idx]
        done = min(self.per_batch[task_idx], total)  # clamp here

        self.pbar_batch.total = total
        self.pbar_batch.set_description(f"{self.desc_batch} {task_idx}")
        self.pbar_batch.n = done
        self.pbar_batch.refresh()

    def _pick_slowest_started_unfinished(self) -> Optional[int]:
        slowest_idx = None
        slowest_frac = float("inf")

        for idx, (done, total) in enumerate(zip(self.per_batch, self.steps_per_batch)):
            if done > 0 and not self.batch_completed[idx]:
                frac = done / total if total > 0 else 1.0
                if frac < slowest_frac:
                    slowest_frac = frac
                    slowest_idx = idx

        return slowest_idx

    # ---- public API ----
    def handle(self, batch_idx: int, inc: int = 1):
        """
        Called whenever any batch makes progress by `inc` steps.
        """
        self.updates_since_switch += 1

        # update this batch's step count
        self.per_batch[batch_idx] += inc
        # clamp to avoid n > total
        if self.per_batch[batch_idx] > self.steps_per_batch[batch_idx]:
            self.per_batch[batch_idx] = self.steps_per_batch[batch_idx]

        # *** NEW: immediately update the active batch bar, if this is it ***
        if self.active_batch == batch_idx:
            total = self.steps_per_batch[batch_idx]
            done = min(self.per_batch[batch_idx], total)
            self.pbar_batch.n = done
            self.pbar_batch.refresh()

        # check for completion of this batch (first time only)
        if (not self.batch_completed[batch_idx] and
                self.per_batch[batch_idx] >= self.steps_per_batch[batch_idx]):
            self.batch_completed[batch_idx] = True
            self.batches_done += 1
            self.pbar_total.update(1)

        # if we don't have an active batch yet, pick one (slowest started)
        if self.active_batch is None:
            slowest_idx = self._pick_slowest_started_unfinished()
            if slowest_idx is not None:
                self._set_active_batch(slowest_idx)
            return

        # if current active batch just finished, force switch to new slowest
        if self.active_batch is not None and self.batch_completed[self.active_batch]:
            slowest_idx = self._pick_slowest_started_unfinished()
            if slowest_idx is not None:
                self._set_active_batch(slowest_idx)
            # if slowest_idx is None, all batches are done; we already
            # set the bar to total above for the last increment
            return

        # otherwise consider switching based on "slowest" with hysteresis
        slowest_idx = self._pick_slowest_started_unfinished()
        if slowest_idx is None:
            return

        # current active's fraction
        active = self.active_batch
        if active is not None:
            active_total = self.steps_per_batch[active]
            active_done = self.per_batch[active]
            active_frac = active_done / active_total if active_total > 0 else 1.0
        else:
            active_frac = 1.0  # force picking slowest

        # slowest fraction
        slow_total = self.steps_per_batch[slowest_idx]
        slow_done = self.per_batch[slowest_idx]
        slow_frac = slow_done / slow_total if slow_total > 0 else 1.0

        should_switch = (
            (self.active_batch is None or
            slow_frac + self.min_fraction_gap < active_frac)
            and self.updates_since_switch >= self.min_updates_between_switches
        )

        if should_switch:
            self._set_active_batch(slowest_idx)
        # no need for an extra "elif active_batch" update block anymore,
        # because we updated the active bar at the top of the function
    def close(self):
        """
        Ensure both bars are visually complete and turn green
        (in notebooks) before closing.
        """
        # --- total bar ---
        # Force it to show all batches done
        if self.pbar_total.n < self.pbar_total.total:
            self.pbar_total.n = self.pbar_total.total
            self.pbar_total.refresh()

        # In notebook mode, tqdm exposes an ipywidgets Progress with bar_style
        if hasattr(self.pbar_total, "bar_style"):
            # 'success' is the green style in ipywidgets
            self.pbar_total.bar_style = "success"

        # --- batch bar ---
        if self.active_batch is not None:
            total = self.steps_per_batch[self.active_batch]
            if self.pbar_batch.total != total:
                self.pbar_batch.total = total

            if self.pbar_batch.n < total:
                self.pbar_batch.n = total
                self.pbar_batch.refresh()

        if hasattr(self.pbar_batch, "bar_style"):
            self.pbar_batch.bar_style = "success"

        # Finally close both
        self.pbar_total.close()
        self.pbar_batch.close()
