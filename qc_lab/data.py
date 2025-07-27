"""
This module defines the Data class, which is used to handle the collection and storage of data
output from algorithms in QC Lab.
"""

import numpy as np
import h5py
from qc_lab._config import DISABLE_H5PY


class Data:
    """
    The data object handles the collection of data from the dynamics driver.
    """

    def __init__(self, seeds=None):
        if seeds is None:
            seeds = np.array([], dtype=int)
        self.data_dict = {"seed": seeds, "norm_factor": 0}

    def add_output_to_data_dict(self, sim, state, t_ind):
        """Add data to the output total arrays.

        Args:
            sim (Simulation): The simulation object containing settings and
                parameters.
            state (Variable): The state object containing the current
                simulation state.
            t_ind (int): The current time index in the simulation.
        """
        # Check if the norm_factor is zero, if it is, save it from the state object.
        if self.data_dict["norm_factor"] == 0:
            if not (hasattr(state, "norm_factor")):
                raise ValueError(
                    "The state object does not have a norm_factor attribute."
                )
            self.data_dict["norm_factor"] = state.norm_factor
        for key, val in state.output_dict.items():
            if not (key in self.data_dict):
                # If the key is not in the data_dict, initialize it with zeros.
                self.data_dict[key] = np.zeros(
                    (len(sim.settings.tdat_output), *np.shape(val)[1:]), dtype=val.dtype
                )
            # store the data in the data_dict at the correct time index
            self.data_dict[key][t_ind // sim.settings.dt_collect_n] = (
                np.sum(val, axis=0) / self.data_dict["norm_factor"]
            )

    def add_data(self, new_data):
        """Add new data to the existing data dictionary.

        Args:
            new_data (Data): A ``Data`` instance containing the new data to
                merge.
        """
        new_norm_factor = (
            new_data.data_dict["norm_factor"] + self.data_dict["norm_factor"]
        )
        for key, val in new_data.data_dict.items():
            if key == "seed":
                self.data_dict[key] = np.concatenate(
                    (self.data_dict[key], val.flatten()), axis=0
                )
            elif key != "norm_factor":
                if key in self.data_dict:
                    self.data_dict[key] = (
                        self.data_dict[key] * self.data_dict["norm_factor"]
                        + val * new_data.data_dict["norm_factor"]
                    ) / new_norm_factor
                else:
                    self.data_dict[key] = val
        self.data_dict["norm_factor"] = new_norm_factor

    def save(self, filename):
        """Save the data to disk.

        If ``h5py`` is available the data is stored as an HDF5 archive; otherwise
        each variable is saved using :func:`numpy.savez`.

        Args:
            filename (str): The file name to save the data to.
        """
        if DISABLE_H5PY:
            np.savez(filename, **self.data_dict)
        else:
            with h5py.File(filename, "w") as h5file:
                self._recursive_save(h5file, "/", self.data_dict)

    def load(self, filename):
        """Load a :class:`Data` object from ``filename``.

        Args:
            filename (str): The file name to load the data from.

        Returns:
            Data: The current instance populated with the loaded data.
        """
        if DISABLE_H5PY:
            loaded = np.load(filename)
            self.data_dict = {key: loaded[key] for key in loaded.files}
            return self
        with h5py.File(filename, "r") as h5file:
            self._recursive_load(h5file, "/", self.data_dict)
        return self

    def _recursive_save(self, h5file, path, dic):
        """Recursively save dictionary contents to an HDF5 group.

        Args:
            h5file (h5py.File): The HDF5 file object to save data into.
            path (str): The current path in the HDF5 file.
            dic (dict): The dictionary to save.
        """
        for key, item in dic.items():
            if isinstance(
                item,
                (
                    np.ndarray,
                    np.int64,
                    np.float64,
                    str,
                    bytes,
                    int,
                    float,
                    bool,
                    complex,
                ),
            ):
                h5file[path + key] = item
            elif isinstance(item, dict):
                self._recursive_save(h5file, path + key + "/", item)
            elif isinstance(item, list):
                h5file[path + key] = np.array(item)
            else:
                raise ValueError(f"Cannot save {key} with type {type(item)}.")

    def _recursive_load(self, h5file, path, dic):
        """Recursively load dictionary contents from an HDF5 group.

        Args:
            h5file (h5py.File): The HDF5 file object to load data from.
            path (str): The current path in the HDF5 file.
            dic (dict): The dictionary to populate with loaded data.
        """
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                dic[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                dic[key] = {}
                self._recursive_load(h5file, path + key + "/", dic[key])
