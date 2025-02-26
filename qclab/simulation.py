"""
This file contains the Simulation, State, and Data classes. It also contains additional 
functions for initializing and handling these objects.
"""

import ctypes
import numpy as np
import h5py
from qclab.parameter import Constants
import warnings

def initialize_vector_objects(sim, batch_seeds):
    # this takes the numpy arrays inside the state and parameter objects
    # and creates a first index corresponding to the number of trajectories

    state_vector = VectorObject()
    state_vector.seed = batch_seeds
    for name in sim.state.__dict__.keys():
        obj = getattr(sim.state, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(batch_seeds), *init_shape), dtype=obj.dtype)
                + obj[np.newaxis]
            )
            setattr(state_vector, name, new_obj)
    parameter_vector = VectorObject()
    parameter_vector.seed = batch_seeds
    for name in sim.model.parameters.__dict__.keys():
        obj = getattr(sim.model.parameters, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(batch_seeds), *init_shape), dtype=obj.dtype)
                + obj[np.newaxis]
            )
            setattr(parameter_vector, name, new_obj)
    return parameter_vector, state_vector


class Data:
    """
    The data object handles the collection of data from the dynamics driver.
    """

    def __init__(self):
        self.data_dic = {"seed": np.array([], dtype=int)}

    def initialize_output_total_arrays_(self, sim, full_state):
        """
        Initialize the output total arrays for data collection.

        Args:
            sim: The simulation object.
            full_state: The full state object.
        """
        self.data_dic["seed"] = np.copy(full_state.get("seed"))
        for key, val in full_state._output_dict.items():
            self.data_dic[key] = np.zeros(
                (len(sim.settings.tdat_output), *np.shape(val)[1:]), dtype=val.dtype
            )

    def add_to_output_total_arrays(self, sim, full_state, t_ind):
        """
        Add data to the output total arrays.

        Args:
            sim: The simulation object.
            full_state: The full state object.
            t_ind: Time index.
        """
        for key, val in full_state._output_dict.items():
            if key in self.data_dic:
                # fill an existing data storage array
                self.data_dic[key][int(t_ind / sim.settings.dt_output_n)] = np.sum(
                    val, axis=0
                )
            else:
                # initialize the data storage array and then fill it
                self.data_dic[key] = np.zeros(
                    (len(sim.settings.tdat_output), *np.shape(val)[1:]), dtype=val.dtype
                )
                self.data_dic[key][int(t_ind / sim.settings.dt_output_n)] = np.sum(
                    val, axis=0
                )

    def add_data(self, new_data):
        """
        Add new data to the existing data dictionary.

        Args:
            new_data: The new data object.
        """
        for key, val in new_data.data_dic.items():
            if key == "seed":
                self.data_dic[key] = np.concatenate(
                    (self.data_dic[key], val.flatten()), axis=0
                )
            else:
                if key in self.data_dic:
                    self.data_dic[key] += val
                else:
                    self.data_dic[key] = val

    def save_as_h5(self, filename):
        """
        Save the data as an h5 archive.
        """
        with h5py.File(filename, "w") as h5file:
            self._recursive_save(h5file, "/", self.data_dic)
        return

    def load_from_h5(self, filename):
        """
        Load a data object from an h5 archive.
        """
        with h5py.File(filename, "r") as h5file:
            self._recursive_load(h5file, "/", self.data_dic)
        return self

    def _recursive_save(self, h5file, path, dic):
        """

        Recursively saves dictionary contents to an HDF5 group.

        Args:
            h5file (h5py.File): The HDF5 file object.
            path (str): The path to the group in the HDF5 file.
            dic (dict): The dictionary to save.
        """
        for key, item in dic.items():
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5file[path + key] = item
            elif isinstance(item, dict):
                self._recursive_save(h5file, path + key + "/", item)
            elif isinstance(item, list):
                h5file[path + key] = np.array(item)
            else:
                raise ValueError(f"Cannot save {type(item)} type")

    def _recursive_load(self, h5file, path, dic):
        """
        Recursively loads dictionary contents from an HDF5 group.

        Args:
            h5file (h5py.File): The HDF5 file object.
            path (str): The path to the group in the HDF5 file.
            dic (dict): The dictionary to load the data into.
        """
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                dic[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                dic[key] = {}
                self._recursive_load(h5file, path + key + "/", dic[key])


class VectorObject:
    def __init__(self):
        self._output_dict = {}

    def __getattr__(self, name):
        """ 
        If an attribute is not in the VectorObject it returns None rather than throwing an error.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            warnings.warn(f"Attribute {name} not found in VectorObject.", UserWarning)
            return None

    def collect_output_variables(self, output_variables):
        """
        Collect output variables for the state.

        Args:
            output_variables: List of output variable names.
        """
        for var in output_variables:
            self._output_dict[var] = getattr(self, var)


class Simulation:
    """
    The simulation object represents the entire simulation process.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = dict(
            tmax=10, dt=0.01, dt_output=0.1, num_trajs=10, batch_size=1
        )
        settings = {**self.default_settings, **settings}
        self.settings = Constants()
        for key, val in settings.items():
            setattr(self.settings, key, val)
        self.algorithm = None
        self.model = None
        self.state = VectorObject()

    def initialize_timesteps(self):
        """
        Initialize the timesteps for the simulation based on the parameters.
        """
        self.settings.tmax_n = np.round(
            self.settings.tmax / self.settings.dt, 1
        ).astype(int)
        self.settings.dt_output_n = np.round(
            self.settings.dt_output / self.settings.dt, 1
        ).astype(int)
        self.settings.tdat = (
            np.arange(0, self.settings.tmax_n + 1, 1) * self.settings.dt
        )
        self.settings.tdat_n = np.arange(0, self.settings.tmax_n + 1, 1)
        self.settings.tdat_output = (
            np.arange(0, self.settings.tmax_n + 1, self.settings.dt_output_n)
            * self.settings.dt
        )
        self.settings.tdat_output_n = np.arange(
            0, self.settings.tmax_n + 1, self.settings.dt_output_n
        )

    def generate_seeds(self, data):
        """
        Generate new seeds for the simulation.

        Args:
            data: The data object containing existing seeds.

        Returns:
            new_seeds: Array of new seeds.
        """
        if len(data.data_dic["seed"]) > 1:
            new_seeds = (
                np.max(data.data_dic["seed"])
                + np.arange(self.settings.num_trajs, dtype=int)
                + 1
            )
        else:
            new_seeds = np.arange(self.settings.num_trajs, dtype=int)
        return new_seeds
