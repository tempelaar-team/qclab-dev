"""
This file contains the Simulation, State, and Data classes. It also contains additional 
functions for initializing and handling these objects.
"""

import ctypes
import numpy as np
import h5py
from qclab.parameter import Constants


def initialize_vector_objects(sim, batch_seeds):
    # Initialize state objects with batch seeds
    state_vector = VectorObject(len(batch_seeds), True)
    state_vector.add("seed", batch_seeds)
    state_vector._element_list = state_vector.to_element_list()
    state_vector.make_consistent()

    # add all initialized state variables to the state vector
    for n, _ in enumerate(batch_seeds):
        for name in sim.state._pointers.keys():
            state_vector._element_list[n].add(name, getattr(sim.state, name))
    state_vector.make_consistent()

    # construct parameter vector
    parameter_vector = VectorObject(len(batch_seeds), True)
    parameter_vector._element_list = parameter_vector.to_element_list()
    parameter_vector.make_consistent()

    # add all initialized parameter variables to the parameter vector
    for n, _ in enumerate(batch_seeds):
        for name in sim.model.parameters._pointers.keys():
            parameter_vector._element_list[n].add(
                name, getattr(sim.model.parameters, name)
            )
    parameter_vector.make_consistent()

    return parameter_vector, state_vector


def check_vars(state_list, full_state):
    """
    Check if the variables in state_list and full_state are correctly aligned.

    Args:
        state_list: List of state objects.
        full_state: The full state object.

    Raises:
        MemoryError: If there is a misalignment in the variables.
    """
    for name in full_state._pointers.keys():
        for n, state in enumerate(state_list):
            if not (
                state.get(name).__array_interface__["data"][0]
                == full_state.get(name)[n].__array_interface__["data"][0]
            ):
                raise MemoryError("Error, variable: ", name)


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


def get_ctypes_type(numpy_array):
    """
    Determines the corresponding ctypes type for a given NumPy array.

    Args:
        numpy_array: The NumPy array.

    Returns:
        The equivalent ctypes type.

    Raises:
        KeyError: If there is no mapping for the given NumPy dtype.
    """
    mapping = {
        "int8": ctypes.c_int8,
        "int16": ctypes.c_int16,
        "int32": ctypes.c_int32,
        "int64": ctypes.c_int64,
        "uint8": ctypes.c_uint8,
        "uint16": ctypes.c_uint16,
        "uint32": ctypes.c_uint32,
        "uint64": ctypes.c_uint64,
        "float32": ctypes.c_float,
        "float64": ctypes.c_double,
        "complex128": ctypes.c_double,
        "bool": ctypes.c_bool,
    }

    dtype = str(numpy_array.dtype)
    ctype = mapping.get(dtype, None)
    if ctype is None:
        raise KeyError(f"Missing type mapping for Numpy dtype: {dtype}")
    return ctype


class VectorObject:
    def __init__(self, size=1, vectorized=False):
        self._pointers = {}
        self._shapes = {}
        self._dtypes = {}
        self._output_dict = {}
        self._reshape_bool = {}
        self._size = size
        if size > 1:
            vectorized = True
        if size == 1:
            vectorized = False
        self._is_vectorized = vectorized
        self._update_list = False
        self._update_vector = False
        self._element_list = []

    def add(self, name, val):
        reshape_bool = False
        if self._is_vectorized:
            if len(np.shape(val)) < 2:
                reshape_bool = True
                val = val.reshape((*np.shape(val), 1))
        else:
            if not isinstance(val, np.ndarray):
                reshape_bool = True
                val = np.array([val])
        self._pointers[name] = val.ctypes.data_as(ctypes.POINTER(get_ctypes_type(val)))
        self._shapes[name] = np.shape(val)
        self._dtypes[name] = val.dtype
        self._reshape_bool[name] = reshape_bool
        if self._is_vectorized:
            if reshape_bool:
                self.__dict__[name] = self.get(name).view()[..., 0]
            else:
                self.__dict__[name] = self.get(name).view()
        else:
            if reshape_bool:
                self.__dict__[name] = self.get(name).view()[..., 0]
            else:
                self.__dict__[name] = self.get(name).view()

        if self._is_vectorized:
            self._update_list = True
        else:
            self._update_vector = True

    def get(self, name):
        ptr = self._pointers[name]
        shape = self._shapes[name]
        dtype = self._dtypes[name]
        dtype_size = np.dtype(dtype).itemsize
        total_bytes = np.prod(shape, dtype=np.int64) * dtype_size
        buffer = (ctypes.c_char * total_bytes).from_address(
            ctypes.addressof(ptr.contents)
        )
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)

    def modify(self, name, val):
        if name in self._pointers:
            if self._is_vectorized and self._reshape_bool[name]:
                val = val.reshape((*np.shape(val), 1))
            if not self._is_vectorized and self._reshape_bool[name]:
                val = np.array([val])
            if val.dtype == self._dtypes[name] and np.shape(val) == self._shapes[name]:
                ctypes.memmove(
                    ctypes.addressof(self._pointers[name].contents),
                    val.ctypes.data_as(ctypes.c_void_p),
                    val.nbytes,
                )
                if self._is_vectorized:
                    if self._reshape_bool[name]:
                        self.__dict__[name] = self.get(name).view()[..., 0]
                    else:
                        self.__dict__[name] = self.get(name).view()
                else:
                    if self._reshape_bool[name]:
                        self.__dict__[name] = self.get(name).view()[..., 0]
                    else:
                        self.__dict__[name] = self.get(name).view()
            else:
                self.add(name, val)
        else:
            self.add(name, val)

    def __setattr__(self, name, val):
        if name[0] == "_":
            super().__setattr__(name, val)
        else:
            self.modify(name, val)

    def copy(self):
        """
        Create a copy of the state object.

        Returns:
            A new state object that is a copy of the current state.
        """
        out = VectorObject(self._size)
        for name in self._pointers:
            out.add(name, np.copy(getattr(self, name)))
        return out

    def to_element_list(self):
        element_list = [VectorObject(1, False) for _ in range(self._size)]
        for ind, state in enumerate(element_list):
            for name in self._pointers.keys():
                state.add(name, getattr(self, name)[ind])
        return element_list

    def from_element_list(self, element_list):
        assert self._size == len(element_list), ValueError(
            f"Size mismatch: {self._size} != {len(element_list)}"
        )
        for name in element_list[0]._pointers:
            self.add(
                name, np.array([getattr(element, name) for element in element_list])
            )
        self._element_list = self.to_element_list()
        self._update_list = False
        self._update_vector = False
        _ = [
            element.__setattr__("_update_vector", False)
            for element in self._element_list
        ]
        _ = [
            element.__setattr__("_update_list", False) for element in self._element_list
        ]

    def make_consistent(self):
        assert self._is_vectorized, ValueError("Object is not vectorized")
        if self._update_list:
            self._element_list = self.to_element_list()
        if self._update_vector:
            self.from_element_list(self._element_list)
        update_vector_list = np.array(
            [element._update_vector for element in self._element_list]
        )
        if np.any(update_vector_list):
            self.from_element_list(self._element_list)

    def collect_output_variables(self, output_variables):
        """
        Collect output variables for the state.

        Args:
            output_variables: List of output variable names.
        """
        for var in output_variables:
            self._output_dict[var] = self.get(var)






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
