"""
This file contains the Simulation, State, and Data classes. It also contains additional 
functions for initializing and handling these objects.
"""
import ctypes
import numpy as np
import h5py
from qclab.parameter import Parameter


def initialize_state_objects(sim, batch_seeds):
    """
    Initialize state objects for the simulation.

    Args:
        sim: The simulation object.
        batch_seeds: List of seeds for batch processing.

    Returns:
        state_list: List of state objects.
        full_state: The full state object.
    """
    batch_seeds_single = np.arange(1)
    state_list = [sim.state.copy() for _ in batch_seeds_single]

    # Initialize seed in each state
    # for n in range(len(batch_seeds_single)):
    #    state_list[n].add('seed', batch_seeds_single[n][np.newaxis])
    for n, seed in enumerate(batch_seeds_single):
        state_list[n].modify('seed', seed[np.newaxis])

    # Create a full_state
    full_state = new_full_state(state_list)
    state_list = new_state_list(full_state)
    sim.algorithm.determine_vectorized()

    # Initialization recipe
    for ind, func in enumerate(sim.algorithm.initialization_recipe):
        if sim.algorithm.initialization_recipe_vectorized_bool[ind]:
            full_state = func(sim, full_state)
            state_list = new_state_list(full_state)
        else:
            state_list = [func(sim, state) for state in state_list]
            full_state = new_full_state(state_list)
            state_list = new_state_list(full_state)

    # Update recipe
    for ind, func in enumerate(sim.algorithm.update_recipe):
        if sim.algorithm.update_recipe_vectorized_bool[ind]:
            full_state = func(sim, full_state)
            state_list = new_state_list(full_state)
        else:
            state_list = [func(sim, state) for state in state_list]
            full_state = new_full_state(state_list)
            state_list = new_state_list(full_state)

    # Output recipe
    for ind, func in enumerate(sim.algorithm.output_recipe):
        if sim.algorithm.output_recipe_vectorized_bool[ind]:
            full_state = func(sim, full_state)
            state_list = new_state_list(full_state)
        else:
            state_list = [func(sim, state) for state in state_list]
            full_state = new_full_state(state_list)
            state_list = new_state_list(full_state)

    # Zero out every variable
    for name in full_state._pointers.keys():
        full_state.modify(name, full_state.get(name) * 0)

    state_list = new_state_list(full_state)
    state_list = [state_list[0].copy() for _ in batch_seeds]

    # Initialize state objects with batch seeds
    for n, seed in enumerate(batch_seeds):
        for name in sim.state._pointers.keys():
            state_list[n].add(name, sim.state.get(name))
        state_list[n].modify('seed', seed[np.newaxis])

    full_state = new_full_state(state_list)
    state_list = new_state_list(full_state)
    check_vars(state_list, full_state)
    full_state.collect_output_variables(sim.algorithm.output_variables)

    return state_list, full_state


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
            if not (state.get(name).__array_interface__['data'][0] ==
                    full_state.get(name)[n].__array_interface__['data'][0]):
                raise MemoryError('Error, variable: ', name)


def new_state_list(full_state):
    batch_size = full_state._size
    state_list = [State() for _ in range(batch_size)]
    attr_names = full_state._pointers.keys()

    for ind, state in enumerate(state_list):
        for name in attr_names:
            state.add(name, full_state.get(name)[ind])

    return state_list

def new_full_state(state_list):
    full_state = State(len(state_list))
    attribute_names = state_list[0]._pointers.keys()
    for name in attribute_names:
        full_var = np.array([state.get(name) for state in state_list])
        full_state.add(name, full_var)
    return full_state


class Data:
    """
    The data object handles the collection of data from the dynamics driver.
    """

    def __init__(self):
        self.data_dic = {'seed': np.array([], dtype=int)}

    def initialize_output_total_arrays(self, sim, full_state):
        """
        Initialize the output total arrays for data collection.

        Args:
            sim: The simulation object.
            full_state: The full state object.
        """
        self.data_dic['seed'] = np.copy(full_state.get('seed'))
        for key, val in full_state._output_dict.items():
            self.data_dic[key] = np.zeros(
                (len(sim.parameters.tdat_output), *np.shape(val)[1:]), dtype=val.dtype)

    def add_to_output_total_arrays(self, sim, full_state, t_ind):
        """
        Add data to the output total arrays.

        Args:
            sim: The simulation object.
            full_state: The full state object.
            t_ind: Time index.
        """
        for key, val in full_state._output_dict.items():
            self.data_dic[key][int(
                t_ind / sim.parameters.dt_output_n)] = np.sum(val, axis=0)

    def add_data(self, new_data):
        """
        Add new data to the existing data dictionary.

        Args:
            new_data: The new data object.
        """
        for key, val in new_data.data_dic.items():
            if key == 'seed':
                self.data_dic[key] = np.concatenate(
                    (self.data_dic[key], val.flatten()), axis=0)
            else:
                if key in self.data_dic:
                    self.data_dic[key] += val
                else:
                    self.data_dic[key] = val

    def save_as_h5(self, filename):
        """ 
        Save the data as an h5 archive.
        """
        with h5py.File(filename, 'w') as h5file:
            self._recursive_save(h5file, '/', self.data_dic)
        return

    def load_from_h5(self, filename):
        """
        Load a data object from an h5 archive.
        """
        with h5py.File(filename, 'r') as h5file:
            self._recursive_load(h5file, '/', self.data_dic)
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
                self._recursive_save(h5file, path + key + '/', item)
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
                self._recursive_load(h5file, path + key + '/', dic[key])


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
        'int8': ctypes.c_int8,
        'int16': ctypes.c_int16,
        'int32': ctypes.c_int32,
        'int64': ctypes.c_int64,
        'uint8': ctypes.c_uint8,
        'uint16': ctypes.c_uint16,
        'uint32': ctypes.c_uint32,
        'uint64': ctypes.c_uint64,
        'float32': ctypes.c_float,
        'float64': ctypes.c_double,
        'complex128': ctypes.c_double,
    }

    dtype = str(numpy_array.dtype)
    ctype = mapping.get(dtype, None)
    if ctype is None:
        raise KeyError(f'Missing type mapping for Numpy dtype: {dtype}')
    return ctype

class State:
    def __init__(self, size=1):
        self._pointers = {}
        self._shapes = {}
        self._dtypes = {}
        self._output_dict = {}
        self._size = size
        if self._size > 1:
            self._is_vectorized = True 
        else:
            self._is_vectorized = False
    def collect_output_variables(self, output_variables):
        """
        Collect output variables for the state.

        Args:
            output_variables: List of output variable names.
        """
        for var in output_variables:
            self._output_dict[var] = self.get(var)

    def copy(self):
        """
        Create a copy of the state object.

        Returns:
            A new state object that is a copy of the current state.
        """
        out = State(self._size)
        for name in self._pointers:
            out.add(name, np.copy(self.get(name)))
        return out
    
    def add(self, name, val):
        if self._is_vectorized:
            if len(np.shape(val)) < 2:
                val = val.reshape((*np.shape(val), 1))
        else:
            if not isinstance(val, np.ndarray):
                val = np.array([val])
        self._pointers[name] = val.ctypes.data_as(ctypes.POINTER(get_ctypes_type(val)))
        self._shapes[name] = np.shape(val)
        self._dtypes[name] = val.dtype
        self.__dict__[name] = self.get(name).view()

    def get(self, name):
        ptr = self._pointers[name]
        shape = self._shapes[name]
        dtype = self._dtypes[name]
        dtype_size = np.dtype(dtype).itemsize
        total_bytes = np.prod(shape, dtype=np.int64) * dtype_size
        buffer = (ctypes.c_char *
                  total_bytes).from_address(ctypes.addressof(ptr.contents))
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)
    
    def modify(self, name, val):
        if name in self._pointers:
            if not self._is_vectorized and not isinstance(val, np.ndarray):
                val = np.array([val])
            assert val.dtype == self._dtypes[name], TypeError(f"Type mismatch: {val.dtype} != {self._dtypes[name]}")
            ctypes.memmove(ctypes.addressof(self._pointers[name].contents),
                           val.ctypes.data_as(ctypes.c_void_p),
                           val.nbytes)
            self.__dict__[name] = self.get(name).view()
        else:
            self.add(name, val)
    
    def __getattr__(self, name):
        if name in self._pointers:
            if self._is_vectorized:
                return self.__dict__[name].reshape(*self._shapes[name][:-1])
            else:
                return self.__dict__[name][...,0]
        else:
            return self.__dict__[name]
        
    def __setattr__(self, name, val):
        if name[0] == '_':
            super().__setattr__(name, val)
        else:
            self.modify(name, val)


class Simulation:
    """
    The simulation object represents the entire simulation process.
    """

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}
        self.default_parameters = dict(
            tmax=10, dt=0.01, dt_output=0.1, num_trajs=10, batch_size=1)
        parameters = {**self.default_parameters, **parameters}
        self.parameters = Parameter()
        for key, val in parameters.items():
            setattr(self.parameters, key, val)
        self.algorithm = None
        self.model = None
        self.state = State()

    def initialize_timesteps(self):
        """
        Initialize the timesteps for the simulation based on the parameters.
        """
        self.parameters.tmax_n = np.round(
            self.parameters.tmax / self.parameters.dt, 1).astype(int)
        self.parameters.dt_output_n = np.round(
            self.parameters.dt_output / self.parameters.dt, 1).astype(int)
        self.parameters.tdat = np.arange(
            0, self.parameters.tmax_n + 1, 1) * self.parameters.dt
        self.parameters.tdat_n = np.arange(0, self.parameters.tmax_n + 1, 1)
        self.parameters.tdat_output = np.arange(0, self.parameters.tmax_n + 1,
                                                self.parameters.dt_output_n) * self.parameters.dt
        self.parameters.tdat_output_n = np.arange(
            0, self.parameters.tmax_n + 1, self.parameters.dt_output_n)

    def generate_seeds(self, data):
        """
        Generate new seeds for the simulation.

        Args:
            data: The data object containing existing seeds.

        Returns:
            new_seeds: Array of new seeds.
        """
        if len(data.data_dic['seed']) > 1:
            new_seeds = np.max(
                data.data_dic['seed']) + np.arange(self.parameters.num_trajs, dtype=int) + 1
        else:
            new_seeds = np.arange(self.parameters.num_trajs, dtype=int)
        return new_seeds
