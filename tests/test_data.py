"""
Tests for :class:`qc_lab.data.Data` save and load functionality.
"""

import numpy as np

from qc_lab import Data


def test_save_load_numpy_scalars_and_iterables(tmp_path):
    """
    Ensure numpy scalar types and iterables persist through save/load.
    """

    data = Data()
    data.data_dict = {
        "int32": np.int32(1),
        "float32": np.float32(2.5),
        "list": [1, 2, 3],
        "tuple": (4, 5, 6),
        "nested": {"inner_int32": np.int32(7), "inner_list": [8, 9]},
    }

    filename = tmp_path / "test.h5"
    data.save(filename)

    loaded = Data().load(filename)

    assert isinstance(loaded.data_dict["int32"], np.int32)
    assert loaded.data_dict["int32"] == np.int32(1)

    assert isinstance(loaded.data_dict["float32"], np.float32)
    assert loaded.data_dict["float32"] == np.float32(2.5)

    assert np.array_equal(loaded.data_dict["list"], np.asarray([1, 2, 3]))
    assert np.array_equal(loaded.data_dict["tuple"], np.asarray([4, 5, 6]))

    nested = loaded.data_dict["nested"]
    assert isinstance(nested["inner_int32"], np.int32)
    assert nested["inner_int32"] == np.int32(7)
    assert np.array_equal(nested["inner_list"], np.asarray([8, 9]))
