import pytest
from qclab import Constants

def test_constants_class():
    """
    Test the ParameterClass functionality.

    This test verifies that the ParameterClass correctly initializes with default parameters,
    triggers the update function when parameters change, and prevents recursion.

    Steps:
    1. Initialize the ParameterClass with default parameters and an update function.
    2. Set the default parameters and mark initialization as complete.
    3. Verify that the parameters are correctly assigned.
    4. Change a parameter and verify that the update function is triggered.
    5. Verify that the update function correctly modifies the parameter.
    6. Change another parameter and verify that the update function is not triggered.
    7. Verify that the parameters are correctly assigned and the update function is triggered only once.

    Assertions:
    - The 'p1' attribute should be 1 after initialization.
    - The 'p2' attribute should be 2 after initialization.
    - The 'p1' attribute should be 20 after setting it to 10 (update function doubles it).
    - The 'p2' attribute should be 10 after setting it to 10.
    - The 'p1' attribute should be 40 after setting 'p2' to 10 (update function doubles it again).
    """
    default_parameters = {'p1': 1, 'p2': 2}

    def update_function():
        constants.p1 *= 2

    constants = Constants(update_function)
    for key, val in default_parameters.items():
        setattr(constants, key, val)
    constants._init_complete = True

    # Verify initial parameter values
    assert constants.p1 == 1
    assert constants.p2 == 2

    # Change parameter and verify update function is triggered
    constants.p1 = 10
    assert constants.p1 == 20

    # Change another parameter and verify update function is not triggered
    constants.p2 = 10
    assert constants.p2 == 10

    # Verify update function is triggered again
    assert constants.p1 == 40

if __name__ == "__main__":
    pytest.main()