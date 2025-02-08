"""
This file contains the Constants class, which is used to handle 
constants and trigger updates automatically when constants change.
"""

class Constants:
    """
    Class to handle constants and trigger updates when constants change.

    Attributes:
        _updating (bool): Flag to prevent recursion.
        _init_complete (bool): Flag to prevent running update_function until initialization 
            is complete.
        _update_function (function): The function to call when constants are updated.
    """

    def __init__(self, update_function=None):
        """
        Initializes the ParameterClass with an update function.

        Args:
            update_function (function): The function to call when constants are updated.
        """
        # Flag to prevent recursion
        self._updating = False
        # Flag to prevent running update_function until initialization is complete
        self._init_complete = False
        self._update_function = update_function

    def __setattr__(self, name, value):
        """
        Overrides attribute setting to call the update function after the attribute is changed,
        preventing recursion.

        Args:
            name (str): The name of the attribute.
            value: The value to assign to the attribute.
        """
        # Set the attribute
        super().__setattr__(name, value)
        # Check if already updating to prevent recursion
        if not self._updating and name not in {'_updating', '_update_function', '_init_complete'}:
            if self._init_complete:
                self._updating = True  # Set the flag to prevent recursion
                if self._update_function is not None:
                    self._update_function()  # Call the update function
                self._updating = False  # Reset the flag
