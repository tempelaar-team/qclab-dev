"""
This file contains the Constants class, which is used to handle 
constants and trigger updates automatically when constants change.
"""


class Constants:
    """
    Class to handle constants and trigger updates when constants change.
    """

    def __init__(self, update_function=None):
        self._updating = False
        self._init_complete = False
        self._update_function = update_function

    def __setattr__(self, name, value):
        """
        Overrides attribute setting to call the update function after the attribute is changed,
        preventing recursion.
        """
        # Set the attribute
        super().__setattr__(name, value)
        # Check if already updating to prevent recursion
        if not self._updating and name not in {
            "_updating",
            "_update_function",
            "_init_complete",
        }:
            if self._init_complete:
                self._updating = True  # Set the flag to prevent recursion
                if self._update_function is not None:
                    self._update_function()  # Call the update function
                self._updating = False  # Reset the flag

    def get(self, name, default=None):
        """
        Get the value of a constant.
        """
        return getattr(self, name, default)
    