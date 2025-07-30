"""
QC Lab package initialization.
"""

from qc_lab.simulation import Simulation
from qc_lab.variable import Variable
from qc_lab.data import Data
from qc_lab.model import Model
from qc_lab.algorithm import Algorithm
from qc_lab.constants import Constants

# Configure logging to capture all messages in memory. This needs to happen
# on import so that any module using the standard logging facilities will have
# its messages captured without printing to ``stdout``.
from qc_lab.utils import configure_memory_logger, get_log_output

# Set up the logger with INFO level by default. The handler returned by this
# function keeps all log messages in memory so they can later be attached to a
# :class:`Data` instance.
configure_memory_logger()

__all__ = [
    "Simulation",
    "Variable",
    "Data",
    "Model",
    "Algorithm",
    "Constants",
    "configure_memory_logger",
    "get_log_output",
]
