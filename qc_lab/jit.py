"""Utilities for optional :mod:`numba` JIT compilation.

This module provides dummy implementations of the ``jit`` and ``njit``
decorators from :mod:`numba`. These no-op implementations are used when the
``DISABLE_NUMBA`` flag is set to ``True`` or when :mod:`numba` is not available.
"""
import warnings
from ._config import DISABLE_NUMBA



def qc_lab_custom_jit(func=None, **kwargs):
    """
    Dummy jit decorator that does nothing if numba is not available.
    """
    warnings.warn(
        "Numba is disabled; using dummy jit decorator.",
        UserWarning,
    )

    def decorator(func):
        return func

    if func is None:
        return decorator
    else:
        return decorator(func)


def qc_lab_custom_njit(func=None, **kwargs):
    """
    Dummy njit decorator that does nothing if numba is not available.
    """
    warnings.warn(
        "Numba is disabled; using dummy njit decorator.",
        UserWarning,
    )

    def decorator(func):
        return func

    if func is None:
        return decorator
    else:
        return decorator(func)


if DISABLE_NUMBA:
    njit = qc_lab_custom_njit
    jit = qc_lab_custom_jit
else:
    try:
        from numba import jit, njit
    except ImportError:
        jit = qc_lab_custom_jit
        njit = qc_lab_custom_njit
