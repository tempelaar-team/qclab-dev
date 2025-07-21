from ._config import DISABLE_NUMBA


def qc_lab_custom_jit(func=None, **kwargs):
    """
    Dummy jit decorator that does nothing if numba is not available.
    """
    print("Using dummy jit decorator.")

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
    print("Using dummy njit decorator.")

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
