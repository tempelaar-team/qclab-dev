import os
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop


DEFAULT_DEPENDENCIES = [
    "numpy",
    "tqdm",
    "h5py",
    "matplotlib",
    "numba",
]

NONUMBA_DEPENDENCIES = [
    "numpy",
    "tqdm",
    "h5py",
    "matplotlib",
]

DISABLE_NUMBA = os.environ.get("QC_LAB_DISABLE_NUMBA", "0") == "1"

install_requires = NONUMBA_DEPENDENCIES if DISABLE_NUMBA else DEFAULT_DEPENDENCIES

class develop(_develop):
    def run(self):
        target = os.path.join(self.install_lib, "qc_lab", "_config.py")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write(f"DISABLE_NUMBA = {DISABLE_NUMBA}\n")
        super().run()

class build_py(_build_py):
    def run(self):
        target = os.path.join(self.build_lib, "qc_lab", "_config.py")
        print(target)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write(f"DISABLE_NUMBA = {DISABLE_NUMBA}\n")
        super().run()



setup(
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "tests": ["pytest", "mpi4py"]
    },
    cmdclass={
        "develop": develop,
        "build_py": build_py,
    },
)
