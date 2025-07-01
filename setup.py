import os
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop


DEPENDENCIES = [
    "numpy",
    "tqdm",
    "h5py",
    "matplotlib",
    "numba",
]


DISABLE_NUMBA = os.environ.get("QC_LAB_DISABLE_NUMBA", "0") == "1"
DISABLE_H5PY = os.environ.get("QC_LAB_DISABLE_H5PY", "0") == "1"

if DISABLE_NUMBA:
    print("Numba is disabled. QC Lab will run without Numba optimizations.")
    DEPENDENCIES.remove("numba")
if DISABLE_H5PY:
    print("H5PY is disabled. QC Lab will run without HDF5 support.")
    DEPENDENCIES.remove("h5py")


install_requires = DEPENDENCIES

class develop(_develop):
    def run(self):
        target = os.path.join(self.install_lib, "qc_lab", "_config.py")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write(f"DISABLE_NUMBA = {DISABLE_NUMBA}\n")
            f.write(f"DISABLE_H5PY = {DISABLE_H5PY}\n")
        super().run()

class build_py(_build_py):
    def run(self):
        target = os.path.join(self.build_lib, "qc_lab", "_config.py")
        print(target)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write(f"DISABLE_NUMBA = {DISABLE_NUMBA}\n")
            f.write(f"DISABLE_H5PY = {DISABLE_H5PY}\n")
        super().run()



setup(
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "tests": ["pytest", "mpi4py",],
        "docs": ["sphinx", "pydata-sphinx-theme", "sphinx-design",],
    },
    cmdclass={
        "develop": develop,
        "build_py": build_py,
    },
)
