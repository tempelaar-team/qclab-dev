"""
setup.py
"""

from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'QC Lab: a python package for quantum-classical modeling.'
LONG_DESCRIPTION = 'QC Lab is a python package for quantum-classical modeling.'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="QC Lab",
    version=VERSION,
    author="Alex Krotz",
    author_email="alexkrotz2024@u.northwestern.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy==1.24.0', 'tqdm','h5py==3.12.1','numba==0.60.0'],
    keywords=['surface hopping', 'mixed quantum-classical dynamics',
              'theoretical chemistry', 'ehrenfest', 'python', 'quantum-classical'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Linux :: Linux",
    ]
)
