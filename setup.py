from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'qc_lab python package'
LONG_DESCRIPTION = 'qc_lab python package'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="qclab", 
        version=VERSION,
        author="Alex Krotz",
        author_email="alexkrotz2024@u.northwestern.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy','numba','ray','tqdm', 'scipy>=0.16', 'dill'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)