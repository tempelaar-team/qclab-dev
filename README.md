# qclab
qclab (pronounced "qc" "lab") is a package for carrying out mixed quantum-classical (MQC) dynamics simulations. It implements popular MQC algorithms in a manner that is agnostic to the underlying physical system and flexible to the mixed quantum-classical algorithm. Along with its dynamics algorithms, qclab comes equipped with a variety of physical models that can be used as starting points for exploring MQC dynamics or to implement more elaborate models. Check out the documentation at ____ to learn more about implementing new models. 

## Capabilities
### Dynamics Algorithms
* Mean-field (Ehrenfest) dynamics
* Fewest-switches surface hopping (FSSH) dynamics
* Coherent fewest-switches surface hopping (FSSH) dynamics
### Model Systems
* Spin-boson model
* Donor-bridge-acceptor model
* Holstein lattice model 
    - 1-D lattice model with a single electronic state and Einstein phonon per site. Electronic sites have nearest-neighbor interactions. 
## installing qc-lab
qclab can be installed with pip 
```
pip install qclab
```
