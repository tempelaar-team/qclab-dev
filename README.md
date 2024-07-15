# qclab
qclab (pronounced "qc" "lab") is a package for carrying out mixed quantum-classical (MQC) dynamics simulations. It implements popular MQC algorithms in a manner that is agnostic to the underlying physical system and flexible to the mixed quantum-classical algorithm. Along with its dynamics algorithms, qclab comes equipped with a variety of physical models that can be used as starting points for exploring MQC dynamics or to implement more elaborate models. Check out the documentation at ____ to learn more about implementing new models. 

## Capabilities
### Dynamics Algorithms
* Mean-field (Ehrenfest) dynamics [1]
* Fewest-switches surface hopping (FSSH) dynamics [2]
* Coherent fewest-switches surface hopping (FSSH) dynamics [3]
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

## Bibliography
[1] Tully, J. C. Mixed Quantum–Classical Dynamics. Faraday Discuss. 110, 1998, 407–419.
(2) Hammes‐Schiffer, S.; Tully, J. C. Proton Transfer in Solution: Molecular Dynamics with Quantum Transitions. J. Chem. Phys. 1994, 101 (6), 4657–4667. https://doi.org/10.1063/1.467455.
(3) Tempelaar, R.; Reichman, D. R. Generalization of Fewest-Switches Surface Hopping for Coherences. The Journal of Chemical Physics 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843.
