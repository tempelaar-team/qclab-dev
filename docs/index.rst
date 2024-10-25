.. qc_lab documentation master file, created by
   sphinx-quickstart on Mon Jul 15 16:14:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to QC-lab!
===================================

**QC-lab** (pronounced "qc" "lab") is a Python package designed for performing mixed quantum-classical (MQC) dynamics simulations. 
It offers a flexible framework for carrying out MQC simulations in a way that is independent of the physical system being studied. 
QC-lab also provides several built-in physical models and MQC algorithms, serving as starting points for exploring MQC dynamics or creating more elaborate simulations.
QC-lab has been designed to accomadate virtually any mixed quantum-classical algorithm and we encourage you to implement your own and contribute it to our public library. 

Capabilities
------------

Dynamics Algorithms
```````````````````

The following algorithms are implemented making use of the complex-classical coordinate formalism established in [1] 
and are formally identical to their original real-space formulation while being invariant to global unitary 
transformations of the classical and quantum bases. 


* :ref:`mf-algorithm` [2]
* :ref:`fssh-algorithm` [3]
* :ref:`cfssh-algorithm` [4]

Model Systems
`````````````

* :ref:`sb-model` [4]
* :ref:`dba-model` [5]
* :ref:`holstein-model` [6]



Using QC-lab
-----------

.. toctree::
   :maxdepth: 2

   introduction
   model_class
   algorithm_class
   dynamics_core

Bibliography
------------

1. Miyazaki, K.; Krotz, A.; Tempelaar, R. Unitary Basis Transformations in Mixed Quantum-Classical Dynamics. arXiv April 23, 2024. http://arxiv.org/abs/2404.15614.
2. Tully, J. C. Mixed Quantum–Classical Dynamics. Faraday Discuss. 1998, 110 (0), 407–419. https://doi.org/10.1039/A801824C.
3. Hammes‐Schiffer, S.; Tully, J. C. Proton Transfer in Solution: Molecular Dynamics with Quantum Transitions. J. Chem. Phys. 1994, 101 (6), 4657–4667. https://doi.org/10.1063/1.467455.
4. Tempelaar, R.; Reichman, D. R. Generalization of Fewest-Switches Surface Hopping for Coherences. The Journal of Chemical Physics 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843.
5. Bondarenko, A. S.; Tempelaar, R. Overcoming Positivity Violations for Density Matrices in Surface Hopping. J. Chem. Phys. 2023, 158 (5), 054117. https://doi.org/10.1063/5.0135456.
6. Krotz, A.; Provazza, J.; Tempelaar, R. A Reciprocal-Space Formulation of Mixed Quantum–Classical Dynamics. J. Chem. Phys. 2021, 154 (22), 224101. https://doi.org/10.1063/5.0053177.
7. Krotz, A; Tempelaar, R. Treating geometric phase effects in nonadiabatic dynamics. Phys. Rev. A. 2024, 109, 032210. https://doi.org/10.1103/PhysRevA.109.032210





