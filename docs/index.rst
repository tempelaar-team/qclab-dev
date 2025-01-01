.. qc_lab documentation master file, created by
   sphinx-quickstart on Mon Jul 15 16:14:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to QC Lab!
===================================

**QC Lab** is a Python package designed for implementing and executing quantum-classical (QC) dynamics simulations. It offers a framework for developing both physical models 
and QC algorithms that enables algorithms and models to be combined arbitrarily. QC Lab comes with a variety of already implemented models and algorithms which we hope 
encourage new researchers to explore the field of quantum-classical dynamics. It also provides a framework for advanced users to implement their own models and algorithms 
which we hope will become part of a growing library of quantum-classical dynamics tools.

Capabilities
------------

Dynamics Algorithms
```````````````````

The following algorithms are implemented in the current version of QC Lab. All methods are implemented making use of the complex-classical coordinate formalism established in [1] 
and are formally identical to their original real-space formulations while using equations of motion that are invariant to global unitary transformations 
of the classical and quantum representations. 

* :ref:`mf-algorithm` [2]
* :ref:`fssh-algorithm` [3]

Model Systems
`````````````

The following models are implemented in the current version of QC Lab.

* :ref:`sb-model` [4]
* :ref:`holstein_model` [5]



Using QC Lab
------------

.. toctree::
   :maxdepth: 1

   quickstart
   model_class
   algorithm_class
   dynamics_core
   ingredients
   holstein_model 
   spin_boson_model

Bibliography
------------

1. Miyazaki, K.; Krotz, A.; Tempelaar, R. Unitary Basis Transformations in Mixed Quantum-Classical Dynamics. arXiv April 23, 2024. http://arxiv.org/abs/2404.15614.
2. Tully, J. C. Mixed Quantum–Classical Dynamics. Faraday Discuss. 1998, 110 (0), 407–419. https://doi.org/10.1039/A801824C.
3. Hammes‐Schiffer, S.; Tully, J. C. Proton Transfer in Solution: Molecular Dynamics with Quantum Transitions. J. Chem. Phys. 1994, 101 (6), 4657–4667. https://doi.org/10.1063/1.467455.
4. Tempelaar, R.; Reichman, D. R. Generalization of Fewest-Switches Surface Hopping for Coherences. The Journal of Chemical Physics 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843.
5. Krotz, A.; Provazza, J.; Tempelaar, R. A Reciprocal-Space Formulation of Mixed Quantum–Classical Dynamics. J. Chem. Phys. 2021, 154 (22), 224101. https://doi.org/10.1063/5.0053177.





