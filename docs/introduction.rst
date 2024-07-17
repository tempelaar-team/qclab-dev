Introduction to QClab
=====================

QClab contains a central dynamics core which is responsible for the low-level execution of dynamics algorithms. 
Rather than directly modifying the dynamics core the user interfaces with it through two primary objects:

* simulation class 
* algorithm class 

The simulation class contains a full description of the phyical system as well as parameters governing the mixed quantum-classical
simulation and any observables to calculate. In order to use QClab to study new physical models, a user has to make their own simulation
class or modify an existing one. The simulation class must posess a minimal set of attributes with prescribed inputs and outputs, by modifying these
standardized attributes the underlying physics or simulation properties can be changed. 

The algorithm class contains a fully specified mixed quantum-classical algorithm and in practice is something not to be modified by typical users. Advanced
or expert users may make modifications to existing algorithms or even implement their own algorithms by following the prescribed structure
of the algorithm class. Like the simulation class, each algorithm class contains a minimal set of attributes that are required for execution by the dynamics
core. By modifying these attributes appropriately, virtually any mixed quantum-classical algorithm can be implemented. 




.. figure:: images/code_structure.svg
    :alt: A diagram of the code structure
    :width: 100%
    :align: center
    
    *Figure 1.* Diagrammatic representation of how QClab is used. The simulation class (a) is equipped with inputs specifying the simulation parameters. 
    The simulation class and algorithm class (b) are then provided to a dynamics driver (c) with a list of seeds uniquely characterizing each trajectory. 
    After running the simulation for each trajectory the dynamics driver returns a data class (d) containing the requested obervables. 