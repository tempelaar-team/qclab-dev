.. _quickstart:
Quickstart Guide 
~~~~~~~~~~~~~~~~~~~

QC Lab is organized into models and algorithms which are combined into a simulation object. The simulation object fully defines a quantum-classical dynamics simulation 
which is then carried out by a dynamics driver. This guide will walk you through the process of setting up a simulation object and running a simulation.

The simulation class is contained in qc_lab.simulation and contains a set of default simulation parameters. 

::
    
    from qc_lab.simulation import Simulation 
    sim = Simulation(parameters = dict())
    ## print out default parameters 
    print(vars(sim.parameters))

Here, the simulation object is instantiated without an input dictionary of parameters and will use the default parameters contained in ::sim.parameters::.