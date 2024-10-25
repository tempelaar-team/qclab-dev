Algorithm Class
===============

The Algorithm class can be thought of as a recipe for the dynamics algorithm and ``qclab.recipes`` is the recipe book. Each algorithm class has a 
well-defined structure that can be arbitrarily populated with "ingredients" constituting different mixed 
quantum-classical algorithms. This guide describes the structure of the algorithm class. To learn about the 
structure of ingredients see the ingredients page #TODO insert ingredients link here. 


Algorithm Attributes 
--------------------

.. py:data:: recipe.model 
    :type: model class 
    
        model class containing physical system and simulation parameters. 

.. py:data:: recipe.initialize
    :type: list 

        A list of ingredients executed before the propagation of the simulation, responsible for initializing the relevant quantities. 

.. py:data:: recipe.upate
    :type: list 
    
        A list of ingredients executed at every propagation timestep, responsible for evolving the total system. 

.. py:data:: recipe.output_names
    :type: list 

        A list of strings denoting variables in the ``state`` object that are retrieved and output at each output timestep. 

.. py:data:: recipe.state
    :type: namespace 

        A namespace containing the dynamical quantities stored in the sysem. The dynamics core loads model into state as state.model 


.. py:function:: recipe.defaults(model)

        A function that loads default values into the model class 

    :param: model class 
    :return: model class


Algorithms included in QC-lab 
-----------------------------

.. toctree::
   :maxdepth: 3

   mf_algorithm_class
   fssh_algorithm_class
   cfssh_algorithm_class