.. _overview:

====================
QC Lab Architecture
====================

We prefer to think of running a quantum-classical simulation as like preparing a delicious meal. 
In that analogy, the simulation object is equipped with a model object that provides "ingredients", 
an algorithm object that provides the "recipe", and a dynamics driver that executes the recipe step by step.
The data object is the output dish that results from the simulation.

A key feature of this architecture is that different models and algorithms can be swapped in and out with nearly no 
modification to the rest of the simulation. This enables users to easily experiment with different
quantum-classical simulation algorithms and models, allowing for rapid prototyping and testing of new ideas.

This construction is depicted diagramatically below, click on any component to learn more about it!


.. graphviz::
        
        digraph flow {
        rankdir=TB;
        
        node [
        shape=box
        style="filled,rounded,shadow"
        fillcolor="#007acc"      // primary blue
        fontcolor=white
        fontsize=12
        margin="0.3,0.2"         // vertical, horizontal padding
        ];

        sim     [label="Simulation Object", URL="../../user_guide/simulation.html"];
        model  [label="Model Object", URL="../../user_guide/models/models.html"];
        algo [label="Algorithm Object",     URL="../../user_guide/algorithms/algorithms.html"];
        driver     [label="Dynamics Driver",        URL="../../user_guide/drivers/drivers.html"];
        data [label="Data Object",    URL="../../user_guide/data_object.html"];

        model -> sim;
        algo -> sim;
        sim -> driver;
        driver -> data;
        }
        

.. container:: graphviz-center

    .. graphviz::
        
        digraph flow {
        rankdir=TB;
        
        node [
        shape=box
        style="filled,rounded,shadow"
        fillcolor="#007acc"      // primary blue
        fontcolor=white
        fontsize=12
        margin="0.3,0.2"         // vertical, horizontal padding
        ];

        sim     [label="Simulation Object", URL="../../user_guide/simulation.html"];
        model  [label="Model Object", URL="../../user_guide/models/models.html"];
        algo [label="Algorithm Object",     URL="../../user_guide/algorithms/algorithms.html"];
        driver     [label="Dynamics Driver",        URL="../../user_guide/drivers/drivers.html"];
        data [label="Data Object",    URL="../../user_guide/data_object.html"];

        model -> sim;
        algo -> sim;
        sim -> driver;
        driver -> data;
        }


.. button-link:: simulation.html
    :color: primary
    :shadow:
    :align: center

    Simulation Object

.. button-link:: model.html
    :color: primary
    :shadow:
    :align: center

    Model Object

.. button-link:: algorithm.html
    :color: primary
    :shadow:
    :align: center

    Algorithm Object

.. button-link:: driver.html
    :color: primary
    :shadow:
    :align: center

    Dynamics Driver

.. button-link:: data.html
    :color: primary
    :shadow:
    :align: center

    Data Object


