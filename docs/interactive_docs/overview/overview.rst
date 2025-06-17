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


.. raw:: html

    <style>
    .diagram {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-family: sans-serif;
    }

    /* make the whole simulation box clickable */
    .simulation-link {
        text-decoration: none;
        color: inherit;
        display: inline-block;
    }

    .simulation-container {
        border: 2px solid #ccc;
        border-radius: 15px;
        padding: 20px;
        background: #f8f8f8;
        text-align: center;       /* <<< center everything inside */
    }
    .simulation-container:hover {
        border-color: #999;
    }

    /* centered, with just a little bottom‐spacing */
    .simulation-title {
        font-weight: bold;
        font-size: 1.1em;
        margin: 0 0 15px;         /* zero top/right/left, 15px bottom */
    }

    .row {
        display: flex;
        justify-content: space-around;
        gap: 20px;
    }

    .box {
        display: inline-block;
        padding: 15px 25px;
        background: #8fbf8f;
        color: #000;
        text-decoration: none;
        border-radius: 5px;
        font-weight: 500;
        text-align: center;
        min-width: 120px;
    }
    .box:hover {
        background: #7fae7f;
    }

    .arrow {
        width: 2px;
        height: 30px;
        background: #333;
        position: relative;
        margin: 20px 0;
    }
    .arrow::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 8px solid #333;
    }
    </style>

    <div class="diagram">
    <a href="simulation.html" class="simulation-link">
        <div class="simulation-container">
        <div class="simulation-title">Simulation Object</div>
        <div class="row">
            <a href="model.html"     class="box">Model Object</a>
            <a href="algorithm.html" class="box">Algorithm Object</a>
        </div>
        </div>
    </a>

    <div class="arrow"></div>

    <a href="driver.html" class="box">Dynamics Driver</a>

    <div class="arrow"></div>

    <a href="data.html" class="box">Data Object</a>
    </div>



