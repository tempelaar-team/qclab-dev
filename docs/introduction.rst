Introduction to QClab
=====================

QClab's Dynamics Core
---------------------

QClab utilizes a central dynamics core for the low-level execution of dynamics algorithms. Users interact with this core primarily through two key objects:

* **Simulation Class:**  Provides a comprehensive description of the physical system, simulation parameters, and observables to be calculated. Users typically create or modify existing simulation classes to model new physical systems.
* **Algorithm Class:** Defines a specific mixed quantum-classical algorithm. While most users won't modify these classes, advanced users can tailor or create algorithms by following the prescribed structure.

Both simulation and algorithm classes posess a set of required and standardized attributes used by the dynamics core. By manipulating these attributes, users can control the underlying physics and simulation properties, or implement diverse mixed quantum-classical algorithms.

To facillitate the execution of the dynamics, QClab comes with a variety of dynamics drivers which are optional (but recommended) functions that control the execution of the
dynamics core. In addition to the simulation and algorithm classes, the dynamics driver must be provided with a list of seeds that uniquely specify each mixed quantum-classical 
trajectory to be executed. 


.. figure:: images/code_structure.svg
   :alt: QClab Code Structure Diagram
   :width: 50%
   :align: center

   **Figure 1.** QClab Usage Diagram. The simulation class (a) is configured with input parameters. The simulation and algorithm classes (b) are passed to a dynamics driver (c), along with a list of seeds that uniquely identify each trajectory. The dynamics driver executes the simulation and returns a data class (d) containing the calculated observables.





