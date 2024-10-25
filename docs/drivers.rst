Dynamics Drivers 
==========================

Drivers act as an interface with QC-lab's dynamics core. 

Serial Driver 
-------------


Ray Parallel Driver 
-----------------------------
The Ray parallel driver is suitable for cases where your machine has multiple cpus. It is not suitable in its current form for parallelization across nodes on a 
cluster, however it is in principle possible to implement such paralellization with Ray. We encourage users to implement their own drivers customized towards their 
particular computing setup. The Ray parallel driver that comes with QC-lab should be suitable for a personal machine or individual nodes on a cluster. 

Only one additional argument is needed, which specifies the number of cpus over which to parallelize::

      from qclab.drivers.ray_driver import dynamics_parallel_ray

      ncpus = 8 # for a machine with 8 processors 
      data_spin_boson_mf = dynamics_parallel_ray(algorithm = MeanFieldDynamics, sim = sim, seeds = seeds, ncpus = ncpus, data = simulation.Data())


SLURM parallel driver 
-------------------------------
