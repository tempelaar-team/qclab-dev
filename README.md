# qc-lab
qc-lab is a package for carrying out mixed quantum-classical dynamics simulations. It implements mean-field (Ehrenfest), fewest-switches surface hopping, and coherent fewest-switches surface hopping algorithms in a complex-valued classical coordinate in a manner that is totally generic. 

## Running pyMQC on a personal (non cluster/HPC) machine
To start a ray cluster install ray and in terminal run 
ray start --head
then run the code as
python main.py inputfile '{"address":"auto"}'
## Running pyMQC on a cluster
