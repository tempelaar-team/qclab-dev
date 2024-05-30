# qc-lab
qc-lab is a flexible dynamics code for carrying out popular mixed quantum-classical algorithms on custom defined physical systems. 
## Running pyMQC on a personal (non cluster/HPC) machine
To start a ray cluster install ray and in terminal run 
ray start --head
then run the code as
python main.py inputfile '{"address":"auto"}'
## Running pyMQC on a cluster
