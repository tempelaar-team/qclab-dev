# pyMQC
Python code for mixed quantum-classical dynamics. This code is formulated in terms of 
complex-valued classical coordinates enabling the dynamics to be carried out in 
arbitrary representations for both the quantum and classical subsystems. 
## Running pyMQC on a personal (non cluster/HPC) machine
To start a ray cluster install ray and in terminal run 
ray start --head
then run the code as
python main.py inputfile '{"address":"auto"}'
## Running pyMQC on a cluster
