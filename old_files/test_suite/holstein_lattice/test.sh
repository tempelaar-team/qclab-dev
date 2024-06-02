rm -r *_output
script_path='/home/akrotz/Research/pyMQC/'
ray stop
echo 'Testing Holstein lattice Mean-Field '
ray start --head
python $script_path/main.py holstein_lattice_input_MF '{"address":"auto"}'
ray stop
echo 'Testing Holstein Fewest-Switches Surface Hopping Deterministic '
ray start --head
python $script_path/main.py holstein_lattice_input_FSSH_deterministic '{"address":"auto"}'
ray stop
echo 'Testing Holstein Fewest-Switches Surface Hopping Stochastic '
ray start --head
python $script_path/main.py holstein_lattice_input_FSSH_stochastic '{"address":"auto"}'
ray stop
echo 'Testing Holstein lattice Coherent Fewest-Switches Surface Hopping Deterministic 0'
ray start --head
python $script_path/main.py holstein_lattice_input_CFSSH_deterministic_dmat_0 '{"address":"auto"}'
ray stop
echo 'Testing Holstein lattice Coherent Fewest-Switches Surface Hopping Deterministic 1'
ray start --head
python $script_path/main.py holstein_lattice_input_CFSSH_deterministic_dmat_1 '{"address":"auto"}'
ray stop
echo 'Testing Holstein lattice Coherent Fewest-Switches Surface Hopping Stochastic 1'
ray start --head
python $script_path/main.py holstein_lattice_input_CFSSH_stochastic_dmat_1 '{"address":"auto"}'
ray stop
echo 'Testing Holstein lattice Coherent Fewest-Switches Surface Hopping Stochastic 0'
ray start --head
python $script_path/main.py holstein_lattice_input_CFSSH_stochastic_dmat_0 '{"address":"auto"}'
ray stop