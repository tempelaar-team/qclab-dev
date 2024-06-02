rm -r *_output
script_path='/home/akrotz/Research/pyMQC/'
ray stop --force
ray start --head
#echo 'Profiling Holstein lattice Mean-Field '
#python -m cProfile -o mf_profile.out $script_path/main.py holstein_lattice_input_MF '{"address":"auto"}'
echo 'Profiling Holstein Fewest-Switches Surface Hopping Deterministic '
python -m cProfile -o fssh_profile.out $script_path/main.py holstein_lattice_input_FSSH_deterministic '{"address":"auto"}'
echo 'Profiling Holstein lattice Coherent Fewest-Switches Surface Hopping Deterministic '
python -m cProfile -o cfssh_profile.out $script_path/main.py holstein_lattice_input_CFSSH_deterministic '{"address":"auto"}'
