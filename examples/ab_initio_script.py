import numpy as np
from qclab.drivers.serial_driver import dynamics_serial
from qclab.recipes import AbInitioMeanFieldDynamicsRecipe
from qclab.models.ab_initio import AbInitioModel
from pyscf import gto

# Define the molecule.
mol = gto.M(atom=""" H  0.0   0.0   0.0\n H  0.0   0.0   2.0""",basis='6-31g')
# Define an initial momentum for each atom.
init_momentum = np.array([[0, 0, 0], [0, 0,-1]])
# Initialize the ab initio model providing the mol object, initial momentum, 
# number of adiabatic surfaces, and electronic structure theory method. 
input_params = dict(pyscf_mol = mol, init_momentum=init_momentum, num_states=5, method='CISD')
model = AbInitioModel(input_params)
# Define an initial wavefunction in the adiabatic basis.
model.wf_adb = np.zeros(model.num_states)+0.0j
model.wf_adb[0] = 1.0+0.0j
# Initialize the dynamics algorithm and set the simulation parameters.
recipe = AbInitioMeanFieldDynamicsRecipe()
recipe.params.tmax = 4
recipe.params.dt_output = 0.01
recipe.params.dt = 0.01
# Execute the simulation using the dynamics driver
data = dynamics_serial(recipe = recipe, model = model)