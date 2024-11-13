import numpy as np
from qclab.drivers.serial_driver import dynamics_serial
from qclab.recipes import AbInitioMeanFieldDynamicsRecipe
from qclab.models.ab_initio import AbInitioModel
from pyscf import gto

# Define the molecule
mol = gto.M(
    atom="""
    H  0.00000000   0.00000000   0.00000000
    H  0.00000000   0.00000000   2.00000000
    """,
    basis='6-31g',
)

# Define an initial momentum for each atom
init_momentum = np.array([[0, 0, 0],
                          [0, 0,-1]])

# initialize the ab initio model providing the mol object, initial momentum, 
# number of adiabatic surfaces, and electronic structure theory method. 
input_params = dict(pyscf_mol = mol, init_momentum=init_momentum, num_states=5, method='CISD')
model = AbInitioModel(input_params)

# define an initial wavefunction in the adiabatic basis
model.wf_adb = np.zeros(model.num_states)+0.0j
model.wf_adb[0] = 1.0+0.0j

# initialize the dynamics algorithm
recipe = AbInitioMeanFieldDynamicsRecipe()

# set the simulation parameters
recipe.params.tmax = 4
recipe.params.dt_output = 0.01
recipe.params.dt = 0.01


# execute the simulation using the dynamics driver
data = dynamics_serial(recipe = recipe, model = model)