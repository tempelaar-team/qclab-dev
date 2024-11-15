import numpy as np
from pyscf import gto
import qc_lab
# Define the molecule.
mol = gto.M(atom="H 0 0 0; H 0 0 2", basis="6-31g")
# Define an initial momentum for each atom.
init_momentum = np.array([[0, 0, 0], [0, 0, -1]])
# Initialize the ab initio model providing the mol object, initial momentum, 
# number of electronic states, and electronic structure theory method. 
inputs = {"pyscf_mol":mol, "init_momentum":init_momentum, "num_states":5, "method":"CISD"}
model = qc_lab.ab_initio_model.AbInitioModel(inputs)
# Define an initial wavefunction in the adiabatic basis.
model.wf_adb = np.zeros(model.num_states) + 0.0j
model.wf_adb[0] = 1.0 + 0.0j
# Initialize the dynamics algorithm and set the simulation parameters.
recipe = qc_lab.recipes.MeanFieldDynamicsRecipe()
recipe.params.tmax = 4
recipe.params.dt_output = 0.01
recipe.params.dt = 0.01
# Execute the simulation using the dynamics driver.
data = qc_lab.drivers.dynamics_serial(recipe=recipe, model=model)