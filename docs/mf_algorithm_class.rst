.. _mf-algorithm:
Mean-Field Dynamics 
~~~~~~~~~~~~~~~~~~~

As an example we consider the recipe for mean-field dynamics. As you can see, all attributes except ``recipe.defaults`` is set in the ``recipe.init`` function.::

        class MeanFieldDynamicsRecipe:
            def __init__(self, model):
                self.model = model
                self.initialize = [
                    ingredients.initialize_wf_db,
                    ingredients.initialize_z_coord,
                    ingredients.update_h_quantum,
                    ingredients.update_quantum_force_wf_db,
                ]
                self.update = [ingredients.update_z_coord_rk4,
                            ingredients.update_wf_db_rk4,
                            ingredients.update_h_quantum,
                            ingredients.update_quantum_force_wf_db,
                            ]
                self.output = [ingredients.update_dm_db_mf,
                            ingredients.update_e_c,
                            ingredients.update_e_q_mf,
                            ]
                self.output_names = ['dm_db',
                                    'e_c',
                                    'e_q',
                                    ]
                self.state = argparse.Namespace()

                return

            @staticmethod
            def defaults(model):
                var_names = list(model.__dict__.keys())
                defaults = {
                    'init_classical': auxiliary.harmonic_oscillator_boltzmann_init_classical,
                    'h_c': auxiliary.harmonic_oscillator_h_c,
                    'dh_c_dz': auxiliary.harmonic_oscillator_dh_c_dz,
                    'dh_c_dzc': auxiliary.harmonic_oscillator_dh_c_dzc,
                    'tmax': 10,
                    'dt_output': 0.1,
                    'dt': 0.01,
                    'temp': 1,
                    'num_states': 2,
                    'num_branches': 1,
                    'num_classical_coordinates': None
                }
                for name in defaults.keys():
                    if not (name in list(var_names)):
                        model.__dict__[name] = defaults[name]
                assert model.num_branches == 1
                return model