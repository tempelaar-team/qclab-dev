.. _ingredients:


Ingredients in QC Lab
===================

In QC Lab, the **ingredients** of a model are the fundamental components that define its physical properties. 
A generic ingredient is a function bound to a Model object that computes a specific property of the model.
As such, the ingredients have the form

.. code-block:: python

    def ingredient_name(model, parameters, **kwargs):
        # Calculate var.
        var = None
        return var

where `model` is the instance of the Model class, `parameters` is a vector object passed to the model which contains potentially 
time-dependent parameters, and `kwargs` can include additional keyword arguments as needed.

For detailed documentation of the different ingredients in QC Lab, as well as the built-in ingredients available from `qc_lab.ingredients`, please refer to the
`Ingredients Reference <../../software_reference/ingredients/ingredients.html>`_.
