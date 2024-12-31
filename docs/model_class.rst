.. _model-class:

Model Class Guide 
~~~~~~~~~~~~~~~~~~~
This page will guide you in the construction of a new Model Class for use in QC Lab.

The model class describes a physical model which is described by a set of functions refered to as "ingredients". QC Lab is designed to accomodate 
a bare minimum model that consists of only a quantum-classical Hamiltonian. However, by incorporating additional ingredients, such as 
analytic gradients, the performance of QC Lab can be greatly improved. We will first describe the construction of a minimal model class, and then 
discuss how to incorporate additional ingredients.

Minimal Model Class
--------------------
A physical model in QC Lab is assumed to consist of a quantum-classical Hamiltonial of the form:


.. math::

    \hat{H}(\boldsymbol{z}) = \hat{H}_{\mathrm{q}} + \hat{H}_{\mathrm{q-c}}(\boldsymbol{z}) + H_{\mathrm{c}}(\boldsymbol{z}) 

where :math:`\hat{H}_{\mathrm{q}}` is the quantum Hamiltonian, :math:`\hat{H}_{\mathrm{q-c}}` is the quantum-classical coupling Hamiltonian,
 and :math:`H_{\mathrm{c}}` is the classical Hamiltonian. :math:`\boldsymbol{z}` is a complex-valued classical coordinate that describes the state 
 of the classical degrees of freedom. 

 Before describing how the model ingredients should be structured in the model class, we will first describe the __init__ method of the model class 
 which is responsible for initializing the default parameters and any input parameters into the set of parameters needed for QC Lab to run. 


::

    from qclab.model import Model # import the model class
    # create a minimal spin-boson model subclass
    class MinimalSpinBoson(Model):
        def __init__(self, parameters = None):
            if parameters is None:
                parameters = {}
            self.default_parameters = {
                'temp': 1, 'V': 0.5, 'E': 0.5, 'A': 100, 'W': 0.1,
                'l_reorg': 0.02 / 4, 'boson_mass': 1
            }
            super().__init__(self.default_parameters, parameters)

In the above example, the __init__ method takes an optional input_parameters dictionary which is added to the default_parameters dictionary by 
super().__init__. The default parameters dictionary contains the default input parameters for the model. These parameters are independent from the 
internal parameters required by QC Lab to function and are instead drawn from the analytic formulatino of the spin-boson model. 


.. math::
    
    \hat{H}_{\mathrm{q}} = \left(\begin{array}{cc} -E & V \\ V & E \end{array}\right)


.. math::

    \hat{H}_{\mathrm{q-c}} = \sigma_{z} \sum_{\alpha}^{A}  \frac{g_{\alpha}}{\sqrt{2mh_{\alpha}}} \left(z^{*}_{\alpha} + z_{\alpha}\right)


.. math::

    H_{\mathrm{c}} = \sum_{\alpha}^{A} \omega_{\alpha} z^{*}_{\alpha} z_{\alpha}

Here :math:`\sigma_{z}` is the Pauli-z matrix (:math:`\sigma_{z}=\vert0\rangle\langle 0\vert - \vert 1\rangle\langle 1\vert`), :math:`g_{\alpha}` is the coupling strength, :math:`m` is the boson mass,
:math:`h_{\alpha}` is the z-coordinate parameter (which here we may take to correspond to the frequencies: :math:`h_{\alpha}=\omega_{\alpha}`), and :math:`A` is the number of bosons.
We sample the frequencies and coupling strengths from a Debye spectral density which is discretized to obtain

.. math::

    \omega_{\alpha} = \Omega\tan\left(\frac{\alpha - 1/2}{2A}\pi\right)


.. math::

    g_{\alpha} = \omega_{\alpha}\sqrt{\frac{2\lambda}{A}}


Where :math:`\Omega` is the characteristic frequency and :math:`\lambda` is the reorganization energy. 

In a spin-boson model, the number of bosons :math:`A` can be quite large (e.g. 100). Rather than specifying every value of :math:`\omega_{\alpha}` 
and :math:`g_{\alpha}` in the input parameters, we can instead specify the characteristic frequency :math:`\Omega` and the reorganization energy :math:`\lambda`.
We can then use an internal function to generate the remaining parameters needed by the model and any parameters needed by QC Lab. 

This is accomplished by specifying a function called "update_model_parameters" as a method of the model class. 


::

    def update_model_parameters(self):
        self.parameters.w = self.parameters.W * np.tan(((np.arange(self.parameters.A) + 1) - 0.5) * np.pi / (2 * self.parameters.A))
        self.parameters.g = self.parameters.w * np.sqrt(2 * self.parameters.l_reorg / self.parameters.A) 

        ### additional parameters required by QC Lab
        self.parameters.pq_weight = self.parameters.w
        self.parameters.num_classical_coordinates = self.parameters.A
        self.parameters.mass = np.ones(self.parameters.A) * self.parameters.boson_mass


Here, we obtain the frequencies and coupling strengths from the Debye spectral density. We then specify the parameters needed by QC Lab. Namely the 
complex-valued coordinate parameter :math:`h_{\alpha}` is denoted as "pq_weight", the number of classical coordinates is denoted as 
"num_classical_coordinates", and the mass is denoted as "mass".

A list of all the parameters required by QC Lab can be found in the :ref:`parameters` section.

Now you can check that the update_model_parameters is functioning properly by changing one of the input parameters (A for example) and then checking that
the coupling strengths are updated appropraitely:


:: 

    model = MinimalSpinBoson()
    model.parameters.A = 10
    print('coupling strengths: ', model.parameters.g) # should be a list of length 10
    model.parameters.A = 5
    print('coupling strengths: ', model.parameters.g) # should be a list of length 5


Now we can add the minimal set of ingredients to the model class. The ingredients are the quantum Hamiltonian, 
the quantum-classical coupling Hamiltonian, and the classical Hamiltonian. The ingredients in a model class 
take a standard form which is required by QC Lab. Any argument (other than the model class itself) is 
passed as a keyword argument to the ingredient.


::

    def h_q(self, **kwargs):
        E = self.parameters.E
        V = self.parameters.V
        return np.array([[-E, V], [V, E]], dtype=complex)

    def h_qc(self, **kwargs):
        z_coord = kwargs['z_coord']
        g = self.parameters.g
        m = self.parameters.mass
        h = self.parameters.pq_weight
        h_qc = np.zeros((2, 2), dtype=complex)
        h_qc[0, 0] = np.sum((g * np.sqrt(1 / (2 * m * h))) * (z_coord + np.conj(z_coord)))
        h_qc[1, 1] = -h_qc[0, 0]
        return h_qc

    def h_c(self, **kwargs):
        z_coord = kwargs['z_coord']
        w = self.parameters.w
        return np.sum(w * np.conj(z_coord) * z_coord)



Now you have a working model class which you can instantiate and use following the instructions in the :ref:`quickstart` section.


Upgrading the Model Class
-------------------------



By default QC Lab assumes that a model's initial z coordinate is sampled from a Boltzmann distribution at temperature "temp" and that the classical 
Hamiltonian is harmonic with frequencies given by "pq_weight" and mass given by "mass". If this is not the case, the most prudent modification to make 
to the minimal model class is to specify how the classical coordinates are to be initialized. 

This is accomplished by defining a method called "init_classical" which has the following form:

::

    def init_classical(self, **kwargs):
        seed = kwargs['seed']
        np.random.seed(seed) # initialize a random seed for any randomness in this function
        z_coord = # here we sample some distribution to obtain a complex array with length sim.model.parameters.num_classical_coordinates
        return z_coord 


The "seed" argument is passed to the method by QC Lab and is used to initialize a random seed for any randomness in the method. The method should
return a complex array of length "sim.model.parameters.num_classical_coordinates". While including "init_classical" ensures that the physical 
results of the model are correct, it does not change the performance of the minimal model. 

The next recommended upgrade to the minimal model is to include analytic gradients for the classical and quantum-classical Hamiltonians with respect to 
the conjugate z coordinate. By default, QC Lab uses finite difference gradients which can be slow and inaccurate.

The gradient of the quantum-classical Hamiltonian is a complex-valued numpy array with the shape (num_classical_coordinates, num_state, num_states) where 
num_states is the dimension of the quantum Hilbert space. This structure appears naturally from the analytic form of the gradient. 
The :math:`(\alpha,i,j)`-th element 
of this array is given by

.. math::

     \left\langle i\left\vert \frac{\partial \hat{H}_{\mathrm{q-c}}}{\partial z^{*}_{\alpha}}\right\vert j \right\rangle = (-1)^{i}\frac{g_{\alpha}}{\sqrt{2mh_{\alpha}}}\delta_{ij}.



When implemented this is:

::

    def dh_qc_dzc(self, **kwargs):
        g = self.parameters.g
        m = self.parameters.mass
        h = self.parameters.pq_weight
        dh_qc_dzc = np.zeros((self.parameters.A, 2, 2), dtype=complex)
        dh_qc_dzc[:, 0, 0] = g * np.sqrt(1 / (2 * m * h))
        dh_qc_dzc[:, 1, 1] = -dh_qc_dzc[:, 0, 0]
        return dh_qc_dzc

We can likewise implement a gradient for the classical Hamiltonian which is a complex-valued numpy array of shape (num_classical_coordinates). For the spin-boson model
the classical Hamiltonian is Harmonic and so has the form,

::

    def dh_c_dzc(self, **kwargs):
        w = self.parameters.w
        z_coord = kwargs['z_coord']
        dh_c_dzc = w * z_coord + 0.0j
        return dh_c_dzc

A more convenient way to incorporate these ingredients is to use the built in set of ingredients available to QC Lab. For example, a model that 
has a classical Hamiltonian that is harmonic where the frequencies are given by "pq_weight" and the mass is given by "mass" can use the 
function :code:`qclab.ingredients.harmonic_oscillator_dh_c_dzc` to generate the harmonic oscillator Hamiltonian and its gradient.



The next recommended upgrate is to include vectorized ingredients. Vectorized ingredients are ingredients that can be computed for a batch of 
trajectories simultaneously. If implemented making use of broadcasting and vectorized numpy functions, vectorized ingredients can greatly improve
the performance of QC Lab. 

As an example let us consider a simulation where the z-coordinate comes as a vector with the shape (batch_size, num_classical_coordinates).
A vectorized version of the classical Hamiltonian would accept the vectorized z-coordinate and return a vector of shape (batch_size) where each
element is the energy of the classical coordiantes in that batch. That general principle can be applied to any ingredient, where the vectorized form 
of an ingredient should output an array with shape (..., np.shape(output)) where np.shape(output) is the shape of the output of the non-vectorized 
ingredient and ... are the additional dimensions of the z-coordinate (e.g. batch_size). 

The vectorized form of the classical Hamiltonian for the spin-boson model is:


::


    def h_c_vectorized(model, **kwargs):
        z_coord = kwargs['z_coord']
        h_c = np.sum(model.parameters.pq_weight[..., :] * np.conjugate(z_coord) * z_coord, axis=-1)
        return h_c

Importantly, the vectorized ingredient has the same name as the non-vectorized ingredient with "_vectorized" appended to the end. 

Like :code:`dh_c_dzc`, there are vectorized ingredients already built into QC Lab. For a full list of the available ingredients see the :ref:`ingredients` section.

The vectorized quantum-classical interaction is implemented as: 


::

    def h_qc_vectorized(self, **kwargs):
        z_coord = kwargs['z_coord']
        g = self.parameters.g 
        m = self.parameters.mass
        h = self.parameters.pq_weight
        h_qc = np.zeros((*np.shape(z_coord)[:-1], 2, 2), dtype=complex)
        h_qc[..., 0, 0] = np.sum((g * np.sqrt(1 / (2 * m * h)))[..., :] * (z_coord + np.conj(z_coord)), axis=-1)
        h_qc[..., 1, 1] = -h_qc[..., 0, 0]
        return h_qc

and its gradient is implemented as:


::

    def dh_qc_dzc_vectorized(self, **kwargs):
        g = self.parameters.g 
        m = self.parameters.mass
        h = self.parameters.pq_weight
        dh_qc_dzc = np.zeros((*np.shape(kwargs['z_coord'])[:-1], self.parameters.A, 2, 2), dtype=complex)
        dh_qc_dzc[..., :, 0, 0] = (g * np.sqrt(1 / (2 * m * h)))[..., :]
        dh_qc_dzc[..., :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
        return dh_qc_dzc


When vectorized ingredients are present, QC Lab no longer uses the non-vectorized ingredients. This means that the non-vectorized ingredients can be
omitted from the model class. The fully optimized spin-boson model class is then:


::

    class SpinBosonModel(Model):
        def __init__(self, parameters=None):
            if parameters is None:
                parameters = {}
            self.default_parameters = {
                'temp': 1, 'V': 0.5, 'E': 0.5, 'A': 100, 'W': 0.1,
                'l_reorg': 0.02 / 4, 'boson_mass': 1
            }
            super().__init__(self.default_parameters, parameters)

        def update_model_parameters(self):
            self.parameters.w = self.parameters.W * np.tan(((np.arange(self.parameters.A) + 1) - 0.5) * np.pi / (2 * self.parameters.A))
            self.parameters.g = self.parameters.w * np.sqrt(2 * self.parameters.l_reorg / self.parameters.A) 

            ### additional parameters required by QC Lab
            self.parameters.pq_weight = self.parameters.w
            self.parameters.num_classical_coordinates = self.parameters.A
            self.parameters.mass = np.ones(self.parameters.A) * self.parameters.boson_mass

            ### additional parameters for built-in ingredients
            self.parameters.two_level_system_a = self.parameters.E  # Diagonal energy of state 0
            self.parameters.two_level_system_b = -self.parameters.E  # Diagonal energy of state 1
            self.parameters.two_level_system_c = self.parameters.V  # Real part of the off-diagonal coupling
            self.parameters.two_level_system_d = 0  # Imaginary part of the off-diagonal coupling

        def h_qc_vectorized(self, **kwargs):
            z_coord = kwargs['z_coord']
            g = self.parameters.g 
            m = self.parameters.mass
            h = self.parameters.pq_weight
            h_qc = np.zeros((*np.shape(z_coord)[:-1], 2, 2), dtype=complex)
            h_qc[..., 0, 0] = np.sum((g * np.sqrt(1 / (2 * m * h)))[..., :] * (z_coord + np.conj(z_coord)), axis=-1)
            h_qc[..., 1, 1] = -h_qc[..., 0, 0]
            return h_qc

        def dh_qc_dzc_vectorized(self, **kwargs):
            g = self.parameters.g 
            m = self.parameters.mass
            h = self.parameters.pq_weight
            dh_qc_dzc = np.zeros((*np.shape(kwargs['z_coord'])[:-1], self.parameters.A, 2, 2), dtype=complex)
            dh_qc_dzc[..., :, 0, 0] = (g * np.sqrt(1 / (2 * m * h)))[..., :]
            dh_qc_dzc[..., :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
            return dh_qc_dzc

        # Assigning functions from ingredients module
        init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
        h_c_vectorized = ingredients.harmonic_oscillator_h_c_vectorized
        h_q_vectorized = ingredients.two_level_system_h_q_vectorized
        dh_c_dzc_vectorized = ingredients.harmonic_oscillator_dh_c_dzc_vectorized