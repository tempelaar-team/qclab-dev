.. _model-constants:



I want to change the reorganization energy.
===========================================

Changing the reorganization energy is easy! 

Using the same simulation object from the previous example, we can modify the `l_reorg` constant in `sim.model.constants`.



.. code-block:: python

    sim.model.constants.l_reorg = 0.05

    # Now let's run the simulation again
    data_fssh_1000_05 = serial_driver(sim)

    plt.plot(data_fssh_1000.data_dict["t"], np.real(data_fssh_1000.data_dict["dm_db"][:,0,0]), label=r'$\lambda = 0.005$')
    plt.plot(data_fssh_1000_05.data_dict["t"], np.real(data_fssh_1000_05.data_dict["dm_db"][:,0,0]), label=r'$\lambda = 0.05$')
    plt.xlabel('Time')
    plt.ylabel('Excited state population')
    #plt.savefig('fssh_lreorg.png')
    plt.legend()
    plt.show()


.. image:: fssh_lreorg.png
    :alt: Population dynamics.
    :align: center
    :width: 80%


For a complete list of model constants and their descriptions, please refer to the `Spin-Boson model documentation <../../user_guide/models/spin_boson_model.html>`_.





.. button-ref:: modify-fssh
    :color: primary
    :shadow:
    :align: center

    I want to modify the FSSH algorithm.
