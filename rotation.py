

def rotate(sim):
    """
    Executes a rotation over the classical and quantum dimensions for a system that is linear
    in the classical coordinates.
    :param sim: simulation object
    :return: simulation object in rotated basis
    """
    U_q = sim.U_q() # quantum rotation matrix
    U_c = sim.U_c() # classical rotation matrix



