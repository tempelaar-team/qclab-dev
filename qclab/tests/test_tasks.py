import pytest
import qclab.tasks as tasks
from qclab import Simulation, VectorObject, Constants, Model, Algorithm
import numpy as np


def test_apply_vectorized_ingredient_over_internal_axes():
    internal_shape = ()
    batch_size = 10
    num_branches = 3
    sim = Simulation()
    sim.settings.batch_size = batch_size
    parameters = VectorObject(size=batch_size, vectorized=True)
    parameters.index = np.arange(batch_size)
    parameters.make_consistent()

    # first test no branch case
    out_shape = (batch_size)
    input_data = np.arange(np.prod(out_shape)).reshape(out_shape)
    class TestModel(Model):
        def test_ingredient_vectorized(model, constants, parameters, **kwargs):
            input = kwargs["input"]
            output = input + parameters.index
            return output
    sim.model = TestModel()

    output_data = tasks.apply_vectorized_ingredient_over_internal_axes(
        sim,
        sim.model.test_ingredient_vectorized,
        Constants(),
        parameters,
        {"input": input_data},
        (),
    )
    known_correct = np.copy(input_data) + np.arange(batch_size)
    assert np.allclose(output_data, known_correct)
    
    # now test one branch case
    out_shape = (batch_size, num_branches, 2,3)
    input_data = np.arange(np.prod(out_shape)).reshape(out_shape)
    class TestModel(Model):
        def test_ingredient_vectorized(model, constants, parameters, **kwargs):
            input = kwargs["input"]
            output = input + parameters.index[:,np.newaxis,np.newaxis]
            return output
    sim.model = TestModel()
        

    output_data = tasks.apply_vectorized_ingredient_over_internal_axes(
        sim,
        sim.model.test_ingredient_vectorized,
        Constants(),
        parameters,
        {"input": input_data},
        (num_branches,),
    )
    known_correct = np.copy(input_data) + np.arange(batch_size)[:,np.newaxis,np.newaxis,np.newaxis]
    assert np.allclose(output_data, known_correct)

    # test multi-index case
    out_shape = (batch_size, num_branches, 2,3,4)
    input_data = np.arange(np.prod(out_shape)).reshape(out_shape)
    class TestModel(Model):
        def test_ingredient_vectorized(model, constants, parameters, **kwargs):
            input = kwargs["input"] + parameters.index[:,np.newaxis,np.newaxis]
            return input
    sim.model = TestModel()

    output_data = tasks.apply_vectorized_ingredient_over_internal_axes(
        sim,
        sim.model.test_ingredient_vectorized,
        Constants(),
        parameters,
        {"input": input_data},
        (num_branches, 2),
    )
    known_correct = np.copy(input_data) + np.arange(batch_size)[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
    assert np.allclose(output_data, known_correct)




def test_apply_nonvectorized_ingredient_over_internal_axes():
    internal_shape = ()
    batch_size = 2
    num_branches = 3
    sim = Simulation()
    sim.settings.batch_size = batch_size
    parameters = VectorObject(size=batch_size, vectorized=True)
    parameters.index = np.arange(batch_size)
    parameters.make_consistent()
    
    # first test no branch case
    out_shape = (batch_size)
    input_data = np.arange(np.prod(out_shape)).reshape(out_shape)
    class TestModel(Model):
        def test_ingredient(model, constants, parameters, **kwargs):
            input = kwargs["input"] + parameters.index
            return input
    sim.model = TestModel()
        

    output_data = tasks.apply_nonvectorized_ingredient_over_internal_axes(
        sim,
        sim.model.test_ingredient,
        Constants(),
        parameters,
        {"input": input_data},
        (),
    )

    known_correct = np.copy(input_data) + np.arange(batch_size)
    assert np.allclose(output_data, known_correct)

    # now test one branch case
    out_shape = (batch_size, num_branches)
    input_data = np.arange(np.prod(out_shape)).reshape(out_shape)

    output_data = tasks.apply_nonvectorized_ingredient_over_internal_axes(
        sim,
        sim.model.test_ingredient,
        Constants(),
        parameters,
        {"input": input_data},
        (num_branches,),
    )

    known_correct = np.copy(input_data) + np.arange(batch_size)[:,np.newaxis]
    assert np.allclose(output_data, known_correct)

    # now test multi-index case
    out_shape = (batch_size, num_branches, 4, 5)
    input_data = np.arange(np.prod(out_shape)).reshape(out_shape)

    output_data = tasks.apply_nonvectorized_ingredient_over_internal_axes(
        sim,
        sim.model.test_ingredient,
        Constants(),
        parameters,
        {"input": input_data},
        (num_branches,4),
    )

    known_correct = np.copy(input_data) + np.arange(batch_size)[:,np.newaxis,np.newaxis,np.newaxis]
    assert np.allclose(output_data, known_correct)


if __name__ == "__main__":
    pytest.main()
