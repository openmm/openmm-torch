import openmm as mm
from openmm import unit
import openmmtorch as ot
import numpy as np
import pytest
import torch as pt
from tempfile import NamedTemporaryFile

@pytest.mark.parametrize('model_file, output_forces',
                        [('../../tests/central.pt', False),
                         ('../../tests/forces.pt', True)])
@pytest.mark.parametrize('platform, use_graph',
                         [('Reference', False),
                          ('CPU', False),
                          ('CUDA', False),
                          ('CUDA', True)])
@pytest.mark.parametrize('precision', ['single', 'mixed', 'double'])
def testEnergyForce(model_file, output_forces, platform, precision, use_graph):

    if pt.cuda.device_count() < 1 and platform == 'CUDA':
        pytest.skip('A CUDA device is not available')

    # Create a system
    numParticles = 10
    system = mm.System()
    positions = np.random.rand(numParticles, 3)
    for _ in range(numParticles):
        system.addParticle(1.0)

    # Create a force
    force = ot.TorchForce(model_file, {'useCUDAGraphs': 'false'})
    assert not force.getOutputsForces() # Check the default
    force.setOutputsForces(output_forces)
    assert force.getOutputsForces() == output_forces
    assert force.getProperty('useCUDAGraphs') == 'false'
    if use_graph:
        force.setProperty('useCUDAGraphs', 'true')
        assert force.getProperty('useCUDAGraphs') == 'true'
    system.addForce(force)

    # Set up a simulation
    integrator = mm.VerletIntegrator(1.0)
    properties = {}
    if platform == 'CUDA':
        properties['Precision'] = precision
    platform = mm.Platform.getPlatformByName(platform)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)

    # Get expected values
    expectedEnergy = np.sum(positions*positions)
    expectedForces = -2*positions

    # Compare energy and forces
    for _ in range(3):
        # Compare just energy
        state = context.getState(getEnergy=True, getForces=False)
        assert np.allclose(expectedEnergy, state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        # Compare energy and forces
        state = context.getState(getEnergy=True, getForces=True)
        assert np.allclose(expectedEnergy, state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        assert np.allclose(expectedForces, state.getForces(asNumpy=True))

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda:0', 'cuda:1'])
@pytest.mark.parametrize('precision', ['single', 'mixed', 'double'])
def testModuleArguments(deviceString, precision):

    if pt.cuda.device_count() < 1 and deviceString == 'cuda:0':
        pytest.skip('A CUDA device is not available')
    if pt.cuda.device_count() < 2 and deviceString == 'cuda:1':
        pytest.skip('Two CUDA devices are not available')

    class TestModule(pt.nn.Module):

        def __init__(self, device, dtype, positions):
            super().__init__()
            self.device = device
            self.dtype = dtype
            self.register_buffer('positions', pt.tensor(positions).to(dtype))

        def forward(self, positions):
            assert self.positions.device == self.device
            assert positions.device == self.device
            assert positions.dtype == self.dtype
            assert pt.all(positions == self.positions)
            return pt.sum(positions)

    with NamedTemporaryFile() as fd:

        numParticles = 10
        system = mm.System()
        positions = np.random.rand(numParticles, 3)
        for _ in range(numParticles):
            system.addParticle(1.0)

        device = pt.device(deviceString)
        if device.type == 'cpu' or precision == 'double':
            dtype = pt.float64
        else:
            dtype = pt.float32
        module = TestModule(device, dtype, positions)
        pt.jit.script(module).save(fd.name)
        force = ot.TorchForce(fd.name)
        system.addForce(force)

        integrator = mm.VerletIntegrator(1.0)
        platform = mm.Platform.getPlatformByName(device.type.upper())
        properties = {}
        if device.type == 'cuda':
            properties['DeviceIndex'] = str(device.index)
            properties['Precision'] = precision
        context = mm.Context(system, integrator, platform, properties)

        context.setPositions(positions)
        context.getState(getEnergy=True, getForces=True)