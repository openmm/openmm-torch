import openmm as mm
import openmm.unit as unit
import openmmtorch as ot
import numpy as np
import pytest
import torch as pt
from tempfile import NamedTemporaryFile

@pytest.mark.parametrize('model_file,',
                        ['../../tests/central.pt',
                         '../../tests/forces.pt'])
def testConstructors(model_file):
    force = ot.TorchForce(model_file)
    model = pt.jit.load(model_file)
    force = ot.TorchForce(pt.jit.load(model_file))
    model = force.getModule()
    force = ot.TorchForce(model)

@pytest.mark.parametrize('model_file, output_forces, use_module_constructor',
                        [('../../tests/central.pt', False, False,),
                         ('../../tests/forces.pt', True, False),
                         ('../../tests/forces.pt', True, True)])
@pytest.mark.parametrize('use_cv_force', [True, False])
@pytest.mark.parametrize('platform', [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())])
def testForce(model_file, output_forces, use_module_constructor, use_cv_force, platform):

    if pt.cuda.device_count() < 1 and platform == 'CUDA':
        pytest.skip('A CUDA device is not available')

    # Create a random cloud of particles.
    numParticles = 10
    system = mm.System()
    positions = np.random.rand(numParticles, 3)
    for _ in range(numParticles):
        system.addParticle(1.0)

    # Create a force
    if use_module_constructor:
        model = pt.jit.load(model_file)
        force = ot.TorchForce(model, {'useCUDAGraphs': 'false'})
    else:
        force = ot.TorchForce(model_file, {'useCUDAGraphs': 'false'})
    assert not force.getOutputsForces() # Check the default
    force.setOutputsForces(output_forces)
    assert force.getOutputsForces() == output_forces
    assert force.getProperties()['useCUDAGraphs'] == 'false'
    if use_cv_force:
        # Wrap TorchForce into CustomCVForce
        cv_force = mm.CustomCVForce('force')
        cv_force.addCollectiveVariable('force', force)
        system.addForce(cv_force)
    else:
        system.addForce(force)

    # Compute the forces and energy.
    integ = mm.VerletIntegrator(1.0)
    try:
        context = mm.Context(system, integ, mm.Platform.getPlatformByName(platform))
    except:
        pytest.skip(f'Unable to create Context with {platform}')
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)

    # See if the energy and forces are correct.  The network defines a potential of the form E(r) = |r|^2
    expectedEnergy = np.sum(positions*positions)
    assert np.allclose(expectedEnergy, state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    assert np.allclose(-2*positions, state.getForces(asNumpy=True))

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
            assert pt.allclose(positions, self.positions)
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


def testProperties():
    """ Test that the properties are correctly set and retrieved """
    force = ot.TorchForce('../../tests/central.pt')
    force.setProperty('useCUDAGraphs', 'true')
    assert force.getProperties()['useCUDAGraphs'] == 'true'
    force.setProperty('useCUDAGraphs', 'false')
    assert force.getProperties()['useCUDAGraphs'] == 'false'
