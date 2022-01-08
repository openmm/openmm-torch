import simtk.openmm as mm
import simtk.unit as unit
import openmmtorch as ot
import numpy as np
import unittest
import pytest
import torch as pt
from tempfile import NamedTemporaryFile

class TestTorchForce(unittest.TestCase):

    def testForce(self):
        # Create a random cloud of particles.

        numParticles = 10
        system = mm.System()
        positions = np.random.rand(numParticles, 3)
        for i in range(numParticles):
            system.addParticle(1.0)
        force = ot.TorchForce("../../tests/central.pt")
        system.addForce(force)

        # Compute the forces and energy.

        integ = mm.VerletIntegrator(1.0)
        context = mm.Context(system, integ, mm.Platform.getPlatformByName('Reference'))
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
            self.positions = pt.tensor(positions).to(self.device).to(self.dtype)

        def forward(self, positions):
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


if __name__ == '__main__':
    unittest.main()

