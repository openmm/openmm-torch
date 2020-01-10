import simtk.openmm as mm
import simtk.unit as unit
import openmmtorch as ot
import numpy as np
import unittest

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


if __name__ == '__main__':
    unittest.main()

