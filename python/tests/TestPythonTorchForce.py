import unittest
from openmm import *
from openmm.unit import *
from openmmtorch import PythonTorchForce
import numpy as np
import torch
import copy

def compute(state, pos):
    """This is a computation function used by the test cases."""
    k = state.getParameters()['k']
    energy = k*torch.sum(pos*pos)
    force = -0.5*k*pos
    return energy, force

class TestPythonTorchForce(unittest.TestCase):
    """Test the PythonTorchForce class"""

    def testComputeForce(self):
        """Test using PythonTorchForce to compute forces."""
        system = System()
        for i in range(5):
            system.addParticle(1.0)
        force = PythonTorchForce(compute, {'k':2.5})
        system.addForce(force)
        positions = np.random.rand(5, 3)
        for i in range(Platform.getNumPlatforms()):
            integrator = VerletIntegrator(0.001)
            try:
                context = Context(system, integrator, Platform.getPlatform(i))
            except OpenMMException:
                if i == 0:
                    raise
                else:
                    # This happens on CI when no GPU is available.
                    continue
            context.setPositions(positions)
            state = context.getState(energy=True, forces=True)
            self.assertAlmostEqual(2.5*np.sum(positions*positions), state.getPotentialEnergy().value_in_unit(kilojoules_per_mole), places=5)
            self.assertTrue(np.allclose(-1.25*positions, state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometer)))

    def testParticleSubset(self):
        """Test a PythonTorchForce appled to a subset of particles."""
        system = System()
        for i in range(10):
            system.addParticle(1.0)
        force = PythonTorchForce(compute, {'k':2.5})
        particles = [1,3,5,7,9]
        force.setParticles(particles)
        system.addForce(force)
        positions = np.random.rand(10, 3)
        for i in range(Platform.getNumPlatforms()):
            integrator = VerletIntegrator(0.001)
            try:
                context = Context(system, integrator, Platform.getPlatform(i))
            except OpenMMException:
                if i == 0:
                    raise
                else:
                    # This happens on CI when no GPU is available.
                    continue
            context.setPositions(positions)
            state = context.getState(energy=True, forces=True)
            filtered = np.zeros(positions.shape)
            filtered[particles] = positions[particles]
            self.assertAlmostEqual(2.5*np.sum(filtered*filtered), state.getPotentialEnergy().value_in_unit(kilojoules_per_mole), places=5)
            self.assertTrue(np.allclose(-1.25*filtered, state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometer)))

    def testPeriodic(self):
        """Test using PythonTorchForce with periodic boundary conditions."""
        def compute2(state, positions):
            vectors = state.getPeriodicBoxVectors().value_in_unit(nanometer)
            boxsize = torch.tensor(vectors, dtype=positions.dtype, device=positions.device).diag()
            positions = positions - torch.floor(positions/boxsize)*boxsize
            energy = torch.sum(positions**2)
            force = -0.5*positions
            return energy, force

        system = System()
        system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2))
        for i in range(10):
            system.addParticle(1.0)
        force = PythonTorchForce(compute2)
        system.addForce(force)
        self.assertFalse(system.usesPeriodicBoundaryConditions())
        force.setUsesPeriodicBoundaryConditions(True)
        self.assertTrue(system.usesPeriodicBoundaryConditions())
        positions = 10*np.random.rand(10, 3)-3
        for i in range(Platform.getNumPlatforms()):
            integrator = VerletIntegrator(0.001)
            try:
                context = Context(system, integrator, Platform.getPlatform(i))
            except OpenMMException:
                if i == 0:
                    raise
                else:
                    # This happens on CI when no GPU is available.
                    continue
            context.setPositions(positions)
            state = context.getState(energy=True, forces=True, positions=True, enforcePeriodicBox=True)
            periodicPositions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            self.assertAlmostEqual(np.sum(periodicPositions**2), state.getPotentialEnergy().value_in_unit(kilojoules_per_mole), places=5)
            self.assertTrue(np.allclose(-0.5*periodicPositions, state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometer)))

    def testExceptions(self):
        """Test that PythonTorchForce handles exceptions correctly."""
        def compute2(state, pos):
            raise ValueError('This should fail')

        system = System()
        system.addParticle(1.0)
        force = PythonTorchForce(compute2)
        system.addForce(force)
        positions = np.random.rand(1, 3)
        for i in range(Platform.getNumPlatforms()):
            integrator = VerletIntegrator(0.001)
            try:
                context = Context(system, integrator, Platform.getPlatform(i))
            except OpenMMException:
                if i == 0:
                    raise
                else:
                    # This happens on CI when no GPU is available.
                    continue
            context.setPositions(positions)
            with self.assertRaises(OpenMMException) as cm:
                context.getState(energy=True)
            self.assertEqual('This should fail', str(cm.exception))

    def testSerialize(self):
        """Test that PythonTorchForce can be serialized."""
        force1 = PythonTorchForce(compute, {'k':2.5})
        force1.setUsesPeriodicBoundaryConditions(True)
        force1.setParticles([1,3,5])

        # Make a copy by serializing and the deserializing it.

        copied = copy.copy(force1)
        force2 = PythonTorchForce.cast(copied)

        # They should be identical.

        self.assertEqual(XmlSerializer.serialize(force1), XmlSerializer.serialize(force2))
        self.assertEqual(dict(force2.getGlobalParameters()), {'k':2.5})
        self.assertEqual(force1.getParticles(), force2.getParticles())
        self.assertTrue(force2.usesPeriodicBoundaryConditions())

        # A locally defined function cannot be pickled.  We should not be able to serialize a force
        # that uses it.

        def compute2(state):
            return 1.0, np.zeros(len(state.getPositions()), 3)

        force3 = PythonTorchForce(compute2)
        with self.assertRaises(OpenMMException):
            XmlSerializer.serialize(force3)

    def testMinimization(self):
        """Test that PythonTorchForce works correctly with the minimizer."""
        system = System()
        for i in range(5):
            system.addParticle(1.0)
        force = PythonTorchForce(compute, {'k':2.5})
        system.addForce(force)
        positions = np.random.rand(5, 3)
        integrator = VerletIntegrator(0.001)
        context = Context(system, integrator, Platform.getPlatform('Reference'))
        context.setPositions(positions)

        # The PythonTorchForce and the MinimizationReporter both involve calling back into Python code,
        # possibly from different threads.  Make sure it doesn't cause any problems.

        class Reporter(MinimizationReporter):
            count = 0
            def report(self, iteration, x, grad, args):
                self.count += 1
                return False

        reporter = Reporter()
        LocalEnergyMinimizer.minimize(context, tolerance=1e-3, reporter=reporter)
        self.assertTrue(reporter.count > 0)
        state = context.getState(energy=True, positions=True)
        self.assertAlmostEqual(0.0, state.getPotentialEnergy().value_in_unit(kilojoules_per_mole))

    def testMemory(self):
        """Test for memory leaks in the Python/C++ interface."""
        try:
            import resource
        except:
            # The resource module is not available on Windows.
            return
        system = System()
        for i in range(1000):
            system.addParticle(1.0)
        force = PythonTorchForce(compute, {'k':2.5})
        system.addForce(force)
        positions = np.random.rand(1000, 3)
        integrator = VerletIntegrator(0.001)
        context = Context(system, integrator, Platform.getPlatform('Reference'))
        context.setPositions(positions)
        integrator.step(5000)
        memory1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        integrator.step(5000)
        memory2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.assertTrue(memory2 < 1.05*memory1)

    def testDtypes(self):
        """Test returning forces with different types."""
        for dtype in [torch.float32, torch.float64]:
            def compute2(state, pos):
                return torch.tensor(1.2, dtype=dtype), torch.tensor([[1,2,3],[4,5,6]], dtype=dtype)

            system = System()
            system.addParticle(1.0)
            system.addParticle(1.0)
            force = PythonTorchForce(compute2)
            system.addForce(force)
            positions = np.random.rand(2, 3)
            for i in range(Platform.getNumPlatforms()):
                integrator = VerletIntegrator(0.001)
                try:
                    context = Context(system, integrator, Platform.getPlatform(i))
                except OpenMMException:
                    if i == 0:
                        raise
                    else:
                        # This happens on CI when no GPU is available.
                        continue
                context.setPositions(positions)
                state = context.getState(forces=True, energy=True)
                forces = state.getForces().value_in_unit(kilojoules_per_mole/nanometer)
                self.assertEqual(Vec3(1,2,3), forces[0])
                self.assertEqual(Vec3(4,5,6), forces[1])
                energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
                self.assertAlmostEqual(1.2, energy)

if __name__ == '__main__':
    unittest.main()
