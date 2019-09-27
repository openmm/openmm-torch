import simtk.openmm as mm
import simtk.unit as unit
import openmmnn as nn
import tensorflow as tf
import unittest

class TestNeuralNetworkForce(unittest.TestCase):

    def testFreezeGraph(self):
        graph = tf.Graph()
        with graph.as_default():
            positions = tf.placeholder(tf.float32, [None, 3], 'positions')
            scale = tf.Variable(5.0)
            energy = tf.multiply(scale, tf.reduce_sum(positions**2), name='energy')
            forces = tf.identity(tf.gradients(-energy, positions), name='forces')
            session = tf.Session()
            session.run(tf.global_variables_initializer())
        force = nn.NeuralNetworkForce(graph, session)
        system = mm.System()
        for i in range(3):
            system.addParticle(1.0)
        system.addForce(force)
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator)
        positions = [mm.Vec3(3, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(3, 4, 0)]
        context.setPositions(positions)
        assert context.getState(getEnergy=True).getPotentialEnergy() == 250.0*unit.kilojoules_per_mole


if __name__ == '__main__':
    unittest.main()
