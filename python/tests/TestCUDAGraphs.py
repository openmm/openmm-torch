__author__ = "Raul P. Pelaez"
import openmmtorch as ot
import torch
import openmm as mm
import numpy as np
import pytest

class UngraphableModule(torch.nn.Module):
    def forward(self, positions):
        torch.cuda.synchronize()
        return (0.5*torch.sum(positions**2), -2.0*positions)

class GraphableModule(torch.nn.Module):
    def forward(self, positions):
        energy=0.5*torch.einsum('ij,ij->i', positions, positions).sum()
        return (energy, -2.0*positions)

class GraphableModuleOnlyEnergy(torch.nn.Module):
    def forward(self, positions):
        energy=0.5*torch.einsum('ij,ij->i', positions, positions).sum()
        return (energy)

def tryToTestForceWithModule(ModuleType, outputsForce):
    module = torch.jit.script(ModuleType())
    torch_force = ot.TorchForce(module, {'useCUDAGraphs': 'true'})
    torch_force.setOutputsForces(outputsForce)
    numParticles = 10
    system = mm.System()
    positions = np.random.rand(numParticles, 3)
    for _ in range(numParticles):
        system.addParticle(1.0)
    system.addForce(torch_force)
    integ = mm.VerletIntegrator(1.0)
    platform = mm.Platform.getPlatformByName('CUDA')
    context = mm.Context(system, integ, platform)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    expectedEnergy = 0.5*np.sum(positions**2)
    energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    force = state.getForces(asNumpy=True).value_in_unit(mm.unit.kilojoules_per_mole/mm.unit.nanometer)
    assert np.allclose(expectedEnergy, energy)
    assert np.allclose(-2.0*positions, force)


def testUnGraphableModelRaises():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    with pytest.raises(mm.OpenMMException):
        tryToTestForceWithModule(UngraphableModule, True)

def testGraphableModelIsCorrect():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tryToTestForceWithModule(GraphableModule, True)

def testGraphableModelOnlyEnergyIsCorrect():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tryToTestForceWithModule(GraphableModuleOnlyEnergy, False)
