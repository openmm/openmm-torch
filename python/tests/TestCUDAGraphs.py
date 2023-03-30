__author__ = "Raul P. Pelaez"
import openmmtorch as ot
import torch
import openmm as mm
import numpy as np
import pytest


class UngraphableModule(torch.nn.Module):
    def forward(self, positions):
        torch.cuda.synchronize()
        return (torch.sum(positions**2), -2.0 * positions)


class GraphableModule(torch.nn.Module):
    def forward(self, positions):
        energy = torch.einsum("ij,ij->i", positions, positions).sum()
        return (energy, -2.0 * positions)


class GraphableModuleOnlyEnergy(torch.nn.Module):
    def forward(self, positions):
        energy = torch.einsum("ij,ij->i", positions, positions).sum()
        return energy


def tryToTestForceWithModule(
    ModuleType, outputsForce, useGraphs=False, warmup=10, numParticles=10
):
    """Test that the force is correctly computed for a given module type.
    Warmup makes OpenMM call TorchForce execution multiple times, which might expose some bugs related to that given that with CUDA graphs the first execution is different from the rest.
    """
    module = torch.jit.script(ModuleType())
    torch_force = ot.TorchForce(
        module, {"useCUDAGraphs": "true" if useGraphs else "false"}
    )
    torch_force.setOutputsForces(outputsForce)
    system = mm.System()
    positions = np.random.rand(numParticles, 3)
    for _ in range(numParticles):
        system.addParticle(1.0)
    system.addForce(torch_force)
    integ = mm.VerletIntegrator(1.0)
    platform = mm.Platform.getPlatformByName("CUDA")
    context = mm.Context(system, integ, platform)
    context.setPositions(positions)
    for _ in range(warmup):
        state = context.getState(getEnergy=True, getForces=True)
    expectedEnergy = np.sum(positions**2)
    expectedForce = -2.0 * positions
    energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    force = state.getForces(asNumpy=True).value_in_unit(
        mm.unit.kilojoules_per_mole / mm.unit.nanometer
    )
    assert np.allclose(expectedEnergy, energy)
    assert np.allclose(expectedForce, force)


def testUnGraphableModelRaises():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    with pytest.raises(mm.OpenMMException):
        tryToTestForceWithModule(UngraphableModule, outputsForce=True, useGraphs=True)


@pytest.mark.parametrize("numParticles", [10, 10000])
@pytest.mark.parametrize("useGraphs", [True, False])
@pytest.mark.parametrize("warmup", [1, 10])
def testGraphableModelOnlyEnergyIsCorrect(useGraphs, warmup, numParticles):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tryToTestForceWithModule(
        GraphableModuleOnlyEnergy,
        outputsForce=False,
        useGraphs=useGraphs,
        warmup=warmup,
        numParticles=numParticles,
    )


@pytest.mark.parametrize("numParticles", [10, 10000])
@pytest.mark.parametrize("useGraphs", [True, False])
@pytest.mark.parametrize("warmup", [1, 10])
def testGraphableModelIsCorrect(useGraphs, warmup, numParticles):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tryToTestForceWithModule(
        GraphableModule,
        outputsForce=True,
        useGraphs=useGraphs,
        warmup=warmup,
        numParticles=numParticles,
    )
