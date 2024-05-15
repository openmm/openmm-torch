import openmm as mm
import openmm.unit as unit
import openmmtorch as ot
import numpy as np
import pytest
import torch as pt
from torch import Tensor


class ForceWithParameters(pt.nn.Module):

    def __init__(self):
        super(ForceWithParameters, self).__init__()

    def forward(
        self, positions: Tensor, parameter1: Tensor, parameter2: Tensor
    ) -> Tensor:
        x2 = positions.pow(2).sum(dim=1)
        u_harmonic = ((parameter1 + parameter2**2) * x2).sum()
        return u_harmonic


@pytest.mark.parametrize("use_cv_force", [False, True])
@pytest.mark.parametrize("platform", ["Reference", "CPU", "CUDA", "OpenCL"])
def testParameterEnergyDerivatives(use_cv_force, platform):

    if pt.cuda.device_count() < 1 and platform == "CUDA":
        pytest.skip("A CUDA device is not available")

    # Create a random cloud of particles.
    numParticles = 10
    system = mm.System()
    positions = np.random.rand(numParticles, 3)
    for _ in range(numParticles):
        system.addParticle(1.0)

    # Create a force
    pt_force = ForceWithParameters()
    model = pt.jit.script(pt_force)
    tforce = ot.TorchForce(model, {"useCUDAGraphs": "false"})
    # Add a parameter
    parameter1 = 1.0
    parameter2 = 1.0
    tforce.setOutputsForces(False)
    tforce.addGlobalParameter("parameter1", parameter1)
    tforce.addEnergyParameterDerivative("parameter1")
    tforce.addGlobalParameter("parameter2", parameter2)
    tforce.addEnergyParameterDerivative("parameter2")
    if use_cv_force:
        # Wrap TorchForce into CustomCVForce
        force = mm.CustomCVForce("force")
        force.addCollectiveVariable("force", tforce)
    else:
        force = tforce
    # Enable energy derivatives for the parameter
    system.addForce(force)
    # Compute the forces and energy.
    integ = mm.VerletIntegrator(1.0)
    platform = mm.Platform.getPlatformByName(platform)
    context = mm.Context(system, integ, platform)
    context.setPositions(positions)
    state = context.getState(
        getEnergy=True, getForces=True, getParameterDerivatives=True
    )

    # See if the energy and forces and the parameter derivative are correct.
    # The network defines a potential of the form E(r) = (parameter1 + parameter2**2)*|r|^2
    r2 = np.sum(positions * positions)
    expectedEnergy = (parameter1 + parameter2**2) * r2
    assert np.allclose(
        expectedEnergy,
        state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
    )
    assert np.allclose(
        -2 * (parameter1 + parameter2**2) * positions, state.getForces(asNumpy=True)
    )
    assert np.allclose(
        r2,
        state.getEnergyParameterDerivatives()["parameter1"],
    )
    assert np.allclose(
        2 * parameter2 * r2,
        state.getEnergyParameterDerivatives()["parameter2"],
    )
