import openmm as mm
import openmm.unit as unit
import openmmtorch as ot
import numpy as np
import pytest
import torch as pt
from torch import Tensor
from typing import Tuple, List, Optional


class EnergyWithParameters(pt.nn.Module):

    def __init__(self):
        super(EnergyWithParameters, self).__init__()

    def forward(
        self, positions: Tensor, parameter1: Tensor, parameter2: Tensor
    ) -> Tensor:
        x2 = positions.pow(2).sum(dim=1)
        u_harmonic = ((parameter1 + parameter2**2) * x2).sum()
        return u_harmonic


class EnergyForceWithParameters(pt.nn.Module):

    def __init__(self, use_backwards=True):
        super(EnergyForceWithParameters, self).__init__()
        self.use_backwards = use_backwards

    def forward(
        self, positions: Tensor, parameter1: Tensor, parameter2: Tensor
    ) -> Tuple[Tensor, Tensor]:
        positions.requires_grad_(True)
        x2 = positions.pow(2).sum(dim=1)
        u_harmonic = ((parameter1 + parameter2**2) * x2).sum()
        # This way of computing the forces forcefully leaves out the parameter derivatives
        if self.use_backwards:
            grad_outputs: List[Optional[Tensor]] = [pt.ones_like(u_harmonic)]
            dy = pt.autograd.grad(
                [u_harmonic],
                [positions],
                grad_outputs=grad_outputs,
                create_graph=False,
                # This must be true, otherwise pytorch will not allow to compute the gradients with respect to the parameters
                retain_graph=True,
            )[0]
            assert dy is not None
            forces = -dy
        else:
            forces = -2 * (parameter1 + parameter2**2) * positions
        return u_harmonic, forces


@pytest.mark.parametrize("use_cv_force", [False, True])
@pytest.mark.parametrize("platform", ["Reference", "CPU", "CUDA", "OpenCL"])
@pytest.mark.parametrize(
    ("return_forces", "use_backwards"), [(False, False), (True, False), (True, True)]
)
def testParameterEnergyDerivatives(
    use_cv_force, platform, return_forces, use_backwards
):

    if pt.cuda.device_count() < 1 and platform == "CUDA":
        pytest.skip("A CUDA device is not available")

    # Create a random cloud of particles.
    numParticles = 10
    system = mm.System()
    positions = np.random.rand(numParticles, 3)
    for _ in range(numParticles):
        system.addParticle(1.0)

    # Create a force
    if return_forces:
        pt_force = EnergyForceWithParameters(use_backwards=use_backwards)
    else:
        pt_force = EnergyWithParameters()
    model = pt.jit.script(pt_force)
    tforce = ot.TorchForce(model, {"useCUDAGraphs": "false"})
    # Add a parameter
    parameter1 = 1.0
    parameter2 = 1.0
    tforce.setOutputsForces(return_forces)
    tforce.addGlobalParameter("parameter1", parameter1)
    tforce.addGlobalParameter("parameter2", parameter2)
    # Enable energy derivatives for the parameters
    tforce.addEnergyParameterDerivative("parameter1")
    tforce.addEnergyParameterDerivative("parameter2")
    if use_cv_force:
        # Wrap TorchForce into CustomCVForce
        force = mm.CustomCVForce("force")
        force.addCollectiveVariable("force", tforce)
    else:
        force = tforce
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
