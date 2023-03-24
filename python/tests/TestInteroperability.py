import openmm as mm
import openmm.unit as unit
import openmmtorch as ot
import platform
import pytest
from tempfile import NamedTemporaryFile
import torch as pt


@pytest.mark.skipif(platform.system() == 'Darwin', reason='There is no NNPOps package for MacOS')
@pytest.mark.parametrize('use_cv_force', [True, False])
@pytest.mark.parametrize('platform', ['Reference', 'CPU', 'CUDA', 'OpenCL'])
def testTorchANI(use_cv_force, platform):

    if pt.cuda.device_count() < 1 and platform == 'CUDA':
        pytest.skip('A CUDA device is not available')

    import NNPOps # There is no NNPOps package for MacOS
    import torchani

    class Model(pt.nn.Module):

        def __init__(self):
            super().__init__()
            self.register_buffer('atomic_numbers', pt.tensor([[1, 1]]))
            self.model = torchani.models.ANI2x(periodic_table_index=True)
            self.model = NNPOps.OptimizedTorchANI(self.model, self.atomic_numbers)

        def forward(self, positions):
            positions = positions.float().unsqueeze(0) * 10 # nm --> Ang
            return self.model((self.atomic_numbers, positions)).energies[0] * 2625.5 # Hartree --> kJ/mol

    # Create a system
    system = mm.System()
    for _ in range(2):
        system.addParticle(1.0)
    positions = pt.tensor([[-5, 0.0, 0.0], [5, 0.0, 0.0]], requires_grad=True)

    with NamedTemporaryFile() as model_file:

        # Save the model
        pt.jit.script(Model()).save(model_file.name)

        # Compute reference energy and forces
        model = pt.jit.load(model_file)
        ref_energy = model(positions)
        ref_energy.backward()
        ref_forces = positions.grad

        # Create a force
        force = ot.TorchForce(model_file.name)
        if use_cv_force:
            # Wrap TorchForce into CustomCVForce
            cv_force = mm.CustomCVForce('force')
            cv_force.addCollectiveVariable('force', force)
            system.addForce(cv_force)
        else:
            system.addForce(force)

        # Compute energy and forces
        integ = mm.VerletIntegrator(1.0)
        platform = mm.Platform.getPlatformByName(platform)
        context = mm.Context(system, integ, platform)
        context.setPositions(positions.detach().numpy())
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometers)

        # Check energy and forces
        assert pt.allclose(ref_energy, pt.tensor(energy, dtype=ref_energy.dtype))
        assert pt.allclose(ref_forces, pt.tensor(forces, dtype=ref_forces.dtype))