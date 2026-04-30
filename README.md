[![GH Actions Status](https://github.com/openmm/openmm-torch/workflows/CI/badge.svg)](https://github.com/openmm/openmm-torch/actions?query=branch%3Amaster+workflow%3ACI)
[![Conda](https://img.shields.io/conda/v/conda-forge/openmm-torch.svg)](https://anaconda.org/conda-forge/openmm-torch)
[![Anaconda Cloud Badge](https://anaconda.org/conda-forge/openmm-torch/badges/downloads.svg)](https://anaconda.org/conda-forge/openmm-torch)

OpenMM PyTorch Plugin
=====================

This is a plugin for [OpenMM](http://openmm.org) that allows using [PyTorch](https://pytorch.org/) models to compute
forces and energy in a simulation.  It provides two force classes, `TorchForce` and `PythonTorchForce`.  `TorchForce`
is deprecated, since it relies on TorchScript which is no longer maintained.  `PythonTorchForce` is the recommended one
to use in all cases.

`PythonTorchForce` is very similar to OpenMM's built in [`PythonForce`](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.PythonForce.html)
class, but it is specialized for use with PyTorch.  In particular, the particles positions and forces are represented
with tensors instead of NumPy arrays.  The benefit is reducing overhead and improving performance.  When the OpenMM
simulation and PyTorch model both run on the same GPU, coordinates and forces can be copied between them directly on the
GPU without ever needing to transfer them to the host.

Installation
============

Installing with pip
-------------------

We provide packages for Linux and macOS, which can be installed with the command:

```bash
pip install openmmtorch
```

Building from source
--------------------

This plugin uses [CMake](https://cmake.org/) as its build system.  Before compiling you must install PyTorch by
following the instructions at https://pytorch.org.  You can then follow these steps:

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or `ccmake`, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Set `OPENMM_DIR` to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.  If you are unsure of what directory this is, the following
script will print it out.

```python
import openmm
import os
print(os.path.dirname(openmm.version.openmm_library_path))
```

5. Usually PyTorch will be found automatically.  If it is not, set `Torch_DIR` to point to the directory containing its
CMake configuration files (e.g. `<PyTorch root directory>/share/cmake/Torch`).

6. Set `CMAKE_INSTALL_PREFIX` to the directory where the plugin should be installed.  Usually,
this will be the same as `OPENMM_DIR`, so the plugin will be added to your OpenMM installation.

7. If you plan to build the OpenCL, CUDA, or HIP platform, make sure that `NN_BUILD_OPENCL_LIB`, `NN_BUILD_CUDA_LIB`,
or `NN_BUILD_HIP_LIB` respectively is selected.  If the installed location of OpenCL, CUDA, or HIP was not found
automatically, set the appropriate CMake variables to locate them.

8. Press "Configure" again if necessary, then press "Generate".

9. Type `make install` to install the plugin, then `make PythonInstall` to install the Python wrapper.

Using PythonTorchForce
======================

To use PythonTorchForce, define a Python function that computes the interaction.  It should take two arguments, a
State object and a Tensor of shape `(# particles, 3)`.  For example,

```python
import torch

def compute(state, positions):
    energy = torch.sum(positions**2)
    forces = -0.5*positions
    return energy, forces
```

The State contains global parameters and periodic box vectors.  The Tensor contains particle positions.  The function
should compute the potential energy and forces, returning them as its two return values.  The energy should be a
scalar Tenor containing the value in kJ/mol.  The forces should be a Tensor of shape `(# particles, 3)` containing
the value in kJ/mol/nm.

Now create a PythonTorchForce, passing the function to the constructor.

```python
from openmmtorch import PythonTorchForce
force = PythonTorchForce(compute)
```

Do not make any assumptions about either the dtype or the device of the tensor containing positions.  Both of them may
vary depending on the platform and precision mode used for the simulation.  If you require a particular dtype or device,
call `to()` to ensure they are correct:

```python
positions = positions.to(dtype=torch.float32, device='cuda:0')
```

### Global Parameters

The force can optionally depend on global parameters stored in the Context.  To do this, pass a dict to the constructor
containing the names and default values of the parameters:

```python
force = PythonTorchForce(compute, {'k':2.5})
```

The computation function can then retrieve the parameter values from the State:

```python
def compute(state, positions):
    k = state.getParameters()['k']
    energy = k*torch.sum(positions**2)
    forces = -0.5*k*positions
    return energy, forces
```

You can change the parameter value at any time by calling `setParameter()` on the Context:

```python
context.setParameter('k', 5.0)
```

### Periodic Boundary Conditions

If you want your force to depend on periodic boundary conditions, call `setUsesPeriodicBoundaryConditions(True)` on the
PythonTorchForce.  This has two effects.  First, `usesPeriodicBoundaryConditions()` will return True, signaling to
other code that your system is periodic.  Second, the State passed to the computation function will contain periodic
box vectors.  You can use them however you want in computing the force.  For example,

```python
def compute2(state, positions):
    vectors = state.getPeriodicBoxVectors().value_in_unit(nanometer)
    boxsize = torch.tensor(vectors, dtype=positions.dtype, device=positions.device).diag()
    positions = positions - torch.floor(positions/boxsize)*boxsize
    energy = torch.sum(positions**2)
    forces = -0.5*positions
    return energy, forces
```

### Restricting to a Subset of Particles

A PythonTorchForce can optionally be applied to only a subset of the particles in a system.  To do
this, call `setParticles()` on it, providing the indices of the particles to apply it to.

```python
force.setParticles(list(range(50)))  # Apply to only the first 50 particles
```

The  computation function should then proceed as if those particles were the entire system.  The positions
passed to it will be a smaller Tensor containing only the positions of those particles, and the returned
forces should similarly contain only those particles.  That is, `forces[i]` should be the force on the i'th
particle passed to `setParticles()`.  When applying forces to only a small fraction of the particles in a
system, this can greatly improve performance.
