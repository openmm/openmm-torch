[![GH Actions Status](https://github.com/openmm/openmm-torch/workflows/CI/badge.svg)](https://github.com/openmm/openmm-torch/actions?query=branch%3Amaster+workflow%3ACI)
[![Conda](https://img.shields.io/conda/v/conda-forge/openmm-torch.svg)](https://anaconda.org/conda-forge/openmm-torch)
[![Anaconda Cloud Badge](https://anaconda.org/conda-forge/openmm-torch/badges/downloads.svg)](https://anaconda.org/conda-forge/openmm-torch)

OpenMM PyTorch Plugin
=====================

This is a plugin for [OpenMM](http://openmm.org) that allows [PyTorch](https://pytorch.org/) static computation graphs
to be used for defining an OpenMM `TorchForce` object, an [OpenMM `Force` class](http://docs.openmm.org/latest/api-python/library.html#forces) that computes a contribution to the potential energy or used as a collective variable via [`CustomCVForce`](http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html#simtk.openmm.openmm.CustomCVForce).  

To use it, you create a PyTorch model that takes a `(nparticles,3)` tensor of particle positions (in nanometers) as input and produces energy (in kJ/mol) or the value of the collective variable as output.  
The `TorchForce` provided by this plugin can then use the model to compute energy contributions or apply forces to particles during a simulation.
`TorchForce` also supports the use of global context parameters that can be fed to the model and changed dynamically during runtime.

Installation
============

Installing with conda
---------------------

We provide [conda](https://docs.conda.io/) packages for Linux and MacOS via [`conda-forge`](https://conda-forge.org/), which can be installed from the [conda-forge channel](https://anaconda.org/conda-forge/openmm-torch):
```bash
conda install -c conda-forge openmm-torch
```
If you don't have `conda` available, we recommend installing [Miniconda for Python 3](https://docs.conda.io/en/latest/miniconda.html) to provide the `conda` package manager.  

Building from source
--------------------

This plugin uses [CMake](https://cmake.org/) as its build system.  
Before compiling you must install [LibTorch](https://pytorch.org/cppdocs/installing.html), which is the PyTorch C++ API, by following the instructions at https://pytorch.org.
You can then follow these steps:

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or `ccmake`, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".  (Do not worry if it produces an error message about not being able to find PyTorch.)

4. Set `OPENMM_DIR` to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.  If you are unsure of what directory this is, the following
script will print it out.

```python
from simtk import openmm
import os
print(os.path.dirname(openmm.version.openmm_library_path))
```

5. Set `PYTORCH_DIR` to point to the directory where you installed the LibTorch.

6. Set `CMAKE_INSTALL_PREFIX` to the directory where the plugin should be installed.  Usually,
this will be the same as `OPENMM_DIR`, so the plugin will be added to your OpenMM installation.

7. If you plan to build the OpenCL platform, make sure that `OPENCL_INCLUDE_DIR` and
`OPENCL_LIBRARY` are set correctly, and that `NN_BUILD_OPENCL_LIB` is selected.

8. If you plan to build the CUDA platform, make sure that `CUDA_TOOLKIT_ROOT_DIR` is set correctly
and that `NN_BUILD_CUDA_LIB` is selected.

9. Press "Configure" again if necessary, then press "Generate".

10. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install` to install the plugin, and `make PythonInstall` to
install the Python wrapper.

Using the OpenMM PyTorch plugin
===============================

Tutorials
---------

- [A simple simulation of alanine dipeptide with ANI-2x using OpenMM-Torch and NNPOps](tutorials/openmm-torch-nnpops.ipynb)

Exporting a PyTorch model for use in OpenMM
-------------------------------------------

The first step is to create a PyTorch model defining the calculation to perform.  
It should take particle positions in nanometers (in the form of a `torch.Tensor` of shape `(nparticles,3)` as input,
and return the potential energy in kJ/mol as a `torch.Scalar` as output.  

The model must then be converted to a [TorchScript](https://pytorch.org/docs/stable/jit.html) module and saved to a file.  
Converting to TorchScript can usually be done with a single call to [`torch.jit.script()`](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script) or [`torch.jit.trace()`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace),
although more complicated models can sometimes require extra steps.  
See the [PyTorch documentation](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) for details.  

Here is a simple Python example that does this for a very simple potential---a harmonic force attracting every particle to the origin:

```python
import torch

class ForceModule(torch.nn.Module):
    """A central harmonic potential as a static compute graph"""
    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        return torch.sum(positions**2)

# Render the compute graph to a TorchScript module
module = torch.jit.script(ForceModule())

# Serialize the compute graph to a file
module.save('model.pt')
```

To use the exported model in a simulation, create a `TorchForce` object and add it to your `System`.  
The constructor takes the path to the saved model as an argument.  
Alternatively, the scripted module can be provided directly.  
For example,
```python
# Create the TorchForce from the serialized compute graph
from openmmtorch import TorchForce
# Construct using a serialized module:
torch_force = TorchForce('model.pt')
# or using an instance of the module:
torch_force = TorchForce(module)

# Add the TorchForce to your System
system.addForce(torch_force)
```

Defining a model that uses periodic boundary conditions
-------------------------------------------------------

When defining the model to perform a calculation, you may want to apply periodic boundary conditions.  

To do this, call `setUsesPeriodicBoundaryConditions(True)` on the `TorchForce`.  
The graph is then expected to take a second input, which contains the current periodic box vectors.  
You can make use of them in whatever way you want for computing the force.  
For example, the following code applies periodic boundary conditions to each
particle position to translate all of them into a single periodic cell:

```python
class ForceModule(torch.nn.Module):
    """A central harmonic force with periodic boundary conditions"""
    def forward(self, positions, boxvectors):
        """The forward method returns the energy computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i
        boxvectors : torch.tensor with shape (3,3)
           boxvectors[i,k] is the box vector component k (in nanometers) of box vector i

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        # Image articles in rectilinear box
        # NOTE: This will not work for non-orthogonal boxes
        boxsize = boxvectors.diag()
        periodicPositions = positions - torch.floor(positions/boxsize)*boxsize
        # Compute central harmonic potential
        return torch.sum(periodicPositions**2)
```

Note that this code assumes a rectangular box.  Applying periodic boundary
conditions with a triclinic box requires a slightly more complicated calculation.

Defining global parameters that can be modified within the Context
------------------------------------------------------------------

The graph can also take arbitrary scalar arguments that are passed in at
runtime.  For example, this model multiplies the energy by `scale`, which is
passed as an argument to `forward()`.

```python
class ForceModule(torch.nn.Module):
    """A central harmonic force with a user-defined global scale parameter"""
    def forward(self, positions, scale):
        """The forward method returns the energy computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i
        scale : torch.Scalar
           A scalar tensor defined by 'TorchForce.addGlobalParameter'.
           Here, it scales the contribution to the potential.
           Note that parameters are passed in the order defined by `TorchForce.addGlobalParameter`, not by name.

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        return scale*torch.sum(positions**2)
```

When you create the `TorchForce`, call `addGlobalParameter()` once for each extra argument.

```python
torch_force.addGlobalParameter('scale', 2.0)
```

This specifies the name of the parameter and its initial value.  The name
does not need to match the argument to `forward()`.  All global parameters
are simply passed to the model in the order you define them.  The advantage
of using global parameters is that you can change their values at any time
by calling `setParameter()` on the `Context`.

```python
context.setParameter('scale', 5.0)
```

Computing forces in the model
-----------------------------

In the examples above, the PyTorch model computes the potential energy.  Backpropagation
can be used to compute the corresponding forces.  That always works, but sometimes you
may have a more efficient way to compute the forces than the generic backpropagation
algorithm.  In that case, you can have the model directly compute forces as well as
energy, returning both of them in a tuple.  Remember that the force is the *negative*
gradient of the energy.

```python
import torch

class ForceModule(torch.nn.Module):
    """A central harmonic potential that computes both energy and forces."""
    def forward(self, positions):
        """The forward method returns the energy and forces computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        forces : torch.Tensor with shape (nparticles,3)
           The force (in kJ/mol/nm) on each particle
        """
        return (torch.sum(positions**2), -2*positions)
```

When you create the `TorchForce`, call `setOutputsForces()` to tell it to expect the model
to return forces.

```python
torch_force.setOutputsForces(True)
```

Recording the model into a CUDA graph
-------------------------------------

You can ask `TorchForce` to run the model using [CUDA graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs). Not every model will be compatible with this feature, but it can be a significant performance boost for some models. To enable it the CUDA platform must be used and an special property must be provided to `TorchForce`:

```python
torch_force.setProperty("useCUDAGraphs", "true")
# The property can also be set at construction
torch_force = TorchForce('model.pt', {'useCUDAGraphs': 'true'})
```

The first time the model is run, it will be compiled (also known as recording) into a CUDA graph. Subsequent runs will use the compiled graph, which can be significantly faster. It is possible that compilation fails, in which case an `OpenMMException` will be raised. If that happens, you can disable CUDA graphs and try again.

It is required to run the model at least once before recording, in what is known as warmup.
By default ```TorchForce``` will run the model just a few times before recording, but controlling warmup steps might be desired. In these cases one can set the property ```CUDAGraphWarmupSteps```:
```python
torch_force.setProperty("CUDAGraphWarmupSteps", "12")
```

List of available properties
----------------------------

Some ```TorchForce``` functionalities can be customized by setting properties on an instance of it. Properties can be set at construction or by using ```setProperty```. A property is a pair of key/value strings. For instance:

```python
torch_force = TorchForce('model.pt', {'useCUDAGraphs': 'true'})
#Alternatively setProperty can be used to configure an already created instance.
#torch_force.setProperty("useCUDAGraphs", "true")
print("Current properties:")
for property in torch_force.getProperties():
    print(property.key, property.value)
```

Currently, the following properties are available:

1. useCUDAGraphs: Turns on the CUDA graph functionality
2. CUDAGraphWarmupSteps: When CUDA graphs are being used, controls the number of warmup calls to the model before recording.

License
=======

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2018-2020 Stanford University and the Authors.

Authors: Peter Eastman

Contributors: Raimondas Galvelis, Jaime Rodriguez-Guerra, Yaoyi Chen, John D. Chodera

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
