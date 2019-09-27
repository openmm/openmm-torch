OpenMM Neural Network Plugin
============================

This is a plugin for [OpenMM](http://openmm.org) that allows neural networks
to be used for defining forces.  It is implemented with [PyTorch](https://pytorch.org/).
To use it, you create a PyTorch model that takes particle positions as input
and produces energy as output.  This plugin uses the model to apply
forces to particles during a simulation.

Installation
============

At present this plugin must be compiled from source.  It uses CMake as its build
system.  Before compiling you must install LibTorch, which is the PyTorch C++ API,
by following the instructions at https://pytorch.org.  You can then
follow these steps.

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".  (Do not worry if it produces an error message about not being able to find PyTorch.)

4. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.

5. Set PYTORCH_DIR to point to the directory where you installed the LibTorch.

6. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

7. If you plan to build the OpenCL platform, make sure that OPENCL_INCLUDE_DIR and
OPENCL_LIBRARY are set correctly, and that NN_BUILD_OPENCL_LIB is selected.

8. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that NN_BUILD_CUDA_LIB is selected.

9. Press "Configure" again if necessary, then press "Generate".

10. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install` to install the plugin, and `make PythonInstall` to
install the Python wrapper.

Usage
=====

The first step is to create a PyTorch model defining the calculation to
perform.  It should take particle positions (in the form of an Nx3 Tensor) as
its input, and return the potential energy as its output.  The model must then be
converted to a TorchScript module and saved to a file.  Converting to TorchScript
can usually be done with a single call to `torch.jit.script()` or `torch.jit.trace()`,
although more complicated models can sometimes require extra steps.  See the
[PyTorch documentation](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
for details.  Here is an example of Python code that does this for a very
simple calculation (a harmonic force attracting every particle to the origin).

```python
import torch

class ForceModule(torch.nn.Module):
    def forward(self, positions):
        return torch.sum(positions**2)

module = torch.jit.script(ForceModule())
module.save('model.pt')
```

To use the model in a simulation, create a `NeuralNetworkForce` object and add
it to your `System`.  The constructor takes the path to the saved model as an
argument.  For example,

```python
from openmmnn import *
f = NeuralNetworkForce('model.pt')
system.addForce(f)
```

When defining the model to perform a calculation, you may want to apply
periodic boundary conditions.  To do this, call `setUsesPeriodicBoundaryConditions(True)`
on the `NeuralNetworkForce`.  The graph is then expected to take a second input,
which contains the current periodic box vectors.  You
can make use of them in whatever way you want for computing the force.  For
example, the following code applies periodic boundary conditions to each
particle position to translate all of them into a single periodic cell.

```python
class ForceModule(torch.nn.Module):
    def forward(self, positions, boxvectors):
        boxsize = boxvectors.diag()
        periodicPositions = positions - torch.floor(positions/boxsize)*boxsize
        return torch.sum(periodicPositions**2)
```

Note that this code assumes a rectangular box.  Applying periodic boundary
conditions with a triclinic box requires a slightly more complicated
calculation.

License
=======

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2018 Stanford University and the Authors.

Authors: Peter Eastman

Contributors:

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

