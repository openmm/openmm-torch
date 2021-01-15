/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "OpenCLTorchKernels.h"
#include "OpenCLTorchKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

OpenCLCalcTorchForceKernel::~OpenCLCalcTorchForceKernel() {
}

void OpenCLCalcTorchForceKernel::initialize(const System& system, const TorchForce& force, torch::jit::script::Module& module) {
    this->module = module;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalNames.push_back(force.getGlobalParameterName(i));
    int numParticles = system.getNumParticles();

    // Inititalize OpenCL objects.

    map<string, string> defines;
    if (cl.getUseDoublePrecision()) {
        networkForces.initialize<double>(cl, 3*numParticles, "networkForces");
        defines["FORCES_TYPE"] = "double";
    }
    else {
        networkForces.initialize<float>(cl, 3*numParticles, "networkForces");
        defines["FORCES_TYPE"] = "float";
    }
    cl::Program program = cl.createProgram(OpenCLTorchKernelSources::torchForce, defines);
    addForcesKernel = cl::Kernel(program, "addForces");
}

double OpenCLCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cl.getNumAtoms();
    torch::Tensor posTensor = torch::from_blob(pos.data(), {numParticles, 3}, torch::kFloat64);
    if (!cl.getUseDoublePrecision())
        posTensor = posTensor.to(torch::kFloat32);
    posTensor.set_requires_grad(true);
    vector<torch::jit::IValue> inputs = {posTensor};
    if (usePeriodic) {
        Vec3 box[3];
        cl.getPeriodicBoxVectors(box[0], box[1], box[2]);
        torch::Tensor boxTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);
        if (!cl.getUseDoublePrecision())
            boxTensor = boxTensor.to(torch::kFloat32);
        inputs.push_back(boxTensor);
    }
    for (const string& name : globalNames)
        inputs.push_back(torch::tensor(context.getParameter(name)));
    torch::Tensor energyTensor = module.forward(inputs).toTensor();
    if (includeForces) {
        energyTensor.backward();
        torch::Tensor forceTensor = posTensor.grad();
        if (cl.getUseDoublePrecision()) {
            if (!(forceTensor.dtype() == torch::kFloat64))
                forceTensor = forceTensor.to(torch::kFloat64);
            double* data = forceTensor.data_ptr<double>();
            networkForces.upload(data);
        }
        else {
            if (!(forceTensor.dtype() == torch::kFloat32))
                forceTensor = forceTensor.to(torch::kFloat32);
            float* data = forceTensor.data_ptr<float>();
            networkForces.upload(data);
        }
        addForcesKernel.setArg<cl::Buffer>(0, networkForces.getDeviceBuffer());
        addForcesKernel.setArg<cl::Buffer>(1, cl.getForceBuffers().getDeviceBuffer());
        addForcesKernel.setArg<cl::Buffer>(2, cl.getAtomIndexArray().getDeviceBuffer());
        addForcesKernel.setArg<cl_int>(3, numParticles);
        cl.executeKernel(addForcesKernel, numParticles);
    }
    return energyTensor.item<double>();
}

