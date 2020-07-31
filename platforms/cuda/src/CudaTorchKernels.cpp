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

#include "CudaTorchKernels.h"
#include "CudaTorchKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcTorchForceKernel::~CudaCalcTorchForceKernel() {
}

void CudaCalcTorchForceKernel::initialize(const System& system, const TorchForce& force, torch::jit::script::Module& module) {
    this->module = module;
    module.to(torch::kCUDA);
    usePeriodic = force.usesPeriodicBoundaryConditions();
    int numParticles = system.getNumParticles();
    torch::TensorOptions options = torch::TensorOptions()
            .device(torch::kCUDA, cu.getDeviceIndex())
            .dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32)
            .requires_grad(true);
    posTensor = torch::empty({numParticles, 3}, options);
    boxTensor = torch::empty({3, 3}, options);

    // Inititalize CUDA objects.

    cu.setAsCurrent();
    map<string, string> defines;
    if (cu.getUseDoublePrecision()) {
        networkForces.initialize<double>(cu, 3*numParticles, "networkForces");
        defines["FORCES_TYPE"] = "double";
    }
    else {
        networkForces.initialize<float>(cu, 3*numParticles, "networkForces");
        defines["FORCES_TYPE"] = "float";
    }
    CUmodule program = cu.createModule(CudaTorchKernelSources::torchForce, defines);
    copyInputsKernel = cu.getKernel(program, "copyInputs");
    addForcesKernel = cu.getKernel(program, "addForces");
}

double CudaCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    int numParticles = cu.getNumAtoms();
    void* posData;
    void* boxData;
    if (cu.getUseDoublePrecision()) {
        posData = posTensor.data_ptr<double>();
        boxData = boxTensor.data_ptr<double>();
    }
    else {
        posData = posTensor.data_ptr<float>();
        boxData = boxTensor.data_ptr<float>();
    }
    cu.setAsCurrent();
    void* inputArgs[] = {&posData, &boxData, &cu.getPosq().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
            &numParticles, cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer()};
    cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
    vector<torch::jit::IValue> inputs = {posTensor};
    if (usePeriodic)
        inputs.push_back(boxTensor);
    torch::Tensor energyTensor = module.forward(inputs).toTensor();
    if (includeForces) {
        energyTensor.backward();
        torch::Tensor forceTensor = posTensor.grad();
        if (cu.getUseDoublePrecision()) {
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
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        cu.setAsCurrent();
        void* forceArgs[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, forceArgs, numParticles);
        posTensor.grad().zero_();
    }
    return energyTensor.item<double>();
}
