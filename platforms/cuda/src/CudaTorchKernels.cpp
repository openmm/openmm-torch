/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
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
#include <cuda_runtime_api.h>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`
#define CHECK_RESULT(result, prefix) \
if (result != CUDA_SUCCESS) { \
    std::stringstream m; \
    m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
    throw OpenMMException(m.str());\
}

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
    void* inputArgs[] = {&posData, &boxData, &cu.getPosq().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
            &numParticles, cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer()};
    cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
    vector<torch::jit::IValue> inputs = {posTensor};
    if (usePeriodic)
        inputs.push_back(boxTensor);
    torch::Tensor energyTensor = module.forward(inputs).toTensor();
    if (includeForces) {
        energyTensor.backward();
        // Note: "forceTensor" needs to be cloned due to a shared context (https://github.com/openmm/openmm-torch/issues/13)
        torch::Tensor forceTensor = posTensor.grad().clone();
        // make sure that all calculations on PyTorch side is properly finished before changing CUDA context or starting the `addForcesKernel` of this plugin
        // cudaDeviceSynchronize();  // synchronizing the whole device is not necessary and may even cause problem
        // synchronizing the current context and check the return status
        CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
        cu.setAsCurrent();
        void* data;
        if (cu.getUseDoublePrecision()) {
            if (!(forceTensor.dtype() == torch::kFloat64))
                forceTensor = forceTensor.to(torch::kFloat64);
            data = forceTensor.data_ptr<double>();
        }
        else {
            if (!(forceTensor.dtype() == torch::kFloat32))
                forceTensor = forceTensor.to(torch::kFloat32);
            data = forceTensor.data_ptr<float>();
        }
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* forceArgs[] = {&data, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, forceArgs, numParticles);
        posTensor.grad().zero_();
    }
    return energyTensor.item<double>();
}
