/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2022 Stanford University and the Authors.      *
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
#include "openmm/common/ContextSelector.h"
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
    usePeriodic = force.usesPeriodicBoundaryConditions();
    outputsForces = force.getOutputsForces();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalNames.push_back(force.getGlobalParameterName(i));
    int numParticles = system.getNumParticles();

    // Initialize CUDA objects for PyTorch
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initialize PyTorch
    module.to(device);
    torch::TensorOptions options = torch::TensorOptions()
        .device(device)
        .dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
    posTensor = torch::empty({numParticles, 3}, options.requires_grad(!outputsForces));
    boxTensor = torch::empty({3, 3}, options);

    // Initialize CUDA objects for OpenMM-Torch
    ContextSelector selector(cu);
    map<string, string> defines;
    CUmodule program = cu.createModule(CudaTorchKernelSources::torchForce, defines);
    copyInputsKernel = cu.getKernel(program, "copyInputs");
    addForcesKernel = cu.getKernel(program, "addForces");
}

double CudaCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    CUcontext primary;
    cuDevicePrimaryCtxRetain(&primary, cu.getDevice());
    cuCtxPushCurrent(primary);
    int numParticles = cu.getNumAtoms();

    // Get pointers to the atomic positions and simulation box
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

    // Copy the atomic positions and simulation box to PyTorch tensors
    {
        ContextSelector selector(cu);
        void* inputArgs[] = {&posData, &boxData, &cu.getPosq().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
                &numParticles, cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer()};
        cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
        CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context"); // Synchronize before switching to the PyTorch context
    }

    // Prepare the input of the PyTorch model
    vector<torch::jit::IValue> inputs = {posTensor};
    if (usePeriodic)
        inputs.push_back(boxTensor);
    for (const string& name : globalNames)
        inputs.push_back(torch::tensor(context.getParameter(name)));

    // Execute the PyTorch model
    torch::Tensor energyTensor, forceTensor;
    if (outputsForces) {
        auto outputs = module.forward(inputs).toTuple();
        energyTensor = outputs->elements()[0].toTensor();
        forceTensor = outputs->elements()[1].toTensor();
    }
    else
        energyTensor = module.forward(inputs).toTensor();

    if (includeForces) {

        // Compute force by backprogating the PyTorch model
        if (!outputsForces) {
            energyTensor.backward();
            forceTensor = posTensor.grad();
        }

        // Get a pointer to the computed forces
        void* forceData;
        if (cu.getUseDoublePrecision()) {
            if (!(forceTensor.dtype() == torch::kFloat64)) // TODO: simplify the logic when support for PyTorch 1.7 is dropped
                forceTensor = forceTensor.to(torch::kFloat64);
            forceData = forceTensor.data_ptr<double>();
        }
        else {
            if (!(forceTensor.dtype() == torch::kFloat32)) // TODO: simplify the logic when support for PyTorch 1.7 is dropped
                forceTensor = forceTensor.to(torch::kFloat32);
            forceData = forceTensor.data_ptr<float>();
        }
        CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context"); // Synchronize before switching to the OpenMM context

        // Add the computed forces to the total atomic forces
        {
            ContextSelector selector(cu);
            int paddedNumAtoms = cu.getPaddedNumAtoms();
            int forceSign = (outputsForces ? 1 : -1);
            void* forceArgs[] = {&forceData, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms, &forceSign};
            cu.executeKernel(addForcesKernel, forceArgs, numParticles);
            CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context"); // Synchronize before switching to the PyTorch context
        }

        // Reset the forces
        if (!outputsForces)
            posTensor.grad().zero_();
    }
    cuCtxPopCurrent(&primary);
    cuDevicePrimaryCtxRelease(cu.getDevice());
    return energyTensor.item<double>(); // This implicitly synchronize the PyTorch context
}
