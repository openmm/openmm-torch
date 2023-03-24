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
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <map>

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

CudaCalcTorchForceKernel::CudaCalcTorchForceKernel(string name, const Platform& platform, CudaContext& cu) :
        CalcTorchForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

CudaCalcTorchForceKernel::~CudaCalcTorchForceKernel() {
    cuDevicePrimaryCtxRelease(cu.getDevice());
}

void CudaCalcTorchForceKernel::initialize(const System& system, const TorchForce& force, torch::jit::script::Module& module) {
    this->module = module;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    outputsForces = force.getOutputsForces();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalNames.push_back(force.getGlobalParameterName(i));
    int numParticles = system.getNumParticles();

    // Enable CUDA Graphs
    const std::string useCUDAGraphs = force.getProperty("useCUDAGraphs");
    if (useCUDAGraphs == "true")
        useGraph = true;
    else if (useCUDAGraphs == "false" || useCUDAGraphs == "")
        useGraph = false;
    else
        throw OpenMMException("TorchForce: invalid value of \"useCUDAGraphs\"");
#if !CUDA_GRAPHS_SUPPORTED
    if (useGraph)
        throw OpenMMException("TorchForce: CUDA Graphs are not supported! "
                              "You need PyTorch 1.10 or newer");
#endif
    // Push the PyTorch context
    // NOTE: Pytorch is always using the primary context.
    //       It makes the primary context current, if it is not a case.
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

    // Initialize CUDA objects for PyTorch
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initialize PyTorch
    module.to(device);
    torch::TensorOptions options = torch::TensorOptions()
        .device(device)
        .dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
    posTensor = torch::empty({numParticles, 3}, options.requires_grad(!outputsForces));
    boxTensor = torch::empty({3, 3}, options);
    energyTensor = torch::empty({1}, options.dtype(torch::kFloat64));
    forceTensor = torch::empty({numParticles, 3}, options);

    // Get pointers to data
    if (cu.getUseDoublePrecision()) {
        posData = posTensor.data_ptr<double>();
        boxData = boxTensor.data_ptr<double>();
        forceData = forceTensor.data_ptr<double>();
    } else {
        posData = posTensor.data_ptr<float>();
        boxData = boxTensor.data_ptr<float>();
        forceData = forceTensor.data_ptr<float>();
    }

    // Pop the PyToch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that PyTorch haven't messed up the context stack

    // Initialize CUDA objects for OpenMM-Torch
    ContextSelector selector(cu); // Switch to the OpenMM context
    map<string, string> defines;
    CUmodule program = cu.createModule(CudaTorchKernelSources::torchForce, defines);
    copyInputsKernel = cu.getKernel(program, "copyInputs");
    addForcesKernel = cu.getKernel(program, "addForces");
}

// CUDA Graphs (https://pytorch.org/docs/master/notes/cuda.html#cuda-graphs)
// require a static graph and persistent input and output tensors.
// These requiments are partially enforced by using a function,
// the tensors are passed by reference.
static void execute_graph(bool outputsForces,
                          bool includeForces,
                          torch::jit::script::Module& module,
                          vector<torch::jit::IValue>& inputs,
                          torch::Tensor& posTensor,
                          torch::Tensor& energyTensor,
                          torch::Tensor& forceTensor) {

    // Execute the PyTorch model
    torch::Tensor energy, forces;
    if (outputsForces) {
        auto outputs = module.forward(inputs).toTuple();
        energy = outputs->elements()[0].toTensor();
        forces = outputs->elements()[1].toTensor();
    }
    else
        energy = module.forward(inputs).toTensor();

    // Compute force by backprogating the PyTorch model
    if (includeForces)
        if (!outputsForces) {
            energy.backward();
            forces = posTensor.grad();
        }

    // Set the output tensors
    energyTensor.copy_(energy.detach().flatten());
    forceTensor.copy_(forces.detach().to(posTensor.dtype()));

    // Reset the forces
    if (!outputsForces)
        posTensor.grad().zero_();
}

double CudaCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    int numParticles = cu.getNumAtoms();
#if CUDA_GRAPHS_SUPPORTED
    bool captureGraph = graphs.find(includeForces) == graphs.end();
#else
    bool captureGraph = false;
#endif

    // Push to the PyTorch context
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

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
        ContextSelector selector(cu); // Switch to the OpenMM context
        void* inputArgs[] = {&posData, &boxData, &cu.getPosq().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
                &numParticles, cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer()};
        cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
    }

    // Prepare an input for the PyTorch model
    vector<torch::jit::IValue> inputs = {posTensor};
    if (!useGraph || captureGraph) {
        if (usePeriodic)
            inputs.push_back(boxTensor);
        for (const string& name : globalNames)
            inputs.push_back(torch::tensor(context.getParameter(name)));
    }

#if CUDA_GRAPHS_SUPPORTED
    // Convert the PyTorch model into a CUDA Graph
    if (useGraph && captureGraph) {

        // Get a stream for a graph capture
        c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false, posTensor.device().index());
        c10::cuda::CUDAStreamGuard guard(stream);

        // Warm up the graph
        // for (int i = 0; i < 3; i++) // TODO debug the multiple executions
            execute_graph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor);

        // Capture the graph
        graphs[includeForces].capture_begin();
        execute_graph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor);
        graphs[includeForces].capture_end();
    }

    // Execute the PyTorch model
    if (useGraph)
        // Execute the corresponding CUDA Graph
        graphs[includeForces].replay();
    else
#endif
        // Execute the PyTorch model directly
        execute_graph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor);

    if (includeForces) {
        CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context"); // Synchronize before switching to the OpenMM context

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
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the OpenMM context

        // Add the computed forces to the total atomic forces
        {
            ContextSelector selector(cu); // Switch to the OpenMM context
            int paddedNumAtoms = cu.getPaddedNumAtoms();
            int forceSign = (outputsForces ? 1 : -1);
            void* forceArgs[] = {&forceData, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms, &forceSign};
            cu.executeKernel(addForcesKernel, forceArgs, numParticles);
            CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
        }
    }

    // Get energy
    const double energy = energyTensor.item<double>(); // This implicitly synchronizes the PyTorch context

    // Pop to the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that the correct context was popped

    return energy;
}
