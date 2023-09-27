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
 * Contributors: Raimondas Galvelis, Raul P. Pelaez                           *
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
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`
#define CHECK_RESULT(result, prefix)                                             \
    if (result != CUDA_SUCCESS) {                                                \
        std::stringstream m;                                                     \
        m << prefix << ": " << cu.getErrorString(result) << " (" << result << ")"\
          << " at " << __FILE__ << ":" << __LINE__;                              \
        throw OpenMMException(m.str());                                          \
    }

CudaCalcTorchForceKernel::CudaCalcTorchForceKernel(string name, const Platform& platform, CudaContext& cu) : CalcTorchForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
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

    // Push the PyTorch context
    // NOTE: Pytorch is always using the primary context.
    //       It makes the primary context current, if it is not a case.
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

    // Initialize CUDA objects for PyTorch
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initialize PyTorch
    module.to(device);
    torch::TensorOptions options = torch::TensorOptions().device(device).dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
    posTensor = torch::empty({numParticles, 3}, options.requires_grad(!outputsForces));
    boxTensor = torch::empty({3, 3}, options);
    energyTensor = torch::empty({0}, options);
    forceTensor = torch::empty({0}, options);
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
    auto properties = force.getProperties();
    const std::string useCUDAGraphsString = properties["useCUDAGraphs"];
    if (useCUDAGraphsString == "true")
        useGraphs = true;
    else if (useCUDAGraphsString == "false" || useCUDAGraphsString == "")
        useGraphs = false;
    else
        throw OpenMMException("TorchForce: invalid value of \"useCUDAGraphs\"");
    if (useGraphs) {
        this->warmupSteps = std::stoi(properties["CUDAGraphWarmupSteps"]);
        if (this->warmupSteps <= 0) {
            throw OpenMMException("TorchForce: \"CUDAGraphWarmupSteps\" must be a positive integer");
        }
    }
}

/**
 * Get a pointer to the data in a PyTorch tensor.
 * The tensor is converted to the correct data type if necessary.
 */
static void* getTensorPointer(OpenMM::CudaContext& cu, torch::Tensor& tensor) {
    void* data;
    if (cu.getUseDoublePrecision()) {
        data = tensor.to(torch::kFloat64).data_ptr<double>();
    } else {
        data = tensor.to(torch::kFloat32).data_ptr<float>();
    }
    return data;
}

/**
 * Prepare the inputs for the PyTorch model, copying positions from the OpenMM context.
 */
std::vector<torch::jit::IValue> CudaCalcTorchForceKernel::prepareTorchInputs(ContextImpl& context) {
    int numParticles = cu.getNumAtoms();
    // Get pointers to the atomic positions and simulation box
    void* posData = getTensorPointer(cu, posTensor);
    void* boxData = getTensorPointer(cu, boxTensor);
    // Copy the atomic positions and simulation box to PyTorch tensors
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        void* inputArgs[] = {&posData,
                             &boxData,
                             &cu.getPosq().getDevicePointer(),
                             &cu.getAtomIndexArray().getDevicePointer(),
                             &numParticles,
                             cu.getPeriodicBoxVecXPointer(),
                             cu.getPeriodicBoxVecYPointer(),
                             cu.getPeriodicBoxVecZPointer()};
        cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
    }
    // Prepare the input of the PyTorch model
    vector<torch::jit::IValue> inputs = {posTensor};
    if (usePeriodic)
        inputs.push_back(boxTensor);
    for (const string& name : globalNames)
        inputs.push_back(torch::tensor(context.getParameter(name)));
    return inputs;
}

/**
 * Add the computed forces to the total atomic forces.
 */
void CudaCalcTorchForceKernel::addForces(torch::Tensor& forceTensor) {
    int numParticles = cu.getNumAtoms();
    // Get a pointer to the computed forces
    void* forceData = getTensorPointer(cu, forceTensor);
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

/**
 * This function launches  the workload in a way  compatible with CUDA
 * graphs as far  as OpenMM-Torch goes.  Capturing  this function when
 * the model  is not  itself graph compatible  (due to,  for instance,
 * implicit synchronizations) will result in a CUDA error.
 */
static void executeGraph(bool outputsForces, bool includeForces, torch::jit::script::Module& module, vector<torch::jit::IValue>& inputs, torch::Tensor& posTensor, torch::Tensor& energyTensor,
                         torch::Tensor& forceTensor) {
    if (outputsForces) {
        auto outputs = module.forward(inputs).toTuple();
        energyTensor = outputs->elements()[0].toTensor();
        forceTensor = outputs->elements()[1].toTensor();
    } else {
        energyTensor = module.forward(inputs).toTensor();
        // Compute force by backpropagating the PyTorch model
        if (includeForces) {
            // CUDA graph capture sometimes fails if backwards is not explicitly requested w.r.t positions
	    // See https://github.com/openmm/openmm-torch/pull/120/
	    auto none = torch::Tensor();
	    energyTensor.backward(none, false, false, posTensor);
	    // This is minus the forces, we change the sign later on
            forceTensor = posTensor.grad().clone();
            // Zero the gradient to avoid accumulating it
            posTensor.grad().zero_();
        }
    }
}

double CudaCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Push to the PyTorch context
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");
    auto inputs = prepareTorchInputs(context);
    if (!useGraphs) {
        executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor);
    } else {
        // Record graph if not already done
        bool is_graph_captured = false;
        if (graphs.find(includeForces) == graphs.end()) {
	    //CUDA graph capture must occur in a non-default stream
            const auto stream = c10::cuda::getStreamFromPool(false, cu.getDeviceIndex());
	    const c10::cuda::CUDAStreamGuard guard(stream);
            // Warmup the graph workload before capturing.  This first
            // run  before  capture sets  up  allocations  so that  no
            // allocations are  needed after.  Pytorch's  allocator is
            // stream  capture-aware and,  after warmup,  will provide
            // record static pointers and shapes during capture.
            try {
                for (int i = 0; i < this->warmupSteps; i++)
                    executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor);
            }
            catch (std::exception& e) {
                throw OpenMMException(string("TorchForce Failed to warmup the model before graph construction. Torch reported the following error:\n") + e.what());
            }
            graphs[includeForces].capture_begin();
            try {
                executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor);
                is_graph_captured = true;
                graphs[includeForces].capture_end();
            }
            catch (std::exception& e) {
                if (!is_graph_captured) {
                    graphs[includeForces].capture_end();
                }
                throw OpenMMException(string("TorchForce Failed to capture the model into a CUDA graph. Torch reported the following error:\n") + e.what());
            }
        }
	// Use the same stream as the OpenMM context, even if it is the default stream
        const auto openmmStream = cu.getCurrentStream();
	const auto stream = c10::cuda::getStreamFromExternal(openmmStream, cu.getDeviceIndex());
	const c10::cuda::CUDAStreamGuard guard(stream);
        graphs[includeForces].replay();
    }
    if (includeForces) {
        addForces(forceTensor);
    }
    // Get energy
    const double energy = energyTensor.item<double>(); // This implicitly synchronizes the PyTorch context
    // Pop to the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that the correct context was popped
    return energy;
}
