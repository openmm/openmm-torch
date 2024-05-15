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

#include "ReferenceTorchKernels.h"
#include "TorchForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return data->periodicBoxVectors;
}

static map<string, double>& extractEnergyParameterDerivatives(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->energyParameterDerivatives;
}

ReferenceCalcTorchForceKernel::~ReferenceCalcTorchForceKernel() {
}

void ReferenceCalcTorchForceKernel::initialize(const System& system, const TorchForce& force, torch::jit::script::Module& module) {
    this->module = module;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    outputsForces = force.getOutputsForces();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalNames.push_back(force.getGlobalParameterName(i));
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++)
        energyParameterDerivatives.push_back(force.getEnergyParameterDerivativeName(i));
}

double ReferenceCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    int numParticles = pos.size();
    torch::Tensor posTensor = torch::from_blob(pos.data(), {numParticles, 3}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    vector<torch::jit::IValue> inputs = {posTensor};
    if (usePeriodic) {
        Vec3* box = extractBoxVectors(context);
        torch::Tensor boxTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);
        inputs.push_back(boxTensor);
    }
    // Store parameter tensors that need derivatives
    vector<torch::Tensor> parameterTensors;
    for (const string& name : globalNames) {
        // Require grad if the parameter is in the list of energy parameter derivatives
        bool requires_grad = std::find(energyParameterDerivatives.begin(), energyParameterDerivatives.end(), name) != energyParameterDerivatives.end();
	auto tensor = torch::tensor(context.getParameter(name), torch::TensorOptions().requires_grad(requires_grad));
	parameterTensors.emplace_back(tensor);
        inputs.push_back(tensor);
    }
    torch::Tensor energyTensor, forceTensor;
    if (outputsForces) {
        auto outputs = module.forward(inputs).toTuple();
        energyTensor = outputs->elements()[0].toTensor();
        forceTensor = outputs->elements()[1].toTensor();
    } else
        energyTensor = module.forward(inputs).toTensor();
    // Compute any gradients by backpropagating the PyTorch model
    std::vector<torch::Tensor> inputs_with_grad;
    if (includeForces && !outputsForces) {
        inputs_with_grad.push_back(posTensor);
    }
    for (int i = 1; i < inputs.size(); i++) { // Skip the positions
        auto& input = inputs[i];
        if (input.isTensor()) {
            auto tensor = input.toTensor();
            if (tensor.requires_grad())
                inputs_with_grad.emplace_back(tensor);
        }
    }
    if (inputs_with_grad.size() > 0) {
        auto none = torch::Tensor();
        energyTensor.backward(none, false, false, inputs_with_grad);
    }
    if (includeForces) {
        if (!outputsForces) {
            forceTensor = posTensor.grad();
        }
        if (!(forceTensor.dtype() == torch::kFloat64))
            forceTensor = forceTensor.to(torch::kFloat64);
        double* outputForces = forceTensor.data_ptr<double>();
        double forceSign = (outputsForces ? 1.0 : -1.0);
        for (int i = 0; i < numParticles; i++)
            for (int j = 0; j < 3; j++)
                force[i][j] += forceSign * outputForces[3 * i + j];
    }
    // Store parameter energy derivatives
    auto& derivs = extractEnergyParameterDerivatives(context);
    for (int i = 0; i < energyParameterDerivatives.size(); i++) {
        // Compute the derivative of the energy with respect to this parameter.
        // The derivative is stored in the gradient of the parameter tensor.
        double derivative = parameterTensors[i].grad().item<double>();
	auto name = energyParameterDerivatives[i];
        derivs[name] = derivative;
    }
    return energyTensor.item<double>();
}
