/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2024 Stanford University and the Authors.      *
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

#include "CommonTorchKernels.h"
#include "CommonTorchKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

CommonCalcTorchForceKernel::~CommonCalcTorchForceKernel() {
}

void CommonCalcTorchForceKernel::initialize(const System& system, const TorchForce& force, torch::jit::script::Module& module) {
    this->module = module;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    outputsForces = force.getOutputsForces();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalNames.push_back(force.getGlobalParameterName(i));
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        paramDerivs.insert(force.getEnergyParameterDerivativeName(i));
        cc.addEnergyParameterDerivative(force.getEnergyParameterDerivativeName(i));
    }
    int numParticles = system.getNumParticles();

    // Inititalize Common objects.

    if (torch::cuda::is_available()) {
        const torch::Device device(torch::kCUDA, 0); // This implicitly initializes PyTorch
        this->module.to(device);
    }
    this->module.eval();
    this->module = torch::jit::freeze(this->module);
    map<string, string> defines;
    if (cc.getUseDoublePrecision()) {
        networkForces.initialize<double>(cc, 3*numParticles, "networkForces");
        defines["FORCES_TYPE"] = "double";
    }
    else {
        networkForces.initialize<float>(cc, 3*numParticles, "networkForces");
        defines["FORCES_TYPE"] = "float";
    }
    ComputeProgram program = cc.compileProgram(CommonTorchKernelSources::torchForce, defines);
    addForcesKernel = program->createKernel("addForces");
    addForcesKernel->addArg(networkForces);
    addForcesKernel->addArg(cc.getLongForceBuffer());
    addForcesKernel->addArg(cc.getAtomIndexArray());
    addForcesKernel->addArg(numParticles);
    addForcesKernel->addArg();
    addForcesKernel->addArg();
}

double CommonCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cc.getNumAtoms();
    torch::Tensor posTensor = torch::from_blob(pos.data(), {numParticles, 3}, torch::kFloat64);
    if (!cc.getUseDoublePrecision())
        posTensor = posTensor.to(torch::kFloat32);
    posTensor.set_requires_grad(true);
    vector<torch::jit::IValue> inputs = {posTensor};
    if (usePeriodic) {
        Vec3 box[3];
        cc.getPeriodicBoxVectors(box[0], box[1], box[2]);
        torch::Tensor boxTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);
        if (!cc.getUseDoublePrecision())
            boxTensor = boxTensor.to(torch::kFloat32);
        inputs.push_back(boxTensor);
    }
    map<string, torch::Tensor> derivInputs;
    for (const string& name : globalNames) {
        bool requiresGrad = (paramDerivs.find(name) != paramDerivs.end());
        torch::Tensor globalTensor = torch::tensor(context.getParameter(name), torch::TensorOptions().requires_grad(requiresGrad));
        inputs.push_back(globalTensor);
        if (requiresGrad)
            derivInputs[name] = globalTensor;
    }
    torch::Tensor energyTensor, forceTensor;
    if (outputsForces) {
        auto outputs = module.forward(inputs).toTuple();
        energyTensor = outputs->elements()[0].toTensor();
        forceTensor = outputs->elements()[1].toTensor();
    }
    else
        energyTensor = module.forward(inputs).toTensor();
    bool hasComputedBackward = false;
    if (includeForces) {
        if (!outputsForces) {
            energyTensor.backward();
            forceTensor = posTensor.grad();
            hasComputedBackward = true;
        }
        if (cc.getUseDoublePrecision()) {
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
        addForcesKernel->setArg(4, cc.getPaddedNumAtoms());
        addForcesKernel->setArg(5, outputsForces ? 1 : -1);
        addForcesKernel->execute(numParticles);
    }
    map<string, double>& energyParamDerivs = cc.getEnergyParamDerivWorkspace();
    for (const string& name : paramDerivs) {
        if (!hasComputedBackward) {
            energyTensor.backward();
            hasComputedBackward = true;
        }
        energyParamDerivs[name] += derivInputs[name].grad().item<double>();
    }
    return energyTensor.item<double>();
}

