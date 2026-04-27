/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2026 Stanford University and the Authors.      *
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
#include "openmm/common/CommonKernelSources.h"
#include "openmm/common/ContextSelector.h"
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

class CommonCalcPythonTorchForceKernel::ReorderListener : public ComputeContext::ReorderListener {
public:
    ReorderListener(CommonCalcPythonTorchForceKernel& owner) : owner(owner) {
    }
    void execute() {
        owner.sortParticles();
    }
private:
    CommonCalcPythonTorchForceKernel& owner;
};

void CommonCalcPythonTorchForceKernel::initialize(const ContextImpl& context, const PythonTorchForce& force) {
    ContextSelector selector(cc);
    computation = &force.getComputation();
    usePeriodic = force.usesPeriodicBoundaryConditions();
    particles = force.getParticles();
    numParticles = particles.size();
    if (numParticles == 0)
        numParticles = context.getSystem().getNumParticles();
    positionsVec.resize(numParticles);
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    positionsArray.initialize(cc, 3*numParticles, elementSize, "positions");
    forcesArray.initialize(cc, 3*numParticles, elementSize, "forces");
    map<string, string> defines;
    defines["NUM_ATOMS"] = cc.intToString(numParticles);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    ComputeProgram program = cc.compileProgram(CommonKernelSources::pythonForce, defines);
    if (particles.size() > 0) {
        particlesArray.initialize<int>(cc, numParticles, "particles");
        reorderedParticles.initialize<int>(cc, numParticles, "reorderedParticles");
        particlesArray.upload(particles);
        reorderedParticles.upload(particles);
        cc.addReorderListener(new ReorderListener(*this));
        copyPositionsKernel = program->createKernel("copyPositions");
        copyPositionsKernel->addArg(cc.getPosq());
        copyPositionsKernel->addArg(positionsArray);
        copyPositionsKernel->addArg(reorderedParticles);
        copyPositionsKernel->addArg(numParticles);
        addForcesKernel = program->createKernel("addForcesSubset");
        addForcesKernel->addArg(forcesArray);
        addForcesKernel->addArg(cc.getLongForceBuffer());
        addForcesKernel->addArg(cc.getAtomIndexArray());
        addForcesKernel->addArg(reorderedParticles);
        addForcesKernel->addArg(numParticles);
    }
    else {
        addForcesKernel = program->createKernel("addForcesAll");
        addForcesKernel->addArg(forcesArray);
        addForcesKernel->addArg(cc.getLongForceBuffer());
        addForcesKernel->addArg(cc.getAtomIndexArray());
    }
}

double CommonCalcPythonTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (cc.getContextIndex() != 0)
        return 0.0;
    torch::Tensor posTensor = getPositions();
    State::StateBuilder builder(contextImpl.getTime(), contextImpl.getStepCount());
    builder.setParameters(contextImpl.getParameters());
    if (usePeriodic) {
        Vec3 a, b, c;
        contextImpl.getPeriodicBoxVectors(a, b, c);
        builder.setPeriodicBoxVectors(a, b, c);
    }
    State state = builder.getState();
    torch::Tensor forceTensor = computation->compute(state, posTensor, energy);
    if (includeForces)
        addForces(forceTensor);
    return includeEnergy ? energy : 0.0;
}

torch::Tensor CommonCalcPythonTorchForceKernel::getPositions() {
    // If the NonbondedUtilities uses periodic boundary conditions, the positions might have been
    // wrapped to the periodic box.  If this force also applies periodic boundary conditions, that's
    // alright.  Otherwise, we need to move them back.

    bool fixPeriodic = usePeriodic || !cc.getNonbondedUtilities().getUsePeriodic();
    if (particles.size() == 0) {
        // The force applies to the whole system, so we can just use the standard getPositions().

        contextImpl.getPositions(positionsVec, fixPeriodic);
    }
    else {
        // Retrieve positions for the subset of particles the force is applied to.

        ContextSelector selector(cc);
        copyPositionsKernel->execute(numParticles);
        if (cc.getUseDoublePrecision()) {
            vector<double> pos(3*numParticles);
            positionsArray.download(pos);
            for (int i = 0; i < numParticles; i++)
                positionsVec[i] = Vec3(pos[3*i], pos[3*i+1], pos[3*i+2]);
        }
        else {
            vector<float> pos(3*numParticles);
            positionsArray.download(pos);
            for (int i = 0; i < numParticles; i++)
                positionsVec[i] = Vec3((double) pos[3*i], (double) pos[3*i+1], (double) pos[3*i+2]);
        }
        if (fixPeriodic) {
            Vec3 boxVectors[3];
            cc.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
            for (int i = 0; i < numParticles; ++i) {
                mm_int4 offset = cc.getPosCellOffsets()[particles[i]];
                positionsVec[i] -= boxVectors[0]*offset.x-boxVectors[1]*offset.y-boxVectors[2]*offset.z;
            }
        }
    }
    return torch::from_blob(positionsVec.data(), {numParticles, 3}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
}

void CommonCalcPythonTorchForceKernel::sortParticles() {
    // Update the list of particles to account for reordering.

    const vector<int>& order = cc.getAtomIndex();
    vector<int> inverseOrder(order.size());
    for (int i = 0; i < cc.getNumAtoms(); i++)
        inverseOrder[order[i]] = i;
    vector<int> reordered(particles.size());
    for (int i = 0; i < particles.size(); i++)
        reordered[i] = inverseOrder[particles[i]];
    reorderedParticles.upload(reordered);
}

void CommonCalcPythonTorchForceKernel::addForces(torch::Tensor forceTensor) {
    // Add in the forces.

    ContextSelector selector(cc);
    if (cc.getUseDoublePrecision()) {
        if (!(forceTensor.dtype() == torch::kFloat64))
            forceTensor = forceTensor.to(torch::kFloat64);
        double* data = forceTensor.data_ptr<double>();
        forcesArray.upload(data);
    }
    else {
        if (!(forceTensor.dtype() == torch::kFloat32))
            forceTensor = forceTensor.to(torch::kFloat32);
        float* data = forceTensor.data_ptr<float>();
        forcesArray.upload(data);
    }
    addForcesKernel->execute(cc.getNumAtoms());
}
