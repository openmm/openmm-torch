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

#include "CudaNeuralNetworkKernels.h"
#include "CudaNeuralNetworkKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcNeuralNetworkForceKernel::~CudaCalcNeuralNetworkForceKernel() {
}

void CudaCalcNeuralNetworkForceKernel::initialize(const System& system, const NeuralNetworkForce& force, torch::jit::script::Module& module) {
    cu.setAsCurrent();
    this->module = module;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    int numParticles = system.getNumParticles();

    // Inititalize CUDA objects.

    map<string, string> defines;
    if (cu.getUseDoublePrecision()) {
        networkForces.initialize<double>(cu, 3*numParticles, "networkForces");
        defines["FORCES_TYPE"] = "double";
    }
    else {
        networkForces.initialize<float>(cu, 3*numParticles, "networkForces");
        defines["FORCES_TYPE"] = "float";
    }
    CUmodule program = cu.createModule(CudaNeuralNetworkKernelSources::neuralNetworkForce, defines);
    addForcesKernel = cu.getKernel(program, "addForces");
}

double CudaCalcNeuralNetworkForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();
    torch::Tensor posTensor = torch::from_blob(pos.data(), {numParticles, 3}, torch::kFloat64);
    if (!cu.getUseDoublePrecision())
        posTensor = posTensor.to(torch::kFloat32);
    posTensor.set_requires_grad(true);
    vector<torch::jit::IValue> inputs = {posTensor};
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        torch::Tensor boxTensor = torch::from_blob(box, {3, 3}, torch::kFloat64);
        if (!cu.getUseDoublePrecision())
            boxTensor = boxTensor.to(torch::kFloat32);
        inputs.push_back(boxTensor);
    }
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
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energyTensor.item<double>();


/*
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();
    if (positionsType == TF_FLOAT) {
        float* positions = reinterpret_cast<float*>(TF_TensorData(positionsTensor));
        for (int i = 0; i < numParticles; i++) {
            positions[3*i] = pos[i][0];
            positions[3*i+1] = pos[i][1];
            positions[3*i+2] = pos[i][2];
        }
    }
    else {
        double* positions = reinterpret_cast<double*>(TF_TensorData(positionsTensor));
        for (int i = 0; i < numParticles; i++) {
            positions[3*i] = pos[i][0];
            positions[3*i+1] = pos[i][1];
            positions[3*i+2] = pos[i][2];
        }
    }
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        if (boxType == TF_FLOAT) {
            float* boxVectors = reinterpret_cast<float*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
        else {
            double* boxVectors = reinterpret_cast<double*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
    }
    vector<TF_Output> inputs, outputs;
    int forceOutputIndex = 0;
    if (includeEnergy)
        outputs.push_back({TF_GraphOperationByName(graph, "energy"), 0});
    if (includeForces) {
        forceOutputIndex = outputs.size();
        outputs.push_back({TF_GraphOperationByName(graph, "forces"), 0});
    }
    vector<TF_Tensor*> inputTensors, outputTensors(outputs.size());
    inputs.push_back({TF_GraphOperationByName(graph, "positions"), 0});
    inputTensors.push_back(positionsTensor);
    if (usePeriodic) {
        inputs.push_back({TF_GraphOperationByName(graph, "boxvectors"), 0});
        inputTensors.push_back(boxVectorsTensor);
    }
    TF_Status* status = TF_NewStatus();
    TF_SessionRun(session, NULL, &inputs[0], &inputTensors[0], inputs.size(),
                  &outputs[0], &outputTensors[0], outputs.size(),
                  NULL, 0, NULL, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(string("Error running TensorFlow session: ")+TF_Message(status));
    TF_DeleteStatus(status);
    double energy = 0.0;
    if (includeEnergy) {
        if (energyType == TF_FLOAT)
            energy = reinterpret_cast<float*>(TF_TensorData(outputTensors[0]))[0];
        else
            energy = reinterpret_cast<double*>(TF_TensorData(outputTensors[0]))[0];
    }
    if (includeForces) {
        const void* data = TF_TensorData(outputTensors[forceOutputIndex]);
        networkForces.upload(data);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;*/
}
