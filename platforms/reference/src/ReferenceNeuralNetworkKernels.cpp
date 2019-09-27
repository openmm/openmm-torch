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

#include "ReferenceNeuralNetworkKernels.h"
#include "NeuralNetworkForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace NNPlugin;
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

ReferenceCalcNeuralNetworkForceKernel::~ReferenceCalcNeuralNetworkForceKernel() {
}

void ReferenceCalcNeuralNetworkForceKernel::initialize(const System& system, const NeuralNetworkForce& force, torch::jit::script::Module& module) {
    this->module = module;
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcNeuralNetworkForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
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
    torch::Tensor energyTensor = module.forward(inputs).toTensor();
    if (includeForces) {
        energyTensor.backward();
        torch::Tensor forceTensor = posTensor.grad();
        if (!(forceTensor.dtype() == torch::kFloat64))
            forceTensor = forceTensor.to(torch::kFloat64);
        double* outputForces = forceTensor.data_ptr<double>();
        for (int i = 0; i < numParticles; i++)
            for (int j = 0; j < 3; j++)
                force[i][j] = -outputForces[3*i+j];
    }
    return energyTensor.item<double>();
}
