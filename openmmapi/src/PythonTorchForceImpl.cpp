/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
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

#include "internal/PythonTorchForceImpl.h"
#include "TorchKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"
#include <set>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

PythonTorchForceImpl::PythonTorchForceImpl(const PythonTorchForce& owner) : owner(owner), computation(owner.getComputation()),
        defaultParameters(owner.getGlobalParameters()), usePeriodic(owner.usesPeriodicBoundaryConditions()) {
    forceGroup = owner.getForceGroup();
}

PythonTorchForceImpl::~PythonTorchForceImpl() {
}

void PythonTorchForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcPythonTorchForceKernel::Name(), context);
    kernel.getAs<CalcPythonTorchForceKernel>().initialize(context, owner);
}

double PythonTorchForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<forceGroup)) != 0)
        return kernel.getAs<CalcPythonTorchForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> PythonTorchForceImpl::getKernelNames() {
    return {CalcCustomCPPForceKernel::Name()};
}

map<string, double> PythonTorchForceImpl::getDefaultParameters() {
    return defaultParameters;
}
