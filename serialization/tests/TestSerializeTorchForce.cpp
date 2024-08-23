/* -------------------------------------------------------------------------- *
 *                                 OpenMM-NN                                    *
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

#include "TorchForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <cstdio>
#include <sstream>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>
using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerTorchSerializationProxies();

void serializeAndDeserialize(TorchForce force) {
    force.setForceGroup(3);
    force.addGlobalParameter("x", 1.3);
    force.addGlobalParameter("y", 2.221);
    force.setUsesPeriodicBoundaryConditions(true);
    force.setOutputsForces(true);
    force.addEnergyParameterDerivative("y");
    force.setProperty("useCUDAGraphs", "true");
    force.setProperty("CUDAGraphWarmupSteps", "5");

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<TorchForce>(&force, "Force", buffer);
    TorchForce* copy = XmlSerializer::deserialize<TorchForce>(buffer);

    // Compare the two forces to see if they are identical.

    TorchForce& force2 = *copy;
    ostringstream bufferModule;
    force.getModule().save(bufferModule);
    ostringstream bufferModule2;
    force2.getModule().save(bufferModule2);
    ASSERT_EQUAL(bufferModule.str(), bufferModule2.str());
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getNumGlobalParameters(), force2.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        ASSERT_EQUAL(force.getGlobalParameterName(i), force2.getGlobalParameterName(i));
        ASSERT_EQUAL(force.getGlobalParameterDefaultValue(i), force2.getGlobalParameterDefaultValue(i));
    }
    ASSERT_EQUAL(force.usesPeriodicBoundaryConditions(), force2.usesPeriodicBoundaryConditions());
    ASSERT_EQUAL(force.getOutputsForces(), force2.getOutputsForces());
    ASSERT_EQUAL(force.getNumEnergyParameterDerivatives(), force2.getNumEnergyParameterDerivatives());
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++)
        ASSERT_EQUAL(force.getEnergyParameterDerivativeName(i), force2.getEnergyParameterDerivativeName(i));
    ASSERT_EQUAL(force.getProperties().size(), force2.getProperties().size());
    for (auto& prop : force.getProperties())
        ASSERT_EQUAL(prop.second, force2.getProperties().at(prop.first));
}

void testSerializationFromModule() {
    string fileName = "tests/forces.pt";
    torch::jit::Module module = torch::jit::load(fileName);
    TorchForce force(module);
    serializeAndDeserialize(force);
}

void testSerializationFromFile() {
    string fileName = "tests/forces.pt";
    TorchForce force(fileName);
    serializeAndDeserialize(force);
}

int main() {
    try {
        registerTorchSerializationProxies();
        testSerializationFromFile();
        testSerializationFromModule();
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
