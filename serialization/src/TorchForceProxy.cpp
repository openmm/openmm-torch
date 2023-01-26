/* -------------------------------------------------------------------------- *
 *                                 OpenMM-NN                                    *
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

#include "TorchForceProxy.h"
#include "TorchForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <openssl/evp.h>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

static string base64Encode(const string& input) {
    // base64 encodes 4 bytes for every 3 bytes in the input. Also it is padded to the next multiple of 4
    const size_t expectedLength = 4 * ((input.size() + 2) / 3);
    // An additional byte is required to store null termination
    std::vector<unsigned char> output(expectedLength + 1, '\0');
    std::vector<unsigned char> uInput(input.begin(), input.end());
    const auto outputLength = EVP_EncodeBlock(output.data(), uInput.data(), input.size());
    if (expectedLength != outputLength) {
        throw OpenMMException("Error during model file encoding");
    }
    // Remove the extra null termination character
    return string(output.begin(), output.end() - 1);
}

static string base64Decode(const string& input) {
    // base64 decoding yields 3 bytes for each 4 bytes of input
    const auto expectedLength = 3 * input.size() / 4;
    // An additional byte is required to store null termination
    std::vector<unsigned char> output(expectedLength + 1, '\0');
    std::vector<unsigned char> uInput(input.begin(), input.end());
    const auto outputLength = EVP_DecodeBlock(output.data(), uInput.data(), uInput.size());
    if (expectedLength != outputLength) {
        throw OpenMMException("Error during model file decoding");
    }
    // Remove the extra null termination character
    return string(output.begin(), output.end() - 1);
}

static string base64EncodeFromFileName(const string& fileName) {
    stringstream ss;
    ss << ifstream(fileName).rdbuf();
    const auto fileContents = ss.str();
    return base64Encode(fileContents);
}

TorchForceProxy::TorchForceProxy() : SerializationProxy("TorchForce") {
}

void TorchForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const TorchForce& force = *reinterpret_cast<const TorchForce*>(object);
    node.setStringProperty("file", force.getFile());
    try {
        node.setStringProperty("encodedFileContents", base64EncodeFromFileName(force.getFile()));
    }
    catch (...) {
        throw OpenMMException("Could not serialize model file.");
    }
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setBoolProperty("outputsForces", force.getOutputsForces());
    SerializationNode& globalParams = node.createChildNode("GlobalParameters");
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParams.createChildNode("Parameter").setStringProperty("name", force.getGlobalParameterName(i)).setDoubleProperty("default", force.getGlobalParameterDefaultValue(i));
    }
}

void* TorchForceProxy::deserialize(const SerializationNode& node) const {
    int storedVersion = node.getIntProperty("version");
    if (storedVersion > 2)
        throw OpenMMException("Unsupported version number");
    string fileName;
    const string storedEncodedFile = node.getStringProperty("encodedFileContents", "");
    if (storedVersion == 1) {
        fileName = node.getStringProperty("file");
        if (!storedEncodedFile.empty()) {
            const auto encodedFileContents = base64EncodeFromFileName(fileName);
            if (storedEncodedFile.compare(encodedFileContents) != 0) {
                throw OpenMMException("The provided model file does not match the stored one");
            }
        }
    }
    if (storedVersion > 2) {
        if (not node.getStringProperty("file", "").empty()) {
            throw OpenMMException("Serializer version is incompatible with file parameter");
        }
        fileName = tmpnam(nullptr); // A unique filename
        ofstream(fileName) << base64Decode(storedEncodedFile);
    }
    TorchForce* force = new TorchForce(fileName);
    if (node.hasProperty("forceGroup"))
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
    if (node.hasProperty("usesPeriodic"))
        force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
    if (node.hasProperty("outputsForces"))
        force->setOutputsForces(node.getBoolProperty("outputsForces"));
    for (const SerializationNode& child : node.getChildren()) {
        if (child.getName() == "GlobalParameters")
            for (auto& parameter : child.getChildren())
                force->addGlobalParameter(parameter.getStringProperty("name"), parameter.getDoubleProperty("default"));
    }
    return force;
}
