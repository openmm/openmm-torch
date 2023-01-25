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

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

// Based in the answers in https://stackoverflow.com/a/34571089/5155484
static constexpr auto base64Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static string base64Encode(const string& in) {
    string out;
    unsigned val = 0;
    int valb = -6;
    for (auto& c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(base64Alphabet[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6)
        out.push_back(base64Alphabet[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4)
        out.push_back('=');
    return out;
}

static string base64Decode(const string& in) {
    string out;
    vector<int> T(256, -1);
    for (int i = 0; i < 64; i++)
        T[base64Alphabet[i]] = i;
    unsigned val = 0;
    int valb = -8;
    for (auto& c : in) {
        if (T[c] == -1)
            break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

string encodeFromFileName(const string& fileName) {
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
        node.setStringProperty("encodedFileContents", encodeFromFileName(force.getFile()));
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
            const auto encodedFileContents = encodeFromFileName(fileName);
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
