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

#include "TorchForceProxy.h"
#include "TorchForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/csrc/jit/serialization/import.h>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

static string hexEncode(const string& input) {
    stringstream ss;
    ss << hex << setfill('0');
    for (const unsigned char& i : input) {
        ss << setw(2) << static_cast<uint64_t>(i);
    }
    return ss.str();
}

static string hexDecode(const string& input) {
    string res;
    res.reserve(input.size() / 2);
    for (size_t i = 0; i < input.length(); i += 2) {
        istringstream iss(input.substr(i, 2));
        uint64_t temp;
        iss >> hex >> temp;
        res += static_cast<unsigned char>(temp);
    }
    return res;
}

static string hexEncodeFromFileName(const string& filename) {
    ifstream inputFile(filename, ios::binary);
    stringstream inputStream;
    inputStream << inputFile.rdbuf();
    return hexEncode(inputStream.str());
}

TorchForceProxy::TorchForceProxy() : SerializationProxy("TorchForce") {
}

void TorchForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 4);
    const TorchForce& force = *reinterpret_cast<const TorchForce*>(object);
    node.setStringProperty("file", force.getFile());
    try {
        auto tempFileName = std::tmpnam(nullptr);
        force.getModule().save(tempFileName);
        node.setStringProperty("encodedFileContents", hexEncodeFromFileName(tempFileName));
        std::remove(tempFileName);
    }
    catch (exception& ex) {
        throw OpenMMException("TorchForceProxy: Could not serialize model. Failed with error: " + string(ex.what()));
    }
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setBoolProperty("outputsForces", force.getOutputsForces());
    SerializationNode& globalParams = node.createChildNode("GlobalParameters");
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalParams.createChildNode("Parameter").setStringProperty("name", force.getGlobalParameterName(i)).setDoubleProperty("default", force.getGlobalParameterDefaultValue(i));
    SerializationNode& paramDerivs = node.createChildNode("ParameterDerivatives");
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++)
        paramDerivs.createChildNode("Parameter").setStringProperty("name", force.getEnergyParameterDerivativeName(i));
    SerializationNode& properties = node.createChildNode("Properties");
    for (auto& prop : force.getProperties())
        properties.createChildNode("Property").setStringProperty("name", prop.first).setStringProperty("value", prop.second);
}

void* TorchForceProxy::deserialize(const SerializationNode& node) const {
    int storedVersion = node.getIntProperty("version");
    if (storedVersion > 4)
        throw OpenMMException("Unsupported version number");
    TorchForce* force;
    if (storedVersion == 1) {
        string fileName = node.getStringProperty("file");
        force = new TorchForce(fileName);
    } else {
        const string storedEncodedFile = node.getStringProperty("encodedFileContents");
        string fileName = tmpnam(nullptr); // A unique filename
        ofstream ofs = ofstream(fileName, ios::binary);
        ofs << hexDecode(storedEncodedFile);
        ofs.close();
        auto model = torch::jit::load(fileName);
        std::remove(fileName.c_str());
        force = new TorchForce(model);
    }
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
        if (child.getName() == "ParameterDerivatives")
            for (auto& parameter : child.getChildren())
                force->addEnergyParameterDerivative(parameter.getStringProperty("name"));
        if (child.getName() == "Properties")
            for (auto& property : child.getChildren())
                force->setProperty(property.getStringProperty("name"), property.getStringProperty("value"));
    }
    return force;
}
