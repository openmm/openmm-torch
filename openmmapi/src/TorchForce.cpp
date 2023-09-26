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
 * Contributors: Raimondas Galvelis, Raul P. Pelaez                           *
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
#include "internal/TorchForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

TorchForce::TorchForce(const torch::jit::Module& module, const map<string, string>& properties) : file(), usePeriodic(false), outputsForces(false), module(module) {
    const std::map<std::string, std::string> defaultProperties = {{"useCUDAGraphs", "false"}, {"CUDAGraphWarmupSteps", "10"}};
    this->properties = defaultProperties;
    for (auto& property : properties) {
        if (defaultProperties.find(property.first) == defaultProperties.end())
            throw OpenMMException("TorchForce: Unknown property '" + property.first + "'");
        this->properties[property.first] = property.second;
    }
}

TorchForce::TorchForce(const std::string& file, const map<string, string>& properties) : TorchForce(torch::jit::load(file), properties) {
    this->file = file;
}

const string& TorchForce::getFile() const {
    return file;
}

const torch::jit::Module& TorchForce::getModule() const {
    return this->module;
}

ForceImpl* TorchForce::createImpl() const {
    return new TorchForceImpl(*this);
}

void TorchForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool TorchForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}

void TorchForce::setOutputsForces(bool outputsForces) {
    this->outputsForces = outputsForces;
}

bool TorchForce::getOutputsForces() const {
    return outputsForces;
}

int TorchForce::addGlobalParameter(const string& name, double defaultValue) {
    globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
    return globalParameters.size() - 1;
}

int TorchForce::getNumGlobalParameters() const {
    return globalParameters.size();
}

const string& TorchForce::getGlobalParameterName(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].name;
}

void TorchForce::setGlobalParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].name = name;
}

double TorchForce::getGlobalParameterDefaultValue(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].defaultValue;
}

void TorchForce::setGlobalParameterDefaultValue(int index, double defaultValue) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].defaultValue = defaultValue;
}

void TorchForce::setProperty(const std::string& name, const std::string& value) {
    if (properties.find(name) == properties.end())
        throw OpenMMException("TorchForce: Unknown property '" + name + "'");
    properties[name] = value;
}

const std::map<std::string, std::string>& TorchForce::getProperties() const {
    return properties;
}
