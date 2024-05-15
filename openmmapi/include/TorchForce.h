#ifndef OPENMM_TORCHFORCE_H_
#define OPENMM_TORCHFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "internal/windowsExportTorch.h"

namespace TorchPlugin {

/**
 * This class implements forces that are defined by user-supplied neural networks.
 * It uses the PyTorch library to perform the computations.
 * The PyTorch module can either be  passed directly as an argument to
 * the  constructor  or loaded  from  a  file.   In either  case,  the
 * constructor makes a copy of the  module in memory. Later changes to
 * the original module or to the file do not affect it.*/

class OPENMM_EXPORT_NN TorchForce : public OpenMM::Force {
public:
    /**
     * Create a TorchForce.  The network is defined by a PyTorch ScriptModule saved
     * to a file.
     *
     * @param file       the path to the file containing the network
     * @param properties optional map of properties
     */
    TorchForce(const std::string& file,
               const std::map<std::string, std::string>& properties = {});
    /**
     * Create a TorchForce.  The network is defined by a PyTorch ScriptModule
     * Note that this constructor makes a copy of the provided module.
     * Any changes to the module  after calling this constructor will be ignored by TorchForce.
     *
     * @param module   an instance of the torch module
     * @param properties optional map of properties
     */
    TorchForce(const torch::jit::Module &module, const std::map<std::string, std::string>& properties = {});
    /**
     * Get the path to the file containing the network.
     * If the TorchForce instance was constructed with a module, instead of a filename,
     * this function returns an empty string.
     */
    const std::string& getFile() const;
    /**
     * Get the torch module currently in use.
     */
    const torch::jit::Module & getModule() const;
    /**
     * Set whether this force makes use of periodic boundary conditions.  If this is set
     * to true, the network must take a 3x3 tensor as its second input, which
     * is set to the current periodic box vectors.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Get whether this force makes use of periodic boundary conditions.
     */
    bool usesPeriodicBoundaryConditions() const;
    /**
     * Set whether the network directly outputs forces.  By default it is expected to produce
     * a single scalar output containing the energy, and backpropagation is used to compute
     * the forces.  Alternatively, you can set this flag to true in which case the network is
     * expected to produce a tuple with two elements: a scalar with the potential energy, and an
     * Nx3 tensor with the force on every atom.  This can be useful when you have a more efficient
     * way to compute the forces than the generic backpropagation algorithm.
     */
    void setOutputsForces(bool outputsForces);
    /**
     * Get whether the network directly outputs forces.
     */
    bool getOutputsForces() const;
    /**
     * Get the number of global parameters that the interaction depends on.
     */
    int getNumGlobalParameters() const;
    /**
     * Get the number of global parameters with respect to which the derivative of the energy
     * should be computed.
     */
    int getNumEnergyParameterDerivatives() const;
    /**
     * Add a new global parameter that the interaction may depend on.  The default value provided to
     * this method is the initial value of the parameter in newly created Contexts.  You can change
     * the value at any time by calling setParameter() on the Context.
     *
     * @param name             the name of the parameter
     * @param defaultValue     the default value of the parameter
     * @return the index of the parameter that was added
     */
    int addGlobalParameter(const std::string& name, double defaultValue);
    /**
     * Get the name of a global parameter.
     *
     * @param index     the index of the parameter for which to get the name
     * @return the parameter name
     */
    const std::string& getGlobalParameterName(int index) const;
    /**
     * Set the name of a global parameter.
     *
     * @param index          the index of the parameter for which to set the name
     * @param name           the name of the parameter
     */
    void setGlobalParameterName(int index, const std::string& name);
    /**
     * Get the default value of a global parameter.
     *
     * @param index     the index of the parameter for which to get the default value
     * @return the parameter default value
     */
    double getGlobalParameterDefaultValue(int index) const;
    /**
     * Set the default value of a global parameter.
     *
     * @param index          the index of the parameter for which to set the default value
     * @param defaultValue   the default value of the parameter
     */
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    /**
     * Request that this Force compute the derivative of its energy with respect to a global parameter.
     * The parameter must have already been added with addGlobalParameter().
     *
     * @param name             the name of the parameter
     */
    void addEnergyParameterDerivative(const std::string& name);
    /**
     * Get the name of a global parameter with respect to which this Force should compute the
     * derivative of the energy.
     *
     * @param index     the index of the parameter derivative, between 0 and getNumEnergyParameterDerivatives()
     * @return the parameter name
     */
    const std::string& getEnergyParameterDerivativeName(int index) const;
    /**
     * Set a value of a property.
     *
     * @param name           the name of the property
     * @param value          the value of the property
     */
    void setProperty(const std::string& name, const std::string& value);
    /**
     * Get the map of properties for this instance.
     * @return A map of property names to values.
     */
    const std::map<std::string, std::string>& getProperties() const;
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class GlobalParameterInfo;
    std::string file;
    bool usePeriodic, outputsForces;
    std::vector<GlobalParameterInfo> globalParameters;
    std::vector<int> energyParameterDerivatives;
    torch::jit::Module module;
    std::map<std::string, std::string> properties;
    std::string emptyProperty;
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class TorchForce::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

} // namespace TorchPlugin

#endif /*OPENMM_TORCHFORCE_H_*/
