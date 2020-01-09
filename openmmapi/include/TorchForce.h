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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include "internal/windowsExportTorch.h"

namespace TorchPlugin {

/**
 * This class implements forces that are defined by user-supplied neural networks.
 * It uses the PyTorch library to perform the computations. */

class OPENMM_EXPORT_NN TorchForce : public OpenMM::Force {
public:
    /**
     * Create a TorchForce.  The network is defined by a PyTorch ScriptModule saved
     * to a file.
     *
     * @param file   the path to the file containing the network
     */
    TorchForce(const std::string& file);
    /**
     * Get the path to the file containing the network.
     */
    const std::string& getFile() const;
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
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    std::string file;
    bool usePeriodic;
};

} // namespace TorchPlugin

#endif /*OPENMM_TORCHFORCE_H_*/
