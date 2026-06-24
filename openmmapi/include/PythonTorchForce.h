#ifndef OPENMM_PYTHONTORCHFORCE_H_
#define OPENMM_PYTHONTORCHFORCE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2025-2026 Stanford University and the Authors.      *
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

#include "openmm/Force.h"
#include "openmm/State.h"
#include "internal/windowsExportTorch.h"
#include <torch/torch.h>
#include <map>
#include <string>

namespace TorchPlugin {

/**
 * This abstract class represents an interface for performing a computation.  It is not intended to
 * be used or subclassed directly by users.  The Python wrapper contains a subclass that implements
 * the interface using a Python function.
 * @private
 */
class OPENMM_EXPORT_NN PythonTorchForceComputation {
public:
    PythonTorchForceComputation() {
    }
    virtual ~PythonTorchForceComputation() {
    }
    /**
     * Compute forces and energy.  The State contains particle parameters and optionally
     * periodic box vectors.  Implementations should store the potential energy into the
     * energy argument and return a tensor of shape (particles, 3) containing the forces.
     */
    virtual torch::Tensor compute(const OpenMM::State& state, const torch::Tensor& positions, double& energy) const = 0;
};

/**
 * This class provides a mechanism for computing forces and energy with Python code.  It is similar to the
 * PythonForce class included with OpenMM, but it is specialized to give better performance when working with
 * PyTorch models.
 *
 * To use it, define a Python function that takes two arguments: a State object and a Tensor of shape (# particles, 3).
 * The State contains global parameters and periodic box vectors.  The Tensor contains particle positions.  The function
 * should compute the potential energy and forces, returning them as its two return values.  The energy should be a
 * scalar Tensor containing the value in kJ/mol.  The forces should be a Tensor of shape (# particles, 3) containing
 * the value in kJ/mol/nm.  For example,
 * 
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: python
 * 
 *    def compute(state, pos):
 *        k = state.getParameters()['k']
 *        energy = k*torch.sum(pos*pos)
 *        force = -0.5*k*pos
 *        return energy, force
 *
 * \endverbatim
 * 
 * Now create a PythonTorchForce, passing the function to the constructor.  If you want the force
 * to depend on global parameters, pass a dict as the second parameter with the names and default
 * values of the parameters.
 * 
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: python
 * 
 *    force = PythonTorchForce(compute, {'k':2.5})
 * 
 * \endverbatim
 * 
 * The default value of a parameter is its value in newly created Contexts.  After a Context is
 * created, you can change the values of parameters by calling setParameter() on it.
 * 
 * The PythonTorchForce cannot tell whether the function you provide makes use of periodic boundary
 * conditions, so you must tell it.  To make the force periodic, call
 * setUsesPeriodicBoundaryConditions(True).  This will cause usesPeriodicBoundaryConditions()
 * to return True, and the State passed to the computation function will contain periodic
 * box vectors.  The positions may also be wrapped into a different periodic box to keep them
 * closer to the origin and improve accuracy.
 *
 * A PythonTorchForce can optionally be applied to only a subset of the particles in a system.  To do
 * this, call setParticles() on it, providing the indices of the particles to apply it to.  The
 * computation function should then proceed as if those particles were the entire system.  The positions
 * passed to it will be a smaller Tensor containing only the positions of those particles, and the returned
 * forces should similarly contain only those particles.  That is, forces[i] should be the force on the i'th
 * particle passed to setParticles().  When applying forces to only a small fraction of the particles in a
 * system, this can greatly improve performance.
 * 
 * When using XmlSerializer to save a PythonTorchForce, it uses the Python pickle module to save
 * the computation function.  If it cannot be pickled, you will not be able to serialize the
 * PythonTorchForce.  Functions defined at the top level of a module can usually be pickled, but local
 * functions defined inside another function cannot.
 */
class OPENMM_EXPORT PythonTorchForce : public OpenMM::Force {
public:
    /**
     * Create a PythonTorchForce.  This constructor is used internally, and is not intended for use
     * by users.  The Python wrapper defines an alternate constructor that takes a Python
     * function instead of a PythonTorchForceComputation.
     *
     * @param computation        an object defining how the forces and energy should be computed
     * @param globalParameters   any global parameters used by the force.  Keys are the parameter
     *                           names, and the corresponding values are their default values.
     * @param particles          the indices of the particles to use when computing the force.  If
     *                           this is empty (the default), all particles in the system will be used.
     * @private
     */
    explicit PythonTorchForce(PythonTorchForceComputation* computation, const std::map<std::string, double>& globalParameters,
                         const std::vector<int>& particles=std::vector<int>());
    ~PythonTorchForce();
    /**
     * Get the PythonTorchForceComputation that defines the computation.
     * @private
     */
    const PythonTorchForceComputation& getComputation() const;
    /**
     * Get all global parameters defined by this force.  Keys are the parameter names, and the
     * corresponding values are their default values.
     */
    const std::map<std::string, double>& getGlobalParameters() const;
    /**
     * Get the indices of the particles to use when computing the force.  If this
     * is empty, all particles in the system will be used.
     */
    const std::vector<int>& getParticles() const {
        return particles;
    }
    /**
     * Set the indices of the particles to use when computing the force.  If this
     * is empty, all particles in the system will be used.
     */
    void setParticles(const std::vector<int>& particles);
    /**
     * Get the pickled representation of the computation function.  If it cannot be pickled,
     * this will be an empty vector.
     */
    const std::vector<char>& getPickledFunction() const;
    /**
     * Set the pickled representation of the computation function.  This is called automatically
     * by the Python constructor.
     * @private
     */
    void setPickledFunction(char* function, int length);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const;
    /**
     * Set whether or not this force makes use of periodic boundary conditions.
     * If this is set to true, periodic box vectors can be retrieved from the
     * State passed to the computation function.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    PythonTorchForceComputation* computation;
    std::map<std::string, double> globalParameters;
    bool usePeriodic;
    std::vector<int> particles;
    std::vector<char> pickled;
};

} // namespace TorchPlugin

#endif /*OPENMM_PYTHONTORCHFORCE_H_*/
