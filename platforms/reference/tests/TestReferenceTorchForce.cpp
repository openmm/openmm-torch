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

/**
 * This tests the Reference implementation of TorchForce.
 */

#include "TorchForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerTorchReferenceKernelFactories();

void testForce(bool outputsForces) {
    // Create a random cloud of particles.

    const int numParticles = 10;
    System system;
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
    }
    TorchForce* force = new TorchForce(outputsForces ? "tests/forces.pt" : "tests/central.pt");
    force->setOutputsForces(outputsForces);
    system.addForce(force);

    // Compute the forces and energy.

    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("Reference");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.  The network defines a potential of the form E(r) = |r|^2

    double expectedEnergy = 0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        double r = sqrt(pos.dot(pos));
        expectedEnergy += r*r;
        ASSERT_EQUAL_VEC(pos*(-2.0), state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testPeriodicForce() {
    // Create a random cloud of particles.

    const int numParticles = 10;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 4));
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
    }
    TorchForce* force = new TorchForce("tests/periodic.pt");
    force->setUsesPeriodicBoundaryConditions(true);
    system.addForce(force);

    // Compute the forces and energy.
    
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("Reference");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.  The network defines a potential of the form E(r) = |r|^2

    double expectedEnergy = 0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        pos[0] -= floor(pos[0]/2.0)*2.0;
        pos[1] -= floor(pos[1]/3.0)*3.0;
        pos[2] -= floor(pos[2]/4.0)*4.0;
        double r = sqrt(pos.dot(pos));
        expectedEnergy += r*r;
        ASSERT_EQUAL_VEC(pos*(-2.0), state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testGlobal() {
    // Create a random cloud of particles.

    const int numParticles = 10;
    System system;
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
    }
    TorchForce* force = new TorchForce("tests/global.pt");
    force->addGlobalParameter("k", 2.0);
    force->addEnergyParameterDerivative("k");
    system.addForce(force);

    // Compute the forces and energy.

    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("Reference");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces | State::ParameterDerivatives);

    // See if the energy is correct.  The network defines a potential of the form E(r) = k*|r|^2

    double expectedEnergy = 0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        double r = sqrt(pos.dot(pos));
        expectedEnergy += 2*r*r;
        ASSERT_EQUAL_VEC(pos*(-4.0), state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);

    // Check the gradient of the energy with respect to the parameter.

    double expected = 0.0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        expected += pos.dot(pos);
    }
    double actual = state.getEnergyParameterDerivatives().at("k");
    ASSERT_EQUAL_TOL(expected, actual, 1e-5);

    // Change the global parameter and see if the forces are still correct.

    context.setParameter("k", 3.0);
    state = context.getState(State::Forces | State::ParameterDerivatives);
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        double r = sqrt(pos.dot(pos));
        ASSERT_EQUAL_VEC(pos*(-6.0), state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expected, state.getEnergyParameterDerivatives().at("k"), 1e-5);
}

int main() {
    try {
        registerTorchReferenceKernelFactories();
        testForce(false);
        testForce(true);
        testPeriodicForce();
        testGlobal();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
