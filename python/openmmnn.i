%module openmmnn

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "NeuralNetworkForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

namespace NNPlugin {

class NeuralNetworkForce : public OpenMM::Force {
public:
    NeuralNetworkForce(const std::string& file);
    const std::string& getFile() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;
};

}
