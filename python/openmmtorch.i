%module openmmtorch

%include "factory.i"
%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "TorchForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <torch/csrc/jit/python/module_python.h>
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%typemap(in) const torch::jit::Module&(torch::jit::Module module) {
    py::object o = py::reinterpret_borrow<py::object>($input);
    module = torch::jit::as_module(o).value();
    $1 = &module;
}

%typemap(out) const torch::jit::Module& {
    auto fileName = std::tmpnam(nullptr);
    $1->save(fileName);
    $result = py::module::import("torch.jit").attr("load")(fileName).release().ptr();
    //This typemap assumes that torch does not require the file to exist after construction
    std::remove(fileName);
}

%typecheck(SWIG_TYPECHECK_POINTER) const torch::jit::Module& {
    py::object o = py::reinterpret_borrow<py::object>($input);
    $1 = torch::jit::as_module(o).has_value() ? 1 : 0;
}

namespace TorchPlugin {

class TorchForce : public OpenMM::Force {
public:
    TorchForce(const std::string& file);
    TorchForce(const torch::jit::Module& module);
    const std::string& getFile() const;
    const torch::jit::Module& getModule() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;
    void setOutputsForces(bool);
    bool getOutputsForces() const;
    int getNumGlobalParameters() const;
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    void setProperty(const std::string& name, const std::string& value);
    const std::string& getProperty(const std::string& name) const;

    /*
     * Add methods for casting a Force to a TorchForce.
    */
    %extend {
        static TorchPlugin::TorchForce& cast(OpenMM::Force& force) {
            return dynamic_cast<TorchPlugin::TorchForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<TorchPlugin::TorchForce*>(&force) != NULL);
        }
    }
};

}
