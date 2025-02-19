%pythonbegin %{
import sys
if sys.platform == 'win32':
    import os
    import torch
    import openmm
    openmmtorch_library_path = openmm.version.openmm_library_path

    _path = os.environ['PATH']
    os.environ['PATH'] = r'%(lib)s;%(lib)s\plugins;%(path)s' % {'lib': openmmtorch_library_path, 'path': _path}

    os.add_dll_directory(openmmtorch_library_path)

%}

%module openmmtorch

%include "factory.i"
%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_map.i>

%{
#include "TorchForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/serialization/import.h>
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

%typemap(in) const torch::jit::Module&(torch::jit::Module mod) {
    py::object o = py::reinterpret_borrow<py::object>($input);
    py::object pybuffer = py::module::import("io").attr("BytesIO")();
    py::module::import("torch.jit").attr("save")(o, pybuffer);
    std::string s = py::cast<std::string>(pybuffer.attr("getvalue")());
    std::stringstream buffer(s);
    mod = torch::jit::load(buffer);
    $1 = &mod;
}

%typemap(out) const torch::jit::Module& {
    std::stringstream buffer;
    $1->save(buffer);
    auto pybuffer = py::module::import("io").attr("BytesIO")(py::bytes(buffer.str()));
    $result = py::module::import("torch.jit").attr("load")(pybuffer).release().ptr();
}

%typecheck(SWIG_TYPECHECK_POINTER) const torch::jit::Module& {
    py::object o = py::reinterpret_borrow<py::object>($input);
    py::handle ScriptModule = py::module::import("torch.jit").attr("ScriptModule");
    $1 = py::isinstance(o, ScriptModule);
}

namespace std {
    %template(property_map) map<string, string>;
}

namespace TorchPlugin {

class TorchForce : public OpenMM::Force {
public:
    TorchForce(const std::string& file, const std::map<std::string, std::string>& properties = {});
    TorchForce(const torch::jit::Module& module, const std::map<std::string, std::string>& properties = {});
    const std::string& getFile() const;
    const torch::jit::Module& getModule() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;
    void setOutputsForces(bool);
    bool getOutputsForces() const;
    int getNumGlobalParameters() const;
    int getNumEnergyParameterDerivatives() const;
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    void addEnergyParameterDerivative(const std::string& name);
    const std::string& getEnergyParameterDerivativeName(int index) const;
    void setProperty(const std::string& name, const std::string& value);
    const std::map<std::string, std::string>& getProperties() const;

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
