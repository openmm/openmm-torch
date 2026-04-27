%newobject TorchPlugin::PythonTorchForce::PythonTorchForce;

%pythonbegin %{
import torch
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
%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_map.i>
%include <std_string.i>
%include <std_vector.i>

%{
#include "TorchForce.h"
#include "PythonTorchForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/serialization/import.h>

namespace TorchPlugin {
    PythonTorchForce* _createPythonTorchForce(PyObject* computation, const std::map<std::string, double>& globalParameters={}, const std::vector<int>& particles={});
}
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
    %template(parameter_map) map<string, double>;
    %template(particle_list) vector<int>;
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

class PythonTorchForce : public OpenMM::Force {
public:
   ~PythonTorchForce();
   const std::map<std::string, double>& getGlobalParameters() const;
   const std::vector<int>& getParticles() const;
   void setParticles(const std::vector<int> &particles);
   const std::vector<char>& getPickledFunction() const;
   virtual bool usesPeriodicBoundaryConditions() const;
   void setUsesPeriodicBoundaryConditions(bool periodic);

    /*
     * Add methods for casting a Force to a PythonTorchForce.
     */
    %extend {
        static TorchPlugin::PythonTorchForce& cast(OpenMM::Force& force) {
            return dynamic_cast<TorchPlugin::PythonTorchForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<TorchPlugin::PythonTorchForce*>(&force) != NULL);
        }
    }
};

}

%inline %{

namespace TorchPlugin {
    /**
     * This is the PythonTorchForceComputation that performs the computation for a PythonTorchForce.  It invokes the function
     * provided by the user, validates the outputs, and converts them to the required format.
     */
    class ComputationWrapper : public PythonTorchForceComputation {
    public:
        ComputationWrapper(PyObject* computation) : computation(computation) {
            Py_INCREF(computation);
        }
        ~ComputationWrapper() {
            Py_XDECREF(computation);
        }
        torch::Tensor compute(const OpenMM::State& state, const torch::Tensor& positions, double& energy) const {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();

            // Invoke the function.

            swig_type_info* info = SWIGTYPE_p_OpenMM__State;
            PyObject* wrappedState = SWIG_NewPointerObj((void*) &state, info, 0);
            PyObject* wrappedPositions = THPVariable_Wrap(positions);
            PyObject* result = PyObject_CallFunctionObjArgs(computation, wrappedState, wrappedPositions, NULL);
            Py_XDECREF(wrappedState);
            Py_XDECREF(wrappedPositions);
            if (result == NULL) {
                // The function raised an exception.  Convert it to an OpenMMException.

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 12
                PyObject *type;
                PyObject *exception;
                PyObject *traceback;
                PyErr_Fetch(&type, &exception, &traceback);
#else
                PyObject *exception = PyErr_GetRaisedException();
#endif
                PyObject *message = PyObject_Str(exception);
                std::string *ptr;
                SWIG_AsPtr_std_string(message, &ptr);
                Py_XDECREF(message);
                PyGILState_Release(gstate);
                throw OpenMM::OpenMMException(*ptr);
            }

            // Extract the return values.

            if (!PyTuple_Check(result) || PyTuple_Size(result) != 2) {
                PyGILState_Release(gstate);
                throw OpenMM::OpenMMException("PythonTorchForce: Expected two return values");
            }
            PyObject* pyenergy = PyTuple_GetItem(result, 0);
            PyObject* pyforces = PyTuple_GetItem(result, 1);
            if (!THPVariable_Check(pyenergy)) {
                PyGILState_Release(gstate);
                throw OpenMM::OpenMMException("PythonTorchForce: Expected the energy to be a Tensor");
            }
            if (!THPVariable_Check(pyforces)) {
                PyGILState_Release(gstate);
                throw OpenMM::OpenMMException("PythonTorchForce: Expected the forces to be a Tensor");
            }
            torch::Tensor forces = THPVariable_Unpack(pyforces);
            energy = THPVariable_Unpack(pyenergy).item<double>();

            // Clean up before returning.

            Py_XDECREF(result);
            Py_XDECREF(pyenergy);
            Py_XDECREF(pyforces);
            PyGILState_Release(gstate);
            return forces;
        }
    private:
        PyObject* computation;
    };

    /**
     * Construct a new PythonTorchForce.
     */
    PythonTorchForce* _createPythonTorchForce(PyObject* computation, const std::map<std::string, double>& globalParameters, const std::vector<int>& particles) {
        PythonTorchForce* force = new PythonTorchForce(new ComputationWrapper(computation), globalParameters, particles);
        PyObject* pickle = PyImport_ImportModule("pickle");
        PyObject* dumps = PyUnicode_FromString("dumps");
        PyObject* result = PyObject_CallMethodOneArg(pickle, dumps, computation);
        if (result == NULL) {
            // It couldn't be pickled.  It will still work, but can't be serialized.  Clear the error flag.
            PyErr_Clear();
        }
        else {
            char* buffer;
            Py_ssize_t len;
            if (PyBytes_AsStringAndSize(result, &buffer, &len) == 0)
                force->setPickledFunction(buffer, len);
        }
        return force;
    }

    /**
     * This is the serialization proxy used to serialize PythonTorchForce objects.
     */
    class PythonTorchForceProxy : public OpenMM::SerializationProxy {
    public:
        PythonTorchForceProxy() : OpenMM::SerializationProxy("PythonTorchForce") {
        }

        static std::string hexEncode(const std::vector<char>& input) {
            std::stringstream ss;
            ss << std::hex << std::setfill('0');
            for (unsigned char i : input)
                ss << std::setw(2) << static_cast<uint64_t>(i);
            return ss.str();
        }

        static std::vector<char> hexDecode(const std::string& input) {
            std::vector<char> res;
            res.reserve(input.size() / 2);
            for (size_t i = 0; i < input.length(); i += 2) {
                std::istringstream iss(input.substr(i, 2));
                uint64_t temp;
                iss >> std::hex >> temp;
                res.push_back(static_cast<unsigned char>(temp));
            }
            return res;
        }

        void serialize(const void* object, OpenMM::SerializationNode& node) const {
            node.setIntProperty("version", 1);
            const PythonTorchForce& force = *reinterpret_cast<const PythonTorchForce*>(object);
            if (force.getPickledFunction().size() == 0)
                throw OpenMM::OpenMMException("PythonTorchForceProxy: Could not serialize PythonTorchForce because its function could not be pickled.");
            node.setStringProperty("function", hexEncode(force.getPickledFunction()));
            node.setIntProperty("forceGroup", force.getForceGroup());
            node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
            OpenMM::SerializationNode& globalParams = node.createChildNode("GlobalParameters");
            for (auto param : force.getGlobalParameters())
                globalParams.createChildNode("Parameter").setStringProperty("name", param.first).setDoubleProperty("default", param.second);
            OpenMM::SerializationNode& particlesNode = node.createChildNode("Particles");
            for (int i : force.getParticles())
               particlesNode.createChildNode("Particle").setIntProperty("index", i);
        }

        void* deserialize(const OpenMM::SerializationNode& node) const {
            int version = node.getIntProperty("version");
            if (version != 1)
                throw OpenMM::OpenMMException("Unsupported version number");
            std::vector<char> pickledFunction = hexDecode(node.getStringProperty("function"));
            PyObject* pickle = PyImport_ImportModule("pickle");
            PyObject* loads = PyUnicode_FromString("loads");
            PyObject *pythonBytes = PyBytes_FromStringAndSize(pickledFunction.data(), pickledFunction.size());
            PyObject *function = PyObject_CallMethodOneArg(pickle, loads, pythonBytes);
            Py_XDECREF(pythonBytes);
            const OpenMM::SerializationNode& paramsNode = node.getChildNode("GlobalParameters");
            std::map<std::string, double> params;
            for (auto& parameter : paramsNode.getChildren())
                params[parameter.getStringProperty("name")] = parameter.getDoubleProperty("default");
            std::vector<int> particles;
            for (auto& particle : node.getChildNode("Particles").getChildren())
                particles.push_back(particle.getIntProperty("index"));
            PythonTorchForce* force = _createPythonTorchForce(function, params, particles);
            if (node.hasProperty("forceGroup"))
                force->setForceGroup(node.getIntProperty("forceGroup", 0));
            if (node.hasProperty("usesPeriodic"))
                force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
            return force;
        }
    };

    /**
     * Register the serialization proxy.  This function is invoked automatically when the openmm module is imported.
     */
    void registerPythonTorchForceProxy() {
        OpenMM::SerializationProxy::registerProxy(typeid(PythonTorchForce), new PythonTorchForceProxy());
    }
}

%}

%extend TorchPlugin::PythonTorchForce {
    %feature("docstring") PythonTorchForce """Create a PythonTorchForce.

Parameters
----------
computation : function
    A function that performs the computation.  It should take two arguments: a State and a
    Tensor containing positions.  It should return two values: the potential energy and the forces,
    both represented as Tensors.
globalParameters : dict
    Any global parameters the function depends on.  Keys are the parameter names, and the
    corresponding values are their default values.
"""
    PythonTorchForce(PyObject* computation, const std::map<std::string, double>& globalParameters={}, const std::vector<int>& particles={}) {
        return TorchPlugin::_createPythonTorchForce(computation, globalParameters, particles);
    }
}

%pythoncode %{
    registerPythonTorchForceProxy()
%}