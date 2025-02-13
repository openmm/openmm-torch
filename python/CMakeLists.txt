set(WRAP_FILE TorchPluginWrapper.cpp)
set(MODULE_NAME openmmtorch)

# Execute SWIG to generate source code for the Python module.
foreach(dir ${TORCH_INCLUDE_DIRS})
    set(torchincs "${torchincs}" "-I${dir}")
endforeach()

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        ${torchincs}
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmtorch.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/openmmtorch.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}"  "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(NN_PLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(NN_PLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
