set(WRAP_FILE NNPluginWrapper.cpp)
set(MODULE_NAME openmmnn)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmnn.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/openmmnn.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(NN_PLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(NN_PLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/nnpops.py" "${CMAKE_CURRENT_BINARY_DIR}/nnpops.py"
    COMMAND "${PYTHON_EXECUTABLE}" setup.py build
    COMMAND "${PYTHON_EXECUTABLE}" setup.py install
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
