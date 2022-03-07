from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
nn_plugin_header_dir = '@NN_PLUGIN_HEADER_DIR@'
nn_plugin_library_dir = '@NN_PLUGIN_LIBRARY_DIR@'
torch_dir, _ = os.path.split('@TORCH_LIBRARY@')

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib', '-rpath', torch_dir]

extension = Extension(name='_openmmtorch',
                      sources=['TorchPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMTorch'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), nn_plugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), nn_plugin_library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib'), torch_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmtorch',
      version='1.0',
      py_modules=['openmmtorch'],
      ext_modules=[extension],
     )

