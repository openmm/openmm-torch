from setuptools import setup, Extension
import os
import platform

openmm_dir = '@OPENMM_DIR@'
torch_include_dirs = '@TORCH_INCLUDE_DIRS@'.split(';')
nn_plugin_header_dir = '@NN_PLUGIN_HEADER_DIR@'
nn_plugin_library_dir = '@NN_PLUGIN_LIBRARY_DIR@'
torch_dir, _ = os.path.split('@TORCH_LIBRARY@')

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++14']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib', '-rpath', torch_dir]

extension = Extension(name='_openmmtorch',
                      sources=['TorchPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMTorch'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), nn_plugin_header_dir] + torch_include_dirs,
                      library_dirs=[os.path.join(openmm_dir, 'lib'), nn_plugin_library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib')],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmtorch',
      version='1.0',
      py_modules=['openmmtorch'],
      ext_modules=[extension],
     )
