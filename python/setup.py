from setuptools import setup, Extension
import os
import platform

version = '@OPENMM_TORCH_VERSION@'
openmm_dir = '@OPENMM_DIR@'
torch_include_dirs = '@TORCH_INCLUDE_DIRS@'.split(';')
nn_plugin_header_dir = '@NN_PLUGIN_HEADER_DIR@'
nn_plugin_library_dir = '@NN_PLUGIN_LIBRARY_DIR@'
torch_dir, _ = os.path.split('@TORCH_LIBRARY@')

extra_compile_args = ['-std=c++17']
extra_link_args = []
libraries = ['OpenMM', 'OpenMMTorch']
runtime_library_dirs = [os.path.join(openmm_dir, 'lib'), torch_dir]

# For Windows change the compiler flag to /std:c++17
if platform.system() == 'Windows':
    extra_compile_args = ['/std:c++17']
    libraries += ['c10', 'torch']
    if os.environ.get("CUDA_HOME", None) is not None:
        libraries += ['torch_cuda']
    else:
        libraries += ['torch_cpu']
    runtime_library_dirs = None

# setup extra compile and link arguments on Mac
if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']

extension = Extension(name='_openmmtorch',
                      sources=['TorchPluginWrapper.cpp'],
                      libraries=libraries,
                      include_dirs=[os.path.join(openmm_dir, 'include'), nn_plugin_header_dir] + torch_include_dirs,
                      library_dirs=[os.path.join(openmm_dir, 'lib'), nn_plugin_library_dir, torch_dir],
                      runtime_library_dirs=runtime_library_dirs,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmtorch',
      version=version,
      py_modules=['openmmtorch'],
      ext_modules=[extension],
      install_requires=['openmm', 'torch']
     )
