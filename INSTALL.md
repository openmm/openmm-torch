Installing with conda
---------------------

We provide [conda](https://docs.conda.io/) packages for Linux and MacOS via [`conda-forge`](https://conda-forge.org/), which can be installed from the [conda-forge channel](https://anaconda.org/conda-forge/openmm-torch):
```bash
conda install -c conda-forge openmm-torch
```
If you don't have `conda` available, we recommend installing [Miniconda for Python 3](https://docs.conda.io/en/latest/miniconda.html) to provide the `conda` package manager. 


Building from source
--------------------

Depending on your environment there are different instructions to follow:
   - Linux (NO CUDA): This is for Linux OS when you do have have, or do not want to use CUDA.
   - Linux (CUDA): This is for Linux OS when you have CUDA installed and have a CUDA device you want to use.
   - MacOS

### Linux (NO CUDA)


#### Prerequisites

- Minconda https://docs.conda.io/en/latest/miniconda.html#linux-installers


#### Build & install

1. Get the source code

   ```
   git clone https://github.com/openmm/openmm-torch.git
   cd openmm-torch
   ```


3. Create and activate a conda environment using the provided environment file


   ```
   conda env create -n openmm-torch -f linux_cpu.yaml
   conda activate openmm-torch
   ```

4. Configure 
   ```
   mkdir build && cd build

   # set the Torch_DIR path
   export Torch_DIR="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"

   cmake .. -DOPENMM_DIR=$CONDA_PREFIX \
         -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
   ```

6. Build
   ```
   make
   make PythonInstall
   ```

7. Test
   ```
   make test
   ```

8. Install
   ```
   make install
   ```

Your built version of openmm-torch will now be available in your conda environment. You can test this by trying to import `openmmtoch` into `python`.

```
python -c "from openmmtorch import TorchForce"
```
Should complete without error.


### Linux (CUDA)


#### Prerequisites

- Minconda https://docs.conda.io/en/latest/miniconda.html#linux-installers
- CUDA Toolkit https://developer.nvidia.com/cuda-downloads

#### Build & install

1. Get the source code

   ```
   git clone https://github.com/openmm/openmm-torch.git
   cd openmm-torch
   ```


2. Make sure your `$CUDA_HOME` path is set correctly to the path of your CUDA installation

   ```
   echo $CUDA_HOME
   ```

3. Create and activate a conda environment using the provided environment file

   ```
   conda env create -n openmm-torch -f linux_cuda.yaml
   conda activate openmm-torch
   ```



4. Configure 
   ```
   mkdir build && cd build

   # set the Torch_DIR path
   export Torch_DIR="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"

   cmake .. -DOPENMM_DIR=$CONDA_PREFIX \
         -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
   ```

6. Build
   ```
   make
   make PythonInstall
   ```

7. Test
   ```
   make test
   ```

8. Install
   ```
   make install
   ```

Your built version of openmm-torch will now be available in your conda environment. You can test this by trying to import `openmmtoch` into `python`.

```
python -c "from openmmtorch import TorchForce"
```
Should complete without error.


### MacOS

#### Prerequisites

- Minconda https://docs.conda.io/en/latest/miniconda.html#macos-installers


1. Get the source code

   ```
   git clone https://github.com/openmm/openmm-torch.git
   cd openmm-torch
   ```

2. Create and activate a conda environment using the provided environment file

   ```
   conda env create -n openmm-torch -f macOS.yaml
   conda activate openmm-torch
   ```

4. Configure 
   ```
   mkdir build && cd build

   # set the Torch_DIR path
   export Torch_DIR="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"

   cmake .. -DOPENMM_DIR=$CONDA_PREFIX \
         -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
   ```

6. Build
   ```
   make
   make PythonInstall
   ```

7. Test
   ```
   make test
   ```

8. Install
   ```
   make install
   ```

Your built version of openmm-torch will now be available in your conda environment. You can test this by trying to import `openmmtoch` into `python`.

```
python -c "from openmmtorch import TorchForce"
```
Should complete without error.
