# This script install CUDA on Ubuntu-based systemws
# It uses the Nvidia repos for Ubuntu 18.04, which as of Dec 2020
# includes packages for CUDA 10.0, 10.1, 10.2, 11.0, 11.1, 11.2
# Future versions might require an updated repo (maybe Ubuntu 20)
# It expects a $CUDA_VERSION environment variable set to major.minor (e.g. 10.0)

set -euxo pipefail

# Enable retrying
echo 'APT::Acquire::Retries "5";' | sudo tee /etc/apt/apt.conf.d/80-retries

sudo wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --tries 5 \
    -O /etc/apt/preferences.d/cuda-repository-pin-600 \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update -qq

CUDA_APT=${CUDA_VERSION/./-}
## cufft changed package names in CUDA 11
if [[ ${CUDA_VERSION} == 10.* ]]; then CUFFT="cuda-cufft"; else CUFFT="libcufft"; fi
sudo apt-get install -y \
    libgl1-mesa-dev cuda-compiler-${CUDA_APT} \
    cuda-drivers cuda-driver-dev-${CUDA_APT} \
    cuda-cudart-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} \
    ${CUFFT}-${CUDA_APT} ${CUFFT}-dev-${CUDA_APT} \
    cuda-nvprof-${CUDA_APT} tree
sudo apt-get clean

if [[ ! -d /usr/local/cuda ]]; then
    sudo ln -s /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda
fi

if [[ -f /usr/local/cuda-${CUDA_VERSION}/lib64/stubs/libcuda.so ]]; then
    sudo ln -s /usr/local/cuda-${CUDA_VERSION}/lib64/stubs/libcuda.so /usr/local/cuda-${CUDA_VERSION}/lib64/stubs/libcuda.so.1
fi

# Remove Nvidia's OpenCL
sudo rm -rf /usr/local/cuda-${CUDA_VERSION}/lib64/libOpenCL.* /usr/local/cuda-${CUDA_VERSION}/include/CL /etc/OpenCL/vendors/nvidia.icd

export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "CUDA_HOME=${CUDA_HOME}" >> ${GITHUB_ENV}
echo "CUDA_PATH=${CUDA_PATH}" >> ${GITHUB_ENV}
echo "PATH=${PATH}" >> ${GITHUB_ENV}
