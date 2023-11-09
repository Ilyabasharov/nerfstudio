# Variables used at build time.
## Base CUDA version. See all supported version at https://hub.docker.com/r/nvidia/cuda/tags?page=2&name=-devel-ubuntu
ARG CUDA_VERSION=11.8.0
## Base Ubuntu version.
ARG OS_VERSION=22.04

# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION} AS base

## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here.
## To speedup the image build process remove all architectures but the one of your explicit GPU.
## Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37

# Dublicate args because of the visibility zone
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CUDA_VERSION
ARG OS_VERSION

# metainformation
LABEL org.opencontainers.image.version="0.1.18" \
    org.opencontainers.image.source="https://github.com/nerfstudio-project/nerfstudio" \
    org.opencontainers.image.licenses="Apache License 2.0" \
    org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive \
    ## Set timezone as it is required by some packages.
    TZ=Europe/Berlin \
    ## CUDA Home, required to find CUDA in some packages.
    CUDA_HOME="/usr/local/cuda" \
    ## Set LD_LIBRARY_PATH for local libs (glog etc.)
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    ## Set TCNN_CUDA_ARCHITECTURES for tinycudann installation
    TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    ## Set makeflags for faster building
    MAKEFLAGS=-j$(nproc) \
    ## Insert to PATH CUDA bin dir 
    PATH="/usr/local/cuda/bin:${PATH}" \
    ## Insert to CPATH CUDA include dir 
    CPATH="/usr/local/cuda/include:${CPATH}" \
    ## Update torch_cuda_list for torch_scatter
    ## See https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    ## Update torch_nvcc_flags for torch_scatter
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf glog

# Install Ceres-solver (required by colmap).
RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF \
             -DBUILD_EXAMPLES=OFF && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf ceres-solver

# Install colmap.
RUN git clone --branch 3.8 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf colmap

# Create non root user that mimic host user and setup environment.
## Create default user settings.
ARG DOCKER_USER=user
ARG DOCKER_UID=1000
ARG DOCKER_GID=root
ARG DOCKER_PW=user
RUN groupadd -g ${DOCKER_GID} ${DOCKER_USER} && \
    useradd -m -d /home/${DOCKER_USER} -g ${DOCKER_GID} -G sudo -u ${DOCKER_UID} ${DOCKER_USER} && \
    usermod -aG sudo ${DOCKER_USER} && \
    ## Set user password
    echo "${DOCKER_USER}:${DOCKER_PW}" | chpasswd && \
    ## Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    ## Add concrete user to the sudoers file
    echo "${DOCKER_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to new user and workdir.
USER ${DOCKER_UID}:${DOCKER_GID}
WORKDIR /home/${DOCKER_USER}

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/${DOCKER_USER}/.local/bin"
SHELL ["/bin/bash", "-c"]

# Base packages installation
## Upgrade pip and install packages.
RUN python3.10 -m pip install \
    --upgrade \
        pip \
        setuptools \
        pathtools \
        promise \
        pybind11 \
        omegaconf \
        ninja

## Install pytorch and submodules
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3.10 -m pip install \
    torch==2.1.0+cu${CUDA_VER} \
    torchvision==0.15.2+cu${CUDA_VER} \
    torch-scatter==2.1.2 \
        --find-links https://data.pyg.org/whl/torch-2.1.0+${CUDA_VER}.html \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

## Install tynyCUDNN (we need to set the target architectures as environment variable first).
RUN python3.10 -m pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

## Install pycolmap, required by hloc.
RUN git clone --branch v0.4.0 --recursive https://github.com/colmap/pycolmap.git && \
    cd pycolmap && \
    python3.10 -m pip install . && \
    cd ..

## Install hloc master (last release (1.3) is too old) as alternative feature detector and matcher option for nerfstudio.
RUN git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git && \
    cd Hierarchical-Localization && \
    python3.10 -m pip install -e . && \
    cd ..

## Install pyceres from source
RUN git clone --branch v1.0 --recursive https://github.com/cvg/pyceres.git && \
    cd pyceres && \
    python3.10 -m pip install -e . && \
    cd ..

## Install pixel perfect sfm.
RUN git clone --branch main --recursive https://github.com/cvg/pixel-perfect-sfm.git && \
    cd pixel-perfect-sfm && \
    python3.10 -m pip install -e . && \
    cd ..

# Copy nerfstudio folder and give ownership to user.
ADD --chown=${DOCKER_UID}:${DOCKER_GID} . /home/${DOCKER_USER}/nerfstudio

# Install nerfstudio dependencies.
RUN cd nerfstudio && \
    python3.10 -m pip install -e . && \
    cd ..

# Change working directory
WORKDIR /workspace

# Install nerfstudio cli auto completion
RUN ns-install-cli --mode install

# Bash as default entrypoint.
CMD ["/bin/bash", "-l"]
