# FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    git \
    wget \
    vim \
    sudo \
    tar \
    unzip \
    openssh-server \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    swig \
    libopenblas-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev\
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python3-venv \
    git-lfs \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# symlink for python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# add user and his password
ENV USER=docker_diffdepth
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}

# create some directories for mounting volumes
RUN mkdir diff_depth_new && chown -R ${UID}:${GID} /home/${USER}
# RUN mkdir Datasets && chown -R ${UID}:${GID} /home/${USER}

USER ${UID}:${GID}

ENV PATH="/home/${USER}/.local/bin:$PATH"

# create venv
ENV VIRTUAL_ENV="/home/${USER}/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV

# Ensure pip, wheel and setuptools are up to date
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.6)
# RUN python3 -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# Install PyTorch with CUDA 11.3
RUN python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install mmcv-full with correct CUDA and PyTorch compatibility
RUN python3 -m pip install mmcv==2.0.0rc3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# upgrade pip
ARG PIP_VERSION=23.3.1
ARG SETUPTOOLS_VERSION=68.2.2
RUN pip install pip==${PIP_VERSION} setuptools==${SETUPTOOLS_VERSION}
