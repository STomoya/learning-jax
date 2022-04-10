# docker image for jax
# modified from: https://github.com/google/jax/discussions/6338

# tensorflow docker image versions: https://hub.docker.com/r/tensorflow/tensorflow/tags
ARG TENSORFLOW_VERSION=2.8.0
FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}-gpu

# packages
# uncomment if using opencv or installing haiku via git
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update -y && \
#     apt-get install -y \
#     libopencv-dev \
#     git

# Usually UID is 1000 if you only have one user on your computer.
# check with `echo ${UID}`
ARG UID=1000
ARG USERNAME=dockeruser

RUN useradd -l -m -u ${UID} ${USERNAME}
USER ${UID}
ENV PATH=/home/${USERNAME}/.local/bin:$PATH

# jaxlib/jax installation
# NOTE: CUDA_VERSION, CUDNN_VERSION confilcts with the base image
# jax versions: https://github.com/google/jax/releases
# reuired CUDA/cuDNN versions and namings: https://github.com/google/jax#pip-installation-gpu-cuda
ARG JAXLIB_VERSION=0.3.2
ARG JAX_CUDA_VERSION=11
ARG JAX_CUDNN_VERSION=82

RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip3 install --user \
    jax[cuda${JAX_CUDA_VERSION}_cudnn${JAX_CUDNN_VERSION}] \
    jaxlib==${JAXLIB_VERSION}+cuda${JAX_CUDA_VERSION}.cudnn${JAX_CUDNN_VERSION} -f https://storage.googleapis.com/jax-releases/jax_releases.html

# jax based ML libraries

# optax
# most jax based ml libraries depend optimizers on optax
RUN pip3 install --user optax

# flax
RUN pip3 install --user flax

# haiku
#RUN pip3 install --user git+https://github.com/deepmind/dm-haiku
# OR
#RUN pip3 install --user dm-haiku

# pix
#RUN pip3 install dm-pix

# other modules
COPY ./docker/jax/requirements.txt requirements.txt
RUN pip3 install --user -r requirements.txt
