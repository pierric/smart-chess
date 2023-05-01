FROM ubuntu:focal

LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN echo "Europe/Berlin" > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libopenblas-dev \
  python3.8 \
  python3-distutils \
  python3.8-dev \
  build-essential \
  wget \
  vim \
  git \
  python3-opencv

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 &&\
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 &&\
    wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python /tmp/get-pip.py

RUN pip install --no-cache-dir \
  torch==1.10.1+cu113 \
  torchvision==0.11.2+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install -y pkg-config libopencv-dev libturbojpeg0-dev
RUN pip install ffcv==1.0.2 python-chess==1.999 tqdm-loggable==0.1.3
RUN pip install accelerate==0.18.0 git+https://github.com/huggingface/pytorch-image-models.git@5e64777804154b510ab41f0ace5ae30551bc991b

COPY . /app
WORKDIR /app
