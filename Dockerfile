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
  python3-opencv \
  pkg-config libopencv-dev libturbojpeg0-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 &&\
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 &&\
  wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
  python /tmp/get-pip.py

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY . /app
WORKDIR /app
