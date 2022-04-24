FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get --no-install-recommends install -y \
    wget \
    python3.8 python3.8-dev python3.8-distutils && \
    rm -rf /var/lib/apt /var/cache/apt && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN update-alternatives --install /usr/bin/python python `which python3.8` 1 && \
    wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    rm get-pip.py

RUN pip install -U --no-cache-dir pip && \
    pip install --no-cache-dir -f https://download.pytorch.org/whl/cu110/torch_stable.html \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    opencv-contrib-python-headless==4.5.5.62

#RUN pip install -U --no-cache-dir pip && \
    #pip install --no-cache-dir -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    #torch==1.10.2+cu113 \
    #torchvision==0.11.3+cu113 \
    #opencv-contrib-python-headless==4.5.5.62

ENV PATH=/usr/local/bin:$PATH

WORKDIR /app

COPY requirements-docker.txt requirements-docker.txt

RUN pip install -r requirements-docker.txt