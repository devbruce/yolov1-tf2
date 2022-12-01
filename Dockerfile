FROM tensorflow/tensorflow:2.5.0-gpu


LABEL maintainer "devbruce"
LABEL email "bruce93k@gmail.com"
LABEL title "yolov1-tf2"

ARG DEBCONF_NOWARNINGS="yes"
ARG DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL "C.UTF-8"
ENV PIP_NO_CACHE_DIR "1"
ENV PYTHONPATH "/yolov1-tf2:${PYTHONPATH}"


RUN apt-get update
RUN apt-get install -y vim

# Requirement of opencv-python
RUN apt-get install -y libgl1-mesa-glx

# Install python packages
RUN pip install --upgrade pip
RUN pip install opencv-python==4.5.2.52
RUN pip install matplotlib==3.3.4
RUN pip install tensorflow-datasets==4.3.0
RUN pip install albumentations==1.0.0
RUN pip install jupyterlab==3.0.16
RUN pip install calc4ap==1.0.1
COPY . /yolov1-tf2
WORKDIR /yolov1-tf2
