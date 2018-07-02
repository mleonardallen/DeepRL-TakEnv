FROM tensorflow/tensorflow:latest-py3
# Dockerfile author / maintainer 
MAINTAINER Mike Allen <mikeleonardallen@gmail>

RUN pip --no-cache-dir install \
        gym \
        networkx==2.1 \
        keras

ADD . /workspace
WORKDIR "/workspace" 
