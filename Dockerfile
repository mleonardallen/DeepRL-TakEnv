FROM tensorflow/tensorflow:latest-py3
# Dockerfile author / maintainer 
MAINTAINER Mike Allen <mikeleonardallen@gmail>

# Needed to forward pygame display
RUN apt-get update -y \
  && apt-get -y install \
    xvfb \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# TAK python dependencies
RUN pip --no-cache-dir install \
        gym \
        networkx==2.1 \
        keras \
        pygame

ADD . /workspace
WORKDIR "/workspace" 

CMD ["python", "main.py", "--render"]