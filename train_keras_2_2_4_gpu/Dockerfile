FROM nvidia/cuda:9.0-base-ubuntu16.04

RUN apt-get update

RUN apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
