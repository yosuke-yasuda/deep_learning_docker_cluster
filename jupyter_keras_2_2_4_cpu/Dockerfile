FROM ubuntu:18.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt