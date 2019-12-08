FROM ubuntu:18.04

# prevent 404 in apt-get update 
# ref: https://stackoverflow.com/questions/37706635/in-docker-apt-get-install-fails-with-failed-to-fetch-http-archive-ubuntu-com#answer-37708927
RUN sed -i'' 's/archive\.ubuntu\.com/us\.archive\.ubuntu\.com/' /etc/apt/sources.list
RUN rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1 git
RUN apt-get install -y locales \
  && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8'
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt