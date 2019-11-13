FROM mongo:3.6.15-xenial

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN pip3 install --upgrade pip

RUN apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash 
RUN apt-get install -y nodejs
RUN npm install -g omniboard

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt