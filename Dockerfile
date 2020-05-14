FROM ubuntu:latest

# baseline setup

RUN apt-get update

RUN apt-get install -y \
  build-essential \
  python3 \
  python3-dev \
  python3-pip

RUN pip3 install --upgrade pip

# custom start

ENV HOME /mltoys
WORKDIR $HOME

# Python modules

ADD requirements.txt $HOME
RUN pip3 install -r $HOME/requirements.txt

# module installation

ADD mltoys/ $HOME/mltoys
ADD setup.py $HOME
RUN python3 setup.py install

# tests

ADD tests/*.py tests/*.sh tests/

# examples

ADD examples/*.py examples/

# compile all python files as sanity check
RUN python3 -m compileall */*.py
