FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update && \
    apt-get -y install gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /setup

COPY requirements.txt .
RUN python3 -m pip install pip Cython==0.29.21
RUN python3 -m pip install --ignore-installed -r requirements.txt

# Setup the Python-RVO2 project
COPY third_party/python_rvo2 . 
RUN python3 setup.py build \
  && python3 setup.py install

WORKDIR /home/code

RUN rm -r /setup