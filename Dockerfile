FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update && \
    apt-get -y install gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /setup

COPY requirements.txt /setup
RUN python3 -m pip install Cython==0.29.21
RUN python3 -m pip install -r requirements.txt

# Setup the Python-RVO2 project
RUN git clone https://github.com/sybrenstuvel/Python-RVO2.git \
  && cd Python-RVO2 \
  && python3 setup.py build \
  && python3 setup.py install

# Setup socialforce project
RUN git clone https://github.com/ChanganVR/socialforce.git \
  && cd socialforce && python -m pip install -e '.[test,plot]'

COPY . /relational_graph_learning

WORKDIR /relational_graph_learning

RUN python3 -m pip install -e .

WORKDIR /home/relational_graph_learning