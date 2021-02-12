# Adversarial Crowd Search


# Setup

docker build -t crowd-search:latest  .

docker run --ipc=host --gpus all -ti --rm -v $PWD:/home/relational_graph_learning -u $(id -u):$(id -g) crowd-search:latest /bin/bash

PYTHONPATH=. crowd_search/train_distributed.py 