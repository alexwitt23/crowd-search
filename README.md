# Adversarial Crowd Search


# Setup

There are two prodived ways to run the code, locally or in a Docker container.


#### Locally

Optionally create and activate a python virtual environment:

```bash
python3 -m venv ~/.envs/crowd-search
source ~/.envs/crowd-search/bin/activate
```

Install the python packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install Cython==0.29.21
python3 -m pip install -r requirements.txt
pushd third_party/python_rvo2
python3 setup.py build && python3 setup.py install
popd
```

#### Docker

The docker container assumes you have a CUDA compatible GPU device. Build
it like so:

```bash
    make image
```

You can run the docker container with the command below. Note, this will
mount the repo into `/home/code` within in the container so you can develop inside
the container. This is where you will first load into when running.

```bash
make run
```

## Training


```bash
    PYTHONPATH=. crowd_search/train_distributed.py 
    tensorboard --logdir=~/runs/crowd-search
```


## Testing
```bash
PY_IGNORE_IMPORTMISMATCH=1 pytest -s --doctest-modules --ignore="third_party"
```

## TODO

- [ ] Make one central module / functions for rewards. Right now rewards are split
between simulation environments and policies.

- [ ] Visualizations of simulations

- [ ] Look to use socialforce human model

- [ ] Look to use ideas from MuZero's MCTS
