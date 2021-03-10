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

Review the training config in `crowd_search/config.yaml`. When ready to train, run:
```bash
PYTHONPATH=. crowd_search/train_distributed.py \
    --config_path crowd_search/config_2gpu.yaml
```

To then visualize the results of your runs, use TensorBoard:

```bash
tensorboard --logdir=~/runs/crowd-search
```

Each training run is timestamped and saved to `~/runs/crowd-search`.

To stop training, run `Ctrl + C`. Note, due to the distributed nature of the training
script, some nodes might hang. You might have to look for the running PIDs
(using `htop` for example) and `kill -9` the process.


## Testing
```bash
PY_IGNORE_IMPORTMISMATCH=1 pytest -s --doctest-modules --ignore="third_party"
```

## TODO

- [ ] Make todo list
