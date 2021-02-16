current_dir := $(shell pwd)

.PHONY: all
all: image

.PHONY: image
image:
	docker build -t crowd-search -f Dockerfile $(DOCKERFLAGS) .

.PHONY: run
run:
	docker run --ipc=host --gpus all -ti --rm -v $PWD:/home/code -u $(id -u):$(id -g) \
    	crowd-search:latest /bin/bash