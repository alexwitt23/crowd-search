current_dir := $(shell pwd)

.PHONY: all
all: image

.PHONY: image
image:
	docker build -t crowd-search -f Dockerfile $(DOCKERFLAGS) .

.PHONY: run
run:
	docker run --ipc=host --gpus all -ti --rm -v $(PWD):/home/code -u $(id -u):$(id -g) \
    	crowd-search:latest /bin/bash

.PHONY: test
test:
	docker build -t crowd-search-test:test -f Dockerfile $(DOCKERFLAGS) .
	docker run --ipc=host --gpus all --rm -v $(PWD):/home/code -u $(id -u):$(id -g) \
		crowd-search:latest /bin/bash pytest -s --doctest-modules --ignore="third_party"

.PHONY: clean
clean:
	docker image -rmi crowd-search-test:test