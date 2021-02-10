current_dir := $(shell pwd)

.PHONY: all
all: image

.PHONY: image
image:
	docker build -t alexwitt/test -f Dockerfile $(DOCKERFLAGS) .

.PHONY: run
run:
	docker run -it --ipc=host --gpus all \
	-v $(current_dir):/home/relational_graph_learning \
	alexwitt/test