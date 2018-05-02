.PHONY: all build_vagrant build_docker

all: build_docker

build_docker: Dockerfile
	(cd ..; git archive --format=zip -o runner/transfer-cleaning.zip master)
	docker build -t cleaning .
