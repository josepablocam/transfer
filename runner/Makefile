.PHONY: all build_vagrant build_docker

all: build_vagrant build_docker

build_vagrant: Vagrantfile
	longjob vagrant up

build_docker: Dockerfile
	docker build -t cleaning .
