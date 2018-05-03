.PHONY: build_vagrant build_docker

# run inside the vagrant VM
build_docker: Dockerfile
	cp ${HOME}/transfer-cleaning.zip transfer-cleaning.zip
	docker build -t cleaning .

# run on host Rhino
build_vagrant: Vagrantfile
	(cd ../; git archive --format=zip -o runner/transfer-cleaning.zip master)
	(cd ../; zip -r runner/transfer_cleaning.zip runner/program_data)
	vagrant up
