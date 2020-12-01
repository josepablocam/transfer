.PHONY: build_vagrant build_docker

# run inside the vagrant VM
build_docker: Dockerfile build_docker.sh
	bash build_docker.sh

# run on host Rhino
build_vagrant: Vagrantfile
	(cd ../; git archive --format=zip -o runner/transfer.zip master)
	(cd ../; zip -r runner/transfer.zip runner/program_data)
	# make sure that virtualbox doesn't store the VM image on AFS
	mkdir -p /raid/jcamsan/virtualbox_vms
	vboxmanage setproperty machinefolder /raid/jcamsan/virtualbox_vms
	vagrant up
