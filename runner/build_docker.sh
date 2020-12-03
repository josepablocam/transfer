#!/usr/bin/env bash
source ../setup.sh

if [[ $(hostname) == "boruca" ]]
then
  # building inside vagrant VM
  cp ${HOME}/transfer.zip transfer.zip
else
  pushd ../
  git archive --format=zip -o runner/transfer.zip master
  zip -r runner/transfer.zip runner/program_data
  popd
fi


# user --network=host otherwise some installs hang
docker build --network=host -t ${DOCKER_RUNNER_IMAGE} .
