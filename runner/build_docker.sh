#!/usr/bin/env bash
source ../setup.sh

if [[ $(hostname) == "boruca" ]]
then
  # building inside vagrant VM
  cp ${HOME}/wranglesearch.zip wranglesearch.zip
else
  pushd ../
  git archive --format=zip -o runner/wranglesearch.zip master
  zip -r runner/wranglesearch.zip runner/program_data
  popd
fi


# user --network=host otherwise some installs hang
docker build --network=host -t ${DOCKER_RUNNER_IMAGE} .
