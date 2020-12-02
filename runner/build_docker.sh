#!/usr/bin/env bash

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


docker build --network=host -t cleaning .
