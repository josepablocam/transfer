FROM ubuntu:18.04

# Prerequisites
RUN apt update
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y wget
RUN apt-get install -y gnupg
RUN apt-get install -y zip
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add -
RUN echo 'deb https://debian.neo4j.com stable latest' | tee -a /etc/apt/sources.list.d/neo4j.list
RUN apt-get update
RUN printf '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d
RUN apt-get install -y neo4j
RUN apt-get install -y python3.7
RUN apt-get install -y python3-pip
RUN apt-get install 2to3
RUN ln -s $(which python3.7) /usr/bin/python
RUN ln -s $(which python3.7) /usr/bin/python3 --force
RUN ln -s $(which pip3) /usr/bin/pip
RUN pip install -U pip

ADD . transfer/
WORKDIR transfer/
RUN bash reproduce.sh
