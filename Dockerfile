FROM ubuntu:18.04

# Basic prereqs
RUN apt update
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y wget
RUN apt-get install -y gnupg
RUN apt-get install -y zip
RUN apt-get install -y vim

# install neo4j 3.3.4
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add -
RUN echo 'deb https://debian.neo4j.com stable legacy' | tee /etc/apt/sources.list.d/neo4j.list
RUN apt-get update
RUN printf '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d
RUN apt-get install -y neo4j=1:3.3.4
RUN echo "dbms.security.auth_enabled=false" >> /etc/neo4j/neo4j.conf

# install/setup python
RUN apt-get install -y python3.7
RUN apt-get install -y python3-pip
RUN apt-get install 2to3
RUN ln -s $(which python3.7) /usr/bin/python
RUN ln -s $(which python3.7) /usr/bin/python3 --force
RUN ln -s $(which pip3) /usr/bin/pip
RUN pip install -U pip

ADD . transfer/
WORKDIR transfer/
# install transfer, run tests and
# build up sample database
RUN bash reproduce.sh

ENTRYPOINT neo4j start && /bin/bash
