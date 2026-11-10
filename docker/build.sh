#!/bin/bash
set -e
cd "$(dirname "$0")"
ARCH=$(uname -m)
docker build . -f Dockerfile.${ARCH} --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t ${ARCH}/rlvla:latest
