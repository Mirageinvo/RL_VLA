#!/bin/bash
set -e

CONTAINER_NAME=rlvla_${USER}

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container is not running. Start it first:"
    echo "bash start.sh"
    exit 1
fi

docker exec -it ${CONTAINER_NAME} bash
