#!/bin/bash
set -e

ARCH=$(uname -m)
CONTAINER_NAME=rlvla_${USER}
IMAGE=${ARCH}/rlvla:latest
REPO=$(cd "$(dirname "$0")/.." && pwd)

# Домашний каталог монтируется ПО ТОМУ ЖЕ ПУТИ, что снаружи. Тогда любой
# абсолютный путь работает внутри и снаружи одинаково, соседние проекты видны,
# и переводить пути в голове не нужно. Образ собран с вашими UID/GID
# (build.sh передаёт --build-arg UID=$(id -u)), поэтому права совпадают.
#
# Учтите: внутрь попадает ВЕСЬ home, включая ~/.ssh и прочие ключи. Если это
# нежелательно, замените "${HOME}":"${HOME}" на каталог с проектами, например
# "${HOME}/code":"${HOME}/code".
HOME_MOUNT=(-v "${HOME}":"${HOME}")

# Данные вне home (на кластерах их часто держат на отдельном разделе из-за
# квоты). Если каталог лежит внутри home, он и так уже виден — переменная
# нужна только для внешних разделов.
DATA_MOUNT=()
if [ -n "${RLVLA_DATA}" ]; then
    if [ ! -d "${RLVLA_DATA}" ]; then
        echo "ОШИБКА: RLVLA_DATA=${RLVLA_DATA} — такого каталога нет" >&2
        exit 1
    fi
    DATA_MOUNT=(-v "$(readlink -f "${RLVLA_DATA}")":"${REPO}/data")
    echo "data: ${RLVLA_DATA} -> ${REPO}/data"
fi

# Монтирования задаются ТОЛЬКО при создании контейнера. У существующего они
# прежние — чтобы сменить, нужно docker rm -f и пересоздать.
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "  (монтирования прежние; чтобы сменить — docker rm -f ${CONTAINER_NAME})"
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Starting existing container ${CONTAINER_NAME}"
    echo "  (монтирования прежние; чтобы сменить — docker rm -f ${CONTAINER_NAME})"
    docker start ${CONTAINER_NAME}
else
    echo "Creating new container ${CONTAINER_NAME}"
    echo "home: ${HOME} -> ${HOME}"
    echo "cwd:  ${REPO}"
    docker run -it -d \
        --name ${CONTAINER_NAME} \
        --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -e MUJOCO_GL=egl \
        --ipc host \
        "${HOME_MOUNT[@]}" \
        "${DATA_MOUNT[@]}" \
        -w "${REPO}" \
        ${IMAGE}
fi
