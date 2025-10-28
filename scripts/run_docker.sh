#!/usr/bin/env sh
#
# Injects useful arguments for running mjlab in docker. See docs/installation_guide.md for usage.
# This is patterned after the uv-in-docker example: https://github.com/astral-sh/uv-docker-example/blob/5748835918ec293d547bbe0e42df34e140aca1eb/run.sh
#
# docker run \
#     --rm \                        Remove the container after exiting
#     --runtime=nvidia \            Use NVIDIA Container runtime to give GPU access
#     --gpus all \                  Expose all GPUs by default
#     --volume .:/app \             (follows uv example) Mount the current directory to `/app` so code changes don't require an image rebuild
#     --volume /app/.venv \         (follows uv example) Mount the virtual environment separately, so the developer's environment doesn't end up in the container
#     --publish 8080:8080 \         publish port 8080 for viewing the mjlab web interface on the host
#     $INTERACTIVE \                if in a terminal, launch in interactive mode. Note that if running training,
#                                   there is a blocking wandb prompt before training will begin.
#     $(docker build -t mjlab) \    build and launch the docker image (tag matches the one in the Makefile)
#     "$@"                          forward all arguments


if [ -t 1 ]; then
    INTERACTIVE="-it"
else
    INTERACTIVE=""
fi

docker run \
    --rm \
    --runtime=nvidia \
    --gpus all \
    --volume .:/app \
    --volume /app/.venv \
    --publish 8080:8080 \
    $INTERACTIVE \
    $(docker build -qt mjlab .) \
    "$@"
