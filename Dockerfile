# Refer to uv-docker-example:
# https://github.com/astral-sh/uv-docker-example/blob/main/standalone.Dockerfile
# Note that we use uv to launch, so we omit the second half of the example (non-UV final image)

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libegl-dev \
    mesa-utils\
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_PREFERENCE=only-managed


RUN uv python install 3.13

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --no-dev


ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev

ENV MUJOCO_GL=egl
ENV WANDB_ENTITY=fratelligpt-sapienza-universit-di-roma
ENV WANDB_API_KEY=2afebd919c3c3e82e91783cae9df8ee5de971b6f

EXPOSE 8080

CMD ["sh", "-c", "uv run python tests/smoke_test.py && exec bash"]
