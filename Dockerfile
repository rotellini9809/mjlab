FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    libegl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mjlab
COPY Makefile ./
COPY pyproject.toml ./
COPY README.md ./
COPY .python-version ./
COPY src ./src
COPY tests ./tests

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv sync --group dev

ENV MUJOCO_GL=egl
EXPOSE 8080

CMD ["uv", "run", "python", "tests/smoke_test.py"]
