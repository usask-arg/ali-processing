FROM ghcr.io/astral-sh/uv:0.12-python3.13-trixie-slim AS build

# setuptools-scm needs git to determine the package version
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# copy source code
WORKDIR /app
COPY . .
# install the package and its runtime dependencies into `/app/.venv`
RUN uv sync --no-dev --no-editable --extra plotting

FROM python:3.13-slim-trixie AS production
WORKDIR /app
# only copy the environment into the production container
# please note that the path needs to stay the same as in the build container
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# kept for compatibility with commands that were run through the old entrypoint script
RUN printf '#!/bin/sh\nexec "$@"\n' > /app/entrypoint.sh && chmod 0755 /app/entrypoint.sh
