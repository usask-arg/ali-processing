FROM ghcr.io/prefix-dev/pixi:0.40.0-mantic AS build

RUN pixi global install git

# copy source code
WORKDIR /app
COPY . .
# install dependencies to `/app/.pixi/envs/default`
RUN pixi install -e default
# create the shell-hook bash script to activate the environment
RUN pixi shell-hook -e default -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /app/entrypoint.sh

FROM ubuntu:24.04 AS production
WORKDIR /app
# only copy the production environment into prod container
# please note that the "prefix" (path) needs to stay the same as in the build container
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
# copy your project code into the container as well
COPY ./src /app/src
