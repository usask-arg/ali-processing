FROM quay.io/condaforge/mambaforge

ADD . /src/

RUN mamba env create -f /src/env.yml -n ali_processing

SHELL ["mamba", "run", "-n", "ali_processing", "/bin/bash", "-c"]

RUN pip install /src/
