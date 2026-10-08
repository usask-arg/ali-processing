# ali-processing

[![Documentation Status](https://readthedocs.org/projects/ali-processing/badge/?version=latest)](https://ali-processing.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/usask-arg/ali-processing/main.svg)](https://results.pre-commit.ci/latest/github/usask-arg/ali-processing/main)

Research and development libraries developed at the University of Saskatchewan for the ALI instrument

## Installation
`pip install aliprocessing`

## Usage
Documentation can be found at  https://ali-processing.readthedocs.io/

## Development
The development environment is managed with [uv](https://docs.astral.sh/uv/)

```bash
uv sync                  # create the environment with the development dependencies
uv run pytest            # run the tests
uv run pre-commit run -a # run the linters and formatters
uv run --group docs sphinx-build -b html docs/source docs/build  # build the documentation
```

## License
This project is licensed under the MIT license
