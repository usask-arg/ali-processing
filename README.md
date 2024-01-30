# python-library-template
A template repository for pure Python modules developed at the University of Saskatchewan
Atmospheric Research Group (usask-arg) that are to be publicly developed on GitHub.

To create a project based on this template, press the green use this template button on the top right.
The template can either be used for personal projects under your own namespace, or projects
in the `usask-arg` namespace.  If the project is created under your namespace then the deployment
of the package to the `usask-arg` anaconda channels will not work, and you should delete the following files:

 - `.github/workflows/nightly.yml`
 - `.github/workflows/release.yml`


## Check list when creating a new repository

### Updating Files
After using this template, there are a few files which need to be updated.

`pyproject.toml`
 - Read through and update all fields in the `[project]` section

`conda.recipe/meta.yaml`
 - Update the `package:` section with the package name
 - Update the `tests: imports:` section with the package name
 - Look through the `about:` section and update accordingly

`src/usask_arg_example/__init__.py`
 - Remove the example function

`src/usask_arg_example`
 - Rename the `usask_arg_example` folder to the name of your project

`tests/test_example.py`
 - Delete this file, or rename and use for your own tests

`docs/source/conf.py`
 - Update lines 9 through 12 accordingly

`README.md`
 - Delete all contents above `# PROJECTNAMEHERE` and read through the remainder, updating accordingly

`LICENSE`
 - The default license is the MIT license, change this if you want a different license.  Also make sure to update the License section in the README

### Code releases
By default, every night at midnight the project is packaged and then uploaded to the `usask-arg-nightly` Anaconda page (https://anaconda.org/usask-arg-nightly),
in addition, every tagged release is uploaded to the `usask-arg` Anaconda page (https://anaconda.org/usask-arg).

If you want the package to be either uploaded to `pypi` or built on `conda-forge`, speak to @dannyzed

If you want to disable the uploading of packages to the `usask-arg` channels, delete the files

 - `.github/workflows/nightly.yml`
 - `.github/workflows/release.yml`

### Automatic Documentation Building
If you want your project documentation to be automatically built and hosted on readthedocs.org, ask @dannyzed.
If you do not want your documentation to be hosted on readthedocs.org you can delete the `.readthedocs.yaml` file.

## Adding dependencies
If you need to add a dependency to the project, it must be added in the following places

- `env.yml`
- `pypoject.toml`, the "dependencies" section
- `conda.recipe/meta.yaml`, the "requirements/run" section

## Updating Python versions
The template is configured to test the code on the latest three available Python versions.  These are defined in

- `pyproject.toml`
- `.github/workflows/test.yml`

## Releasing a New Package Version
To release an official version of the package, click on Releases on the home page of your project, then draft a new release.
Here, click on "Choose a tag" and name your tag `vX.Y.Z` where `X.Y.Z` is the version number.  For `usask-arg` projects we
recommend using Calendar Versioning (https://calver.org) which has the form `YYYY.MM.MICRO` where

- `YYYY` is the current year, e.g. 2024
- `MM` is the current month as a two element string, e.g. 08
- `MICRO` starts at 0 and increments by one each release that month.  The first release of 2024-08 would be `2024.08.0` and the second release would be `2024.08.1`

You can then write some release notes and make the release.

# PROJECTNAMEHERE

[![Anaconda-Server Badge](https://anaconda.org/usask-arg/usask_arg_example/badges/version.svg)](https://anaconda.org/usask-arg/usask_arg_example)
[![Documentation Status](https://readthedocs.org/projects/usask_arg_example/badge/?version=latest)](https://usask_arg_example.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/usask-arg/usask_arg_example/main.svg)](https://results.pre-commit.ci/latest/github/usask-arg/usask_arg_example/main)

Short description of project here. MAKE SURE TO UPDATE THE 6 instances of usask_arg_example in the urls above to the name of your project.

## Installation
The package can be installed through `conda` with

`conda install -c usask-arg PACKAGENAME`

and the latest nightly available version is available through

`conda install -c usask-arg-nightly PACKAGENAME`

## Usage
Documentation can be found at  https://PROJECTNAMEHERE.readthedocs.io/

## License
This project is licensed under the MIT license
