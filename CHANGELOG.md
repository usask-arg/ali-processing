# Changelog

Versions follow the release tags (`vYYYY.MM.N`).

## Unreleased

### Data formats

- Draft schemas for L1A, L1B, the calibration database and L2, in
  `aliprocessing.schemas`, with a validator, `apply_metadata`, and Markdown pages
  rendered into the documentation. All are `status: draft`, schema version 0.1.0.

### Instrument model and L1A → L1B

- A simple instrument forward model that writes L1A, and a placeholder
  calibration database.
- L1A to L1B processing for the simple instrument model.
- Default spectral channels are now 525, 745, 869, 1020 and 1560 nm.

### L1B → L2

- `process_l1b_to_l2` retrieves every observation set and across-track bin of an
  L1B granule and writes an L2 granule checked against the L2 schema.
- The retrieval is configured by `retrieval_config.yaml`; load it, with
  overrides, through `aliprocessing.l2.config.load_retrieval_config`.
- A two-stage retrieval with cloud detection, enabled by default.
- Extinction can be retrieved in log space.
- The solar spectrum of each channel comes from the calibration database's
  spectral response functions.
- De-weighting noisy DoLP (`max_dolp_noise`) is opt-in, off by default.
- The L1B reference longitude is averaged as angles, so it is continuous across
  the dateline.
- An image whose retrieval fails no longer stops the granule: a warning is
  issued and the profile is stored as a fill with the new `retrieval_failed`
  variable set and `optimizer_status` -2.
- The optimizer's progress output (`minimizer.verbose`) is off by default.
- `load_retrieval_config` rejects prior settings that skretrieval would ignore.
  The unused `prior.extinction_per_m: {type: testing}` and O3 `prior` entries are
  removed from the defaults; the extinction prior is set with `prior_state`.

### Packaging

- Requires numpy >= 2, xarray >= 2024.10, sasktran2 >= 2026.10.1 (earlier versions
  disagree with the Stokes sign convention test) and
  skretrieval >= 2026.11.0.
- The YAML schemas and the default retrieval configuration are declared as
  package data.
