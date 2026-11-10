# ALI data product schemas

One YAML file per product. Each file is the reviewable definition of that product;
validation code and the documentation pages are generated from it.

| File          | Product                                                       |
| ------------- | ------------------------------------------------------------- |
| `l1a.yaml`    | L1A raw detector frames with spacecraft position and attitude |
| `l1b.yaml`    | L1B calibrated radiance and polarization on an altitude grid  |
| `cal_db.yaml` | Calibration database: instrument characterization             |

These are **drafts** (`status: draft`) of the target formats. The simple
instrument model follows them: `aliprocessing.forward_model` writes L1A,
`aliprocessing.cal_db` builds a placeholder calibration database, and
`aliprocessing.processing.l1a_to_l1b` reads both and writes L1B, checking each
against its schema. The simulator package has not been changed to match yet.

## Instrument summary

The schemas describe a simple model of ALI. One detector frame holds three
sub-images of the same scene, side by side with gaps between them. Each sub-image
is seen through a linear polarizer (analyzer) at a different angle, nominally
−60°, 0° and 60°. A filter wheel selects the spectral channel; one frame is taken
at one wheel position, and the instrument steps through the channels in turn. The
wheel also has a blank position, where no light reaches the detector. An **observation set** is
one frame at each spectral channel.

## Conventions

- **Granules.** L1A and L1B files are granules: a collection of consecutive frames
  (L1A) or observation sets (L1B), for example a quarter orbit.
- **Names** are lowercase snake_case with no unit suffixes. Units go only in the
  `units` attribute.
- **Units** follow the CF conventions and must be parsable by UDUNITS-2. Counts of
  electrons have units `"1"`, and the description says they are electrons.
- **Geolocation** is WGS84 throughout. Tangent points are geometric (unrefracted).
- **Time** is the midpoint of the exposure.
- **Detector axes.** `detector_row` and `detector_column` index the full frame,
  including the gaps between sub-images. Which pixels belong to which sub-image is
  given by `pixel_analyzer` in the calibration database.

## Schema vocabulary

Top-level keys:

- `product`, `schema_version`, `status`, `description`
- `dimensions`: each dimension name with a `description`. `coordinate: true` means
  a variable of the same name holds the coordinate values.
- `variables`: see below.
- `attributes`: global netCDF attributes, using the same keys as variables where
  they apply (`dtype`, `units`, `required`, `allowed`, `description`).
- `model`: the physical model the parameters feed, in plain text with equations.
- `groups`: netCDF groups, each with its own `description` and `variables`.
- `file`: on-disk layout (granule definition, filename pattern).
- `constraints`: relationships between fields that a per-field check cannot express.
  Each has an `id` and a plain-language `rule`; the validator implements them by `id`.
- `open_questions`: unresolved decisions, each with an `id`, a `question`, and
  `context`. Fields reference them with `open_questions: [ID, ...]`.

Variable keys:

| Key             | Meaning                                                              |
| --------------- | -------------------------------------------------------------------- |
| `dims`          | Ordered dimension names. `[]` is a scalar.                           |
| `dtype`         | `float`, `int`, `bool`, `datetime`, or `str`.                        |
| `units`         | CF / UDUNITS-2 unit string; `"1"` is dimensionless.                  |
| `required`      | `true` or `false`.                                                   |
| `required_when` | Required when the listed variables have the listed values.           |
| `default`       | Value assumed when an optional field is absent.                      |
| `valid_min`/`valid_max` | Inclusive bounds on every element.                           |
| `allowed`       | Enumerated allowed values.                                           |
| `role`          | What kind of field this is (below).                                  |
| `description`   | One or two sentences; rendered into the docs.                        |
| `open_questions`| Ids of open questions that affect this field.                        |

Roles:

- `measurement`: the observed signal and its uncertainty.
- `geometry`: viewing and solar geometry, time, and location.
- `acquisition`: settings that can change from one frame to the next.
- `instrument`: properties of the instrument rather than of one acquisition.
- `quality`: flags and diagnostics about the measurement.
- `processing`: records how the product was made.
- `simulator_truth`: only a simulator can know this; it cannot exist for flight data.
