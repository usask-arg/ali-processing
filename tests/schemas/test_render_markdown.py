from __future__ import annotations

import copy

import pytest

from aliprocessing.schemas import PRODUCTS, load_schema, render_markdown


def _row(markdown: str, name: str) -> str:
    """The table row for a field named ``name``."""
    return next(
        line for line in markdown.splitlines() if line.startswith(f"| `{name}` ")
    )


@pytest.mark.parametrize("product", PRODUCTS)
def test_title_and_version(product):
    schema = load_schema(product)
    markdown = render_markdown(schema)
    assert markdown.startswith(f"# {schema['product']}\n")
    assert f"Schema version {schema['schema_version']}" in markdown


def test_variable_row():
    row = _row(render_markdown(load_schema("l1a")), "image")
    assert "time, detector_row, detector_column" in row
    assert "`count`" in row
    assert "| yes |" in row
    assert "Raw detector signal" in row


def test_scalar_range_and_default():
    markdown = render_markdown(load_schema("cal_db"))
    assert "scalar" in _row(markdown, "detector_gain")
    assert "[-1, 1]" in _row(markdown, "q_sensitivity")
    assert "no (default 0)" in _row(markdown, "dark_current")
    assert "≥ -1" in _row(markdown, "pixel_analyzer")


def test_allowed_values():
    row = _row(render_markdown(load_schema("l1b")), "radiance_noise_model")
    assert "none, detector" in row


def test_conditional_requirement():
    schema = copy.deepcopy(load_schema("l1b"))
    schema["attributes"]["detector_noise_source"] = {
        "dtype": "str",
        "required_when": {"radiance_noise_model": "detector"},
        "role": "processing",
        "description": "Test attribute.",
    }
    row = _row(render_markdown(schema), "detector_noise_source")
    assert "when radiance_noise_model = detector" in row


def test_groups_are_rendered():
    markdown = render_markdown(load_schema("l1a"))
    assert "## Group `simulation`" in markdown
    assert "time, detector_row, detector_column" in _row(markdown, "image_noiseless")


def test_model_is_rendered_as_text_blocks():
    markdown = render_markdown(load_schema("cal_db"))
    assert "## Model" in markdown
    assert "### Forward" in markdown
    assert "### Inverse" in markdown
    assert "```text\n" in markdown
    assert "Φ_w(p)" in markdown


def test_constraints_are_listed():
    markdown = render_markdown(load_schema("cal_db"))
    assert "- **`demodulable`**:" in markdown


def test_open_questions_are_listed_only_when_present():
    assert "## Open questions" not in render_markdown(load_schema("l1b"))
    markdown = render_markdown(load_schema("cal_db"))
    assert "## Open questions" in markdown
    assert "- **dark-model**:" in markdown


def test_pipes_in_text_do_not_break_tables():
    schema = copy.deepcopy(load_schema("l1b"))
    schema["variables"]["radiance"]["description"] = "uses |x| here"
    assert "uses \\|x\\| here" in _row(render_markdown(schema), "radiance")
