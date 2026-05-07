"""Tests for OMERO helper constructors."""

import pytest

from zarrnii import make_omero, make_omero_channels


def test_make_omero_defaults():
    """Helper should create channels with default colors and windows."""
    labels = ["DAPI", "GFP", "RFP", "BF"]
    omero = make_omero(labels)

    assert [ch.label for ch in omero.channels] == labels
    assert [ch.color for ch in omero.channels] == [
        "0000FF",
        "00FF00",
        "FF0000",
        "FFFF00",
    ]
    for channel in omero.channels:
        assert channel.window.min == 0.0
        assert channel.window.max == 1.0
        assert channel.window.start == 0.0
        assert channel.window.end == 1.0


def test_make_omero_accepts_plain_windows_and_colors():
    """Helper should accept plain color and window inputs."""
    omero = make_omero(
        ["A", "B"],
        channel_colors=["#abcdef", "123456"],
        channel_windows=[
            {"min": 0, "max": 4095, "start": 10, "end": 1000},
            (0, 65535, 100, 2000),
        ],
    )

    assert [ch.color for ch in omero.channels] == ["ABCDEF", "123456"]
    assert omero.channels[0].window.max == 4095.0
    assert omero.channels[1].window.end == 2000.0


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        (
            {"channel_labels": ["A", "B"], "channel_colors": ["FFFFFF"]},
            "channel_colors",
        ),
        ({"channel_labels": ["A", "B"], "channel_windows": [None]}, "channel_windows"),
    ],
)
def test_make_omero_validates_input_lengths(kwargs, expected_message):
    """Helper should validate list lengths against label count."""
    with pytest.raises(ValueError, match=expected_message):
        make_omero(**kwargs)


def test_make_omero_channels_alias():
    """Alias should return OMERO metadata like make_omero."""
    omero = make_omero_channels(["DAPI"])
    assert len(omero.channels) == 1
    assert omero.channels[0].label == "DAPI"


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"channel_labels": []}, "at least one"),
        ({"channel_labels": ["A"], "channel_colors": ["nothex"]}, "6-digit hex"),
        (
            {"channel_labels": ["A"], "channel_windows": [{"min": 0, "max": 1}]},
            "min/max/start/end",
        ),
    ],
)
def test_make_omero_validates_content(kwargs, expected_message):
    """Helper should validate labels, colors, and window payload content."""
    with pytest.raises(ValueError, match=expected_message):
        make_omero(**kwargs)
