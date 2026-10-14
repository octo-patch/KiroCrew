"""The shared type-strict comparator over decoded JSON documents."""

from __future__ import annotations

import sys

import pytest

from kiro_crew.user_json import strict_json_equal


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1, True),
        (0, False),
        (1, 1.0),
        ({"a": 1}, {"a": True}),
        ([1, 2], [1, 2.0]),
        ({"a": 1}, {"a": 1, "b": 2}),
        ([1], [1, 1]),
        ("1", 1),
    ],
)
def test_values_python_calls_equal_are_two_documents(a: object, b: object) -> None:
    assert not strict_json_equal(a, b)
    assert not strict_json_equal(b, a)


@pytest.mark.parametrize(
    "value",
    [
        {"name": "x", "tools": ["read"], "flag": True, "n": 2, "f": 2.5, "none": None},
        [{"a": [1, 2, {"b": False}]}],
        "text",
    ],
)
def test_a_document_equals_an_independent_copy_of_itself(value: object) -> None:
    import copy

    assert strict_json_equal(value, copy.deepcopy(value))


def test_key_order_is_not_content() -> None:
    assert strict_json_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})


def _nested(depth: int, leaf: object) -> object:
    value: object = leaf
    for _ in range(depth):
        value = {"m": [value]}
    return value


@pytest.mark.parametrize("leaf_a, leaf_b, expected", [(1, 1, True), (1, True, False)])
def test_a_document_nested_past_the_interpreter_limit_is_compared_not_raised(
    leaf_a: object, leaf_b: object, expected: bool
) -> None:
    """A walk that costs one interpreter frame per level raises
    ``RecursionError`` on a document a few hundred levels deep; the comparison
    answers instead, down to a leaf that differs only in type."""
    depth = 5000
    assert depth > sys.getrecursionlimit()
    assert strict_json_equal(_nested(depth, leaf_a), _nested(depth, leaf_b)) is expected
