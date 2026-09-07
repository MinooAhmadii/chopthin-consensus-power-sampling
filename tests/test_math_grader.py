"""Regression and safety tests for the symbolic math grader."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader_utils.math_grader import _sympy_parse, grade_answer  # noqa: E402

EQUAL = [
    ("\\frac{1}{2}", "0.5"), ("0.5", "\\frac{1}{2}"), ("\\dfrac{3}{4}", "3/4"), ("2\\sqrt{2}", "2\\sqrt2"),
    ("\\boxed{7}", "7"), ("x^2+1", "1+x^2"), ("(1,2)", "(1, 2)"), ("10\\%", "10"), ("\\$5", "5"),
    ("\\pi", "pi"), ("1,000", "1000"),
]
DIFFERENT = [("1/2", "1/3"), ("7", "8"), ("x+1", "x-1"), ("(1,2)", "(2,1)"), ("", "3"), ("3.5", "3")]
INJECTIONS = ["__import__('os').system('true')", "().__class__.__bases__[0]", "1; import os", 'eval("1")',
              "open('/etc/passwd')", "lambda: 1", "1 # comment", "`x`"]


@pytest.mark.parametrize("given,gold", EQUAL)
def test_equal(given, gold):
    assert grade_answer(given, gold)


@pytest.mark.parametrize("given,gold", DIFFERENT)
def test_different(given, gold):
    assert not grade_answer(given, gold)


@pytest.mark.parametrize("bad", INJECTIONS)
def test_refuses_non_math(bad):
    with pytest.raises(ValueError):
        _sympy_parse(bad)
    assert not grade_answer(bad, "0")      # graded as wrong, nothing executed
