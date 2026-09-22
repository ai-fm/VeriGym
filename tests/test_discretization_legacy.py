"""Comparisons of `verigym.abstraction.discretization` against the old stack.

The pre-refactor index arithmetic is kept frozen in `tests/legacy_reference.py`
purely so these comparisons survive the deletion of the modules it came from.
Every test here imports from it; nothing else in the test suite should.

The plain tests for the current implementation live in
`tests/test_discretization.py` and must stay free of legacy references.
"""

import numpy as np
import pytest
from gymnasium.spaces import Box

from legacy_reference import legacy_factored_to_index, legacy_index_to_factored
from verigym.abstraction.discretization import generate_box_bins


def test_equivalence_vs_old_stack_forward_and_backward():
    """The new codecs give exactly the same results as the old hand-written arithmetic,
    forwards (random samples) and backwards (every enum value)."""
    space = Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), seed=42)
    n_bins = np.array([4, 3, 5])

    be = generate_box_bins(space, np.linspace, n_bins)

    n_forward = 300
    forward_mismatches = 0
    for _ in range(n_forward):
        x = space.sample()
        if be.orig_to_enum(x) != legacy_factored_to_index(x, be):
            forward_mismatches += 1
    assert forward_mismatches == 0, f"{forward_mismatches}/{n_forward} forward mismatches"

    total = int(np.prod(be.lengths))
    assert total == 4 * 3 * 5  # == 60
    backward_mismatches = 0
    for e in range(total):
        v_new = be.enum_to_value(e)
        v_old = legacy_index_to_factored(e, be)
        if not np.allclose(v_new, v_old):
            backward_mismatches += 1
    assert backward_mismatches == 0, f"{backward_mismatches}/{total} backward mismatches"


@pytest.mark.parametrize("shape", [(2, 2), (2, 3, 4)])
def test_legacy_arithmetic_crashes_for_ndim_gt_1(shape):
    """Documents the bug this refactor fixed: the old arithmetic reversed only axis 0,
    so it raised `TypeError` for any space with more than one dimension. The new
    codecs handle these shapes (see `test_ndim_gt_1` in `test_discretization.py`)."""
    space = Box(low=-1.0, high=1.0, shape=shape, seed=7)
    be = generate_box_bins(space, np.linspace, 4)

    with pytest.raises(TypeError):
        legacy_factored_to_index(space.sample(), be)
