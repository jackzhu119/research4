#!/usr/bin/env python3
"""Analytical checks of the stability definition and coordinate/rank rules."""
import itertools
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

from e2_utils import alpha_mask, pair_summary, snv, spaced
from e2_stability import paired_draws


def main():
    identical = np.array([[1, 0, 1, 0]] * 3)
    value, _ = pair_summary(identical, 4)
    assert value["Nogueira_Phi"] == 1.
    complement = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    value, _ = pair_summary(complement, 4)
    assert value["Nogueira_Phi"] == -1., value
    assert value["intersection"]["mean"] == 0.
    full = np.ones((2, 4))
    assert pair_summary(full, 4)[0]["Nogueira_Phi"] is None
    partial = np.array([[1, 1, 0, 0], [0, 1, 1, 0]])
    assert np.isclose(pair_summary(partial, 4)[0]["exact_Jaccard"]["mean"], 1 / 3)
    assert alpha_mask({"selected_indices": [1223, 612, 700]}).tolist() == [0, 88, 611]
    try:
        alpha_mask({"selected_indices": [0, 700]})
    except ValueError:
        pass
    else:
        raise AssertionError("n offset wrongly accepted in alpha-only mask")
    scores = np.array([3., 5., 2., 8., 4., 6., 1.])
    chosen = spaced(scores, 3, 2)
    feasible = [ix for ix in itertools.combinations(range(len(scores)), 3) if all(np.diff(ix) >= 2)]
    assert np.isclose(scores[chosen].sum(), max(scores[list(ix)].sum() for ix in feasible))
    y = np.repeat(np.arange(5), 48)
    first, second = paired_draws(y, 2026111600), paired_draws(y, 2026111600)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(first) == 20
    assert all(np.array_equal(np.bincount(y[ix], minlength=5), np.repeat(39, 5)) for ix in first)
    rng = np.random.default_rng(4)
    alpha = snv(rng.normal(size=(20, 612)))
    n = snv(rng.normal(size=(20, 612)))
    scaled_a = StandardScaler().fit_transform(alpha)
    scaled_b = StandardScaler().fit_transform(np.column_stack([n, alpha]))[:, 612:]
    assert np.allclose(scaled_a, scaled_b, rtol=1e-13, atol=1e-13)
    result = {"status": "PASS", "checks": ["identical masks Phi1", "negative Phi preserved", "degenerate Phi undefined",
                                                  "intersection and Jaccard", "global alpha coordinate offset", "exact spacing DP brute force",
                                                  "paired actual MC draw indices and class counts", "shared alpha-column training transform"],
              "new_classifier_fits": 0, "new_pls_coefficient_fits": 0}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
