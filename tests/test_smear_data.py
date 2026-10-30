"""MathUtils.smear_data: an untruncated Gaussian per value, centred on its grid point."""

import numpy as np

from spinlabs.utils import MathUtils


def _gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def test_smear_data_keeps_the_tail_beyond_four_sigma():
    """A mode at 40 meV with sigma 5 meV still has weight near omega = 0 (7.8 sigma away)."""
    res, sigma = 0.02, 5.0
    x, y = MathUtils.smear_data(np.array([40.0]), np.array([3.0]), res, sigma)
    for omega in (1.0, 10.0, 39.0, 40.0, 55.0):
        i = int(round(omega / res))
        assert np.isclose(y[i], 3.0 * _gaussian(x[i], 40.0, sigma), rtol=1e-9)
    assert y[int(round(1.0 / res))] > 0


def test_smear_data_conserves_weight_and_centres_each_mode():
    """The smeared values integrate to the summed input and peak at the modes."""
    res, sigma = 0.1, 2.0
    freqs, values = np.array([30.0, 61.0]), np.array([1.5, 0.5])
    x, y = MathUtils.smear_data(freqs, values, res, sigma)
    assert np.isclose(np.sum(y) * res, values.sum(), rtol=1e-6)
    assert np.isclose(x[np.argmax(y)], 30.0)
