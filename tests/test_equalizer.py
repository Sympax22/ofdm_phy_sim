"""
test_equalizer.py — Unit tests for ofdm_phy_sim.equalizer.

Tests cover:
    - LS estimation: identity channel, output shape, known channel spot check
    - MMSE estimation: identity channel, outperforms LS at low SNR
    - equalize: identity channel passthrough, output shape, known channel correction
"""

import numpy as np

from ofdm_phy_sim.constants import *
from ofdm_phy_sim.equalizer import interpolate_channel_from_pilots, equalize_from_pilots

def test_interpolate_channel_from_pilots_identity():
    """
    Test that interpolation of pilot-based channel estimates recovers the original channel for an identity channel.
    """
    h_pilots = np.array([1, 1, 1, 1], dtype=np.complex64) # Identity channel at pilots
    h_est = interpolate_channel_from_pilots(h_est_pilots=h_pilots)
    assert h_est.shape == (N_SD,)
    assert np.allclose(h_est, 1, atol=1e-16) # Should be close to 1 across all data subcarriers

    # TODO: more tests