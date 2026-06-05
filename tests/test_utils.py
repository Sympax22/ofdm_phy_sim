"""
test_utils.py — Unit tests for ofdm_phy_sim.utils.

Tests cover:
    - random_bits: output length, value range, dtype, and
    both fixed and random-length modes.
"""

import numpy as np
import pytest

from ofdm_phy_sim.constants import *
from ofdm_phy_sim.utils import random_bits, ebno_to_noise_var, compute_ber, theoretical_ber_awgn


def test_random_bits_default_length_in_range():
    bits = random_bits()
    assert 64 <= len(bits) <= 4096

def test_random_bits_fixed_length():
    bits = random_bits(n_bits=128)
    assert len(bits) == 128

def test_random_bits_only_binary():
    bits = random_bits(n_bits=1000)
    assert set(np.unique(bits)).issubset({0, 1})

def test_random_bits_dtype():
    bits = random_bits(n_bits=64)
    assert bits.dtype in [np.int32, np.int64]

####################################################

def test_ebno_to_noise_var_bpsk_0db():
    # At 0dB, ebno_linear=1, bps=1 → N0=1
    assert np.isclose(ebno_to_noise_var(0.0, 1), 1.0)

def test_ebno_to_noise_var_decreases_with_snr():
    # Higher Eb/N0 → less noise
    assert ebno_to_noise_var(10.0, 1) < ebno_to_noise_var(0.0, 1)

def test_ebno_to_noise_var_scales_with_bps():
    # More bits/symbol → less noise per symbol at same Eb/N0
    assert ebno_to_noise_var(0.0, 4) < ebno_to_noise_var(0.0, 1)

def test_theoretical_ber_high_snr():
    # At very high SNR, BER → 0
    for mod in MODULATIONS:
        assert theoretical_ber_awgn(mod, np.array([30.0])) < 1e-6

def test_theoretical_ber_low_snr():
    # At very low SNR, BER should approach its theoretical maximum
    # BPSK/QPSK → 0.5, 16-QAM → 3/8
    expected_max = {'BPSK': 0.5, 'QPSK': 0.5, '16-QAM': 3/8}
    for mod in MODULATIONS:
        assert theoretical_ber_awgn(mod, np.array([-30.0])) > 0.9 * expected_max[mod]

def test_theoretical_ber_bpsk_equals_qpsk():
    for ebno in [-5.0, 0.0, 5.0, 10.0]:
        assert np.isclose(theoretical_ber_awgn('BPSK', np.array([ebno])),
                          theoretical_ber_awgn('QPSK', np.array([ebno])))

def test_theoretical_ber_ordering():
    # At same Eb/N0, higher order modulation → worse BER
    ebno = 5.0
    assert theoretical_ber_awgn('BPSK', np.array([ebno])) < theoretical_ber_awgn('16-QAM', np.array([ebno]))

def test_theoretical_ber_invalid_modulation():
    with pytest.raises(ValueError):
        theoretical_ber_awgn('64-QAM', np.array([10.0]))

def test_compute_ber_high_snr_near_zero():
    # At 30dB, almost no errors
    bits = np.random.randint(0, 2, 10000)
    for mod in MODULATIONS:
        assert compute_ber(mod, bits, ebno_db=30.0, seed=42) < 1e-3

def test_compute_ber_low_snr_near_half():
    # At -10dB, BER close to 0.5
    bits = np.random.randint(0, 2, 10000)
    for mod in MODULATIONS:
        assert compute_ber(mod, bits, ebno_db=-10.0, seed=42) > 0.3

def test_compute_ber_reproducible():
    bits = np.random.randint(0, 2, 1000)
    ber1 = compute_ber('BPSK', bits, ebno_db=5.0, seed=42)
    ber2 = compute_ber('BPSK', bits, ebno_db=5.0, seed=42)
    assert ber1 == ber2

def test_compute_ber_empty_bits():
    assert compute_ber('BPSK', np.array([]), ebno_db=5.0, seed=42) == 0.0

def test_compute_ber_decreases_with_snr():
    bits = np.random.randint(0, 2, 10000)
    ber_low  = compute_ber('QPSK', bits, ebno_db=0.0,  seed=42)
    ber_high = compute_ber('QPSK', bits, ebno_db=10.0, seed=42)
    assert ber_high < ber_low