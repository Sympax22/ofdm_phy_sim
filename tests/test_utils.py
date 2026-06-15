"""
test_utils.py — Unit tests for ofdm_phy_sim.utils.

Tests cover:
    - random_bits: output length, value range, dtype, and
    both fixed and random-length modes.
"""

import numpy as np
import pytest

from ofdm_phy_sim.constants import *
from ofdm_phy_sim.utils import random_bits, \
ebno_to_noise_var, compute_ber, theoretical_ber_awgn, \
zero_centered_to_zero_start, apply_fractional_delay

def test_random_bits_default_length_in_range():
    bits = random_bits()
    assert 64 <= len(bits) <= 8192

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

def test_zero_centered_to_zero_start():
    # Test that negative indices are correctly converted to positive indices
    assert zero_centered_to_zero_start(-32) == -32 + N_FFT//2
    assert zero_centered_to_zero_start(-1)  == -1  + N_FFT//2
    assert zero_centered_to_zero_start(0)   ==  0  + N_FFT//2
    assert zero_centered_to_zero_start(31)  == 31  + N_FFT//2

def test_fractional_delay_no_delay():
    signal = np.random.rand(100)
    delayed_signal = apply_fractional_delay(signal, delay_samples=0.0)
    # No delay: output length = input length, values match
    assert delayed_signal.shape[0] == signal.shape[0]
    assert np.allclose(signal, delayed_signal)

def test_fractional_delay_nonzero_delay():
    signal = np.ones(100)  # Use constant signal for clearer testing
    delay_samples = 2.5
    delayed_signal = apply_fractional_delay(signal, delay_samples=delay_samples)
    
    # Output length = input + ceil(delay) - 1
    expected_len = 100 + int(np.ceil(delay_samples)) - 1
    assert delayed_signal.shape[0] == expected_len
    
    # After the delay period, should be close to 1 (the constant signal)
    assert np.allclose(delayed_signal[5:], 1.0, atol=0.1)

def test_multipath_fading_shape():
    signal = np.ones(100)
    max_delay = 2.5
    delayed_signal = apply_fractional_delay(signal, delay_samples=max_delay)
    
    # Output length = input length + ceil(delay)
    expected_len = 100 + int(np.ceil(max_delay)) - 1
    assert delayed_signal.shape[0] == expected_len

def test_multipath_fading_zero_delay():
    signal = np.random.rand(100)
    delayed_signal = apply_fractional_delay(signal, delay_samples=0.0)
    
    assert delayed_signal.shape[0] == signal.shape[0]
    assert np.allclose(signal, delayed_signal)

def test_multipath_fading_large_delay():
    signal = np.random.rand(100)
    delay_samples = 150.0
    delayed_signal = apply_fractional_delay(signal, delay_samples=delay_samples)
    
    # Output length should be 100 + 150 = 250
    expected_len = 100 + int(np.ceil(delay_samples))
    assert delayed_signal.shape[0] == expected_len
    
    # First 150 samples should be near zero (prepended delay), rest from signal
    assert np.allclose(delayed_signal[:150], 0.0, atol=1e-10)

