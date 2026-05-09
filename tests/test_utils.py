"""
test_utils.py — Unit tests for ofdm_phy_sim.utils.

Tests cover:
    - random_bits: output length, value range, dtype, and
    both fixed and random-length modes.
"""

import numpy as np
import pytest
from ofdm_phy_sim.utils import random_bits

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