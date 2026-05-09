"""
test_modem.py — Unit tests for ofdm_phy_sim.modem.

Tests cover:
    - modulate: error handling for invalid modulation, output length,
    output dtype, padding behaviour, and correct Grey-coded symbol
    mapping for BPSK, QPSK, and 16-QAM.
"""

import numpy as np
import pytest
from ofdm_phy_sim.modem import modulate
from ofdm_phy_sim.constants import BITS_PER_SYMBOL

def test_modulate_invalid_modulation():
    bits = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        modulate(bits, modulation='64-QAM')

def test_modulate_output_length():
    for mod, bps in BITS_PER_SYMBOL.items():
        bits = np.zeros(4 * bps, dtype=int)
        symbols = modulate(bits, modulation=mod)
        assert len(symbols) == 4

def test_modulate_output_is_complex():
    for mod in BITS_PER_SYMBOL:
        bits = np.zeros(16, dtype=int)
        symbols = modulate(bits, modulation=mod)
        assert np.iscomplexobj(symbols)

def test_modulate_bpsk_mapping():
    assert np.isclose(modulate(np.array([0]), 'BPSK')[0], -1+0j)
    assert np.isclose(modulate(np.array([1]), 'BPSK')[0], +1+0j)

def test_modulate_qpsk_mapping():
    expected = {
        (0,0): (-1-1j) / np.sqrt(2),
        (0,1): (-1+1j) / np.sqrt(2),
        (1,0): (+1-1j) / np.sqrt(2),
        (1,1): (+1+1j) / np.sqrt(2),
    }
    for bits, symbol in expected.items():
        out = modulate(np.array(bits), 'QPSK')
        assert np.isclose(out[0], symbol), f"QPSK {bits} → expected {symbol}, got {out[0]}"

def test_modulate_16qam_mapping():
    expected = {
        (0,0,0,0): (-3-3j) / np.sqrt(10),
        (0,1,0,1): (-1-1j) / np.sqrt(10),
        (1,1,1,1): (+1+1j) / np.sqrt(10),
        (1,0,1,0): (+3+3j) / np.sqrt(10),
    }
    for bits, symbol in expected.items():
        out = modulate(np.array(bits), '16-QAM')
        assert np.isclose(out[0], symbol), f"16-QAM {bits} → expected {symbol}, got {out[0]}"

def test_modulate_padding():
    # 5 bits with QPSK (bps=2) should produce 3 symbols (padded to 6 bits)
    bits = np.ones(5, dtype=int)
    symbols = modulate(bits, 'QPSK')
    assert len(symbols) == 3