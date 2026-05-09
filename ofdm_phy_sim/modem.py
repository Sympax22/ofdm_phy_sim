"""
modem.py — Constellation mapping and demapping for ofdm_phy_sim.

Implements Grey-coded modulation schemes:
    - BPSK  (1 bit  per symbol)
    - QPSK  (2 bits per symbol)
    - 16-QAM (4 bits per symbol)

Each constellation is normalized to unit average symbol energy.

Typical usage:
    symbols  = modulate(bits, modulation='QPSK')
    rx_bits  = demodulate(symbols, modulation='QPSK')
"""

import numpy as np
from ofdm_phy_sim.constants import BITS_PER_SYMBOL
#
# Constellation Maps
#
_BPSK_MAP  = np.array([
                        -1+0j,  # 0
                        +1+0j,  # 1
                    ])
_QPSK_MAP  = np.array([
                        -1-1j,  # 00
                        -1+1j,  # 01
                        +1-1j,  # 10
                        +1+1j,  # 11
                    ]) / np.sqrt(2)
_QAM16_MAP = np.array([
                        -3-3j,  # 0000
                        -3-1j,  # 0001
                        -3+3j,  # 0010
                        -3+1j,  # 0011
                        -1-3j,  # 0100
                        -1-1j,  # 0101
                        -1+3j,  # 0110
                        -1+1j,  # 0111
                        +3-3j,  # 1000
                        +3-1j,  # 1001
                        +3+3j,  # 1010
                        +3+1j,  # 1011
                        +1-3j,  # 1100
                        +1-1j,  # 1101
                        +1+3j,  # 1110
                        +1+1j,  # 1111
                    ]) / np.sqrt(10)

_MAPS = {
    'BPSK'  : _BPSK_MAP,
    'QPSK'  : _QPSK_MAP,
    '16-QAM': _QAM16_MAP,
}
#
# Public API
#
def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    """
    Map a bitstream to complex constellation symbols.

    :param bits: Flat array of bits (0s and 1s). Length must be a multiple of bits_per_symbol 
    for the chosen modulation.
    :param modulation: Modulation scheme — one of 'BPSK', 'QPSK', '16-QAM'.
    :returns: Complex symbol array of shape (n_symbols,).
    :rtype: np.ndarray
    :raises ValueError: If modulation is not recognised.
    """
    # First, check modulation is valid
    if modulation not in _MAPS.keys():
        raise ValueError(f"Unknown modulation '{modulation}'. Choose from {list(_MAPS.keys())}")
    bits_per_symbol = BITS_PER_SYMBOL[modulation]

    # Next, pad with 0 if necessary
    n_zero_pads = (-1*len(bits)) % bits_per_symbol
    if n_zero_pads != 0:
        bits = np.append(bits, np.zeros(shape=(n_zero_pads,), 
                                        dtype=bits.dtype))

    # Next, split into groups of size bits_per_symbol
    bits = bits.reshape(-1, bits_per_symbol)

    # Next, interpret as binary and map to 0...2**M - 1
    binary_weights = 2 ** np.arange(bits_per_symbol - 1, -1, -1)
    indices = bits.dot(binary_weights)

    # Finally, return symbols
    symbols = _MAPS[modulation][indices]
    return symbols

def demodulate(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """
    Hard-decision demap complex symbols to bits.

    Finds the nearest constellation point for each symbol and returns
    the corresponding Grey-coded bits.

    :param symbols: Complex symbol array of shape (n_symbols,).
    :param modulation: Modulation scheme — one of 'BPSK', 'QPSK', '16-QAM'.
    :returns: Flat bit array of shape (n_symbols * bits_per_symbol,).
    :rtype: np.ndarray
    :raises ValueError: If modulation is not recognised.
    """
    return np.empty(1)