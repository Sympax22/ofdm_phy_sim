"""
utils.py — Simulation utilities for ofdm_phy_sim.

Provides helper functions used across the simulation pipeline:
- Random bit generation
- BER computation
- Eb/N0 and SNR conversions
- Theoretical BER references
"""
from ofdm_phy_sim.constants import *
import numpy as np


def random_bits(n_bits: int | None = None) -> np.ndarray:
    """
    Generate an equiprobable random bitstring.

    :param n_bits: Number of bits. If None, a random length between ```MIN_BITSTRING_LENGTH``` and ```MAX_BITSTRING_LENGTH``` is used.
    :returns: Equiprobable random bitstring of 0s and 1s.
    :rtype: np.ndarray
    """
    if n_bits is None or n_bits <= 0:
        n_bits = np.random.randint(low=MIN_BITSTRING_LENGTH, 
                                   high=MAX_BITSTRING_LENGTH+1, 
                                   size=None)
    return np.random.randint(low=0, 
                             high=2, 
                             size=n_bits)

def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """
    Compute the Bit Error Rate between transmitted and received bits.

    :param tx_bits: Transmitted bit array.
    :param rx_bits: Received bit array.
    :returns: Fraction of bits in error.
    :rtype: float
    :raises ValueError: If arrays have different lengths.
    """
    return 0.0


def ebno_to_noise_std(ebno_db: float, bits_per_symbol: int, n_subcarriers: int) -> float:
    """
    Convert Eb/N0 (dB) to noise standard deviation for AWGN.

    :param ebno_db: Eb/N0 in decibels.
    :param bits_per_symbol: Bits per constellation symbol (1=BPSK, 2=QPSK, 4=16-QAM).
    :param n_subcarriers: Number of active OFDM subcarriers.
    :returns: Standard deviation of AWGN noise to apply.
    :rtype: float
    """
    return 0.0


def theoretical_ber_awgn(ebno_db: float, modulation: str) -> float:
    """
    Compute the theoretical BER over AWGN for a given modulation scheme.

    :param ebno_db: Eb/N0 in decibels.
    :param modulation: One of 'BPSK', 'QPSK', '16-QAM'.
    :returns: Theoretical BER.
    :rtype: float
    :raises ValueError: If modulation scheme is not recognised.
    """
    return 0.0
