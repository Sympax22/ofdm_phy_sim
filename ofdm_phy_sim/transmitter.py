"""
transmitter.py — OFDM transmitter chain for ofdm_phy_sim.

Implements the physical layer transmitter pipeline:
    1. Subcarrier mapping: place data and pilot symbols onto active subcarriers
    2. IFFT: convert frequency-domain frame to time-domain OFDM symbol
    3. Cyclic prefix insertion: append CP to guard against multipath delay spread

Follows 802.11a subcarrier layout (N_FFT=64, N_CP=16, N_DATA=48, N_PILOTS=4).
"""

import numpy as np

from ofdm_phy_sim.constants import *


def map_symbols_to_subcarriers(data_symbols: np.ndarray, 
                               pilot_symbols: np.ndarray) -> np.ndarray:
    """
    Map data and pilot symbols to the appropriate subcarriers in the frequency domain.

    :param data_symbols: Array of shape (N_DATA,) containing the modulated data symbols.
    :param pilot_symbols: Array of shape (N_PILOTS,) containing the modulated pilot symbols.
    :return: Array of shape (N_FFT,) representing the frequency-domain OFDM symbol.
    """ 
    subcarrier_array_f = np.zeros(shape=(N_FFT,), dtype=np.complex64)
    subcarrier_array_f[DATA_INDICES]  = data_symbols
    subcarrier_array_f[PILOT_INDICES] = pilot_symbols
    return subcarrier_array_f