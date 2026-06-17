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


def map_symbols_to_subcarriers_f(data_symbols: np.ndarray, 
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

def ofdm_ifft(subcarrier_array_f: np.ndarray) -> np.ndarray:
    """
    Perform IFFT to convert the frequency-domain subcarrier array into a time-domain OFDM symbol.

    :param subcarrier_array_f: Array of shape (N_FFT,) representing the frequency-domain OFDM symbol.
    :return: Array of shape (N_FFT,) representing the time-domain OFDM symbol.
    """
    return np.fft.ifft(np.fft.ifftshift(subcarrier_array_f))

def insert_cyclic_prefix(ofdm_symbol_td: np.ndarray) -> np.ndarray:
    """
    Insert a cyclic prefix (CP) at the beginning of the time-domain OFDM symbol.

    :param ofdm_symbol_td: Array of shape (N_FFT,) representing the time-domain OFDM symbol.
    :return: Array of shape (N_FFT + N_CP,) representing the OFDM symbol with cyclic prefix.
    """
    cyclic_prefix = ofdm_symbol_td[-N_CP:]  # Last N_CP samples become the CP
    return np.concatenate((cyclic_prefix, ofdm_symbol_td))


