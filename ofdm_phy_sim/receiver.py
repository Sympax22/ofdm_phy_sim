"""
receiver.py — OFDM receiver chain for ofdm_phy_sim.

Implements the physical layer receiver pipeline:
    1. Cyclic prefix removal: strip CP before FFT processing
    2. FFT: convert time-domain received signal to frequency domain
    3. Subcarrier extraction: separate data and pilot subcarriers

Channel estimation and equalization are handled in equalizer.py.
Follows 802.11a subcarrier layout (N_FFT=64, N_CP=16, N_DATA=48, N_PILOTS=4).
"""
from typing import Tuple
import numpy as np

from ofdm_phy_sim.constants import *

def remove_cyclic_prefix(ofdm_symbol: np.ndarray, 
                         n_cp=N_CP) -> np.ndarray:
    """
    Remove cyclic prefix from a received OFDM symbol.

    :param ofdm_symbol: Array of shape (N_FFT+N_CP,).
    :type ofdm_symbol: np.ndarray with entry type np.complex64
    :param n_cp: Length of cyclic prefix.
    :type n_cp: int 
    :returns: Array of shape (N_FFT,); N_FFT < N_FFT+N_CP.
    :rtype: np.ndarray with entry type np.complex64
    """
    return ofdm_symbol[n_cp:]


def ofdm_fft(subcarrier_array_t: np.ndarray) -> np.ndarray:
    """
    Perform FFT to convert the time-domain subcarrier array into a frequency-domain OFDM symbol.

    :param subcarrier_array_t: Array of shape (N_FFT,) representing the time-domain OFDM symbol.
    :type subcarrier_array_t: np.ndarray, dtype=np.complex64

    :returns: Array of shape (N_FFT,) representing the frequency-domain OFDM symbol.
    :rtype:   np.ndarray, dtype=np.complex64
    """
    return np.fft.fftshift(np.fft.fft(subcarrier_array_t))  


def extract_subcarriers_to_symbols_f(subcarrier_array_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract data and pilot symbols from the frequency-domain subcarrier array.

    :param subcarrier_array_f: Array of shape (N_FFT,) representing the frequency-domain OFDM symbol.
    :type subcarrier_array_f: np.ndarray, dtype=np.complex64
    :returns: Tuple of two arrays:
        - data_symbols: Array of shape (N_DATA,) containing the extracted data symbols.
        - pilot_symbols: Array of shape (N_PILOTS,) containing the extracted pilot symbols.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    data_symbols  = subcarrier_array_f[DATA_INDICES]
    pilot_symbols = subcarrier_array_f[PILOT_INDICES] 
    return data_symbols, pilot_symbols