"""
equalizer.py — Channel estimation and equalization for ofdm_phy_sim.

Implements pilot-based frequency-domain channel estimation and
one-tap equalization for OFDM systems.

Estimation methods:
    - Least Squares (LS): simple pilot-based estimate, no noise regularisation
    - MMSE: minimum mean square error estimate using noise variance

Equalization:
    - One-tap frequency-domain equalizer: divides each subcarrier by
      its estimated channel coefficient

Designed to be swappable — equalizer variants can be compared
independently of the transmitter and receiver chain.
"""

import numpy as np

from ofdm_phy_sim.constants import *
from ofdm_phy_sim.utils import get_nth_pilots

def interpolate_channel_from_pilots(h_est_pilots: np.ndarray) -> np.ndarray:
    """
    Interpolate channel estimates for data subcarriers from pilot subcarriers.

    :param pilot_subcarriers: Array of shape (N_PILOTS,) containing the received pilot subcarriers.
    :type pilot_subcarriers: np.ndarray, dtype=np.complex64
    :returns: Array of shape (N_DATA,) containing the interpolated channel estimates for data subcarriers.
    :rtype: np.ndarray, dtype=np.complex64
    """
    return np.interp(x=DATA_INDICES, xp=PILOT_INDICES, fp=h_est_pilots)

def equalize_from_pilots(data_subcarriers:  np.ndarray,
                         pilot_subcarriers: np.ndarray,
                         n: int) -> np.ndarray:
    """
    Equalize data subcarriers using pilot subcarriers.

    :param data_subcarriers: Array of shape (N_DATA,) containing the received data subcarriers.
    :type data_subcarriers: np.ndarray, dtype=np.complex64
    :param pilot_subcarriers: Array of shape (N_PILOTS,) containing the received pilot subcarriers.
    :type pilot_subcarriers: np.ndarray, dtype=np.complex64
    :param n: Index of the OFDM symbol.
    :type n: int
    :returns: Array of shape (N_DATA,) containing the equalized data symbols.
    :rtype: np.ndarray, dtype=np.complex64
    """
    h_est_pilots = pilot_subcarriers * get_nth_pilots(n) # normally we divide, but since we only have +- 1, we can multiply
    h_est = interpolate_channel_from_pilots(h_est_pilots=h_est_pilots)
    data_est = data_subcarriers * np.conj(h_est) / np.square(np.abs(h_est))
    return data_est
    