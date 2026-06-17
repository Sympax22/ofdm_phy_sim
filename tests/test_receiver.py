"""
test_receiver.py — Unit tests for ofdm_phy_sim.receiver.

Tests cover:
    - cyclic prefix removal: output shape, correct samples retained
    - FFT: output shape, inversion of transmitter IFFT
    - subcarrier extraction: correct data and pilot indices extracted
"""

import numpy as np

from ofdm_phy_sim.transmitter import insert_cyclic_prefix
from ofdm_phy_sim.receiver    import remove_cyclic_prefix
from ofdm_phy_sim.constants   import *

def test_remove_cyclic_prefix():
    test_array = np.arange(0,N_FFT)
    test_array_CP = insert_cyclic_prefix(test_array)

    result_array = remove_cyclic_prefix(test_array_CP)

    assert (result_array.shape[0] == N_FFT)
    assert np.array_equal(result_array, test_array)
    assert np.array_equal(result_array, np.sort(result_array))
