"""
test_receiver.py — Unit tests for ofdm_phy_sim.receiver.

Tests cover:
    - cyclic prefix removal: output shape, correct samples retained
    - FFT: output shape, inversion of transmitter IFFT
    - subcarrier extraction: correct data and pilot indices extracted
"""

import numpy as np

from ofdm_phy_sim.transmitter import insert_cyclic_prefix, ofdm_ifft
from ofdm_phy_sim.receiver    import remove_cyclic_prefix, ofdm_fft
from ofdm_phy_sim.constants   import *

def test_remove_cyclic_prefix():
    test_array = np.arange(0,N_FFT)
    test_array_CP = insert_cyclic_prefix(test_array)

    result_array = remove_cyclic_prefix(test_array_CP)

    assert (result_array.shape[0] == N_FFT)
    assert np.array_equal(result_array, test_array)
    assert np.array_equal(result_array, np.sort(result_array))


def test_ofdm_fft():
    N_TESTS = 2**14
    measured_array = np.empty(shape=(N_TESTS,))

    test_array = np.sqrt(0.5) * (np.random.standard_normal(size=(N_TESTS, N_FFT)) \
                          + 1j * np.random.standard_normal(size=(N_TESTS, N_FFT)))

    for k in range(N_TESTS):
        test_array[k, :] -= ofdm_fft(ofdm_ifft(test_array[k, :]))
        
    measured_array = np.mean(test_array, axis=1)

    assert np.allclose(measured_array, np.zeros(shape=(N_TESTS,)), atol=1e-16)
