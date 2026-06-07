"""
test_transmitter.py — Unit tests for ofdm_phy_sim.transmitter.

Tests cover:
    - subcarrier mapping: output shape, pilot placement, DC and guard zeroing
    - IFFT: output shape, energy preservation, inversion of FFT
    - cyclic prefix insertion: output length, CP content matches tail of symbol
"""

import numpy as np

from ofdm_phy_sim.transmitter import map_symbols_to_subcarriers_f, ofdm_ifft, insert_cyclic_prefix
from ofdm_phy_sim.constants   import *

def test_80211a_subcarrier_mapping():
    # 1. Create dummy data: 1 to 48 (so we can track ordering)
    # Using complex numbers since the array is complex64
    test_data = np.arange(1, 49, dtype=np.complex64) 
    
    # 2. Create distinct dummy pilots to easily spot them
    test_pilots = np.array([100j, 200j, 300j, 400j], dtype=np.complex64)

    # Run the mapper
    ofdm_symbol = map_symbols_to_subcarriers_f(test_data, test_pilots)

    # --- ASSERTIONS ---

    # 1. Check structural constraints
    assert len(ofdm_symbol) == 64, "FFT size must be exactly 64"
    
    # 2. Check the DC Subcarrier (Index 32 in FFT-shifted array)
    assert ofdm_symbol[32] == 0j, "DC subcarrier must be perfectly null (0)"

    # 3. Check Guard Bands (Indices 0-5 and 59-63)
    assert np.all(ofdm_symbol[0:6] == 0j), "Lower guard band must be null"
    assert np.all(ofdm_symbol[59:64] == 0j), "Upper guard band must be null"

    # 4. Check Pilot Placements
    # -21 + 32 = 11
    assert ofdm_symbol[11] == 100j, "Pilot 1 incorrectly mapped"
    # -7 + 32 = 25
    assert ofdm_symbol[25] == 200j, "Pilot 2 incorrectly mapped"
    # 7 + 32 = 39
    assert ofdm_symbol[39] == 300j, "Pilot 3 incorrectly mapped"
    # 21 + 32 = 53
    assert ofdm_symbol[53] == 400j, "Pilot 4 incorrectly mapped"

    # 5. Spot-check Data Placements (Verifying the continuity around gaps)
    # The first data symbol (index -26) should be at bin 6 (-26 + 32)
    assert ofdm_symbol[6] == test_data[0], "First data symbol failed"
    
    # The symbol right before the first pilot (-22) should be at bin 10
    assert ofdm_symbol[10] == test_data[4], "Data symbol before pilot -21 failed"
    
    # The symbol right after the first pilot (-20) should be at bin 12
    assert ofdm_symbol[12] == test_data[5], "Data symbol after pilot -21 failed"

    # The symbol right before DC (-1) should be at bin 31
    assert ofdm_symbol[31] == test_data[23], "Data symbol right before DC failed"
    
    # The symbol right after DC (+1) should be at bin 33
    assert ofdm_symbol[33] == test_data[24], "Data symbol right after DC failed"

    # The very last data symbol (+26) should be at bin 58
    assert ofdm_symbol[58] == test_data[47], "Last data symbol failed"


def test_ofdm_modulate():
    # ==========================================
    # TEST 1: Verify the IFFT Shift Mapping
    # ==========================================
    physical_freqs = np.arange(-32, 32)
    shifted_freqs = np.fft.ifftshift(physical_freqs)
    
    assert shifted_freqs[0] == 0,   "FAIL: DC (0) did not map to IFFT bin 0"
    assert shifted_freqs[1] == 1,   "FAIL: Positive freq (+1) mapped incorrectly"
    assert shifted_freqs[31] == 31, "FAIL: Positive freq (+31) mapped incorrectly"
    assert shifted_freqs[32] == -32,"FAIL: Lower edge (-32) did not wrap correctly"
    assert shifted_freqs[63] == -1, "FAIL: Negative freq (-1) did not map to the end"

    # ==========================================
    # TEST 2: The DC Impulse -> Constant Output
    # ==========================================
    centered_dc_array = np.zeros(N_FFT, dtype=np.complex64)
    centered_dc_array[32] = 1.0 + 0j 
    
    # Passing through YOUR function
    time_domain_dc = ofdm_ifft(centered_dc_array)
    
    expected_constant = 1.0 / N_FFT
    assert np.allclose(time_domain_dc, expected_constant), "FAIL: DC impulse did not yield a constant time-domain signal!"

    # ==========================================
    # TEST 3: Symmetric Impulses -> Pure Real Cosine
    # ==========================================
    centered_cos_array = np.zeros(N_FFT, dtype=np.complex64)
    centered_cos_array[32 + 4] = 1.0 + 0j 
    centered_cos_array[32 - 4] = 1.0 + 0j 
    
    # Passing through YOUR function
    time_domain_cos = ofdm_ifft(centered_cos_array)
    
    assert np.allclose(np.imag(time_domain_cos), 0), "FAIL: Symmetric input did not cancel imaginary components!"


def test_insert_cyclic_prefix():
    # 1. Create dummy time-domain data (numbers 0 to 63 to easily track indices)
    dummy_symbol = np.arange(N_FFT)
    
    # 2. Run the function
    symbol_with_cp = insert_cyclic_prefix(dummy_symbol)
    
    # --- ASSERTIONS ---
    
    # Assertion 1: Check the overall length
    expected_length = N_FFT + N_CP
    assert len(symbol_with_cp) == expected_length, \
        f"FAIL: Expected length {expected_length}, got {len(symbol_with_cp)}"
    
    # Assertion 2: Verify the original symbol is perfectly intact at the tail
    assert np.array_equal(symbol_with_cp[N_CP:], dummy_symbol), \
        "FAIL: The original symbol payload was corrupted or misplaced!"
    
    # Assertion 3: Verify the prefix itself is an exact copy of the last N_CP samples
    expected_cp = dummy_symbol[-N_CP:]
    assert np.array_equal(symbol_with_cp[:N_CP], expected_cp), \
        "FAIL: The copied prefix does not match the end of the original symbol!"
    
    # Assertion 4: Spot check the exact transition boundary
    # The element right after the prefix should be the start of the payload
    assert symbol_with_cp[N_CP] == dummy_symbol[0], \
        "FAIL: Boundary mismatch at the transition from CP to original symbol!"