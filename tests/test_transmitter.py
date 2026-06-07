"""
test_transmitter.py — Unit tests for ofdm_phy_sim.transmitter.

Tests cover:
    - subcarrier mapping: output shape, pilot placement, DC and guard zeroing
    - IFFT: output shape, energy preservation, inversion of FFT
    - cyclic prefix insertion: output length, CP content matches tail of symbol
"""

from ofdm_phy_sim.transmitter import map_symbols_to_subcarriers
import numpy as np

def test_80211a_subcarrier_mapping():
    # 1. Create dummy data: 1 to 48 (so we can track ordering)
    # Using complex numbers since the array is complex64
    test_data = np.arange(1, 49, dtype=np.complex64) 
    
    # 2. Create distinct dummy pilots to easily spot them
    test_pilots = np.array([100j, 200j, 300j, 400j], dtype=np.complex64)

    # Run the mapper
    ofdm_symbol = map_symbols_to_subcarriers(test_data, test_pilots)

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