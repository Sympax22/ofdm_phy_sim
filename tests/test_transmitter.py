"""
test_transmitter.py — Unit tests for ofdm_phy_sim.transmitter.

Tests cover:
    - subcarrier mapping: output shape, pilot placement, DC and guard zeroing
    - IFFT: output shape, energy preservation, inversion of FFT
    - cyclic prefix insertion: output length, CP content matches tail of symbol
"""