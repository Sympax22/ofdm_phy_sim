"""
test_equalizer.py — Unit tests for ofdm_phy_sim.equalizer.

Tests cover:
    - LS estimation: identity channel, output shape, known channel spot check
    - MMSE estimation: identity channel, outperforms LS at low SNR
    - equalize: identity channel passthrough, output shape, known channel correction
"""