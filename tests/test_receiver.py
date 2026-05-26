"""
test_receiver.py — Unit tests for ofdm_phy_sim.receiver.

Tests cover:
    - cyclic prefix removal: output shape, correct samples retained
    - FFT: output shape, inversion of transmitter IFFT
    - subcarrier extraction: correct data and pilot indices extracted
"""