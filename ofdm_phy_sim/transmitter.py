"""
transmitter.py — OFDM transmitter chain for ofdm_phy_sim.

Implements the physical layer transmitter pipeline:
    1. Subcarrier mapping: place data and pilot symbols onto active subcarriers
    2. IFFT: convert frequency-domain frame to time-domain OFDM symbol
    3. Cyclic prefix insertion: append CP to guard against multipath delay spread

Follows 802.11a subcarrier layout (N_FFT=64, N_CP=16, N_DATA=48, N_PILOTS=4).
"""