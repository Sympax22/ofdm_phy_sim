"""
receiver.py — OFDM receiver chain for ofdm_phy_sim.

Implements the physical layer receiver pipeline:
    1. Cyclic prefix removal: strip CP before FFT processing
    2. FFT: convert time-domain received signal to frequency domain
    3. Subcarrier extraction: separate data and pilot subcarriers

Channel estimation and equalization are handled in equalizer.py.
Follows 802.11a subcarrier layout (N_FFT=64, N_CP=16, N_DATA=48, N_PILOTS=4).
"""