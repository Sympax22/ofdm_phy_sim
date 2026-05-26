"""
ber_multipath.py — BER vs Eb/N0 simulation over a multipath fading channel.

Sweeps Eb/N0 from EBNO_DB_RANGE[0] to EBNO_DB_RANGE[1] and computes
simulated BER for BPSK, QPSK, and 16-QAM with a tapped delay line
multipath channel and frequency-domain equalization.

LS and MMSE equalization curves are plotted alongside the AWGN
reference and saved to plots/ber_multipath.png.

Usage:
    python simulations/ber_multipath.py
"""