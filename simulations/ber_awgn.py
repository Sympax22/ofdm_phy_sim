"""
ber_awgn.py — BER vs Eb/N0 simulation over an AWGN channel.

Sweeps Eb/N0 from EBNO_DB_RANGE[0] to EBNO_DB_RANGE[1] and computes
simulated BER for BPSK, QPSK, and 16-QAM via Monte Carlo trials.

Simulated curves are plotted alongside theoretical BER references
and saved to plots/ber_awgn.png.

Usage:
    python simulations/ber_awgn.py
"""