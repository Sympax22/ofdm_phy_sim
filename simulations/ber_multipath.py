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

import time
from tqdm import tqdm
import numpy as np

from ofdm_phy_sim.constants import *
from ofdm_phy_sim.utils import simulate_ber_for_modulation
import matplotlib.pyplot as plt


def plot_all_modulations_ber_multipath():
    """
    Simulates and plots BER vs Eb/N0 for all modulation schemes defined in MODULATIONS.
    """
    ebno_db_range = np.linspace(EBNO_DB_RANGE[0], EBNO_DB_RANGE[1], num=N_POINTS_PER_CURVE)
    simulated_ber_curves   = dict(zip(MODULATIONS, [np.empty(N_POINTS_PER_CURVE) for _ in range(len(MODULATIONS))]))
    #theoretical_ber_curves = dict(zip(MODULATIONS, [np.empty(N_POINTS_PER_CURVE) for _ in range(len(MODULATIONS))]))
    for modulation in tqdm(MODULATIONS, desc="Modulation", position=0):
        simulated_ber_curves[modulation]   = simulate_ber_for_modulation(modulation=modulation,
                                                                         ebno_db_range=ebno_db_range,
                                                                         n_trials=N_TRIALS_PER_POINT,
                                                                         fading=True)
        #theoretical_ber_curves[modulation] = theoretical_ber_awgn(modulation=modulation, 
        #                                                          ebno_db_range=ebno_db_range)
    # plot the curves
    plt.figure(figsize=(10, 6))
    for modulation in MODULATIONS:
        plt.semilogy(ebno_db_range, simulated_ber_curves[modulation], label=f"{modulation} Simulated", marker='o', linestyle='-', markersize=4)
        #plt.semilogy(ebno_db_range, theoretical_ber_curves[modulation], label=f"{modulation} Theoretical", linestyle='--')      
    plt.title("BER vs Eb/N0 for Multipath Fading Channel")
    plt.xlabel("Eb/N0 (dB)")        
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.xlim(EBNO_DB_RANGE[0], EBNO_DB_RANGE[1])
    fig_timecode = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"plots/ber_multipath_{fig_timecode}.png")
    plt.show()

if __name__ == "__main__":
    plot_all_modulations_ber_multipath()