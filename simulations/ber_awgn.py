"""
ber_awgn.py — BER vs Eb/N0 simulation over an AWGN channel.

Sweeps Eb/N0 from EBNO_DB_RANGE[0] to EBNO_DB_RANGE[1] and computes
simulated BER for BPSK, QPSK, and 16-QAM via Monte Carlo trials.

Simulated curves are plotted alongside theoretical BER references
and saved to plots/ber_awgn.png.

Usage:
    python simulations/ber_awgn.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from ofdm_phy_sim.constants import *
from ofdm_phy_sim.utils     import simulate_ber_for_modulation, theoretical_ber_awgn



def plot_all_modulations_ber_awgn():
    """
    Simulates and plots BER vs Eb/N0 for all modulation schemes defined in MODULATIONS.
    """
    ebno_db_range = np.linspace(EBNO_DB_RANGE[0], EBNO_DB_RANGE[1], num=N_POINTS_PER_CURVE)
    simulated_ber_curves   = dict(zip(MODULATIONS, [np.empty(N_POINTS_PER_CURVE) for _ in range(len(MODULATIONS))]))
    theoretical_ber_curves = dict(zip(MODULATIONS, [np.empty(N_POINTS_PER_CURVE) for _ in range(len(MODULATIONS))]))
    for modulation in tqdm(MODULATIONS, desc="Modulation", position=0):
        simulated_ber_curves[modulation]   = simulate_ber_for_modulation(modulation=modulation,
                                                                         ebno_db_range=ebno_db_range,
                                                                         n_trials=N_TRIALS_PER_POINT)
        theoretical_ber_curves[modulation] = theoretical_ber_awgn(modulation=modulation, 
                                                                  ebno_db_range=ebno_db_range)
    # plot the curves
    plt.figure(figsize=(10, 6))
    for modulation in MODULATIONS:
        plt.semilogy(ebno_db_range, simulated_ber_curves[modulation], label=f"{modulation} Simulated", marker='o', linestyle='-', markersize=4)
        plt.semilogy(ebno_db_range, theoretical_ber_curves[modulation], label=f"{modulation} Theoretical", linestyle='--')      
    plt.title("BER vs Eb/N0 for AWGN Channel")
    plt.xlabel("Eb/N0 (dB)")        
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.xlim(EBNO_DB_RANGE[0], EBNO_DB_RANGE[1])
    fig_timecode = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"plots/ber_awgn_{fig_timecode}.png")
    plt.show()

if __name__ == "__main__":
    plot_all_modulations_ber_awgn()