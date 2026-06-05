"""
utils.py — Simulation utilities for ofdm_phy_sim.

Provides helper functions used across the simulation pipeline:
- Random bit generation
- BER computation
- Eb/N0 and SNR conversions
- Theoretical BER references
"""

import numpy as np
from scipy.special import erfc
from tqdm import tqdm

from ofdm_phy_sim.modem     import modulate, demodulate
from ofdm_phy_sim.channel   import apply_cawgn
from ofdm_phy_sim.constants import *



def random_bits(n_bits: int | None = None) -> np.ndarray:
    """
    Generate an equiprobable random bitstring.

    :param n_bits: Number of bits. If None, a random length between ```MIN_BITSTRING_LENGTH``` and ```MAX_BITSTRING_LENGTH``` is used.
    :returns: Equiprobable random bitstring of 0s and 1s.
    :rtype: np.ndarray
    """
    if n_bits is None or n_bits <= 0:
        n_bits = np.random.randint(low=MIN_BITSTRING_LENGTH, 
                                   high=MAX_BITSTRING_LENGTH+1, 
                                   size=None)
    return np.random.randint(low=0, 
                             high=2, 
                             size=n_bits)


def ebno_to_noise_var(ebno_db: float, bits_per_symbol: int) -> float:
    """
    Converts Eb/N0 (in dB) to noise variance (N0).

    Assumes unit symbol energy (Es = 1) for simplicity, so Eb = Es / bits_per_symbol.

    :param ebno_db: Eb/N0 in dB.
    :param bits_per_symbol: Number of bits per symbol for the modulation scheme.
    :returns: Noise variance N0 corresponding to the given Eb/N0.
    :rtype: float
    :raises ValueError: If bits_per_symbol is <= 0.
    """
    if bits_per_symbol <= 0:
        raise ValueError("In simulations/ber_awgn.py: ebno_to_noise_var: bits_per_symbol must be a positive integer.")

    ebno_linear = 10 ** (ebno_db / 10)
    N0 = 1 / (bits_per_symbol * ebno_linear)
    return N0


def compute_ber(modulation: str, 
                bits: np.ndarray,
                ebno_db: float,
                seed: int):
    """
    Simulates transmission of bits over an AWGN channel and computes BER.

    :param modulation: Modulation scheme to use (e.g., 'BPSK', 'QPSK', '16QAM').
    :param bits: Flat array of bits to transmit (0s and 1s).
    :param ebno_db: Eb/N0 (in dB) for the AWGN channel.
    :param seed: Seed for random number generation to ensure reproducibility.
    :returns: Simulated BER for the given modulation and noise level.
    :rtype: float
    :raises ValueError: If modulation scheme is not supported or if bits array is empty.
    """
    if bits.size == 0:
        return 0.0
    
    symbols = modulate(bits=bits, 
                       modulation=modulation)
    symbols = apply_cawgn(raw_symbols=symbols, 
                          N0=ebno_to_noise_var(ebno_db, BITS_PER_SYMBOL[modulation]),
                          seed=seed)
    noisy_bits = demodulate(symbols=symbols,
                            modulation=modulation)
    # Note: demodulate() should return an array of bits (0s and 1s) corresponding to the noisy symbols.
    # We XOR here because the bits are binary (0 and 1), so a mismatch will yield 1 and a match will yield 0.
    return np.count_nonzero(bits ^ noisy_bits[:len(bits)])/bits.size


def simulate_ber_for_modulation(modulation: str,
                                ebno_db_range=np.linspace(EBNO_DB_RANGE[0], EBNO_DB_RANGE[1], num=N_POINTS_PER_CURVE),
                                n_trials=N_TRIALS_PER_POINT) -> np.ndarray:
    """
    Simulates BER vs Eb/N0 for a given modulation scheme over a specified range of Eb/N0 values.

    :param modulation: Modulation scheme to simulate (e.g., 'BPSK', 'QPSK', '16QAM').
    :param ebno_db_range: Array of Eb/N0 values in dB to simulate over.
    :param n_trials: Number of Monte Carlo trials to perform for each Eb/N0 value.
    :returns: Array of simulated BER values corresponding to the input Eb/N0 range.
    :rtype: np.ndarray
    """
    ber_values = np.zeros_like(ebno_db_range)
    
    for i, ebno_db in enumerate(tqdm(ebno_db_range, desc=modulation, position=1, leave=False)):
        ber_mean = 0.0
        for trial in tqdm(range(n_trials), desc=f"Trial", position=2, leave=False):
            bits = np.random.randint(0, 2, size=2**13)  # Simulate transmission of 8'192 bits per trial
            ber = compute_ber(modulation=modulation,
                              bits=bits,
                              ebno_db=ebno_db,
                              seed=trial)  # Use trial index as seed for reproducibility
            ber_mean += ber
        ber_values[i] = ber_mean / n_trials
    
    return ber_values


def theoretical_ber_awgn(modulation: str, ebno_db_range: np.ndarray) -> np.ndarray:
    """
    Vectorized version of theoretical_ber_awgn to compute BER over a range of Eb/N0 values.

    :param modulation: Modulation scheme to compute BER for (e.g., 'BPSK', 'QPSK', '16QAM').
    :param ebno_db_range: Array of Eb/N0 values in dB.
    :returns: Array of theoretical BER values corresponding to the input Eb/N0 range.
    :rtype: np.ndarray
    """
    ebno_linear = 10 ** (ebno_db_range / 10)
    if modulation == 'BPSK' or modulation == 'QPSK':
        return 0.5 * erfc(np.sqrt(ebno_linear))
    elif modulation == '16-QAM':
        return (3/8) * erfc(np.sqrt((2/5)*ebno_linear))
    else:
        raise ValueError(f"Unsupported modulation scheme: {modulation}")