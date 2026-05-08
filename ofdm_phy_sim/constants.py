"""
constants.py — System-level constants for ofdm_phy_sim.

Defines the core OFDM physical layer parameters shared across
the simulation pipeline.
"""

# Bit & Block specific settings
MIN_BITSTRING_LENGTH = 64
MAX_BITSTRING_LENGTH = 4096

# FFT / OFDM Frame 

N_FFT = 64              # Total FFT size (includes null subcarriers)
N_CP = 16               # Cyclic prefix length (samples)
N_SUBCARRIERS = 52      # Number of active data + pilot subcarriers
N_PILOTS = 4            # Number of pilot subcarriers
N_DATA = 48             # Number of data subcarriers

# Subcarrier Indices (relative to FFT, 802.11a convention)

PILOT_INDICES = [-21, -7, 7, 21]          # Pilot subcarrier positions
NULL_INDICES  = [0]                       # DC subcarrier (unused)

# Modulation

MODULATIONS = ['BPSK', 'QPSK', '16-QAM']
BITS_PER_SYMBOL = {'BPSK': 1, 'QPSK': 2, '16-QAM': 4}

# Simulation

EBNO_DB_RANGE = (-2, 20)    # Default Eb/N0 sweep range (dB)
N_TRIALS      = 1000        # Default number of Monte Carlo trials per Eb/N0 point