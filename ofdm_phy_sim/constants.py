"""
constants.py — System-level constants for ofdm_phy_sim.

Defines the core OFDM physical layer parameters shared across
the simulation pipeline.
"""
import numpy as np


# Bit & Block specific settings
MIN_BITSTRING_LENGTH = 64
MAX_BITSTRING_LENGTH = 8192

# FFT / OFDM Frame 
N_FFT       = 64            # Total FFT size (includes null subcarriers)
N_CP        = 16            # Cyclic prefix length (samples)
N_SP        = 4             # Number of pilot subcarriers
N_SD        = 48            # Number of data subcarriers
N_ST        = N_SP + N_SD   # = 52 Total number of active subcarriers (data + pilots)

BW          = 20e6          # Channel bandwidth (Hz)
FS          = BW            # Sampling frequency (Hz), set equal to bandwidth for simplicity
TS          = 1 / FS        # Sampling period (s)
BW_MHZ      = BW / 1e6      # Bandwidth in MHz
DELTA_F     = BW / N_FFT    # Subcarrier spacing (Hz)
DELTA_F_MHZ = DELTA_F / 1e6 # Subcarrier spacing in MHz


# Subcarrier Indices (relative to FFT, 802.11a convention)
PILOT_INDICES_CENTERED      = np.array([-21, -7, 7, 21])          # Pilot subcarrier positions
DC_INDEX_CENTERED           = np.array([0])                       # DC subcarrier (unused)
FIRST_DATA_INDICES_CENTERED = np.arange(-26,-21)
LAST_DATA_INDICES_CENTERED  = np.arange(22, 27)
DATA_INDICES_CENTERED       = np.concatenate([
                                np.arange(-26, -21),
                                np.arange(-20,  -7),
                                np.arange( -6,   0),
                                np.arange(  1,   7),
                                np.arange(  8,  21),
                                np.arange( 22,  27)
                                ])
GUARD_INDICES_CENTERED = np.block([np.arange(-N_FFT//2, -26), 
                                   np.arange(27, N_FFT//2)]) # Unused guard subcarriers
GUARD_AND_DC_INDICES_CENTERED = np.sort(np.concatenate([GUARD_INDICES_CENTERED, 
                                                        DC_INDEX_CENTERED])) # All unused subcarriers

# Convert to FFT bin indices (0 to 63)
PILOT_INDICES        = PILOT_INDICES_CENTERED        + N_FFT // 2
DC_INDEX             = DC_INDEX_CENTERED             + N_FFT // 2
DATA_INDICES         = DATA_INDICES_CENTERED         + N_FFT // 2
GUARD_INDICES        = GUARD_INDICES_CENTERED        + N_FFT // 2
GUARD_AND_DC_INDICES = GUARD_AND_DC_INDICES_CENTERED + N_FFT // 2 # All unused subcarriers
FIRST_DATA_INDICES   = FIRST_DATA_INDICES_CENTERED   + N_FFT // 2
LAST_DATA_INDICES    = FIRST_DATA_INDICES_CENTERED   + N_FFT // 2

# Modulation
MODULATIONS = ['BPSK', 'QPSK', '16-QAM']
BITS_PER_SYMBOL = {'BPSK': 1, 'QPSK': 2, '16-QAM': 4}

# Monte Carlo BER Simulation
EBNO_DB_RANGE      = (-10, 20) # Default Eb/N0 sweep range (dB)
N_TRIALS_PER_POINT = 2**10     # (=1024) Default number of Monte Carlo trials per Eb/N0 point
N_POINTS_PER_CURVE = 2**5      # (=32)   Default number of points in the Eb/N0 sweep for BER curves

CHANNEL_MODELS = {
    'A': {
        'delays': np.array([0, 10, 20, 30]),           # ns (converted to samples)
        'power': np.array([1.0, 0.8, 0.6, 0.4]),
        'description': 'Indoor small delay spread'
    },
    'B': {
        'delays': np.array([0, 10, 20, 35]),
        'power': np.array([1.0, 0.7, 0.5, 0.3]),
        'description': 'Indoor medium delay spread'
    },
    'C': {
        'delays': np.array([0, 15, 35, 60, 100, 150]),
        'power': np.array([1.0, 0.85, 0.7, 0.55, 0.4, 0.2]),
        'description': 'Indoor large delay spread'
    },
    'random': {
        'delays': lambda: np.array([0] + list(np.sort(np.random.normal(20, 5, 3)))),
        # ^ means: direct path, then 3 random reflections centered at 20ns with 5ns std dev
        'power': lambda: np.exp(-np.arange(4) / 1.5),  # exponential decay
        'description': 'Random delays (mean 20ns), random Rayleigh amplitudes'
    }
}

# Pilots
PILOT_SEQUENCE = np.array([ 1, 1, 1, 1,  -1,-1,-1, 1,  -1,-1,-1,-1,  1, 1,-1, 1,
                           -1,-1, 1, 1,  -1, 1, 1,-1,   1, 1, 1, 1,  1, 1,-1, 1,
                            1, 1,-1, 1,   1,-1,-1, 1,   1, 1,-1, 1, -1,-1,-1, 1,
                           -1, 1,-1,-1,   1,-1,-1, 1,   1, 1, 1, 1, -1,-1, 1, 1,
                           -1,-1, 1,-1,   1,-1, 1, 1,  -1,-1,-1, 1,  1,-1,-1,-1,
                           -1, 1,-1,-1,   1,-1, 1, 1,   1, 1,-1, 1, -1, 1,-1, 1,
                           -1,-1,-1,-1,  -1, 1,-1, 1,   1,-1, 1,-1,  1, 1 ,1,-1,
                           -1, 1,-1,-1,  -1, 1, 1, 1,  -1,-1,-1,-1, -1,-1,-1],
                           dtype=np.complex64)
