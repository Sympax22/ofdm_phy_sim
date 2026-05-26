"""
channel.py — Applies the channel to the modulated symbols.

Can be CAWGN, fading (typically flat-fading), frequency-selective etc.
Channel estimation is part of the receiver/equalizer.
"""

import numpy as np

def apply_cawgn(raw_symbols: np.ndarray, N0: float, seed=-1) -> np.ndarray:
    """
    Add complex white gaussian noise element-wise to ```raw_symbols``` with variance ```N0```.

    :param raw_symbols: Flat array or matrix of complex numbers.
    :param N0: specify the complex noise variance E[|z|²] of the CAWGN.
    :param seed: (optional) specify the seed with which to generate the random numbers for reproducibility. If negative, unknown seed is used.
    :returns: Complex symbol array of shape (n_symbols,).
    :rtype: np.ndarray
    :raises ValueError: If N0 is <= 0.
    """
    if N0 <= 0.0:
        raise ValueError(f"ERROR: In apply_cawgn: expected noise variance N0 > 0 but was {N0}.")
    
    rng = np.random.default_rng(seed if seed >= 0 else None)
    
    cawgn = rng.standard_normal(raw_symbols.shape) \
    + 1j * rng.standard_normal(raw_symbols.shape)

    cawgn *= np.sqrt(0.5*N0)
    
    return raw_symbols + cawgn