"""
equalizer.py — Channel estimation and equalization for ofdm_phy_sim.

Implements pilot-based frequency-domain channel estimation and
one-tap equalization for OFDM systems.

Estimation methods:
    - Least Squares (LS): simple pilot-based estimate, no noise regularisation
    - MMSE: minimum mean square error estimate using noise variance

Equalization:
    - One-tap frequency-domain equalizer: divides each subcarrier by
      its estimated channel coefficient

Designed to be swappable — equalizer variants can be compared
independently of the transmitter and receiver chain.
"""