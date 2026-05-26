"""
test_channel.py — Unit tests for ofdm_phy_sim.channel.

Tests cover:
    - apply_cawgn: noise variance correctness, output shape,
      seed reproducibility, ValueError for invalid N0,
      and passthrough behaviour at near-zero noise.
"""