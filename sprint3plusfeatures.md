# Sprint 3+ Features for ofdm_phy_sim

## Sprint 3: Performance & Optimization

### CPU/GPU Auto-Detection Benchmark
- Implement `run_ber_simulation(use_gpu=None)` that auto-detects CUDA availability
- Benchmark both CPU and GPU implementations on same workload
- Choose faster backend at runtime
- Print timing comparison
- **Justification:** Determine if CuPy GPU acceleration is worth complexity on your ThinkPad T470p

### GPU Acceleration (CuPy)
- Replace NumPy with CuPy in BER simulation paths
- Minimal code changes — CuPy is a drop-in NumPy replacement
- Focus on high-data-volume operations (modulate, noise, demodulate)
- **Note:** May not be faster than CPU+JIT on this hardware; benchmark will tell

### Numba JIT Compilation
- Add `@njit` decorator to `compute_ber()` inner loops
- Target: modulate → AWGN → demodulate tight loop
- Expected speedup: 10-50x

### joblib Parallelization
- Parallelize BER sweep across Eb/N0 points using `joblib.Parallel`
- Each point is independent → linear scaling with CPU cores
- Expected speedup: ~4x (on 4-core i7-6700HQ)

**Sprint 3 Total Expected Speedup:** 10-100x (JIT + parallelization + GPU if beneficial)

---

## Sprint 4-5: Protocol Stack Features

### 64-QAM Modulation
- Add Grey-coded 64-QAM (8×8 constellation)
- 6 bits per symbol
- Normalize to unit energy
- Add to `modem.py` constellation maps
- Update tests

### Frame Structure Classes (OOP Refactor)
- `OFDMSymbol` — wraps subcarrier layout
- `PPDUHeader` — rate, length fields
- `TxFrame` / `RxFrame` — pipeline objects
- Move from bare functions to cleaner OOP API
- **Scope:** Keep focused on PHY layer, not full protocol stack

### APSK Modulation
- Amplitude Phase Shift Keying with ring structure (from your bachelor thesis background)
- Configurable ring radii
- Grey coding
- Round-trip test

---

## Sprint 5-6: Advanced DSP & ML Extensions

### Convolutional FEC Encoder & Viterbi Decoder
- Convolutional encoder (rate 1/2, constraint length 7 typical)
- Viterbi decoder for hard-decision decoding
- Coded BER simulation
- ~8 story points

### Bit Interleaver & Deinterleaver
- Interleaving pattern matching 802.11a specification
- Spread burst errors across codeword
- Depends on FEC implementation
- ~3 story points

### Soft LLR Demodulation
- Log-likelihood ratio output per bit (not hard-decision)
- Requires noise variance
- Viterbi decoder fed with soft values
- Outperforms hard-decision at low SNR
- Depends on FEC
- ~5 story points

### 5G NR Numerology Profile
- Multiple subcarrier spacings (15 kHz, 30 kHz, 60 kHz, 120 kHz)
- Flexible resource grid
- LDPC instead of convolutional codes
- Refactor `constants.py` to support profiles
- ~3 story points

### MIMO 2×2 with MRC Combining
- 2 transmit antennas, 2 receive antennas
- Maximum Ratio Combining on receiver
- Spatial diversity gain in BER curves
- ~8 story points

### Probabilistic Constellation Shaping
- Non-uniform symbol probabilities (from your 802.11 knowledge)
- Approach Shannon limit
- Optimized bit-to-symbol mapping
- Depends on APSK
- ~8 story points

### Autoencoder-Based Learned Constellation
- End-to-end learned TX/RX via autoencoder (ML extension)
- Trained over AWGN channel
- Visualize learned constellation
- Compare BER vs classical schemes
- ~8 story points

---

## Implementation Notes

- **GPU work (Sprint 3):** Only proceed if benchmark shows GPU > CPU. Otherwise focus on JIT + parallelization.
- **FEC (Sprint 5):** Gateway to soft demodulation and interleaving; prioritize if time permits.
- **ML extensions (Sprint 6):** Use PyTorch; can be showcased as separate notebook.
- **5G NR (Sprint 6):** Lower priority; demonstrates extensibility but not core to 802.11a focus.
