# ofdm_phy_sim

End-to-end OFDM transceiver simulation with multipath fading, channel 
estimation, and BER vs. Eb/N0 analysis — implemented in Python/NumPy.

## Overview

This project simulates a complete OFDM PHY chain:

- **Modulation:** BPSK, QPSK, 16-QAM constellation mapping/demapping
- **Transmitter:** serial-to-parallel conversion, IFFT, cyclic prefix insertion
- **Channel:** AWGN and frequency-selective multipath fading models
- **Receiver:** cyclic prefix removal, FFT, pilot-based channel estimation
- **Equalization:** Least Squares (LS) and MMSE
- **Analysis:** BER vs. Eb/N0 curves with theoretical reference

## Structure

ofdm_phy_sim/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── ofdm_phy_sim/
│   ├── __init__.py
│   ├── modem.py            # Constellation mapping/demapping (BPSK, QPSK, 16-QAM)
│   ├── transmitter.py      # OFDM TX: serial→parallel, IFFT, CP insertion
│   ├── channel.py          # AWGN + multipath fading models
│   ├── receiver.py         # CP removal, FFT, equalization
│   ├── equalizer.py        # LS and MMSE channel estimation
│   └── utils.py            # BER computation, Eb/N0 helpers, bit generation
├── simulations/
│   ├── ber_awgn.py         # BER vs Eb/N0 over AWGN
│   └── ber_multipath.py    # BER vs Eb/N0 over multipath channel
├── plots/                  # output figures (gitignored or committed)
├── tests/                  # Tests (quality and speed)
|   ├── test_all.py         # Run all tests
│   ├── test_channel.py     
|   ├── test_equalizer.py
|   ├── test_transceiver.py # Test the whole chain from bit to bit 
|   ├── test_modem.py
|   ├── test_receiver.py
|   ├── test_transmitter.py
|   └── test_utils.py

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python simulations/ber_awgn.py
python simulations/ber_multipath.py
```

## Planned

- Pilot insertion and interpolation-based channel estimation
- MIMO extension (2x2, MRC combining)
- ML-based channel estimation / equalization

## License

MIT