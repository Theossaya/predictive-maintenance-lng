# Predictive Maintenance Portfolio — CMAPSS & PRONOSTIA

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

End-to-end predictive maintenance pipeline for industrial machinery degradation modelling, using two internationally recognised benchmark datasets. Implements the complete workflow: raw sensor ingestion → signal processing → feature engineering → model training → evaluation.

**Datasets used:**
- **NASA CMAPSS** — turbofan engine run-to-failure data (Remaining Useful Life regression)
- **FEMTO-ST PRONOSTIA** — bearing vibration accelerometer data (failure detection classification)

> The PRONOSTIA dataset was created at **FEMTO-ST's AS2M department**, Besançon — a world-leading lab in mechatronics and micro-robotics research.

---

## Results

### RUL Estimation — CMAPSS (NASA Turbofan Engines)

| Model | RMSE (cycles) | MAE (cycles) | vs. Benchmark (45 cycles) |
|-------|:---:|:---:|:---:|
| **LSTM** | **15.93** | **11.66** | ✅ **65% better** |
| CNN | 19.54 | 16.36 | ✅ 57% better |
| Random Forest | 21.79 | 16.63 | ✅ 52% better |
| Weighted Ensemble | 22.48 | 19.15 | ✅ 50% better |

*Evaluated on held-out test set using time-series aware split (engine unit level)*

### Bearing Failure Detection — PRONOSTIA

| Model | Accuracy | Precision | Recall |
|-------|:---:|:---:|:---:|
| **Random Forest** | **99.39%** | 97.9% | 95.9% |
| Neural Network | 98.98% | 90.2% | 93.9% |
| Gradient Boosting | 98.37% | — | — |

---

## Technical Implementation

### Signal Processing Pipeline (PRONOSTIA)

Raw accelerometer data sampled at **25,600 Hz** is processed per-file into 26 features:

**Time domain:** RMS, kurtosis, skewness, entropy, crest factor, peak-to-peak  
**Frequency domain:** FFT band energies across three bands:
- Band 1: 0–1 kHz (structural resonance)
- Band 2: 1–5 kHz (bearing defect characteristic frequencies)
- Band 3: 5–10 kHz (high-frequency wear signatures)

### RUL Prediction Pipeline (CMAPSS)

```
Raw sensor data (26 channels, 100 engines, 20,631 readings)
    ↓
Feature selection (13 informative sensors via correlation analysis)
    ↓
RUL labelling with piecewise linear cap at 125 cycles
    ↓
Sequence creation (window = 30 cycles) → 17,631 sequences
    ↓
Time-series aware train/test split (engine-unit level)
    ↓
LSTM / CNN / Random Forest training
    ↓
Weighted ensemble (LSTM: 39.3%, CNN: 32.0%, RF: 28.7%)
```

### Architecture Details

**LSTM Model:**
- Input: (30 time steps × 13 features)
- Architecture: LSTM layers with dropout regularisation
- Training: Early stopping, ReduceLROnPlateau callback
- Optimiser: Adam (lr=0.001 → 0.0001)

**Neural Network (Bearing Classification):**
- Architecture: 256 → 128 → 64 → 32 → 1 (sigmoid)
- Dropout: 0.4 / 0.3 / 0.2
- Class weighting to handle imbalance (10:1 ratio)
- Final validation AUC: 0.9879

### Key Feature Correlations with RUL (CMAPSS)

Top sensors by absolute Pearson correlation:
1. `sensor_12`: 0.749
2. `sensor_7`: 0.733
3. `sensor_20`: 0.705

### PRONOSTIA — Feature Changes: Healthy → Failing State

| Feature | Healthy Mean | Failing Mean | Change |
|---------|:---:|:---:|:---:|
| FFT Band 2 (Horiz) | 100.66 | 266.91 | +165% |
| Kurtosis (Horiz) | 1.82 | 4.xx | significant |
| RMS (Vert) | baseline | elevated | +ve |

*RMS_Vert_accel is the single most predictive feature (p-value: 3.82×10⁻¹⁰⁵)*

---

## Project Structure

```
predictive-maintenance-portfolio/
├── predictive_maintenance_portfolio.ipynb   # Main notebook (all code + results)
├── README.md                                # This file
├── requirements.txt                         # Dependencies
└── results/                                 # Exported plots (auto-generated)
    ├── cmapss_rul_trends.png
    ├── feature_correlations.png
    ├── temporal_degradation.png
    ├── pronostia_signal_analysis.png
    └── model_performance_comparison.png
```

> **Note:** The raw datasets are not included in this repository due to size.
> - CMAPSS: Download from [NASA Prognostics Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
> - PRONOSTIA: Download from [IEEE DataPort / FEMTO-ST](https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge)

---

## Installation

```bash
git clone https://github.com/Theossaya/predictive-maintenance-lng.git
cd predictive-maintenance-lng
pip install -r requirements.txt
jupyter notebook predictive_maintenance_portfolio.ipynb
```

**requirements.txt:**
```
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
tensorflow>=2.8
scipy>=1.7
jupyter>=1.0
xgboost>=1.5
```

---

## Relevance to Industrial Applications

This work directly maps to real industrial use cases:

| Application | Dataset | Technique |
|-------------|---------|-----------|
| Aircraft engine maintenance scheduling | CMAPSS | LSTM RUL prediction |
| Wind turbine bearing monitoring | PRONOSTIA | FFT + failure classification |
| Predictive maintenance in manufacturing | Both | Ensemble modelling |
| Condition monitoring systems | PRONOSTIA | Real-time vibration analysis |

---

## Limitations & Notes

- PRONOSTIA classification trained on a single bearing (Bearing3_2) due to local data availability. Full generalisation would require training across all 6 bearings in the learning set.
- CMAPSS FD001 used (single operating condition). Extension to FD002–FD004 (multiple conditions) is a natural next step.
- Models saved to `../models/` directory during training (not included in repo).

---

## Author

**Eric Oghenefejiro Favour**
- GitHub: [@Theossaya](https://github.com/Theossaya)
- LinkedIn: [eric-favour-1459ht](https://linkedin.com/in/eric-favour-1459ht)
- IEEE Publication: [10645268](https://ieeexplore.ieee.org/document/10645268)

## References

1. Saxena, A. et al. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. *NASA CMAPSS Dataset.*
2. Nectoux, P. et al. (2012). PRONOSTIA: An Experimental Platform for Bearings Accelerated Degradation Tests. *IEEE PHM 2012 Data Challenge. FEMTO-ST Institute.*
3. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation.*
