# 📊 Data-Driven Algorithm for Dynamical System Analysis

This repository implements a data-driven algorithm to extract patterns, modes, or reduced representations from time-series simulation data. The approach is ideal for analysing dynamic systems where physical models are expensive or complex, such as CFD, structural vibrations, or thermal systems.

---

## 🎯 Project Objective

To use simulation or experimental data (e.g., from ANSYS or other solvers) and apply data-driven algorithms like **Dynamic Mode Decomposition (DMD)** to:
- Reduce dimensionality
- Extract dominant spatial-temporal modes
- Predict or reconstruct system behaviour

---

## 📁 Data Source

- ✅ Simulation Data from **ANSYS**
- Format: `.txt` or `.csv` (e.g., velocity, pressure, temperature vs time or space)
- Preprocessed into structured NumPy arrays before analysis

---

## 🔧 Techniques Used

| Method                     | Purpose                               |
|---------------------------|----------------------------------------|
| Dynamic Mode Decomposition (DMD) | Identify coherent structures in time-varying data |
| Singular Value Decomposition (SVD) | For dimensionality reduction and rank truncation |
| NumPy & Matplotlib         | Core scientific stack for Python      |

---

## 🛠️ Tools & Libraries

- Python 3.x
- NumPy
- Matplotlib
- SciPy

---

## 🚀 How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/data-driven-analysis.git
   cd data-driven-analysis
