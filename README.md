# DENFIS: Dynamic Evolving Neural-Fuzzy Inference System

### US Accidents Severity Prediction Web Application

**Instructor:** Dr. Mona Nagy ElBedwehy

---

## Overview

This project implements **DENFIS** (Dynamic Evolving Neural-Fuzzy Inference System) to predict the severity of traffic accidents in the US based on weather conditions. It consists of:

- **Flask Backend** — Serves the web UI and handles prediction requests
- **DENFIS Model** — Dynamic neural-fuzzy system using ECM and RLS
- **React Frontend** — User-friendly interface for severity prediction

## Architecture

```
DENFIS/
├── app.py                    # Flask web server
├── model/
│   ├── train_model.py        # DENFIS implementation (ECM + RLS)
│   ├── predict_severity.py   # Prediction API
│   └── model.pkl             # Trained model (generated)
├── templates/
│   └── index.html            # Main HTML entry point
├── static/                   # Compiled frontend assets
│   ├── index-*.js
│   ├── vendor-*.js
│   ├── motion-*.js
│   └── index-*.css
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+ (for frontend development)

### Installation

```bash
# Install Python dependencies
pip install flask numpy pandas scikit-learn

# (Optional) Train the model from scratch
# Download US_Accidents_March23.csv from Kaggle
# python -m model.train_model
```

### Running the Application

```bash
# Start the Flask server
python app.py
```

Open http://localhost:5000 in your browser.

---

## How It Works

### 1. DENFIS Model

The model uses two key algorithms:

| Algorithm | Purpose |
|-----------|---------|
| **ECM** (Evolving Clustering Method) | Dynamically creates fuzzy rules based on data distribution |
| **RLS** (Recursive Least Squares) | Online parameter optimization without gradient descent |

The model processes data **one sample at a time** (online learning), making it efficient for large datasets.

### 2. Features

The model takes three weather inputs:

| Feature | Range | Description |
|---------|-------|-------------|
| Temperature | -60°F to 160°F | Ambient temperature |
| Visibility | 0 to 10 miles | Road visibility |
| Humidity | 0% to 100% | Relative humidity |

### 3. Output

Predicted severity on a scale of **1 (Minor)** to **4 (Severe)**.

---

## Training (Optional)

To train the model from scratch:

1. Download the dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
2. Place `US_Accidents_March23.csv` in the project root
3. Run:

```bash
python -m model.train_model
```

This will generate `model/model.pkl` with the trained DENFIS model.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the web UI |
| `/assets/*` | GET | Serve static files |
| `/predict` | POST | Predict severity |

**Predict Request:**
```json
POST /predict
{
  "temperature": 72.0,
  "visibility": 10.0,
  "humidity": 65.0
}
```

**Response:**
```json
{
  "severity": 2.14
}
```

---

## Theory: Why DENFIS is "Third Generation"

| Generation | Example | Structure | Learning |
|------------|---------|-----------|----------|
| 1st | ANFIS | Fixed rules | Gradient descent |
| 2nd | FALCON | Pre-clustered, then frozen | Gradient descent |
| **3rd** | **DENFIS** | **Evolves online** | **RLS — no gradient needed** |

**DENFIS vs MLP:**

| Property | MLP | DENFIS |
|----------|-----|--------|
| Architecture | Fixed layers | Grows dynamically (ECM) |
| Learning | Back-propagation | Recursive Least Squares |
| Passes | Many epochs | **One pass** through data |
| Interpretability | Black-box weights | Readable IF-THEN rules |

---

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `D_thr` | 0.30 | Smaller → more rules (finer granularity) |
| `sigma_scale` | 1.0 | Wider Gaussians = softer rules |
| `lambda_forget` | 0.99 | < 1 = favor recent data |
| `rls_init_cov` | 1000 | Large = open prior for new rules |

---

## References

- Kasabov, N. & Song, Q. (2002). *DENFIS: Dynamic evolving neural-fuzzy inference system and its application for time-series prediction.* IEEE TFS, 10(2), 144–154.
- Moosavi, S. et al. (2019). *A Countrywide Traffic Accident Dataset.* arXiv:1906.05409.