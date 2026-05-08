# DENFIS Project Documentation

## Full Technical Reference Guide

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Running the Application](#running-the-application)
5. [DENFIS Algorithm Deep Dive](#denfis-algorithm-deep-dive)
6. [Model Training](#model-training)
7. [API Reference](#api-reference)
8. [Frontend Integration](#frontend-integration)
9. [Project Structure](#project-structure)
10. [Hyperparameters Reference](#hyperparameters-reference)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

## Project Overview

### What is DENFIS?

**DENFIS** (Dynamic Evolving Neural-Fuzzy Inference System) is a third-generation neuro-fuzzy system developed by Kasabov and Song (2002). Unlike traditional neural networks with fixed architectures, DENFIS evolves its structure dynamically during online learning.

### Key Characteristics

| Property | Description |
|----------|-------------|
| **Online Learning** | Processes data sample-by-sample in a single pass |
| **Dynamic Architecture** | Automatically creates and updates fuzzy rules |
| **No Gradient Descent** | Uses Recursive Least Squares (RLS) for parameter optimization |
| **Interpretable** | Extracts human-readable IF-THEN rules |

### Project Context

This implementation predicts **traffic accident severity** (1-4 scale) based on three weather conditions:

- **Temperature (°F)**: Ambient air temperature
- **Visibility (miles)**: Distance at which objects are visible
- **Humidity (%)**: Relative humidity percentage

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                         │
│                    (React + Vite, served by Flask)             │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP POST /predict
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Flask Backend (app.py)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Endpoint: POST /predict                                │   │
│  │  - Validates input JSON                                │   │
│  │  - Calls predict_severity.predict()                   │   │
│  │  - Returns JSON response                               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │ function call
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Prediction Module (predict_severity.py)         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - Loads model.pkl on startup (cached)                  │   │
│  │  - Normalizes input using fitted MinMaxScaler          │   │
│  │  - Calls model.predict_one()                           │   │
│  │  - Denormalizes output to 1-4 scale                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │ prediction
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              DENFIS Model (trained, stored in model.pkl)       │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │  ECM (Clusters)   │  │  RLS (Weights)   │                  │
│  │  - Rule centers  │  │  - Theta params  │                  │
│  │  - Rule radii    │  │  - Covariance P  │                  │
│  └──────────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| Web Server | `app.py` | HTTP routing, request handling |
| Prediction API | `predict_severity.py` | Input normalization, model invocation, output denormalization |
| DENFIS Core | `train_model.py` | ECM clustering, RLS optimization, fuzzy inference |
| Trained Model | `model.pkl` | Persisted model state (pickle) |

---

## Installation & Setup

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | ≥ 3.8 | Runtime environment |
| pip | Latest | Package management |
| Git | Any | Version control (optional) |

### Python Dependencies

Install all required packages:

```bash
pip install flask numpy pandas scikit-learn
```

| Package | Version | Purpose |
|---------|---------|---------|
| **flask** | ≥ 2.0 | Web server and routing |
| **numpy** | ≥ 1.20 | Numerical computations |
| **pandas** | ≥ 1.3 | CSV processing, data handling |
| **scikit-learn** | ≥ 1.0 | MinMaxScaler preprocessing |

### Dataset Preparation (Optional)

To train the model from scratch:

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
2. Save as `US_Accidents_March23.csv` in the project root
3. Run training: `python -m model.train_model`

---

## Running the Application

### Starting the Server

```bash
python app.py
```

Expected output:
```
 * Running on http://localhost:5000
```

### Accessing the Application

Open your browser and navigate to:

```
http://localhost:5000
```

### Testing the API

Using curl:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 72.0, "visibility": 10.0, "humidity": 65.0}'
```

Response:
```json
{
  "severity": 2.14
}
```

---

## DENFIS Algorithm Deep Dive

### Overview

DENFIS combines two key algorithms:

1. **ECM (Evolving Clustering Method)** — Manages the rule base
2. **RLS (Recursive Least Squares)** — Optimizes consequent parameters

### 1. Evolving Clustering Method (ECM)

ECM dynamically creates fuzzy rules based on input data distribution.

#### Algorithm

```
For each input sample x:
    if no clusters exist:
        Create first cluster with center = x
    else:
        Find nearest cluster center c_k
        Calculate distance d = ||x - c_k||
        
        if d > D_thr (threshold):
            CREATE new cluster with center = x
        else:
            UPDATE cluster center: c_k = (c_k * n + x) / (n + 1)
            UPDATE cluster radius if d > current radius
```

#### Key Properties

| Property | Description |
|----------|-------------|
| `D_thr` | Distance threshold for creating new rules |
| `centres` | List of cluster centers (rule antecedents) |
| `radii` | Coverage radius of each cluster |
| `counts` | Number of samples assigned to each cluster |

#### Gaussian Membership Function

For Takagi-Sugeno inference, each cluster computes membership:

```
μ_k(x) = exp(-||x - c_k||² / (2 * σ_k²))
```

Where σ_k = max(radius_k, 1e-6) × sigma_scale

### 2. Recursive Least Squares (RLS)

RLS optimizes the consequent parameters of Takagi-Sugeno rules.

#### Takagi-Sugeno Form

Each rule has the form:

```
IF x is in cluster k
THEN y = w_k0 + w_k1*x_1 + w_k2*x_2 + ... + w_kn*x_n
```

Where:
- `w_k0` is the bias (constant term)
- `w_k1...w_kn` are feature weights

#### Regression Vector

For input x with K clusters and n features:

```
h = [φ_0, φ_0*x_1, ..., φ_0*x_n, φ_1, φ_1*x_1, ...]
```

Where φ_k = μ_k(x) / Σ_j(μ_j(x)) (normalized firing strength)

#### RLS Update Equations

```
Prediction: ŷ = h · θ

Error: e = y - ŷ

Kalman Gain: K = P · h / (λ + h · P · h)

Parameter Update: θ = θ + K · e

Covariance Update: P = (P - K · h · P) / λ
```

| Symbol | Description |
|--------|-------------|
| θ | Parameter vector (weights) |
| P | Covariance matrix |
| λ | Forgetting factor (0 < λ ≤ 1) |
| K | Kalman gain vector |

### 3. Complete DENFIS Flow

```
For each sample (x, y):
    
    ┌─────────────────────────────────────┐
    │  ECM Layer                         │
    │  - Find nearest cluster            │
    │  - Create or update rule          │
    │  - Update radii                   │
    └─────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │  Fuzzy Inference                   │
    │  - Compute membership μ_k(x)       │
    │  - Normalize to φ_k                │
    │  - Build regression vector h       │
    │  - Compute prediction ŷ = h·θ     │
    └─────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │  RLS Update                         │
    │  - Calculate error e = y - ŷ      │
    │  - Compute Kalman gain K           │
    │  - Update parameters θ             │
    │  - Update covariance P             │
    └─────────────────────────────────────┘
```

### 4. Rule Extraction

After training, interpretable rules can be extracted:

```
Rule #1 (cluster 3, covers 18,241 samples)
  IF  Temperature is Mild 
      AND Visibility is High 
      AND Humidity is Moderate
  THEN Severity = 0.2341 + 0.1823·Temp + -0.0412·Vis + 0.2011·Hum
       → Severity at center ≈ 2.14
```

Linguistic labels are derived from normalized values:

| Feature | < 0.30 | < 0.55 | < 0.75 | ≥ 0.75 |
|---------|--------|--------|--------|--------|
| Temperature | Cold | Mild | Warm | Hot |
| Visibility | Very Low | Low | Moderate | High |
| Humidity | Low | Moderate | High | High |

---

## Model Training

### Training Pipeline

The training process consists of three passes:

#### Pass 1: Data Scanning

- Stream through entire dataset
- Compute min/max for each feature (for MinMaxScaler)
- Collect reservoir sample for median calculation
- Output: Fitted scaler, median values, total row count

#### Pass 2: Online Training

- First 80% of data → training set
- Last 20% of data → test set
- Process samples one-by-one through DENFIS
- Log progress every 50,000 samples
- Output: Trained model, test set

#### Pass 3: Evaluation

- Predict on test set
- Calculate MSE, RMSE, MAE
- Extract and display sample rules

### Training Command

```bash
python -m model.train_model
```

### Expected Output

```
[PASS 1/3]  Scanning 'US_Accidents_March23.csv' …
            (reads every chunk once to fit the scaler)
  … 480,000 rows scanned
[PASS 1/3]  Done.  Valid rows: 480,000
            Medians : {'Temperature(F)': 62.0, 'Visibility(mi)': 10.0, 'Humidity(%)': 65.0}
            Ranges  : {'Temperature(F)': (-58.0, 120.0), 'Visibility(mi)': (0.0, 10.0), 'Humidity(%)': (0.0, 100.0)}

[PASS 2/3]  Streaming training …
            D_thr=0.3  sigma=1.0  lambda=0.99
            Train budget : 384,000 rows (80 % of 480,000)
            Test set     : ~96,000 rows (last 20 %)
  Chunk    1 (~100,000 raw rows)  |  Trained:  100,000  |  Rules:   35  |  26.0%
  Chunk    2 (~200,000 raw rows)  |  Trained:  200,000  |  Rules:   41  |  52.1%
  Chunk    3 (~300,000 raw rows)  |  Trained:  300,000  |  Rules:   47  |  78.1%
  Chunk    4 (~400,000 raw rows)  |  Trained:  384,000  |  Rules:   52  | 100.0%
[PASS 2/3]  Done.
            Rows trained : 384,000
            Test rows    : 96,000
            Fuzzy rules  : 52

─────────────────────────────────────────────
  DENFIS Performance Report
─────────────────────────────────────────────
  MSE  (normalised)      : 0.0312
  RMSE (normalised)      : 0.1766
  MAE  (normalised)      : 0.1401
  MAE  (Severity 1–4)    : 0.42
  Total fuzzy rules      : 52
─────────────────────────────────────────────

=================================================================
  DENFIS EXTRACTED RULES  (showing 3 of 52 rules)
=================================================================

  Rule #1  (cluster 3, covers 18,241 samples, radius=0.2301)
  IF   Temperature(F) is Mild AND
       Visibility(mi) is High AND
       Humidity(%) is Moderate
  THEN Severity = 0.2341 + 0.1823·Temp + -0.0412·Vis + 0.2011·Hum
       → Severity at cluster centre ≈ 2.14

[SAVE]  Model exported → 'model.pkl'  (24.5 KB)
        Bundled: DENFIS (52 rules) + scaler + metrics

[DONE]  Run  python test_model.py  to test the model.
```

### Model Serialization

The trained model is saved as a pickle file containing:

```python
{
    "model": DENFIS,          # Trained model object
    "scaler": MinMaxScaler,   # Feature normalizer
    "metrics": dict,          # MSE, RMSE, MAE scores
    "meta": {
        "n_rules": int,       # Number of fuzzy rules
        "n_features": int,   # Input dimension (3)
        "feature_cols": list,# Feature names
        "target_col": str,   # Target column name
        "D_thr": float,      # Distance threshold
        "sigma_scale": float,# Gaussian width multiplier
        "lambda": float,     # Forgetting factor
    }
}
```

---

## API Reference

### Endpoints

#### GET /

Serves the main HTML page.

| Property | Value |
|----------|-------|
| URL | `http://localhost:5000/` |
| Method | GET |
| Response | HTML (index.html) |

#### GET /assets/<filename>

Serves static files (JavaScript, CSS).

| Property | Value |
|----------|-------|
| URL | `http://localhost:5000/assets/<filename>` |
| Method | GET |
| Response | JavaScript or CSS file |

#### POST /predict

Predicts accident severity from weather inputs.

| Property | Value |
|----------|-------|
| URL | `http://localhost:5000/predict` |
| Method | POST |
| Content-Type | application/json |

**Request Body:**

```json
{
  "temperature": 72.0,
  "visibility": 10.0,
  "humidity": 65.0
}
```

| Field | Type | Range | Required |
|-------|------|-------|----------|
| temperature | float | -60 to 160 | Yes |
| visibility | float | 0 to 10 | Yes |
| humidity | float | 0 to 100 | Yes |

**Response:**

```json
{
  "severity": 2.14
}
```

| Field | Type | Description |
|-------|------|-------------|
| severity | float | Predicted severity (1.0 to 4.0) |

**Error Response (400):**

```json
{
  "error": "Missing required field"
}
```

---

## Frontend Integration

### Static Files

The frontend consists of compiled assets served from the `static/` directory:

| File | Description |
|------|-------------|
| `index-*.js` | Main React application bundle |
| `vendor-*.js` | Third-party dependencies (React, etc.) |
| `motion-*.js` | Animation library |
| `index-*.css` | Compiled CSS styles |

### Loading Mechanism

1. Browser loads `index.html`
2. HTML references assets at `/assets/<filename>`
3. Flask serves files from the `static/` folder

### Rebuilding Frontend

To rebuild the frontend (requires source):

```bash
# Navigate to frontend source
cd DENFIS-Site

# Install dependencies
npm install

# Build for production
npm run build

# Copy output to static folder
cp dist/assets/* ../static/
```

---

## Project Structure

```
DENFIS/
├── app.py                      # Flask application entry point
├── model/
│   ├── __init__.py
│   ├── train_model.py          # DENFIS implementation + training
│   ├── predict_severity.py     # Prediction function
│   ├── test_model.py           # Evaluation script
│   └── model.pkl               # Trained model (generated)
├── templates/
│   └── index.html              # HTML entry point
├── static/                     # Frontend assets
│   ├── index-*.js
│   ├── index-*.css
│   ├── vendor-*.js
│   └── motion-*.js
├── US_Accidents_March23.csv   # Dataset (user-provided)
├── README.md                   # Quick start guide
└── DOCUMENTATION.md           # This file
```

---

## Hyperparameters Reference

### DENFIS Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `D_thr` | 0.30 | 0.1 - 1.0 | Distance threshold for new rule creation. Smaller = more rules |
| `sigma_scale` | 1.0 | 0.5 - 2.0 | Multiplier for Gaussian width. Larger = softer membership |
| `lambda_forget` | 0.99 | 0.9 - 1.0 | Forgetting factor. < 1 gives more weight to recent data |
| `rls_init_cov` | 1000 | 100 - 10000 | Initial covariance for new rules. Larger = more adaptation |

### Data Processing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 100,000 | Rows processed per CSV chunk |
| `FEATURE_CLIP` | (see below) | Outlier clipping ranges |
| `DTYPE_MAP` | float32/int8 | Memory-efficient data types |

**Feature Clipping Ranges:**

| Feature | Min | Max |
|---------|-----|-----|
| Temperature(F) | -60.0 | 160.0 |
| Visibility(mi) | 0.0 | 10.0 |
| Humidity(%) | 0.0 | 100.0 |

---

## Troubleshooting

### Issue: 404 on Static Files

**Symptom:** Browser shows 404 for `/assets/*.js` or `/assets/*.css`

**Cause:** Assets in wrong folder

**Solution:**
```bash
# Ensure Flask static configuration is correct in app.py:
app = Flask(__name__, static_url_path='/assets', static_folder='static')

# Verify files exist in static/ folder:
ls static/
```

### Issue: Model Not Found

**Symptom:** `FileNotFoundError: model.pkl not found`

**Cause:** Model file doesn't exist

**Solution:**
```bash
# Train the model:
python -m model.train_model

# Or download pre-trained model
```

### Issue: Lower Not Defined (JavaScript Error)

**Symptom:** `Uncaught ReferenceError: lower is not defined`

**Cause:** Frontend build has errors

**Solution:** Rebuild frontend from source:
```bash
cd DENFIS-Site
npm run build
cp dist/assets/* ../static/
```

### Issue: Prediction Returns NaN

**Symptom:** `severity: NaN` in response

**Cause:** Input values outside training range

**Solution:** Ensure inputs are within valid ranges:
- Temperature: -60 to 160
- Visibility: 0 to 10
- Humidity: 0 to 100

### Issue: CORS Errors

**Symptom:** Cross-origin request blocked

**Cause:** Browser blocks requests from different origin

**Solution:** Add CORS headers to Flask (if needed):
```python
from flask_cors import CORS
CORS(app)
```

---

## References

### Primary Sources

1. **Kasabov, N. & Song, Q. (2002).** DENFIS: Dynamic evolving neural-fuzzy inference system and its application for time-series prediction. *IEEE Transactions on Fuzzy Systems*, 10(2), 144-154.

2. **Moosavi, S. et al. (2019).** A Countrywide Traffic Accident Dataset. *arXiv:1906.05409*.

### Related Work

- **ANFIS** (Jang, 1993) — First-generation neuro-fuzzy, fixed architecture
- **FALCON** (Lin & Lee, 1991) — Second-generation, pre-clustered rules
- **DENFIS** — Third-generation, online evolving architecture

### Dataset

US Accidents Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

---

## Appendix: Mathematical Formulas

### Normalization

```
x_norm = (x - x_min) / (x_max - x_min)
```

### Denormalization

```
y_severity = y_norm * 3.0 + 1.0  # Maps [0,1] → [1,4]
```

### Euclidean Distance

```
d(a, b) = sqrt(sum((a_i - b_i)²))
```

### Gaussian Membership

```
μ(x, c, σ) = exp(-d(x, c)² / (2σ²))
```

### RLS Parameter Update

```
θ_new = θ_old + K * (y - h·θ_old)
K = P·h / (λ + h·P·h)
P_new = (P - K·h·P) / λ
```

---

*Last Updated: May 2026*
*Version: 1.0*