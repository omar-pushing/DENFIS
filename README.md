# DENFIS — US Accidents Severity Prediction
### Soft Computing Project | Dr. Mona Nagy ElBedwehy

---

## Quick Start

```bash
# 1. Install dependencies (only standard scientific Python needed)
pip install numpy pandas scikit-learn

# 2. Download the dataset from Kaggle
#    https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
#    Place  US_Accidents_March23.csv  in the same folder as the script.

# 3. Train the model
python train_model.py

# 4. Run the test scenarios
python test_model.py
```

Expected runtime: ~3–6 minutes for 600 000 rows on a modern laptop.

---

## Theory: Why DENFIS is "Third Generation"

| Generation | Example    | Structure    | Learning          |
|------------|------------|--------------|-------------------|
| 1st        | ANFIS      | Fixed rules  | Gradient (ANFIS uses hybrid LSE+BP) |
| 2nd        | FALCON     | Pre-clustered, then frozen | Gradient |
| **3rd**    | **DENFIS** | **Evolves online** | **RLS — no gradient needed** |

**DENFIS vs. MLP (your textbook):**

| Property              | MLP (Ch. 4-5 of your book)         | DENFIS                          |
|-----------------------|-------------------------------------|---------------------------------|
| Architecture          | Fixed layers & neuron count         | Grows dynamically (ECM)         |
| Learning rule         | Back-propagation / gradient descent | Recursive Least Squares (RLS)   |
| Number of passes      | Many epochs required                | **One pass** through data       |
| Interpretability      | Black-box weights                   | Each cluster = readable IF-THEN rule |
| Rule extraction       | Not possible                        | Built-in (see extract_rules)    |

---

## Algorithm Flow

```
For each sample (x, y):
  ┌──────────────────────────────────────────────────┐
  │  ECM Layer                                        │
  │  Find nearest cluster centre c_k                 │
  │  if dist(x, c_k) > D_thr  →  CREATE new rule    │
  │  else                      →  UPDATE cluster     │
  └──────────────────────────────────────────────────┘
               ↓
  ┌──────────────────────────────────────────────────┐
  │  Fuzzy Inference (Takagi-Sugeno)                 │
  │  μ_k(x) = Gaussian(x, centre_k, σ_k)            │
  │  φ_k    = μ_k / Σμ_j          (normalise)       │
  │  ŷ_k    = w_k0 + w_k·x        (TS consequent)   │
  │  ŷ      = Σ φ_k · ŷ_k         (crisp output)    │
  └──────────────────────────────────────────────────┘
               ↓
  ┌──────────────────────────────────────────────────┐
  │  RLS Update                                      │
  │  h  = [φ_k · [1, x]]  for all k                 │
  │  K  = P·h / (λ + h·P·h)                         │
  │  θ  = θ + K·(y − ŷ)                             │
  │  P  = (P − K·h·P) / λ                           │
  └──────────────────────────────────────────────────┘
```

---

## Key Hyperparameters

| Parameter      | Default | Effect |
|----------------|---------|--------|
| `D_thr`        | 0.25    | Smaller → more rules (finer) |
| `sigma_scale`  | 1.0     | Wider Gaussians = softer rules |
| `lambda_forget`| 0.99    | < 1 = favour recent data |
| `rls_init_cov` | 1000    | Large = open prior for new rules |

---

## Sample Output

```
[TRAIN] Starting DENFIS online training on 480,000 samples …
        Step  50,000 / 480,000  |  Rules so far:   41
        Step 100,000 / 480,000  |  Rules so far:   47
        ...
[TRAIN] Training complete.  Total rules: 52

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

Rule #2  (cluster 9, covers 14,832 samples, radius=0.2158)
  IF   Temperature(F) is Cold AND
       Visibility(mi) is Low AND
       Humidity(%) is High
  THEN Severity = 0.4102 + -0.0312·Temp + -0.3821·Vis + 0.1243·Hum
       → Severity at cluster centre ≈ 2.87

Rule #3  (cluster 17, covers 11,003 samples, radius=0.1984)
  IF   Temperature(F) is Hot AND
       Visibility(mi) is Very Low AND
       Humidity(%) is High
  THEN Severity = 0.6234 + 0.0821·Temp + -0.5012·Vis + 0.0341·Hum
       → Severity at cluster centre ≈ 3.21
```

---

## References

- Kasabov, N. & Song, Q. (2002). *DENFIS: Dynamic evolving neural-fuzzy inference system and its application for time-series prediction.* IEEE TFS, 10(2), 144–154.
- Moosavi, S. et al. (2019). *A Countrywide Traffic Accident Dataset.* arXiv:1906.05409.
