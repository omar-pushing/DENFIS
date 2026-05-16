# Training Workflow

## train_model.py — `main()`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PASS 1: fit_scaler_streaming(DATA_FILE)                            │
│  ─────────────────────────────────────────────────────────────────── │
│     Stream CSV in chunks of 100 000 rows                             │
│     For each chunk:                                                  │
│       • Drop rows with missing target or Severity ∉ [1,4]           │
│       • Clip Temperature, Visibility, Humidity to sensor bounds     │
│       • Track per-feature global min / max                          │
│       • Reservoir-sample up to 500 000 values per feature           │
│     After stream:                                                    │
│       • Compute per-feature median from reservoir                   │
│       • Fit MinMaxScaler on [global_min] and [global_max]           │
│     Returns: scaler, medians dict, n_total (valid row count)        │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  INSTANTIATE DENFIS                                                  │
│  ─────────────────────────────────────────────────────────────────── │
│     model = DENFIS(D_thr=0.30, sigma_scale=1.0,                     │
│                    lambda_forget=0.99, rls_init_cov=1000.0)         │
│     • Creates internal ECM instance (rule base grows from empty)    │
│     • RLS weight vector θ and covariance P start empty              │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PASS 2: stream_train(model, DATA_FILE, scaler, medians, n_total)   │
│  ─────────────────────────────────────────────────────────────────── │
│  Train budget = 80 % of n_total; last 20 % → held-out test set      │
│                                                                     │
│     Stream CSV again in chunks:                                     │
│       • Clean each chunk via _clean_chunk()                         │
│       • Normalise features with scaler.transform()                  │
│       • Normalise target: y_norm = (Severity - 1) / 3               │
│                                                                     │
│     For each row in train-budget portion:                           │
│       ┌─────────────────────────────────────────────────────────┐   │
│       │  model.train_one(x, y)                                  │   │
│       │  ─────────────────────────                               │   │
│       │  1. ECM.update(x)  ← cluster evolves or new rule born   │   │
│       │  2. If new cluster → _expand_rls() to grow θ and P      │   │
│       │  3. Build regression vector h = _regression_vector(x)   │   │
│       │  4. y_hat = hᵀ·θ                                        │   │
│       │  5. error = y − y_hat                                   │   │
│       │  6. RLS: θ ← θ + Kalman_gain · error                   │   │
│       │     P ← (P − gain·hᵀ·P) / λ                             │   │
│       │  7. Discard update if NaN detected                      │   │
│       └─────────────────────────────────────────────────────────┘   │
│                                                                     │
│     Remaining rows → accumulate X_test, y_test                      │
│                                                                     │
│     Returns: X_test, y_test                                         │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EVALUATE & DISPLAY                                                  │
│  ─────────────────────────────────────────────────────────────────── │
│     evaluate(model, X_test, y_test)                                  │
│       → MSE, RMSE, MAE, severity-scale MAE                          │
│     extract_rules(model, FEATURE_COLS, n_rules=3)                    │
│       → prints top 3 rules by coverage count                        │
│     Run one example prediction (72°F, 10 mi, 65%)                   │
│     save_model(model, scaler, metrics)                               │
│       → pickles bundle to model.pkl                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## test_model.py — `main()`

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. load_model("model.pkl") → model, scaler, metrics, meta          │
│  2. For each of the 12 predefined TEST_CASES:                       │
│       • Scale input through scaler                                  │
│       • model.predict_one(x_norm) → normalised severity             │
│       • Denormalise to 1–4 scale                                   │
│       • Identify winning rule via membership_vector()              │
│       • Print detailed per-case output                             │
│  3. Print summary table with pass/fail against expected ranges      │
│  4. extract_rules(model, ...)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## predict_severity.py — `predict()`

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Load model from model/model.pkl at import time                  │
│  2. predict(temp, vis, hum):                                        │
│       • scaler.transform([temp, vis, hum])                         │
│       • model.predict_one(scaled)  → severity [0,1]                │
│       • y * 3 + 1 → severity [1,4], clipped                        │
└─────────────────────────────────────────────────────────────────────┘
```
