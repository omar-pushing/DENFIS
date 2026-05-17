# Import core libraries for data processing and modeling
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# File and memory configuration
DATA_FILE  = "US_Accidents_March23.csv"
CHUNK_SIZE = 100_000

FEATURE_COLS = ["Temperature(F)", "Visibility(mi)", "Humidity(%)"]
TARGET_COL   = "Severity"

# Sensor outlier clipping ranges
FEATURE_CLIP = {
    "Temperature(F)": (-60.0, 160.0),
    "Visibility(mi)": (  0.0,  10.0),
    "Humidity(%)":    (  0.0, 100.0),
}

DTYPE_MAP = {
    "Temperature(F)": "float32",
    "Visibility(mi)": "float32",
    "Humidity(%)":    "float32",
    "Severity":       "int8",
}

# PASS 1: Stream through data to find global min/max and medians
def fit_scaler_streaming(filepath: str) -> tuple:
    print(f"\n[PASS 1/3]  Scanning '{filepath}' …")
    print( "            (reads every chunk once to fit the scaler)")

    col_min    = {c:  float("inf") for c in FEATURE_COLS}
    col_max    = {c: float("-inf") for c in FEATURE_COLS}
# Storage for median estimation
    RESERVOIR  = 500_000
    reservoirs = {c: [] for c in FEATURE_COLS}
    n_total    = 0

    for chunk in pd.read_csv(filepath,
                              usecols=FEATURE_COLS + [TARGET_COL],
                              dtype=DTYPE_MAP,
                              chunksize=CHUNK_SIZE,
                              low_memory=True):

# Filter for valid severity records
        chunk = chunk.dropna(subset=[TARGET_COL])
        chunk = chunk[chunk[TARGET_COL].between(1, 4)]
        if chunk.empty:
            continue

        for col in FEATURE_COLS:
            vals = chunk[col].dropna()
            lo, hi = FEATURE_CLIP[col]
            vals   = vals.clip(lo, hi)
            if vals.empty:
                continue
            col_min[col] = min(col_min[col], float(vals.min()))
            col_max[col] = max(col_max[col], float(vals.max()))
            space = RESERVOIR - len(reservoirs[col])
            if space > 0:
                reservoirs[col].extend(vals.iloc[:space].tolist())

        n_total += len(chunk)
        print(f"  … {n_total:>8,} rows scanned", end="\r")

    print(f"\n[PASS 1/3]  Done.  Valid rows: {n_total:,}")

    medians = {c: float(np.median(v)) if v else 0.0
               for c, v in reservoirs.items()}
    print(f"            Medians : { {c: round(medians[c], 2) for c in FEATURE_COLS} }")
    print(f"            Ranges  : { {c: (round(col_min[c],1), round(col_max[c],1)) for c in FEATURE_COLS} }")

    scaler = MinMaxScaler()
    scaler.fit(np.array([[col_min[c] for c in FEATURE_COLS],
                         [col_max[c] for c in FEATURE_COLS]], dtype=np.float64))
    return scaler, medians, n_total

# Helper to handle NaNs and outliers within data chunks
def _clean_chunk(chunk: pd.DataFrame, medians: dict) -> pd.DataFrame:
    for col in FEATURE_COLS:
        chunk[col].fillna(medians[col], inplace=True)
        lo, hi = FEATURE_CLIP[col]
        chunk[col] = chunk[col].clip(lo, hi)
    chunk = chunk.dropna()
    chunk = chunk[chunk[TARGET_COL].between(1, 4)]
    return chunk

# PASS 2: Train the model online and set aside a test set
def stream_train(model, filepath: str,
                 scaler, medians: dict, n_total: int) -> tuple:
    train_budget = int(0.8 * n_total)
    est_chunks   = max(1, n_total // CHUNK_SIZE)

    print(f"\n[PASS 2/3]  Streaming training …")
    print(f"            D_thr={model.ecm.D_thr}  "
          f"sigma={model.sigma_scale}  lambda={model.lam}")
    print(f"            Train budget : {train_budget:,} rows  "
          f"(80 % of {n_total:,})")
    print(f"            Test set     : ~{n_total - train_budget:,} rows  "
          f"(last 20 %)")

    trained_rows  = 0
    X_test_parts  = []
    y_test_parts  = []
    chunk_idx     = 0

    for chunk in pd.read_csv(filepath,
                              usecols=FEATURE_COLS + [TARGET_COL],
                              dtype=DTYPE_MAP,
                              chunksize=CHUNK_SIZE,
                              low_memory=True):

        chunk = _clean_chunk(chunk, medians)
        if chunk.empty:
            continue

# Normalize features and target (target mapped to 0-1)
        chunk_idx += 1
        X_raw  = chunk[FEATURE_COLS].values.astype(np.float64)
        X_norm = scaler.transform(X_raw)
        y_norm = (chunk[TARGET_COL].values.astype(np.float64) - 1.0) / 3.0
        n_rows = len(X_norm)

        if trained_rows >= train_budget:
# Accumulate test set
            X_test_parts.append(X_norm)
            y_test_parts.append(y_norm)

        elif trained_rows + n_rows <= train_budget:
# Training zone: update model sample by sample
            for i in range(n_rows):
                model.train_one(X_norm[i], y_norm[i])
            trained_rows += n_rows

        else:
# Straddle chunk between train and test
            cut = train_budget - trained_rows
            for i in range(cut):
                model.train_one(X_norm[i], y_norm[i])
            trained_rows += cut
            # remainder goes to test
            X_test_parts.append(X_norm[cut:])
            y_test_parts.append(y_norm[cut:])

        pct = min(trained_rows / train_budget * 100, 100.0)
        print(f"  Chunk {chunk_idx:>4} (~{chunk_idx*CHUNK_SIZE:>8,} raw rows)  |  "
              f"Trained: {trained_rows:>8,}  |  "
              f"Rules: {model.ecm.n_clusters():>4}  |  {pct:5.1f}%")

    X_test = np.vstack(X_test_parts)
    y_test = np.concatenate(y_test_parts)

    print(f"\n[PASS 2/3]  Done.")
    print(f"            Rows trained : {trained_rows:,}")
    print(f"            Test rows    : {len(X_test):,}")
    print(f"            Fuzzy rules  : {model.ecm.n_clusters()}")
    return X_test, y_test

# Evolving Clustering Method: Grows the rule base dynamically
class ECM:
    def __init__(self, D_thr: float = 0.3):
        self.D_thr   = D_thr
        self.centres: list[np.ndarray] = []
        self.radii:   list[float]      = []
        self.counts:  list[int]        = []

    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def n_clusters(self) -> int:
        return len(self.centres)

    def update(self, x: np.ndarray) -> int:
# Handle first sample
        if self.n_clusters() == 0:
            self.centres.append(x.copy())
            self.radii.append(0.0)
            self.counts.append(1)
            return 0

# Find nearest rule center
        dists = [self._euclidean(x, c) for c in self.centres]
        winner_idx = int(np.argmin(dists))
        d_min = dists[winner_idx]

        if d_min > self.D_thr:
            self.centres.append(x.copy())
            self.radii.append(0.0)
            self.counts.append(1)
            return self.n_clusters() - 1
        else:
# Update existing rule center and radius
            n = self.counts[winner_idx]
            self.centres[winner_idx] = (
                self.centres[winner_idx] * n + x
            ) / (n + 1)
            self.counts[winner_idx] += 1

            new_dist = self._euclidean(x, self.centres[winner_idx])
            if new_dist > self.radii[winner_idx]:
                self.radii[winner_idx] = new_dist

            return winner_idx

    def gaussian_membership(self, x: np.ndarray, cluster_idx: int,
                            sigma_scale: float = 1.0) -> float:
        c     = self.centres[cluster_idx]
        sigma = max(self.radii[cluster_idx], 1e-6) * sigma_scale
        dist2 = np.sum((x - c) ** 2)
        return float(np.exp(-dist2 / (2.0 * sigma ** 2)))

    def membership_vector(self, x: np.ndarray,
                          sigma_scale: float = 1.0) -> np.ndarray:
        return np.array(
            [self.gaussian_membership(x, i, sigma_scale)
             for i in range(self.n_clusters())]
        )

# DENFIS: Neural-Fuzzy system with Recursive Least Squares (RLS) learning
class DENFIS:
    def __init__(self,
                 D_thr:           float = 0.25,
                 sigma_scale:     float = 1.0,
                 lambda_forget:   float = 0.99,
                 rls_init_cov:    float = 1000.0):
        self.ecm          = ECM(D_thr)
        self.sigma_scale  = sigma_scale
        self.lam          = lambda_forget
        self.init_cov     = rls_init_cov

        self.n_features: int            = 0
        self.theta:      np.ndarray     = np.empty(0)
        self.P:          np.ndarray     = np.empty((0, 0))

    def _n_params(self) -> int:
        K = self.ecm.n_clusters()
        return K * (self.n_features + 1)

# Builds the vector used for Takagi-Sugeno linear regression
    def _regression_vector(self, x: np.ndarray) -> np.ndarray:
        phi   = self.ecm.membership_vector(x, self.sigma_scale)
        phi_s = phi.sum()
        if phi_s < 1e-10:
            phi_s = 1e-10
        phi = phi / phi_s

        n = self.n_features
        K = self.ecm.n_clusters()
        h = np.zeros(K * (n + 1))
        for k in range(K):
            base = k * (n + 1)
            h[base]          = phi[k]
            h[base+1:base+1+n] = phi[k] * x
        return h

# Grow the weight vector and covariance matrix when a new rule is born
    def _expand_rls(self, old_K: int, new_K: int):
        n        = self.n_features
        n_new    = (new_K - old_K) * (n + 1)
        n_total  = new_K * (n + 1)

        self.theta = np.append(self.theta, np.zeros(n_new))

        P_new         = np.zeros((n_total, n_total))
        old_size      = old_K * (n + 1)
        P_new[:old_size, :old_size] = self.P
        for i in range(old_size, n_total):
            P_new[i, i] = self.init_cov
        self.P = P_new

    def predict_one(self, x: np.ndarray) -> float:
        if self.ecm.n_clusters() == 0:
            return 0.0
        h = self._regression_vector(x)
        return float(h @ self.theta)

# Core online learning step
    def train_one(self, x: np.ndarray, y: float):
        if self.n_features == 0:
            self.n_features = len(x)

        old_K = self.ecm.n_clusters()

        self.ecm.update(x)
        new_K = self.ecm.n_clusters()

        if old_K == 0:
            n_params  = new_K * (self.n_features + 1)
            self.theta = np.zeros(n_params)
            self.P     = np.eye(n_params) * self.init_cov
        elif new_K > old_K:
            self._expand_rls(old_K, new_K)

# RLS calculation: mathematically optimal weight update
        h       = self._regression_vector(x)
        y_hat   = float(h @ self.theta)
        error   = y - y_hat

        Ph      = self.P @ h
        denom   = self.lam + h @ Ph

        # Stability check: prevent division by zero or tiny values that cause NaNs
        if abs(denom) < 1e-12:
            return

        K_gain  = Ph / denom

        new_theta = self.theta + K_gain * error
        # Guard against NaN propagation
        if not np.isnan(new_theta).any():
            self.theta = new_theta
            new_P = (self.P - np.outer(K_gain, h @ self.P)) / self.lam
            # Enforce symmetry to maintain numerical stability
            self.P = (new_P + new_P.T) * 0.5

    def fit(self, X: np.ndarray, y: np.ndarray,
            log_every: int = 50_000) -> "DENFIS":
        n = len(X)
        print(f"\n[TRAIN] Starting DENFIS online training on {n:,} samples …")
        print(f"        D_thr={self.ecm.D_thr}, σ_scale={self.sigma_scale}, "
              f"λ={self.lam}")

        for i in range(n):
            self.train_one(X[i], y[i])
            if (i + 1) % log_every == 0:
                print(f"        Step {i+1:>7,} / {n:,}  |  "
                      f"Rules so far: {self.ecm.n_clusters():>4}")

        print(f"[TRAIN] Training complete.  Total rules: {self.ecm.n_clusters()}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(x) for x in X])

# Linguistic translation helpers for rule extraction
# Maps normalized values to text labels for readability
def _label_feature(feat_name: str, value: float) -> str:
    if feat_name == "Temperature(F)":
        if value < 0.30:   return "Cold"
        if value < 0.55:   return "Mild"
        if value < 0.75:   return "Warm"
        return "Hot"
    elif feat_name == "Visibility(mi)":
        if value < 0.25:   return "Very Low"
        if value < 0.50:   return "Low"
        if value < 0.75:   return "Moderate"
        return "High"
    elif feat_name == "Humidity(%)":
        if value < 0.30:   return "Low"
        if value < 0.60:   return "Moderate"
        return "High"
    return f"{value:.2f}"

def _denorm_severity(y_norm: float) -> float:
    return max(1.0, min(4.0, y_norm * 3.0 + 1.0))

# Converts internal fuzzy rules into human-readable text
def extract_rules(model: DENFIS,
                  feature_names: list[str],
                  n_rules: int = 3):
    K = model.ecm.n_clusters()
    n = model.n_features
    print(f"\n{'='*65}")
    print(f"  DENFIS EXTRACTED RULES  (showing {min(n_rules, K)} of {K} rules)")
    print(f"{'='*65}")

# Prioritize rules that cover the most data
    order = np.argsort(model.ecm.counts)[::-1]
    shown = 0
    for rank, k in enumerate(order):
        if shown >= n_rules:
            break

        centre   = model.ecm.centres[k]
        radius   = model.ecm.radii[k]
        count    = model.ecm.counts[k]

# Retrieve weights for the linear consequent
        base      = k * (n + 1)
        w         = model.theta[base: base + n + 1]

        antecedent_parts = []
        for j, fname in enumerate(feature_names):
            label = _label_feature(fname, float(centre[j]))
            antecedent_parts.append(f"{fname} is {label}")
        antecedent = " AND\n      ".join(antecedent_parts)

        y_at_centre = float(w[0] + w[1:] @ centre)
        severity    = _denorm_severity(y_at_centre)

        print(f"\n  Rule #{rank+1}  (cluster {k}, covers {count:,} samples, "
              f"radius={radius:.4f})")
        print(f"  IF   {antecedent}")
        print(f"  THEN Severity = {w[0]:.4f}"
              + "".join(f" + {w[j+1]:+.4f}·{feature_names[j]}"
                        for j in range(n)))
        print(f"       → Severity at cluster centre ≈ {severity:.2f}  "
              f"(scale 1=minor … 4=severe)")
        shown += 1

    print(f"\n{'='*65}\n")

# Statistics for evaluating model accuracy
def evaluate(model: DENFIS,
             X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
    print("\n[EVAL]  Running predictions on test set …")
    y_pred = model.predict(X_test)

    mse  = float(np.mean((y_pred - y_test) ** 2))
    mae  = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(mse))

    y_pred_sev = np.clip(y_pred * 3.0 + 1.0, 1.0, 4.0)
    y_true_sev = y_test * 3.0 + 1.0
    mae_sev    = float(np.mean(np.abs(y_pred_sev - y_true_sev)))

    print(f"\n{'─'*45}")
    print(f"  DENFIS Performance Report")
    print(f"{'─'*45}")
    print(f"  MSE  (normalised)      : {mse:.6f}")
    print(f"  RMSE (normalised)      : {rmse:.6f}")
    print(f"  MAE  (normalised)      : {mae:.6f}")
    print(f"  MAE  (Severity 1–4)    : {mae_sev:.4f}")
    print(f"  Total fuzzy rules      : {model.ecm.n_clusters()}")
    print(f"{'─'*45}\n")

    return {"mse": mse, "rmse": rmse, "mae": mae, "mae_severity": mae_sev}

MODEL_PATH = "model.pkl"

# Save model and scaler to a file
def save_model(model: "DENFIS", scaler: MinMaxScaler,
               metrics: dict, path: str = MODEL_PATH) -> None:
    bundle = {
        "model":   model,
        "scaler":  scaler,
        "metrics": metrics,
        "meta": {
            "n_rules":      model.ecm.n_clusters(),
            "n_features":   model.n_features,
            "feature_cols": FEATURE_COLS,
            "target_col":   TARGET_COL,
            "D_thr":        model.ecm.D_thr,
            "sigma_scale":  model.sigma_scale,
            "lambda":       model.lam,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = os.path.getsize(path) / 1024
    print(f"\n[SAVE]  Model exported → '{path}'  ({size_kb:.1f} KB)")
    print(f"        Bundled: DENFIS ({model.ecm.n_clusters()} rules) + "
          f"scaler + metrics")

# Load model and fix module scope issues for unpickling
def load_model(path: str = MODEL_PATH) -> tuple:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'.\n"
            f"Run  python train_model.py  first to train and export."
        )
# Handle case where main module name changes during import
    import model.train_model as _self_module # type: ignore

    class _FixedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__":
                return getattr(_self_module, name)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        bundle = _FixedUnpickler(f).load()

    model   = bundle["model"]
    scaler  = bundle["scaler"]
    metrics = bundle["metrics"]
    meta    = bundle["meta"]

    print(f"\n[LOAD]  Model loaded from '{path}'")
    print(f"        Rules: {meta['n_rules']}  |  "
          f"Features: {meta['feature_cols']}  |  "
          f"D_thr: {meta['D_thr']}")
    return model, scaler, metrics, meta

# Execution entry point
def main():
    import time
    t0 = time.time()

    scaler, medians, n_total = fit_scaler_streaming(DATA_FILE)
# Pass 1: Statistics

    model = DENFIS(
        D_thr         = 0.30,
        sigma_scale   = 1.0,
        lambda_forget = 0.99,
        rls_init_cov  = 1000.0,
    )

# Pass 2: Training
    X_test, y_test = stream_train(model, DATA_FILE, scaler, medians, n_total)

    elapsed = time.time() - t0
    print(f"\n[PASS 3/3]  Training wall-clock time: {elapsed/60:.1f} min")

# Evaluation and Rule Display
    metrics = evaluate(model, X_test, y_test)

    extract_rules(model, FEATURE_COLS, n_rules=3)

# Quick test on a typical sample
    print("─── Example prediction ──────────────────────────────────────")
    sample_raw  = np.array([[72.0, 10.0, 65.0]])
    sample_norm = scaler.transform(sample_raw)[0]
    pred_norm   = model.predict_one(sample_norm)
    pred_sev    = np.clip(pred_norm * 3.0 + 1.0, 1.0, 4.0)
    print(f"  Input : Temp=72 °F, Visibility=10 mi, Humidity=65 %")
    print(f"  DENFIS Severity ≈ {pred_sev:.2f}  (1=minor … 4=critical)")
    print("─────────────────────────────────────────────────────────────\n")

# Persistence
    save_model(model, scaler, metrics, path=MODEL_PATH)
    print(f"[DONE]  Run  python test_model.py  to test the model.\n")

    return model, scaler, metrics

if __name__ == "__main__":
    model, scaler, metrics = main()
