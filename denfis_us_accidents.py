"""
============================================================
  DENFIS – Dynamic Evolving Neural-Fuzzy Inference System
  Applied to: US Accidents Dataset (2016-2023)
  Course   : Soft Computing  |  Dr. Mona Nagy ElBedwehy
  Author   : Omar (student implementation)
============================================================

THEORY NOTE — Why DENFIS is "Third Generation" Neuro-Fuzzy
------------------------------------------------------------
Generation 1 (e.g. ANFIS): Static structure. Rules and membership
  functions are fixed BEFORE training. The network learns only the
  parameters, not the topology.

Generation 2 (e.g. FALCON): The structure can be optimised by
  clustering before training, but the rule base is still frozen
  during online operation.

Generation 3 — DENFIS (Kasabov & Song, 2002): The rule base
  GROWS dynamically during training via the Evolving Clustering
  Method (ECM). New fuzzy rules are born on-line when the data
  demands them, and existing rules are refined by Recursive Least
  Squares (RLS). The system therefore starts with zero rules and
  self-organises its own architecture.

Contrast with MLP (from your textbook):
  ─ MLP:   fixed layers & neurons decided before training;
           learns only weights via back-propagation (gradient descent).
  ─ DENFIS: topology (clusters / rules) evolves online;
           consequent weights updated by RLS (no gradient needed);
           every cluster IS a fuzzy rule — interpretable & extractable.
"""

# ─────────────────────────────────────────────────────────────
#  0. IMPORTS
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  1. CONFIGURATION
# ─────────────────────────────────────────────────────────────

DATA_FILE  = "US_Accidents_March23.csv"   # ← put your Kaggle CSV here

# Rows read per chunk.  100 000 rows ≈ 30 MB RAM — safe on any machine.
# The CSV is never fully loaded; only one chunk lives in memory at a time.
CHUNK_SIZE = 100_000

FEATURE_COLS = ["Temperature(F)", "Visibility(mi)", "Humidity(%)"]
TARGET_COL   = "Severity"   # integer 1–4

# Hard physical limits — silently drops sensor outliers before scaling
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


# ─────────────────────────────────────────────────────────────
#  PASS 1 — fit_scaler_streaming
#
#  Reads the full CSV once, one CHUNK_SIZE chunk at a time.
#  Never stores more than one chunk in RAM.
#  Tracks global min/max per column → builds MinMaxScaler.
#  Tracks per-column medians via a capped reservoir → used to
#  fill NaN values during training.
#  Also counts total valid rows so we can compute 80/20 split.
# ─────────────────────────────────────────────────────────────

def fit_scaler_streaming(filepath: str) -> tuple:
    """
    Single-pass scan of the full CSV.

    Returns
    -------
    scaler  : MinMaxScaler  fitted on global [min, max] per column
    medians : dict          {col_name: median_value}  for NaN filling
    n_total : int           total valid (non-NaN, severity 1-4) rows
    """
    print(f"\n[PASS 1/3]  Scanning '{filepath}' …")
    print( "            (reads every chunk once to fit the scaler)")

    col_min    = {c:  float("inf") for c in FEATURE_COLS}
    col_max    = {c: float("-inf") for c in FEATURE_COLS}
    RESERVOIR  = 500_000          # cap reservoir at 500 k values per column
    reservoirs = {c: [] for c in FEATURE_COLS}
    n_total    = 0

    for chunk in pd.read_csv(filepath,
                              usecols=FEATURE_COLS + [TARGET_COL],
                              dtype=DTYPE_MAP,
                              chunksize=CHUNK_SIZE,
                              low_memory=True):

        # Drop rows where the target is missing or invalid
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


# ─────────────────────────────────────────────────────────────
#  PASS 2 — stream_train
#
#  Reads the full CSV a SECOND time, chunk by chunk.
#  The first 80 % of valid rows → training (fed to DENFIS online).
#  The remaining 20 %          → accumulated as the test set.
#
#  How the 80/20 boundary works without loading everything:
#    train_budget = int(0.8 * n_total)
#    We count valid rows as we go.  While trained_rows < train_budget
#    every row goes to model.train_one().  Once we cross the boundary
#    the rest are collected into X_test / y_test lists and returned.
#
#  Peak RAM at any moment = one chunk (~30 MB) + test set rows
#  accumulated so far.  For 7.7 M rows the test set is ~1.54 M rows
#  × 3 features × 8 bytes ≈ 37 MB.  Total peak ≈ 70 MB.
# ─────────────────────────────────────────────────────────────

def _clean_chunk(chunk: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Fill NaNs with medians, clip sensor outliers, drop bad rows."""
    for col in FEATURE_COLS:
        chunk[col].fillna(medians[col], inplace=True)
        lo, hi = FEATURE_CLIP[col]
        chunk[col] = chunk[col].clip(lo, hi)
    chunk = chunk.dropna()
    chunk = chunk[chunk[TARGET_COL].between(1, 4)]
    return chunk


def stream_train(model, filepath: str,
                 scaler, medians: dict, n_total: int) -> tuple:
    """
    True streaming training — only one chunk in RAM at a time.

    The 80/20 train/test split is enforced by a running counter:
    rows are sent to model.train_one() until train_budget is reached,
    then accumulated into the test arrays.

    Returns
    -------
    X_test : np.ndarray  shape (n_test, 3)  normalised to [0, 1]
    y_test : np.ndarray  shape (n_test,)    normalised to [0, 1]
    """
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

        chunk_idx += 1
        X_raw  = chunk[FEATURE_COLS].values.astype(np.float64)
        X_norm = scaler.transform(X_raw)
        y_norm = (chunk[TARGET_COL].values.astype(np.float64) - 1.0) / 3.0
        n_rows = len(X_norm)

        if trained_rows >= train_budget:
            # ── entirely in the test zone ────────────────────────────
            X_test_parts.append(X_norm)
            y_test_parts.append(y_norm)

        elif trained_rows + n_rows <= train_budget:
            # ── entirely in the training zone ───────────────────────
            for i in range(n_rows):
                model.train_one(X_norm[i], y_norm[i])
            trained_rows += n_rows

        else:
            # ── this chunk straddles the boundary ───────────────────
            cut = train_budget - trained_rows      # rows that still go to train
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


# ─────────────────────────────────────────────────────────────
#  Legacy helper — kept so test_denfis.py still works
# ─────────────────────────────────────────────────────────────

def load_and_preprocess(filepath: str, nrows: int) -> tuple:
    """Load a fixed-size subset (used only by the unit-test script)."""
    df = pd.read_csv(filepath, nrows=nrows,
                     usecols=FEATURE_COLS + [TARGET_COL],
                     dtype=DTYPE_MAP, low_memory=True)
    for col in FEATURE_COLS:
        df[col].fillna(df[col].median(), inplace=True)
    df.dropna(inplace=True)
    df = df[df[TARGET_COL].between(1, 4)]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[FEATURE_COLS].values).astype(np.float64)
    y = (df[TARGET_COL].values.astype(np.float64) - 1.0) / 3.0
    return X, y, scaler, df


# ─────────────────────────────────────────────────────────────
#  2. ECM – EVOLVING CLUSTERING METHOD
#     Creates / updates clusters (≡ fuzzy rules) on-line.
# ─────────────────────────────────────────────────────────────

class ECM:
    """
    Evolving Clustering Method (Kasabov & Song, 2002).

    A new cluster is created when the nearest existing cluster centre
    is further than the threshold distance D_thr from the input vector.
    Cluster radii grow to cover member points, enabling soft/overlapping
    membership functions.

    Attributes
    ----------
    D_thr   : distance threshold controlling rule granularity.
              Smaller → more rules (finer partition).
              Larger  → fewer rules (coarser partition).
    centres : list of cluster centre vectors  (shape: n_features)
    radii   : list of cluster radii           (scalar per cluster)
    counts  : list of how many samples belong to each cluster
    """

    def __init__(self, D_thr: float = 0.3):
        self.D_thr   = D_thr
        self.centres: list[np.ndarray] = []
        self.radii:   list[float]      = []
        self.counts:  list[int]        = []

    # ── helpers ──────────────────────────────────────────────────────
    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def n_clusters(self) -> int:
        return len(self.centres)

    # ── main update step ─────────────────────────────────────────────
    def update(self, x: np.ndarray) -> int:
        """
        Process one data point.  Returns the index of the winning cluster.
        """
        if self.n_clusters() == 0:
            # Very first point → create the first cluster
            self.centres.append(x.copy())
            self.radii.append(0.0)
            self.counts.append(1)
            return 0

        # Find nearest cluster centre
        dists = [self._euclidean(x, c) for c in self.centres]
        winner_idx = int(np.argmin(dists))
        d_min = dists[winner_idx]

        if d_min > self.D_thr:
            # ── CREATE a new cluster ──────────────────────────────────
            self.centres.append(x.copy())
            self.radii.append(0.0)
            self.counts.append(1)
            return self.n_clusters() - 1
        else:
            # ── UPDATE the winning cluster (online mean update) ───────
            n = self.counts[winner_idx]
            self.centres[winner_idx] = (
                self.centres[winner_idx] * n + x
            ) / (n + 1)
            self.counts[winner_idx] += 1

            # Expand radius to cover the new point
            new_dist = self._euclidean(x, self.centres[winner_idx])
            if new_dist > self.radii[winner_idx]:
                self.radii[winner_idx] = new_dist

            return winner_idx

    # ── membership function ──────────────────────────────────────────
    def gaussian_membership(self, x: np.ndarray, cluster_idx: int,
                            sigma_scale: float = 1.0) -> float:
        """
        Gaussian membership: μ = exp(−‖x − c‖² / (2σ²))
        σ = max(radius, epsilon) * sigma_scale
        """
        c     = self.centres[cluster_idx]
        sigma = max(self.radii[cluster_idx], 1e-6) * sigma_scale
        dist2 = np.sum((x - c) ** 2)
        return float(np.exp(-dist2 / (2.0 * sigma ** 2)))

    def membership_vector(self, x: np.ndarray,
                          sigma_scale: float = 1.0) -> np.ndarray:
        """Return membership degree for x w.r.t. every cluster."""
        return np.array(
            [self.gaussian_membership(x, i, sigma_scale)
             for i in range(self.n_clusters())]
        )


# ─────────────────────────────────────────────────────────────
#  3. DENFIS  (full model)
#     Neural layer: Takagi-Sugeno (TS) inference
#     Learning    : Recursive Least Squares (RLS) for consequents
# ─────────────────────────────────────────────────────────────

class DENFIS:
    """
    Dynamic Evolving Neural-Fuzzy Inference System.

    Architecture
    ────────────
    Layer 1 — Input nodes  (pass x through unchanged)
    Layer 2 — ECM clusters  = fuzzy rule antecedents
              Each cluster k has a Gaussian MF: μ_k(x)
    Layer 3 — Normalisation  φ_k = μ_k / Σμ_j   (firing strength)
    Layer 4 — Takagi-Sugeno consequents
              ŷ_k = w_k0 + w_k1·x1 + … + w_kn·xn   (linear in x)
    Layer 5 — Crisp output
              ŷ = Σ_k  φ_k · ŷ_k

    Learning — RLS
    ──────────────
    The TS consequent coefficients w_k form a vector θ of size
    K × (n+1) where K = number of rules, n = number of features.
    RLS maintains a covariance matrix P and updates θ recursively:

        K_gain = P · h / (λ + h^T · P · h)
        θ      = θ + K_gain · (y − h^T · θ)
        P      = (P − K_gain · h^T · P) / λ

    where h is the regression vector (stacked φ_k · [1, x]) and
    λ ∈ (0,1] is the forgetting factor (1 = no forgetting).

    Advantage over Back-Propagation (textbook MLP):
    ─ RLS converges in ONE pass; no epochs needed.
    ─ No learning-rate hyper-parameter.
    ─ Exact least-squares solution at each step (within the
      evolving structure).
    """

    def __init__(self,
                 D_thr:           float = 0.25,
                 sigma_scale:     float = 1.0,
                 lambda_forget:   float = 0.99,
                 rls_init_cov:    float = 1000.0):
        """
        Parameters
        ----------
        D_thr         : ECM distance threshold
        sigma_scale   : width multiplier for Gaussian MFs
        lambda_forget : RLS forgetting factor (0 < λ ≤ 1)
        rls_init_cov  : initial covariance diagonal value (large → open prior)
        """
        self.ecm          = ECM(D_thr)
        self.sigma_scale  = sigma_scale
        self.lam          = lambda_forget
        self.init_cov     = rls_init_cov

        # Will be initialised on-the-fly as clusters are born
        self.n_features: int            = 0
        self.theta:      np.ndarray     = np.empty(0)   # consequent weights
        self.P:          np.ndarray     = np.empty((0, 0))  # RLS covariance

    # ── private helpers ──────────────────────────────────────────────
    def _n_params(self) -> int:
        """Total number of TS consequent parameters."""
        K = self.ecm.n_clusters()
        return K * (self.n_features + 1)   # K × (1 + n)

    def _regression_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Build the regression vector h for RLS.
        For each cluster k: [φ_k, φ_k·x1, φ_k·x2, … , φ_k·xn]
        h has length K × (n+1).
        """
        phi   = self.ecm.membership_vector(x, self.sigma_scale)
        phi_s = phi.sum()
        if phi_s < 1e-10:
            phi_s = 1e-10
        phi = phi / phi_s          # normalised firing strengths

        n = self.n_features
        K = self.ecm.n_clusters()
        h = np.zeros(K * (n + 1))
        for k in range(K):
            base = k * (n + 1)
            h[base]          = phi[k]          # bias term
            h[base+1:base+1+n] = phi[k] * x   # linear terms
        return h

    def _expand_rls(self, old_K: int, new_K: int):
        """
        A new cluster was created → expand θ and P to accommodate
        the extra (n+1) consequent parameters.
        """
        n        = self.n_features
        n_new    = (new_K - old_K) * (n + 1)
        n_total  = new_K * (n + 1)

        # Expand theta: append zeros for new cluster
        self.theta = np.append(self.theta, np.zeros(n_new))

        # Expand P: block-diagonal extension
        P_new         = np.zeros((n_total, n_total))
        old_size      = old_K * (n + 1)
        P_new[:old_size, :old_size] = self.P
        # New block: large diagonal → high uncertainty for new parameters
        for i in range(old_size, n_total):
            P_new[i, i] = self.init_cov
        self.P = P_new

    # ── predict ─────────────────────────────────────────────────────
    def predict_one(self, x: np.ndarray) -> float:
        """Predict for a single sample."""
        if self.ecm.n_clusters() == 0:
            return 0.0
        h = self._regression_vector(x)
        return float(h @ self.theta)

    # ── online train step ────────────────────────────────────────────
    def train_one(self, x: np.ndarray, y: float):
        """
        Process ONE sample:
          1. ECM: possibly create a new cluster (new rule).
          2. RLS: update consequent weights.
        """
        if self.n_features == 0:
            self.n_features = len(x)

        old_K = self.ecm.n_clusters()

        # ── Step 1: ECM update ────────────────────────────────────────
        self.ecm.update(x)
        new_K = self.ecm.n_clusters()

        # ── Step 2: Initialise / expand RLS matrices if needed ────────
        if old_K == 0:
            # Very first sample → initialise from scratch
            n_params  = new_K * (self.n_features + 1)
            self.theta = np.zeros(n_params)
            self.P     = np.eye(n_params) * self.init_cov
        elif new_K > old_K:
            # A new cluster was born → expand
            self._expand_rls(old_K, new_K)

        # ── Step 3: RLS update ────────────────────────────────────────
        h       = self._regression_vector(x)
        y_hat   = float(h @ self.theta)
        error   = y - y_hat

        # Kalman gain
        Ph      = self.P @ h
        denom   = self.lam + h @ Ph
        K_gain  = Ph / denom

        # Update weights and covariance
        self.theta = self.theta + K_gain * error
        self.P     = (self.P - np.outer(K_gain, h @ self.P)) / self.lam

    # ── batch interface ──────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray,
            log_every: int = 50_000) -> "DENFIS":
        """
        One-pass online training (the defining property of DENFIS).
        """
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


# ─────────────────────────────────────────────────────────────
#  4. HUMAN-READABLE RULE EXTRACTION
# ─────────────────────────────────────────────────────────────

FEATURE_NAMES = ["Temperature(F)", "Visibility(mi)", "Humidity(%)"]

# Linguistic labels for normalised [0,1] values
def _label_feature(feat_name: str, value: float) -> str:
    """Convert a normalised cluster-centre value to a linguistic label."""
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
    """Convert normalised output back to Severity scale 1-4."""
    return max(1.0, min(4.0, y_norm * 3.0 + 1.0))


def extract_rules(model: DENFIS,
                  feature_names: list[str],
                  n_rules: int = 3):
    """
    Print human-readable Takagi-Sugeno fuzzy rules from the trained DENFIS.
    """
    K = model.ecm.n_clusters()
    n = model.n_features
    print(f"\n{'='*65}")
    print(f"  DENFIS EXTRACTED RULES  (showing {min(n_rules, K)} of {K} rules)")
    print(f"{'='*65}")

    # Sort rules by cluster membership count (most-used first)
    order = np.argsort(model.ecm.counts)[::-1]

    shown = 0
    for rank, k in enumerate(order):
        if shown >= n_rules:
            break

        centre   = model.ecm.centres[k]
        radius   = model.ecm.radii[k]
        count    = model.ecm.counts[k]

        # Consequent: ŷ_k = w0 + w1·T + w2·V + w3·H
        base      = k * (n + 1)
        w         = model.theta[base: base + n + 1]

        # Build antecedent string
        antecedent_parts = []
        for j, fname in enumerate(feature_names):
            label = _label_feature(fname, float(centre[j]))
            antecedent_parts.append(f"{fname} is {label}")
        antecedent = " AND\n      ".join(antecedent_parts)

        # Compute example output at cluster centre
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


# ─────────────────────────────────────────────────────────────
#  5. EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(model: DENFIS,
             X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
    """Compute MSE and MAE on a held-out test set."""
    print("\n[EVAL]  Running predictions on test set …")
    y_pred = model.predict(X_test)

    mse  = float(np.mean((y_pred - y_test) ** 2))
    mae  = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(mse))

    # Convert back to original Severity scale for interpretability
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


# ─────────────────────────────────────────────────────────────
#  6. MODEL EXPORT & IMPORT
# ─────────────────────────────────────────────────────────────

MODEL_PATH = "denfis_trained_model.pkl"   # ← file that gets saved/loaded


def save_model(model: "DENFIS", scaler: MinMaxScaler,
               metrics: dict, path: str = MODEL_PATH) -> None:
    """
    Serialise the trained DENFIS + scaler + training metrics to a .pkl file.

    Everything needed to make predictions on new data is bundled together:
      • model   — the DENFIS object (ECM clusters + RLS weights)
      • scaler  — the fitted MinMaxScaler (must use the SAME scaler that
                  was used during training, otherwise inputs are on a
                  different scale and predictions will be garbage)
      • metrics — MSE / MAE recorded at evaluation time
      • meta    — hyper-parameters and dataset info for reproducibility
    """
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


def load_model(path: str = MODEL_PATH) -> tuple:
    """
    Load a previously saved DENFIS bundle.

    Returns
    -------
    model   : DENFIS  — ready for predict_one() / predict()
    scaler  : MinMaxScaler — use to normalise raw inputs before predicting
    metrics : dict    — training-time performance numbers
    meta    : dict    — hyper-parameters and dataset info
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'.\n"
            f"Run  python denfis_us_accidents.py  first to train and export."
        )
    # ── Pickle fix ────────────────────────────────────────────────────────
    # pickle saves class objects by their *module path*.  When the pkl was
    # created, DENFIS and ECM lived in __main__ (denfis_us_accidents.py).
    # When test_trained_model.py loads the file, __main__ is the tester,
    # so pickle cannot find the classes and raises AttributeError.
    # Solution: a custom Unpickler that redirects any class stored under
    # __main__ to denfis_us_accidents (where DENFIS and ECM always live).
    import denfis_us_accidents as _self_module

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


# ─────────────────────────────────────────────────────────────
#  7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()

    # ── Step 1: Scan full CSV, fit scaler ─────────────────────────────
    #   Reads the file once.  No model training.  No full-dataset load.
    scaler, medians, n_total = fit_scaler_streaming(DATA_FILE)

    # ── Step 2: Build DENFIS ──────────────────────────────────────────
    #   D_thr = 0.30 keeps the rule count manageable across 7.7 M rows.
    #   A smaller D_thr (e.g. 0.20) creates more rules but slows the
    #   RLS matrix operations proportionally.
    model = DENFIS(
        D_thr         = 0.30,
        sigma_scale   = 1.0,
        lambda_forget = 0.99,
        rls_init_cov  = 1000.0,
    )

    # ── Step 3: Stream-train over the full dataset ────────────────────
    #   Reads the file a second time, chunk by chunk.
    #   80 % of valid rows → model.train_one() (online, one row at a time)
    #   20 % of valid rows → held back as the test set
    #   Peak RAM ≈ one chunk + test set ≈ 70 MB total.
    X_test, y_test = stream_train(model, DATA_FILE, scaler, medians, n_total)

    elapsed = time.time() - t0
    print(f"\n[PASS 3/3]  Training wall-clock time: {elapsed/60:.1f} min")

    # ── Step 4: Evaluate on the held-back 20 % ───────────────────────
    metrics = evaluate(model, X_test, y_test)

    # ── Step 5: Print 3 human-readable fuzzy rules ───────────────────
    extract_rules(model, FEATURE_NAMES, n_rules=3)

    # ── Step 6: Quick sanity prediction ──────────────────────────────
    print("─── Example prediction ──────────────────────────────────────")
    sample_raw  = np.array([[72.0, 10.0, 65.0]])
    sample_norm = scaler.transform(sample_raw)[0]
    pred_norm   = model.predict_one(sample_norm)
    pred_sev    = np.clip(pred_norm * 3.0 + 1.0, 1.0, 4.0)
    print(f"  Input : Temp=72 °F, Visibility=10 mi, Humidity=65 %")
    print(f"  DENFIS Severity ≈ {pred_sev:.2f}  (1=minor … 4=critical)")
    print("─────────────────────────────────────────────────────────────\n")

    # ── Step 7: Export ───────────────────────────────────────────────
    save_model(model, scaler, metrics, path=MODEL_PATH)
    print(f"[DONE]  Run  python test_trained_model.py  to test the model.\n")

    return model, scaler, metrics


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, scaler, metrics = main()
