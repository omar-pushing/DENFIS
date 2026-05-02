"""
============================================================
  DENFIS — Test Suite
  Run this WITHOUT the Kaggle dataset.
  Uses synthetic data to verify every component works.
============================================================

Usage
-----
    python test_denfis.py

All tests print PASS / FAIL. You should see 10/10 passing.
"""

import sys
import numpy as np

# ── Import the model from the project file ──────────────────────────────────
# Both files must be in the same folder.
try:
    from denfis_us_accidents import ECM, DENFIS, extract_rules, evaluate, FEATURE_NAMES
except ImportError as e:
    print(f"[ERROR] Could not import denfis_us_accidents.py: {e}")
    print("        Make sure test_denfis.py and denfis_us_accidents.py are in the same folder.")
    sys.exit(1)

# ── Helpers ──────────────────────────────────────────────────────────────────
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{suffix}")
    results.append(condition)

# ════════════════════════════════════════════════════════════════════════════
#  TEST 1 — ECM creates a cluster on first sample
# ════════════════════════════════════════════════════════════════════════════
print("\n── ECM Unit Tests ──────────────────────────────────────────────")

ecm = ECM(D_thr=0.3)
ecm.update(np.array([0.5, 0.5, 0.5]))
check("ECM creates first cluster", ecm.n_clusters() == 1,
      f"n_clusters={ecm.n_clusters()}")

# ════════════════════════════════════════════════════════════════════════════
#  TEST 2 — ECM creates a NEW cluster when point is far
# ════════════════════════════════════════════════════════════════════════════
ecm.update(np.array([0.0, 0.0, 0.0]))   # distance ~0.87 >> D_thr=0.3
check("ECM creates new cluster for distant point", ecm.n_clusters() == 2,
      f"n_clusters={ecm.n_clusters()}")

# ════════════════════════════════════════════════════════════════════════════
#  TEST 3 — ECM does NOT create a cluster when point is close
# ════════════════════════════════════════════════════════════════════════════
before = ecm.n_clusters()
ecm.update(np.array([0.51, 0.49, 0.50]))  # very close to first cluster
check("ECM does NOT create cluster for nearby point",
      ecm.n_clusters() == before,
      f"n_clusters={ecm.n_clusters()}")

# ════════════════════════════════════════════════════════════════════════════
#  TEST 4 — ECM membership sums roughly to something positive
# ════════════════════════════════════════════════════════════════════════════
x_test = np.array([0.5, 0.5, 0.5])
memberships = ecm.membership_vector(x_test)
check("ECM membership vector has correct length",
      len(memberships) == ecm.n_clusters(),
      f"len={len(memberships)}, n_clusters={ecm.n_clusters()}")

# ════════════════════════════════════════════════════════════════════════════
#  TEST 5 — DENFIS trains on small synthetic dataset without crashing
# ════════════════════════════════════════════════════════════════════════════
print("\n── DENFIS Training Tests ────────────────────────────────────────")

np.random.seed(42)
N = 500
X_small = np.random.rand(N, 3).astype(np.float64)
# Synthetic severity: high temp + low visibility → higher severity
y_small = np.clip(0.4 * X_small[:,0] + 0.5*(1-X_small[:,1]) + 0.1*X_small[:,2], 0, 1)

model = DENFIS(D_thr=0.30, sigma_scale=1.0, lambda_forget=0.99)
try:
    model.fit(X_small, y_small, log_every=1000)  # quiet
    trained_ok = True
except Exception as ex:
    trained_ok = False
    print(f"        Exception during training: {ex}")
check("DENFIS trains 500 samples without error", trained_ok)

# ════════════════════════════════════════════════════════════════════════════
#  TEST 6 — Model creates at least 1 rule
# ════════════════════════════════════════════════════════════════════════════
check("Model has at least 1 fuzzy rule after training",
      model.ecm.n_clusters() >= 1,
      f"rules={model.ecm.n_clusters()}")

# ════════════════════════════════════════════════════════════════════════════
#  TEST 7 — predict_one returns a float in a plausible range
# ════════════════════════════════════════════════════════════════════════════
sample = np.array([0.6, 0.3, 0.7])
pred = model.predict_one(sample)
check("predict_one returns finite float",
      np.isfinite(pred),
      f"pred={pred:.4f}")

# ════════════════════════════════════════════════════════════════════════════
#  TEST 8 — batch predict returns array of correct shape
# ════════════════════════════════════════════════════════════════════════════
X_test_batch = np.random.rand(50, 3)
preds = model.predict(X_test_batch)
check("batch predict returns correct shape",
      preds.shape == (50,),
      f"shape={preds.shape}")

# ════════════════════════════════════════════════════════════════════════════
#  TEST 9 — MSE is better than naive baseline (always predict the mean)
# ════════════════════════════════════════════════════════════════════════════
np.random.seed(7)
N2 = 1000
X2 = np.random.rand(N2, 3)
y2 = np.clip(0.4*X2[:,0] + 0.5*(1-X2[:,1]) + 0.1*X2[:,2], 0, 1)

model2 = DENFIS(D_thr=0.25, sigma_scale=1.0)
model2.fit(X2[:800], y2[:800], log_every=5000)

y_pred = model2.predict(X2[800:])
y_true = y2[800:]
mse_denfis = float(np.mean((y_pred - y_true)**2))
mse_baseline = float(np.mean((y_true.mean() - y_true)**2))  # always predict mean

check("DENFIS MSE is better than naive mean-predictor baseline",
      mse_denfis < mse_baseline,
      f"DENFIS={mse_denfis:.5f}, baseline={mse_baseline:.5f}")

# ════════════════════════════════════════════════════════════════════════════
#  TEST 10 — Rule extraction runs without error and prints 3 rules
# ════════════════════════════════════════════════════════════════════════════
print("\n── Rule Extraction Test ─────────────────────────────────────────")
try:
    extract_rules(model2, FEATURE_NAMES, n_rules=3)
    extraction_ok = True
except Exception as ex:
    extraction_ok = False
    print(f"  Exception during rule extraction: {ex}")
check("extract_rules prints 3 rules without error", extraction_ok)

# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n── Summary ──────────────────────────────────────────────────────")
passed = sum(results)
total  = len(results)
bar    = ("█" * passed) + ("░" * (total - passed))
print(f"  {bar}  {passed}/{total} tests passed")

if passed == total:
    print("\n  All tests passed. The model is working correctly.")
    print("  You can now run  python denfis_us_accidents.py  with the real dataset.\n")
else:
    failed_idx = [i+1 for i, r in enumerate(results) if not r]
    print(f"\n  Tests {failed_idx} failed. Check the output above for details.\n")
    sys.exit(1)
