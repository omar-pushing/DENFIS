"""
============================================================
  test_trained_model.py
  DENFIS — Load the exported model and test it
  Soft Computing Project | Dr. Mona Nagy ElBedwehy
============================================================

WORKFLOW
────────
  Step 1:  python denfis_us_accidents.py        ← trains and saves the model
  Step 2:  python test_trained_model.py         ← YOU ARE HERE

  This script does three things:
    A. Loads denfis_trained_model.pkl from disk.
    B. Runs a battery of 12 hand-crafted test cases covering
       different weather scenarios (clear day, ice storm, fog, etc.)
    C. Prints a full prediction report for every case, plus a
       summary table at the end.

REQUIREMENTS
────────────
  - denfis_us_accidents.py  (in the same folder — needed for unpickling)
  - denfis_trained_model.pkl (produced by running denfis_us_accidents.py)
"""

import sys
import numpy as np

# ── Guard: make sure the source file is importable (needed for pickle) ──────
try:
    from denfis_us_accidents import load_model, FEATURE_NAMES, _label_feature
except ImportError:
    print("\n[ERROR] Cannot import denfis_us_accidents.py")
    print("        Both files must be in the same folder.\n")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
#  TERMINAL COLOURS
# ════════════════════════════════════════════════════════════════════════════
GREEN  = "\033[92m"
YELLOW = "\033[93m"
ORANGE = "\033[33m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

SEV_COLOR = {1: GREEN, 2: YELLOW, 3: ORANGE, 4: RED}
SEV_LABEL = {
    1: "Minor       — small delay, clears fast",
    2: "Moderate    — noticeable congestion",
    3: "Significant — major disruption",
    4: "Severe      — road likely blocked/closed",
}

# ════════════════════════════════════════════════════════════════════════════
#  TEST CASES
#  Each entry is a realistic accident scenario with a short description.
#  Raw values: (Temperature °F, Visibility mi, Humidity %)
# ════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # ── Low-risk conditions ──────────────────────────────────────────────
    {"name": "Clear summer day",
     "temp": 78.0, "vis": 10.0, "hum": 40.0,
     "expect_range": (1, 2),
     "note": "Dry, warm, high visibility — low accident risk"},

    {"name": "Mild spring morning",
     "temp": 62.0, "vis":  9.5, "hum": 50.0,
     "expect_range": (1, 2),
     "note": "Pleasant conditions, minor risk"},

    {"name": "Warm evening commute",
     "temp": 85.0, "vis":  8.0, "hum": 35.0,
     "expect_range": (1, 2),
     "note": "Hot but clear — heat alone is a mild factor"},

    # ── Moderate conditions ───────────────────────────────────────────────
    {"name": "Light rain, cool",
     "temp": 50.0, "vis":  5.0, "hum": 80.0,
     "expect_range": (2, 3),
     "note": "Reduced visibility and wet road surface"},

    {"name": "Overcast autumn day",
     "temp": 45.0, "vis":  6.0, "hum": 70.0,
     "expect_range": (2, 3),
     "note": "Moderate risk — leaves and low light"},

    {"name": "High humidity, warm",
     "temp": 88.0, "vis":  7.0, "hum": 90.0,
     "expect_range": (2, 3),
     "note": "Haze and heat stress"},

    # ── High-risk conditions ──────────────────────────────────────────────
    {"name": "Dense fog",
     "temp": 42.0, "vis":  1.0, "hum": 95.0,
     "expect_range": (3, 4),
     "note": "Very low visibility is the dominant risk factor"},

    {"name": "Winter ice storm",
     "temp": 22.0, "vis":  2.0, "hum": 92.0,
     "expect_range": (3, 4),
     "note": "Ice on road + poor visibility + cold"},

    {"name": "Heavy snowfall",
     "temp": 28.0, "vis":  0.5, "hum": 88.0,
     "expect_range": (3, 4),
     "note": "Near-zero visibility in blizzard conditions"},

    # ── Extreme / edge cases ──────────────────────────────────────────────
    {"name": "Extreme heat, desert road",
     "temp": 115.0, "vis": 10.0, "hum": 10.0,
     "expect_range": (1, 3),
     "note": "Extreme temp but perfect visibility — engine issues likely"},

    {"name": "Arctic cold snap",
     "temp": -15.0, "vis":  3.0, "hum": 60.0,
     "expect_range": (2, 4),
     "note": "Black ice, frozen mechanisms"},

    {"name": "Thick fog, freezing rain",
     "temp": 31.0, "vis":  0.2, "hum": 98.0,
     "expect_range": (3, 4),
     "note": "Worst-case scenario: near-zero vis + freezing rain"},
]


# ════════════════════════════════════════════════════════════════════════════
#  PREDICTION HELPER
# ════════════════════════════════════════════════════════════════════════════

def predict_case(model, scaler, temp_f, vis_mi, hum_pct):
    """
    Normalise raw inputs using the SAME scaler used during training,
    run DENFIS inference, and return a result dict.
    """
    raw     = np.array([[temp_f, vis_mi, hum_pct]])
    x_norm  = scaler.transform(raw)[0]              # same scale as training

    y_norm  = model.predict_one(x_norm)
    sev_f   = float(np.clip(y_norm * 3.0 + 1.0, 1.0, 4.0))
    sev_i   = int(round(sev_f))
    sev_i   = max(1, min(4, sev_i))

    # Winning fuzzy rule
    mu      = model.ecm.membership_vector(x_norm, model.sigma_scale)
    mu_sum  = mu.sum() or 1e-9
    phi     = mu / mu_sum
    winner  = int(np.argmax(phi))
    centre  = model.ecm.centres[winner]

    rule_antecedent = " AND ".join(
        f"{fname} is {_label_feature(fname, float(centre[j]))}"
        for j, fname in enumerate(FEATURE_NAMES)
    )

    return {
        "sev_float":   round(sev_f, 3),
        "sev_int":     sev_i,
        "rule_idx":    winner,
        "firing_pct":  round(phi[winner] * 100, 1),
        "antecedent":  rule_antecedent,
    }


# ════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _sev_bar(sev_i, width=24):
    filled = round(width * sev_i / 4)
    return "█" * filled + "░" * (width - filled)


def print_case(idx, case, result):
    c = SEV_COLOR[result["sev_int"]]
    in_range = case["expect_range"][0] <= result["sev_int"] <= case["expect_range"][1]
    flag = "✓" if in_range else "?"

    print(f"\n  ┌─ Case {idx:>2}: {case['name']}")
    print(f"  │  {DIM}{case['note']}{RESET}")
    print(f"  │")
    print(f"  │  Input  : Temp={case['temp']:.1f}°F  "
          f"Visibility={case['vis']:.1f}mi  Humidity={case['hum']:.0f}%")
    print(f"  │  Rule   : IF {result['antecedent']}")
    print(f"  │           (rule #{result['rule_idx']}, "
          f"firing strength {result['firing_pct']:.1f}%)")
    print(f"  │")
    print(f"  │  Severity: {c}{BOLD}{result['sev_float']:.2f}/4.00{RESET}  "
          f"{c}{_sev_bar(result['sev_int'])}{RESET}")
    print(f"  │  Class   : {c}{BOLD}Severity {result['sev_int']} — "
          f"{SEV_LABEL[result['sev_int']]}{RESET}")
    print(f"  └─ Expected range: {case['expect_range']}   {flag}")


def print_summary(results_log):
    print(f"\n\n  {'═'*64}")
    print(f"  {BOLD}SUMMARY TABLE{RESET}")
    print(f"  {'═'*64}")
    header = f"  {'#':>2}  {'Scenario':<28} {'Temp':>6} {'Vis':>5} {'Hum':>5}  {'Pred':>5}  {'Class':>3}"
    print(header)
    print(f"  {'─'*64}")

    within_range = 0
    for entry in results_log:
        case, result = entry["case"], entry["result"]
        c   = SEV_COLOR[result["sev_int"]]
        ok  = case["expect_range"][0] <= result["sev_int"] <= case["expect_range"][1]
        if ok:
            within_range += 1
        flag = "✓" if ok else "?"
        print(f"  {entry['idx']:>2}  {case['name']:<28} "
              f"{case['temp']:>5.0f}°  "
              f"{case['vis']:>4.1f}  "
              f"{case['hum']:>4.0f}%  "
              f"{c}{result['sev_float']:>5.2f}{RESET}  "
              f"{c}S{result['sev_int']}{RESET}  {flag}")

    pct = within_range / len(results_log) * 100
    print(f"  {'─'*64}")
    print(f"  Cases within expected severity range: "
          f"{BOLD}{within_range}/{len(results_log)} ({pct:.0f}%){RESET}")
    print(f"  {'═'*64}\n")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load the exported model ──────────────────────────────────────────
    model, scaler, metrics, meta = load_model("denfis_trained_model.pkl")

    # ── Print model info ─────────────────────────────────────────────────
    print(f"\n  {BOLD}Model info{RESET}")
    print(f"  {'─'*40}")
    print(f"  Fuzzy rules (ECM clusters) : {meta['n_rules']}")
    print(f"  Features                   : {', '.join(meta['feature_cols'])}")
    print(f"  D_thr                      : {meta['D_thr']}")
    print(f"  Training MSE (normalised)  : {metrics['mse']:.6f}")
    print(f"  Training MAE (Severity 1-4): {metrics['mae_severity']:.4f}")

    # ── Run all test cases ───────────────────────────────────────────────
    print(f"\n\n  {BOLD}Running {len(TEST_CASES)} test cases …{RESET}")

    results_log = []
    for idx, case in enumerate(TEST_CASES, 1):
        result = predict_case(model, scaler,
                              case["temp"], case["vis"], case["hum"])
        print_case(idx, case, result)
        results_log.append({"idx": idx, "case": case, "result": result})

    print_summary(results_log)

    # ── Show extracted rules from the loaded model ───────────────────────
    from denfis_us_accidents import extract_rules
    extract_rules(model, FEATURE_NAMES, n_rules=3)

    print("  To predict your own accident conditions, run:")
    print("      python predict_severity.py\n")


if __name__ == "__main__":
    main()
