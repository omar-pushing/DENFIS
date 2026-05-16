# Functions Reference

## train_model.py

### Top-Level Functions

| Function | Description |
|---|---|
| `fit_scaler_streaming(filepath)` | Pass 1 — streams the CSV once to collect per-feature min/max and reservoir-sampled medians; fits and returns a `MinMaxScaler` |
| `_clean_chunk(chunk, medians)` | Fills NaN with feature median, clips outliers to sensor bounds, drops remaining NaN rows and invalid severity values |
| `stream_train(model, filepath, scaler, medians, n_total)` | Pass 2 — streams the CSV, normalises features & target, feeds first 80 % of rows to `model.train_one()` sample-by-sample, accumulates last 20 % as a held-out test set |
| `_label_feature(feat_name, value)` | Maps a normalised feature value to a human-readable label (e.g. `Cold`, `Low`, `High`) |
| `_denorm_severity(y_norm)` | Converts normalised severity [0,1] back to the 1–4 scale |
| `extract_rules(model, feature_names, n_rules)` | Prints the top-N fuzzy rules ranked by coverage count, showing antecedents, linear consequents, and inferred severity at cluster centre |
| `evaluate(model, X_test, y_test)` | Computes MSE, RMSE, MAE, and severity-scale MAE on the test set; prints a performance report |
| `save_model(model, scaler, metrics, path)` | Pickles a dict with model, scaler, metrics, and metadata to disk |
| `load_model(path)` | Unpickles the bundle with a custom unpickler that remaps `__main__` imports to the `model.train_model` module |
| `main()` | Entry point — orchestrates Pass 1, model instantiation, Pass 2, evaluation, rule extraction, example prediction, and saving |

### ECM Class

| Method | Description |
|---|---|
| `__init__(D_thr)` | Sets the distance threshold for cluster creation |
| `_euclidean(a, b)` | Euclidean distance between two vectors |
| `n_clusters()` | Returns current rule count |
| `update(x)` | Adds a new cluster if distance to nearest centre exceeds `D_thr`; otherwise updates nearest centre via moving average and expands radius if needed |
| `gaussian_membership(x, idx, sigma_scale)` | Gaussian membership value for sample `x` against cluster `idx` |
| `membership_vector(x, sigma_scale)` | Returns membership values for all clusters |

### DENFIS Class

| Method | Description |
|---|---|
| `__init__(D_thr, sigma_scale, lambda_forget, rls_init_cov)` | Initialises ECM, feature count, RLS weights `theta`, and covariance matrix `P` |
| `_n_params()` | Total number of RLS parameters (= K * (n_features + 1)) |
| `_regression_vector(x)` | Builds the normalised membership-weighted regression vector for the Takagi–Sugeno consequent |
| `_expand_rls(old_K, new_K)` | Grows `theta` (zero-pad) and `P` (block-diagonal with `init_cov` on new diagonals) when a new rule is born |
| `predict_one(x)` | Single-sample forward pass: `theta @ regression_vector(x)` |
| `train_one(x, y)` | Single-sample online update: grows RLS if new rule appears, computes prediction error, applies RLS gain update |
| `fit(X, y, log_every)` | Batch-style wrapper — loops `train_one` over all samples with periodic logging |
| `predict(X)` | Batch wrapper — returns array of `predict_one` results |

## test_model.py

| Function | Description |
|---|---|
| `predict_case(model, scaler, temp_f, vis_mi, hum_pct)` | Runs one inference, denormalises severity, identifies the winning rule, returns a dict with severity float/int, rule index, firing strength, and antecedent string |
| `_sev_bar(sev_i, width)` | Returns a block-character progress bar for severity |
| `print_case(idx, case, result)` | Pretty-prints one test case with colour-coded severity bar |
| `print_summary(results_log)` | Prints a compact tabular summary of all test cases and pass/fail rate against expected ranges |
| `main()` | Loads the model, iterates all `TEST_CASES`, prints detailed output and summary |

## predict_severity.py

| Function | Description |
|---|---|
| `predict(temperature, visibility, humidity)` | One-shot prediction — scales input, runs model, returns severity in [1.0, 4.0] |
