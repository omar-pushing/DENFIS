# DENFIS — Dynamic Evolving Neural-Fuzzy Inference System for Accident Severity Prediction

Predicts US accident severity (1–4) from weather conditions (temperature, visibility, humidity) using an online-streaming neuro-fuzzy model that grows its own rule base from data.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Streaming data pipeline, ECM clustering, DENFIS model definition, online training, evaluation, rule extraction, and model persistence |
| `test_model.py` | 12 predefined weather scenarios with expected severity ranges; prints detailed per-case breakdown, summary table, and extracted fuzzy rules |
| `predict_severity.py` | Lightweight inference API — loads the saved model and exposes a `predict(temperature, visibility, humidity)` function |
| `model.pkl` | Serialised model bundle (DENFIS + scaler + metrics + metadata), produced by `train_model.py` and consumed by the other two scripts |

## Quickstart

```bash
python train_model.py      # Train (expects US_Accidents_March23.csv in cwd)
python test_model.py       # Evaluate on 12 hand-crafted scenarios
python predict_severity.py # Interactive one-shot prediction
```
