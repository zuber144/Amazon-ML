# Experiments

Each experiment should be documented here with:

| Field | Description |
|-------|-------------|
| `name` | Short experiment identifier |
| `date` | Date run |
| `blocking_strategies` | List of strategies used |
| `features` | Feature set description |
| `model` | Model type and key hyperparameters |
| `threshold` | Decision threshold used |
| `val_precision` | Validation precision |
| `val_recall` | Validation recall |
| `val_f0_5` | Validation F0.5 (primary metric) |
| `candidate_recall` | Blocking candidate recall |
| `runtime_minutes` | Total pipeline runtime |
| `notes` | Key observations |

## Experiment Log

| Name | Date | Blocking | Model | Threshold | Precision | Recall | F0.5 | CandRecall | Runtime | Notes |
|------|------|----------|-------|-----------|-----------|--------|------|------------|---------|-------|
| baseline-001 | TBD | exact_name+token_name | logistic_regression | tuned | TBD | TBD | TBD | TBD | TBD | Initial baseline |

## Hypothesis Tracking

- [ ] Does TF-IDF blocking significantly improve candidate recall vs. token blocking?
- [ ] Does gradient_boosting outperform logistic_regression on F0.5?
- [ ] Does country-aware blocking help or hurt recall?
- [ ] What is the optimal MAX_CANDIDATES_PER_S1 cap?
