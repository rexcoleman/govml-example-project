# Experiment Contract

**Project:** Iris Classification govML Demo
**Date:** 2026-03-16
**Status:** PRE-REGISTERED

## Design

| Parameter | Value |
|---|---|
| Dataset | scikit-learn Iris (150 samples, 4 features, 3 classes) |
| Split | 80/20 stratified train/test, seed-controlled |
| Validation | Stratified 5-fold CV on training set only |
| Seeds | 42, 123, 456, 789, 1024 |
| Metric | Accuracy (macro, since classes are balanced) |

## Algorithms

| ID | Algorithm | Key Hyperparameters |
|---|---|---|
| A1 | Logistic Regression | C=1.0, max_iter=1000, solver=lbfgs |
| A2 | Random Forest | n_estimators=100, max_depth=None |
| A3 | SVM (RBF kernel) | C=1.0, gamma=scale |

## Stopping Rules

- All 3 algorithms x 5 seeds x 5 folds = 75 fits must complete without error.
- No algorithm may score below the stratified dummy baseline.
- If any seed produces accuracy < 0.80, investigate before proceeding.

## Test-Access Barrier

The held-out test set (20%) is accessed ONLY in `final_eval.py`, AFTER all CV experiments and hypothesis adjudication are complete. No script other than `final_eval.py` may load or evaluate on the test split.
