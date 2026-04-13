---
fp: "FP-00"
title: "Iris Classification with govML Governance"
quality_score: 5.0
last_scored: "2026-03-20"
status: "COMPLETE"
audience_side: "of-ai"
sharing_tier: "1_publish"
---

# Findings

**Project:** Iris Classification with govML Governance
**Date:** _FILL IN_
**Author:** _FILL IN_

## 1. Data Summary

- Dataset: scikit-learn Iris (150 samples, 4 features, 3 balanced classes)
- Train/test split: 120/30, stratified, seed-controlled
- No missing values, no preprocessing required

## 2. Sanity Baselines

| Baseline | Accuracy |
|---|---|
| Dummy (stratified) | _RUN run_sanity_baselines.py_ |
| RF on shuffled labels | _RUN run_sanity_baselines.py_ |

**Interpretation:** _All models must beat the dummy baseline. Shuffled-label accuracy should be near chance (0.33)._

## 3. Cross-Validation Results (5 seeds x 5-fold)

| Algorithm | Grand Mean | Grand Std |
|---|---|---|
| Logistic Regression | _RUN train.py_ | _RUN train.py_ |
| Random Forest | _RUN train.py_ | _RUN train.py_ |
| SVM (RBF) | _RUN train.py_ | _RUN train.py_ |

## 4. Hypothesis Adjudication

### H-1: Random Forest > Logistic Regression
- **Result:** _CONFIRMED / REJECTED / INCONCLUSIVE_
- **Evidence:** _Grand mean difference and std overlap_

### H-2: SVM-RBF > LogReg (linear proxy)
- **Result:** _CONFIRMED / REJECTED / INCONCLUSIVE_
- **Evidence:** _Grand mean difference and std overlap_

## 5. Learning Curves

See `outputs/figures/learning_curves.png`.

**Interpretation:** _Do models improve with more data? Any signs of overfitting (train >> val)?_

## 6. Final Test Set Evaluation

| Metric | Value |
|---|---|
| Best algorithm | _RUN final_eval.py_ |
| Test accuracy | _RUN final_eval.py_ |
| CV grand mean | _RUN final_eval.py_ |
| CV-test gap | _COMPUTE_ |

**Interpretation:** _A small gap between CV and test accuracy indicates the CV estimate was reliable._

## 7. Governance Checklist

- [ ] Experiment contract followed (3 algos x 5 seeds x 5 folds)
- [ ] Hypotheses pre-registered before experiments
- [ ] Sanity baselines computed and beaten
- [ ] Leakage tests pass (pytest tests/test_leakage.py)
- [ ] Test set accessed only in final_eval.py
- [ ] All results reproducible (reproduce.sh)
