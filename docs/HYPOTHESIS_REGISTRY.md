# Hypothesis Registry

**Pre-registered before any experiments are run.**

## H-1: Random Forest outperforms Logistic Regression

- **Rationale:** Iris has known nonlinear decision boundaries between versicolor and virginica. RF captures these without explicit feature engineering.
- **Test:** Mean CV accuracy (RF) > Mean CV accuracy (LogReg), averaged across 5 seeds.
- **Threshold:** Difference > 0.01 (1 percentage point) to be considered meaningful.
- **Result:** _TO BE FILLED AFTER EXPERIMENT_

## H-2: SVM-RBF outperforms SVM-Linear (implicit via LogReg comparison)

- **Rationale:** RBF kernel maps features to higher-dimensional space, capturing the nonlinear versicolor/virginica boundary.
- **Test:** Mean CV accuracy (SVM-RBF) > Mean CV accuracy (LogReg), since LogReg is equivalent to a linear SVM for this comparison.
- **Threshold:** Difference > 0.005 to be considered meaningful.
- **Result:** _TO BE FILLED AFTER EXPERIMENT_

## Adjudication Protocol

Results are adjudicated using the mean +/- std across 5 seeds. If standard deviations overlap substantially, the hypothesis is marked INCONCLUSIVE rather than CONFIRMED or REJECTED.
