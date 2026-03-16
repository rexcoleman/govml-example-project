# Data Contract

**Dataset:** scikit-learn built-in Iris dataset
**Source:** `sklearn.datasets.load_iris()`

## Schema

| Feature | Type | Unit | Range |
|---|---|---|---|
| sepal_length | float64 | cm | [4.3, 7.9] |
| sepal_width | float64 | cm | [2.0, 4.4] |
| petal_length | float64 | cm | [1.0, 6.9] |
| petal_width | float64 | cm | [0.1, 2.5] |
| target | int64 | class | {0: setosa, 1: versicolor, 2: virginica} |

## Assumptions

1. **No missing values.** Iris is complete by construction.
2. **Balanced classes.** 50 samples per class (150 total).
3. **No temporal component.** Samples are i.i.d.
4. **No leakage risk from features.** All 4 features are physical measurements with no target encoding.

## Integrity Check

Before any experiment, verify:
- Shape is (150, 4)
- Target has exactly 3 unique values
- No NaN values
- Class balance is 50/50/50
