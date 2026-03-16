# govML Example Project: Iris Classification

**This is what govML looks like in practice. Clone, run, see governance in action.**

[govML](https://github.com/rexcoleman/ml-governance-templates) is a contract-driven governance framework for machine learning projects. Instead of bolting on documentation after the fact, govML bakes reproducibility, leakage prevention, and hypothesis discipline into your workflow from day one.

This tiny project classifies Iris flowers using three algorithms. The dataset is built into scikit-learn -- no downloads, no API keys, no GPU. The entire pipeline runs in under 30 seconds on a laptop.

## What You'll See

| govML Practice | Where It Lives |
|---|---|
| Experiment contract (pre-registered design) | `docs/EXPERIMENT_CONTRACT.md` |
| Hypothesis registry (testable claims) | `docs/HYPOTHESIS_REGISTRY.md` |
| Data contract (schema + assumptions) | `docs/DATA_CONTRACT.md` |
| Leakage prevention tests | `tests/test_leakage.py` |
| Sanity baselines (dummy + shuffled) | `scripts/run_sanity_baselines.py` |
| Multi-seed reproducibility (5 seeds) | `scripts/train.py` |
| Learning curves | `scripts/run_learning_curves.py` |
| Publication-ready figures | `scripts/make_report_figures.py` |
| Test-access barrier (no peeking) | `scripts/final_eval.py` |

## Quick Start (< 5 minutes)

```bash
# Clone
git clone https://github.com/rexcoleman/govml-example-project.git
cd govml-example-project

# No special environment needed -- just base Python with scikit-learn + matplotlib
pip install scikit-learn matplotlib

# Run the full pipeline
bash reproduce.sh

# Or step by step:
python scripts/train.py                  # Train 3 algos x 5 seeds
python scripts/run_sanity_baselines.py   # Dummy + shuffled baselines
python scripts/run_learning_curves.py    # Performance vs training size
python scripts/make_report_figures.py    # Generate figures
python scripts/final_eval.py            # Final test-set evaluation
pytest tests/ -v                         # Verify governance checks pass
```

## Results

After running the pipeline, check:
- `outputs/results.json` -- per-algorithm, per-seed cross-validation scores
- `outputs/sanity_baselines.json` -- dummy and shuffled-label baselines
- `outputs/learning_curves.json` -- performance at 5 training fractions
- `outputs/figures/` -- publication-ready PNG figures
- `outputs/final_eval.json` -- held-out test set metrics
- `FINDINGS.md` -- template for writing up your conclusions

## Why This Matters

Most ML tutorials skip governance entirely. You train a model, get a number, and move on. In production and in research, that leads to:
- Leaked test data inflating metrics
- Cherry-picked seeds hiding instability
- No baseline proving the model actually learned something
- Irreproducible results

govML prevents all of these with lightweight contracts and automated checks. This project is the proof.

## License

MIT

## Links

- [govML repository](https://github.com/rexcoleman/ml-governance-templates)
- [govML documentation](https://github.com/rexcoleman/ml-governance-templates#readme)
