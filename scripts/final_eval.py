"""
Final evaluation on the held-out test set.
This script is the ONLY place where test data is accessed.
Run AFTER all CV experiments and hypothesis adjudication are complete.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SEED = 42  # Use primary seed for final eval


def main():
    # Load results to find best algorithm
    results_path = OUTPUTS_DIR / "results.json"
    if not results_path.exists():
        print("ERROR: Run train.py first to generate results.json")
        return

    with open(results_path) as f:
        data = json.load(f)

    summary = data["summary"]
    best_algo = max(summary, key=lambda k: summary[k]["grand_mean"])
    print(f"Best algorithm by CV grand mean: {best_algo} "
          f"({summary[best_algo]['grand_mean']:.4f})")

    # Load data and create the SAME split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Train best model on full training set
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=SEED
        ),
        "SVM_RBF": SVC(C=1.0, kernel="rbf", gamma="scale", random_state=SEED),
    }

    model = models[best_algo]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=iris.target_names,
                                   output_dict=True)

    print(f"\n--- Final Test Set Evaluation ---")
    print(f"Algorithm: {best_algo}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Save
    output = {
        "best_algorithm": best_algo,
        "cv_grand_mean": summary[best_algo]["grand_mean"],
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "seed": SEED,
        "test_size": len(y_test),
        "train_size": len(y_train),
    }
    out_path = OUTPUTS_DIR / "final_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Final evaluation saved to {out_path}")


if __name__ == "__main__":
    main()
