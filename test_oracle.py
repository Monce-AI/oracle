"""
Test Oracle — exercises every method on real data.
Run: python3.11 test_oracle.py
"""
import sys
sys.path.insert(0, ".")

from monce import Oracle

# ─── Classification dataset (Iris-style) ───
classification_data = [
    {"sepal_l": 5.1, "sepal_w": 3.5, "petal_l": 1.4, "petal_w": 0.2, "species": "setosa"},
    {"sepal_l": 4.9, "sepal_w": 3.0, "petal_l": 1.4, "petal_w": 0.2, "species": "setosa"},
    {"sepal_l": 4.7, "sepal_w": 3.2, "petal_l": 1.3, "petal_w": 0.2, "species": "setosa"},
    {"sepal_l": 5.0, "sepal_w": 3.6, "petal_l": 1.4, "petal_w": 0.2, "species": "setosa"},
    {"sepal_l": 5.4, "sepal_w": 3.9, "petal_l": 1.7, "petal_w": 0.4, "species": "setosa"},
    {"sepal_l": 4.6, "sepal_w": 3.4, "petal_l": 1.4, "petal_w": 0.3, "species": "setosa"},
    {"sepal_l": 5.0, "sepal_w": 3.4, "petal_l": 1.5, "petal_w": 0.2, "species": "setosa"},
    {"sepal_l": 7.0, "sepal_w": 3.2, "petal_l": 4.7, "petal_w": 1.4, "species": "versicolor"},
    {"sepal_l": 6.4, "sepal_w": 3.2, "petal_l": 4.5, "petal_w": 1.5, "species": "versicolor"},
    {"sepal_l": 6.9, "sepal_w": 3.1, "petal_l": 4.9, "petal_w": 1.5, "species": "versicolor"},
    {"sepal_l": 5.5, "sepal_w": 2.3, "petal_l": 4.0, "petal_w": 1.3, "species": "versicolor"},
    {"sepal_l": 6.5, "sepal_w": 2.8, "petal_l": 4.6, "petal_w": 1.5, "species": "versicolor"},
    {"sepal_l": 5.7, "sepal_w": 2.8, "petal_l": 4.5, "petal_w": 1.3, "species": "versicolor"},
    {"sepal_l": 6.3, "sepal_w": 3.3, "petal_l": 6.0, "petal_w": 2.5, "species": "virginica"},
    {"sepal_l": 5.8, "sepal_w": 2.7, "petal_l": 5.1, "petal_w": 1.9, "species": "virginica"},
    {"sepal_l": 7.1, "sepal_w": 3.0, "petal_l": 5.9, "petal_w": 2.1, "species": "virginica"},
    {"sepal_l": 6.3, "sepal_w": 2.9, "petal_l": 5.6, "petal_w": 1.8, "species": "virginica"},
    {"sepal_l": 6.5, "sepal_w": 3.0, "petal_l": 5.8, "petal_w": 2.2, "species": "virginica"},
    {"sepal_l": 7.6, "sepal_w": 3.0, "petal_l": 6.6, "petal_w": 2.1, "species": "virginica"},
    {"sepal_l": 7.2, "sepal_w": 3.6, "petal_l": 6.1, "petal_w": 2.5, "species": "virginica"},
]

# ─── Regression dataset ───
regression_data = [
    {"sqft": 800, "beds": 1, "location": "downtown", "price": 250000},
    {"sqft": 1200, "beds": 2, "location": "downtown", "price": 400000},
    {"sqft": 1500, "beds": 3, "location": "suburbs", "price": 350000},
    {"sqft": 2000, "beds": 4, "location": "suburbs", "price": 450000},
    {"sqft": 900, "beds": 1, "location": "suburbs", "price": 200000},
    {"sqft": 1100, "beds": 2, "location": "downtown", "price": 380000},
    {"sqft": 1800, "beds": 3, "location": "downtown", "price": 550000},
    {"sqft": 2200, "beds": 4, "location": "suburbs", "price": 500000},
    {"sqft": 1000, "beds": 2, "location": "suburbs", "price": 280000},
    {"sqft": 1600, "beds": 3, "location": "downtown", "price": 480000},
    {"sqft": 700, "beds": 1, "location": "suburbs", "price": 180000},
    {"sqft": 2500, "beds": 5, "location": "suburbs", "price": 600000},
    {"sqft": 1300, "beds": 2, "location": "downtown", "price": 420000},
    {"sqft": 1900, "beds": 3, "location": "suburbs", "price": 400000},
    {"sqft": 3000, "beds": 5, "location": "downtown", "price": 800000},
]


def test_classification():
    print("=" * 60)
    print("  CLASSIFICATION TEST (Iris-style)")
    print("=" * 60)

    oracle = Oracle(classification_data, n_layers=3, bucket=10)
    print(f"\n{oracle}")

    X = {"sepal_l": 5.1, "sepal_w": 3.5, "petal_l": 1.4, "petal_w": 0.2}

    print(f"\n--- predict('species', X) ---")
    print(oracle.predict("species", X))

    print(f"\n--- probability('species', X) ---")
    print(oracle.probability("species", X))

    print(f"\n--- lookalikes('species', X) ---")
    lk = oracle.lookalikes("species", X)
    print(f"  {len(lk)} lookalikes found")

    print(f"\n--- lookalikes_labeled('species', X) ---")
    lkl = oracle.lookalikes_labeled("species", X)
    print(f"  {len(lkl)} labeled lookalikes")
    if lkl:
        print(f"  First: {lkl[0]}")

    print(f"\n--- augmented('species', X) ---")
    aug = oracle.augmented("species", X)
    print(f"  Keys: {list(aug.keys())}")

    print(f"\n--- audit('species', X) ---")
    audit = oracle.audit("species", X)
    print(audit[:500] + "..." if len(audit) > 500 else audit)

    print(f"\n--- anomaly_score(row) ---")
    row = classification_data[0]
    score = oracle.anomaly_score(row)
    print(f"  Overall: {score['overall']:.2f}")

    print(f"\n--- fill(row, 'species') ---")
    incomplete = {"sepal_l": 5.1, "sepal_w": 3.5, "petal_l": 1.4, "petal_w": 0.2}
    print(f"  Filled: {oracle.fill(incomplete, 'species')}")

    print(f"\n--- correlations() ---")
    for col, score, ctype in oracle.correlations():
        print(f"  {col}: {score:.2f} ({ctype})")

    print(f"\n--- ask('what determines species?') ---")
    print(oracle.ask("what determines species?"))

    print(f"\n--- formula(X) ---")
    print(oracle.formula(X, col="species"))

    print(f"\n--- formula() [all training data] ---")
    print(oracle.formula())


def test_regression():
    print("\n" + "=" * 60)
    print("  REGRESSION TEST (Housing)")
    print("=" * 60)

    oracle = Oracle(regression_data, n_layers=3, bucket=8)
    print(f"\n{oracle}")

    X = {"sqft": 1500, "beds": 3, "location": "downtown"}

    print(f"\n--- predict('price', X) ---")
    print(oracle.predict("price", X))

    print(f"\n--- regression('price', X) ---")
    print(oracle.regression("price", X))

    print(f"\n--- candle('price', X) ---")
    print(oracle.candle("price", X))

    print(f"\n--- probability('price', X) ---")
    prob = oracle.probability("price", X)
    top3 = sorted(prob.items(), key=lambda x: -x[1])[:3]
    print(f"  Top 3: {top3}")

    print(f"\n--- audit('price', X) ---")
    audit = oracle.audit("price", X)
    print(audit[:500] + "..." if len(audit) > 500 else audit)

    print(f"\n--- fill(row, 'price') ---")
    incomplete = {"sqft": 2000, "beds": 4, "location": "suburbs"}
    print(f"  Filled: {oracle.fill(incomplete, 'price')}")

    print(f"\n--- correlations() ---")
    for col, score, ctype in oracle.correlations():
        print(f"  {col}: {score:.2f} ({ctype})")

    print(f"\n--- formula() ---")
    print(oracle.formula())


def test_context():
    print("\n" + "=" * 60)
    print("  CONTEXT TEST (LLM-ready output)")
    print("=" * 60)

    oracle = Oracle(classification_data, n_layers=3, bucket=10)
    ctx = oracle.context()
    print(f"\n--- context() [{len(ctx)} chars] ---")
    print(ctx)


if __name__ == "__main__":
    test_classification()
    test_regression()
    test_context()
    print("\n✓ All methods exercised successfully")
