"""
Titanic showcase — Oracle discovers survival rules from the Kaggle dataset.
Run: python3.11 test_titanic.py
"""
import sys
import csv
sys.path.insert(0, ".")

from monce import Oracle

# Load Titanic CSV
with open("titanic.csv", "r") as f:
    reader = csv.DictReader(f)
    data = list(reader)

print(f"Loaded {len(data)} passengers, {len(data[0])} columns")
print(f"Columns: {list(data[0].keys())}")

# Train Oracle on Titanic — focus on key columns
# Keep it manageable: select useful features
keep = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
records = []
for row in data:
    r = {}
    for k in keep:
        v = row.get(k, "")
        if v == "":
            continue
        r[k] = v
    if len(r) == len(keep):
        records.append(r)

print(f"\nUsing {len(records)} complete rows × {len(keep)} columns")
print()

# Train
print("Training Oracle...")
oracle = Oracle(records, n_layers=5, bucket=50)
print(f"{oracle}")

# ─── Formula: what determines survival? ───
print("\n" + "=" * 70)
print("  FORMULAS — What determines survival?")
print("=" * 70)
print(oracle.formula(col="Survived"))

# ─── Context: LLM-ready snippet ───
print("\n" + "=" * 70)
print("  CONTEXT — LLM-ready snippet")
print("=" * 70)
print(oracle.context(col="Survived"))

# ─── Predict a specific passenger ───
print("\n" + "=" * 70)
print("  PREDICT — Rose (1st class, female, young)")
print("=" * 70)
rose = {"Pclass": "1", "Sex": "female", "Age": "17", "SibSp": "0", "Parch": "1", "Fare": "110", "Embarked": "S"}
print(f"Prediction: {oracle.predict('Survived', rose)}")
print(f"Probability: {oracle.probability('Survived', rose)}")

# ─── Predict Jack ───
print("\n" + "=" * 70)
print("  PREDICT — Jack (3rd class, male, young)")
print("=" * 70)
jack = {"Pclass": "3", "Sex": "male", "Age": "20", "SibSp": "0", "Parch": "0", "Fare": "8", "Embarked": "S"}
print(f"Prediction: {oracle.predict('Survived', jack)}")
print(f"Probability: {oracle.probability('Survived', jack)}")

# ─── Audit for Jack ───
print("\n" + "=" * 70)
print("  AUDIT — Why Jack dies (first 600 chars)")
print("=" * 70)
audit = oracle.audit("Survived", jack)
print(audit[:600])

# ─── Correlations ───
print("\n" + "=" * 70)
print("  CORRELATIONS — Which columns are most predictable?")
print("=" * 70)
for col, score, ctype in oracle.correlations():
    print(f"  {col:12s} {score:.3f} ({ctype})")

# ─── Formula for row ───
print("\n" + "=" * 70)
print("  FORMULA(rose) — What rules fire for Rose?")
print("=" * 70)
print(oracle.formula(rose, col="Survived"))

# ─── Fare regression ───
print("\n" + "=" * 70)
print("  FARE REGRESSION — What determines ticket price?")
print("=" * 70)
print(f"Fare regression (1st class): {oracle.regression('Fare', {'Pclass': '1', 'Sex': 'male', 'Age': '30', 'SibSp': '0', 'Parch': '0', 'Survived': '1', 'Embarked': 'C'})}")
print(f"Fare regression (3rd class): {oracle.regression('Fare', {'Pclass': '3', 'Sex': 'male', 'Age': '25', 'SibSp': '0', 'Parch': '0', 'Survived': '0', 'Embarked': 'S'})}")
print()
print("Fare formulas:")
print(oracle.formula(col="Fare"))

# ─── Accuracy check ───
print("\n" + "=" * 70)
print("  ACCURACY — How good is the survival model?")
print("=" * 70)
correct = 0
for record in records:
    features = {k: v for k, v in record.items() if k != "Survived"}
    pred = oracle.predict("Survived", features)
    if str(pred) == str(record["Survived"]):
        correct += 1
acc = correct / len(records)
print(f"  Training accuracy (Survived): {acc:.1%} ({correct}/{len(records)})")
