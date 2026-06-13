"""
Verify README claims about Titanic survival and fare pricing.
Run: python test/titanic_analysis.py
"""
from pathlib import Path
import pandas as pd
from monce import Oracle

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "titanic.csv")
oracle = Oracle(df, n_layers=5)

print("=" * 60)
print("CLAIM 1: 98.5% survival accuracy")
print("=" * 60)
correct = 0
for _, row in df.iterrows():
    features = {k: str(v) for k, v in row.items() if k != "Survived"}
    pred = oracle.predict("Survived", features)
    if str(pred) == str(row["Survived"]):
        correct += 1
acc = correct / len(df)
print(f"  Accuracy: {acc:.1%} ({correct}/{len(df)})")
print(f"  PASS" if acc >= 0.98 else f"  FAIL")
print()

print("=" * 60)
print("CLAIM 2: Women in 1st/2nd class survived at ~100%")
print("=" * 60)
subset = df[(df["Sex"] == "female") & (df["Pclass"].isin([1, 2]))]
survived = subset["Survived"].mean()
print(f"  Women 1st/2nd class survival rate: {survived:.1%} ({int(subset['Survived'].sum())}/{len(subset)})")
print(f"  PASS" if survived >= 0.95 else f"  FAIL")
print()

print("=" * 60)
print("CLAIM 3: Men in 3rd class died at ~100%")
print("=" * 60)
subset = df[(df["Sex"] == "male") & (df["Pclass"] == 3)]
died = 1 - subset["Survived"].mean()
print(f"  Men 3rd class death rate: {died:.1%} ({int((1-subset['Survived']).sum())}/{len(subset)})")
print(f"  PASS" if died >= 0.85 else f"  FAIL")
print()

print("=" * 60)
print("CLAIM 4: 3rd class from Southampton paid ~£9.50")
print("=" * 60)
subset = df[(df["Pclass"] == 3) & (df["Embarked"] == "S")]
median_fare = subset["Fare"].median()
print(f"  Median fare: £{median_fare:.2f}")
print(f"  PASS" if 7 <= median_fare <= 12 else f"  FAIL")
print()

print("=" * 60)
print("CLAIM 5: Cabin D passengers paid £53-63")
print("=" * 60)
subset = df[df["Cabin"].str.startswith("D", na=False)]
median_fare = subset["Fare"].median()
print(f"  Median fare: £{median_fare:.2f} (n={len(subset)})")
print(f"  PASS" if 40 <= median_fare <= 70 else f"  FAIL")
print()

print("=" * 60)
print("CLAIM 6: Fare regression works")
print("=" * 60)
jack = {"Pclass": "3", "Sex": "male", "Age": "20", "SibSp": "0", "Parch": "0", "Survived": "0", "Embarked": "S"}
fare = oracle.regression("Fare", jack)
print(f"  Predicted fare for 3rd class male: £{fare:.2f}")
print(f"  PASS" if 5 <= fare <= 40 else f"  FAIL")
