"""
pip install -e .
python example.py
"""
import pandas as pd
from monce import Oracle

oracle = Oracle(pd.read_csv("train.csv"))

print("# Survival Formulas")
print(oracle.formula(col="Survived"))
print()

print("# Fare Regression")
print(oracle.formula(col="Fare"))
print()

rose = {"Pclass": "1", "Sex": "female", "Age": "17", "SibSp": "1", "Parch": "1", "Fare": "512", "Embarked": "C"}
print(f"Rose survived? {oracle.predict('Survived', rose)}")
print(f"Probability:   {oracle.probability('Survived', rose)}")
print()

jack = {"Pclass": "3", "Sex": "male", "Age": "20", "SibSp": "0", "Parch": "0", "Survived": "0", "Embarked": "S"}
print(f"Jack's fare:   {oracle.regression('Fare', jack):.2f}")
print(f"Fare candle:   {oracle.candle('Fare', jack)}")
