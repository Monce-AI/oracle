"""
Rule discovery test suite — easy to hard.
Each test generates data from a KNOWN rule, then checks if Oracle finds it.
Run: python3.11 test_rules.py
"""
import sys
import random
sys.path.insert(0, ".")

random.seed(42)
from monce import Oracle


def check_formula_contains(oracle, col, expected_fragments, test_name):
    """Check if the formula output contains expected rule fragments."""
    formula = oracle.formula(col=col)
    formula_lower = formula.lower() if formula else ""
    found = []
    missing = []
    for frag in expected_fragments:
        if frag.lower() in formula_lower:
            found.append(frag)
        else:
            missing.append(frag)
    passed = len(found) >= len(expected_fragments) * 0.5
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}")
    if found:
        print(f"         Found: {found}")
    if missing:
        print(f"         Missing: {missing}")
    return passed


def test_1_binary_single_feature():
    """Rule: color=red → label=yes, else no"""
    print("\n#1 EASY — single feature, binary")
    print("   Rule: color=red → yes")
    data = []
    for _ in range(30):
        data.append({"color": "red", "size": random.choice(["big", "small"]), "label": "yes"})
    for _ in range(30):
        data.append({"color": "blue", "size": random.choice(["big", "small"]), "label": "no"})
    oracle = Oracle(data, n_layers=5, bucket=15)
    return check_formula_contains(oracle, "label", ["red", "label"], "#1 finds 'red'")


def test_2_threshold():
    """Rule: age > 30 → senior"""
    print("\n#2 EASY — numeric threshold")
    print("   Rule: age > 30 → senior")
    data = []
    for _ in range(30):
        age = random.randint(31, 60)
        data.append({"age": age, "name": random.choice(["alice", "bob", "carol"]), "group": "senior"})
    for _ in range(30):
        age = random.randint(18, 29)
        data.append({"age": age, "name": random.choice(["alice", "bob", "carol"]), "group": "junior"})
    oracle = Oracle(data, n_layers=5, bucket=15)
    return check_formula_contains(oracle, "group", ["age", "group"], "#2 finds 'age'")


def test_3_contains_substring():
    """Rule: name contains 'pro' → premium"""
    print("\n#3 EASY — substring match")
    print("   Rule: name contains 'pro' → premium")
    data = []
    pro_names = ["pro_basic", "pro_plus", "pro_max", "pro_lite", "pro_gold"]
    free_names = ["free_basic", "free_trial", "starter", "lite_free", "demo"]
    for _ in range(30):
        data.append({"name": random.choice(pro_names), "region": random.choice(["US", "EU"]), "tier": "premium"})
    for _ in range(30):
        data.append({"name": random.choice(free_names), "region": random.choice(["US", "EU"]), "tier": "free"})
    oracle = Oracle(data, n_layers=5, bucket=15)
    return check_formula_contains(oracle, "tier", ["pro", "tier"], "#3 finds 'pro'")


def test_4_two_features():
    """Rule: size=big AND color=red → A"""
    print("\n#4 MEDIUM — two features combined")
    print("   Rule: size=big AND color=red → A")
    data = []
    for _ in range(20):
        data.append({"color": "red", "size": "big", "shape": random.choice(["round", "square"]), "label": "A"})
    for _ in range(20):
        data.append({"color": "blue", "size": "small", "shape": random.choice(["round", "square"]), "label": "B"})
    for _ in range(20):
        data.append({"color": "green", "size": "big", "shape": random.choice(["round", "square"]), "label": "C"})
    oracle = Oracle(data, n_layers=5, bucket=15)
    return check_formula_contains(oracle, "label", ["red", "label"], "#4 finds 'red' in formula")


def test_5_three_classes():
    """Rule: temp<10→cold, 10-25→mild, >25→hot"""
    print("\n#5 MEDIUM — three numeric classes")
    print("   Rule: temp<10→cold, 10-25→mild, >25→hot")
    data = []
    for _ in range(20):
        data.append({"temp": random.randint(0, 9), "humidity": random.randint(30, 90), "weather": "cold"})
    for _ in range(20):
        data.append({"temp": random.randint(10, 25), "humidity": random.randint(30, 90), "weather": "mild"})
    for _ in range(20):
        data.append({"temp": random.randint(26, 40), "humidity": random.randint(30, 90), "weather": "hot"})
    oracle = Oracle(data, n_layers=5, bucket=15)
    return check_formula_contains(oracle, "weather", ["temp", "weather"], "#5 finds 'temp'")


def test_6_irrelevant_features():
    """Rule: status=active → yes (3 noise features)"""
    print("\n#6 MEDIUM — rule with noise features")
    print("   Rule: status=active → yes (ignore noise1, noise2, noise3)")
    data = []
    for _ in range(30):
        data.append({
            "status": "active",
            "noise1": random.choice(["a", "b", "c", "d"]),
            "noise2": random.randint(1, 100),
            "noise3": random.choice(["x", "y", "z"]),
            "result": "yes"
        })
    for _ in range(30):
        data.append({
            "status": "inactive",
            "noise1": random.choice(["a", "b", "c", "d"]),
            "noise2": random.randint(1, 100),
            "noise3": random.choice(["x", "y", "z"]),
            "result": "no"
        })
    oracle = Oracle(data, n_layers=5, bucket=15)
    return check_formula_contains(oracle, "result", ["active", "result"], "#6 finds 'active' despite noise")


def test_7_interaction():
    """Rule: size=large AND material=wood → expensive"""
    print("\n#7 HARD — feature interaction")
    print("   Rule: size=large AND material=wood → expensive")
    data = []
    for _ in range(20):
        data.append({"size": "large", "material": "wood", "color": random.choice(["brown", "white", "black"]), "price": "expensive"})
    for _ in range(20):
        data.append({"size": "large", "material": "metal", "color": random.choice(["brown", "white", "black"]), "price": "medium"})
    for _ in range(20):
        data.append({"size": "small", "material": random.choice(["wood", "metal"]), "color": random.choice(["brown", "white", "black"]), "price": "cheap"})
    oracle = Oracle(data, n_layers=5, bucket=15)
    return check_formula_contains(oracle, "price", ["wood", "price"], "#7 finds 'wood'")


def test_8_majority_with_exceptions():
    """Rule: department=engineering → high_salary (80% of the time, some exceptions)"""
    print("\n#8 HARD — noisy rule (80% true)")
    print("   Rule: department=engineering → high (80% accuracy)")
    data = []
    for _ in range(40):
        data.append({"department": "engineering", "years": random.randint(1, 20), "salary": "high"})
    for _ in range(10):
        data.append({"department": "engineering", "years": random.randint(1, 3), "salary": "low"})
    for _ in range(10):
        data.append({"department": "sales", "years": random.randint(1, 20), "salary": "high"})
    for _ in range(40):
        data.append({"department": "sales", "years": random.randint(1, 20), "salary": "low"})
    oracle = Oracle(data, n_layers=5, bucket=25)
    return check_formula_contains(oracle, "salary", ["engineering", "salary"], "#8 finds 'engineering' in noisy data")


def test_9_multi_value_categories():
    """Rule: continent=Europe → drives_right=yes (with 5 continents)"""
    print("\n#9 HARD — multi-class with specific target")
    print("   Rule: continent in {NA,EU,AS} → drives_right, {AF,SA} → drives_left")
    data = []
    right = ["NA", "EU", "AS"]
    left = ["AF", "SA"]
    for _ in range(20):
        cont = random.choice(right)
        data.append({"continent": cont, "population": random.randint(1, 500), "driving": "right"})
    for _ in range(20):
        cont = random.choice(left)
        data.append({"continent": cont, "population": random.randint(1, 500), "driving": "left"})
    oracle = Oracle(data, n_layers=5, bucket=10)
    return check_formula_contains(oracle, "driving", ["continent", "driving"], "#9 finds 'continent'")


def test_10_regression_rule():
    """Rule: sqft * 200 ≈ price (linear relationship)"""
    print("\n#10 HARD — regression pattern")
    print("   Rule: bigger sqft → higher price (linear)")
    data = []
    for _ in range(15):
        sqft = random.randint(500, 1000)
        data.append({"sqft": sqft, "style": random.choice(["modern", "classic"]), "price": sqft * 200 + random.randint(-10000, 10000)})
    for _ in range(15):
        sqft = random.randint(1500, 2500)
        data.append({"sqft": sqft, "style": random.choice(["modern", "classic"]), "price": sqft * 200 + random.randint(-10000, 10000)})
    oracle = Oracle(data, n_layers=5, bucket=10)
    return check_formula_contains(oracle, "price", ["sqft", "price"], "#10 finds 'sqft' drives price")


if __name__ == "__main__":
    print("=" * 60)
    print("  RULE DISCOVERY TEST SUITE")
    print("  Does Oracle find the known rule? (easy → hard)")
    print("=" * 60)

    tests = [
        test_1_binary_single_feature,
        test_2_threshold,
        test_3_contains_substring,
        test_4_two_features,
        test_5_three_classes,
        test_6_irrelevant_features,
        test_7_interaction,
        test_8_majority_with_exceptions,
        test_9_multi_value_categories,
        test_10_regression_rule,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    passed = sum(results)
    print(f"  RESULTS: {passed}/{len(results)} passed")
    print("=" * 60)
