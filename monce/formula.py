"""
Formula extraction from Snake models.
Discovers human-readable IF→THEN rules, ranked by lift × p-value.
"""

import math
from collections import defaultdict


def _binomial_p_value(successes, trials, base_rate):
    """One-sided binomial test: P(X >= successes) given trials and base_rate."""
    if trials == 0 or base_rate >= 1.0:
        return 1.0
    if successes <= trials * base_rate:
        return 1.0
    mu = trials * base_rate
    sigma = math.sqrt(trials * base_rate * (1 - base_rate))
    if sigma == 0:
        return 0.0 if successes > mu else 1.0
    z = (successes - 0.5 - mu) / sigma
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return max(p, 1e-300)


class Rule:
    __slots__ = ("target_col", "target_val", "conditions", "coverage", "accuracy",
                 "support", "p_value", "lift", "layer_idx", "bucket_idx",
                 "is_regression", "candle_stats")

    def __init__(self, target_col, target_val, conditions, coverage, accuracy,
                 support, p_value, lift, layer_idx, bucket_idx,
                 is_regression=False, candle_stats=None):
        self.target_col = target_col
        self.target_val = target_val
        self.conditions = conditions
        self.coverage = coverage
        self.accuracy = accuracy
        self.support = support
        self.p_value = p_value
        self.lift = lift
        self.layer_idx = layer_idx
        self.bucket_idx = bucket_idx
        self.is_regression = is_regression
        self.candle_stats = candle_stats

    @property
    def formula_text(self):
        cond_str = " AND ".join(self.conditions)
        if self.is_regression and self.candle_stats:
            cs = self.candle_stats
            return f"IF {cond_str} → {self.target_col} ≈ {cs['median']:.2f} (IQR [{cs['q1']:.2f}, {cs['q3']:.2f}])"
        return f"IF {cond_str} → {self.target_col} = {self.target_val}"

    @property
    def significance(self):
        if self.p_value < 1e-10:
            return "★★★"
        elif self.p_value < 1e-5:
            return "★★"
        elif self.p_value < 0.01:
            return "★"
        return ""

    def __repr__(self):
        sig = self.significance
        stars = f" {sig}" if sig else ""
        if self.is_regression:
            return f"{self.formula_text}  [lift={self.lift:.1f}x, n={self.coverage}{stars}]"
        return f"{self.formula_text}  [acc={self.accuracy:.0%}, lift={self.lift:.1f}x, n={self.coverage}{stars}]"


def extract_formulas_for_row(model, X, target_col):
    """Extract the active formulas (rules) that fire for a single row X."""
    X = model._normalize_features(model._as_single(X))
    rules = []
    for layer_idx, chain in enumerate(model.layers):
        matched_bucket_idx = None
        routing_parts = []

        for b_idx, entry in enumerate(chain):
            cond = entry["condition"]
            if cond is None:
                matched_bucket_idx = b_idx
            else:
                if all(model.apply_literal(X, lit) for lit in cond):
                    matched_bucket_idx = b_idx
                    for lit in cond:
                        routing_parts.append(model._format_literal_text(lit))
                    break
                else:
                    fail_lit = model._first_failing_literal(X, cond)
                    if fail_lit is not None:
                        negated = model._negate_literal(fail_lit)
                        routing_parts.append(model._format_literal_text(negated))

        if matched_bucket_idx is None:
            continue

        bucket = chain[matched_bucket_idx]
        clause_bool = [model.apply_clause(X, c) for c in bucket["clauses"]]
        negated_set = {i for i, val in enumerate(clause_bool) if not val}

        lookalike_targets = defaultdict(list)
        for l in bucket["lookalikes"]:
            for condition in bucket["lookalikes"][l]:
                if all(c_idx in negated_set for c_idx in condition):
                    global_idx = bucket["members"][int(l)]
                    target_val = model.targets[global_idx]
                    and_parts = []
                    seen = set()
                    for c_idx in condition:
                        if c_idx < len(bucket["clauses"]):
                            for lit in bucket["clauses"][c_idx]:
                                negated = model._negate_literal(lit)
                                desc = model._format_literal_text(negated)
                                if desc not in seen:
                                    seen.add(desc)
                                    and_parts.append(desc)
                    lookalike_targets[str(target_val)].append(and_parts)

        for target_val, all_parts in lookalike_targets.items():
            combined = routing_parts.copy()
            literal_counts = defaultdict(int)
            for parts in all_parts:
                for p in parts:
                    literal_counts[p] += 1
            frequent = [p for p, c in literal_counts.items() if c >= len(all_parts) * 0.5]
            combined.extend(frequent)

            if combined:
                rules.append(Rule(
                    target_col=target_col,
                    target_val=target_val,
                    conditions=combined,
                    coverage=len(all_parts),
                    accuracy=1.0,
                    support=len(bucket["members"]),
                    p_value=0.0,
                    lift=1.0,
                    layer_idx=layer_idx,
                    bucket_idx=matched_bucket_idx,
                ))

    return rules


def extract_formulas_for_dataset(model, records, target_col, top_n=12):
    """Extract top classification formulas ranked by lift × p-value."""
    rule_bank = defaultdict(lambda: {"hits": 0, "correct": 0, "target_val": None})

    n_classes = len(set(str(model.targets[i]) for i in range(len(model.targets))))
    base_rate = 1.0 / n_classes if n_classes > 0 else 0.5

    for record in records:
        features = {k: v for k, v in record.items() if k != target_col}
        features = model._normalize_features(model._as_single(features))
        actual = str(record.get(target_col, ""))

        for layer_idx, chain in enumerate(model.layers):
            matched_bucket_idx = None
            routing_parts = []

            for b_idx, entry in enumerate(chain):
                cond = entry["condition"]
                if cond is None:
                    matched_bucket_idx = b_idx
                else:
                    if all(model.apply_literal(features, lit) for lit in cond):
                        matched_bucket_idx = b_idx
                        for lit in cond:
                            routing_parts.append(model._format_literal_text(lit))
                        break
                    else:
                        fail_lit = model._first_failing_literal(features, cond)
                        if fail_lit is not None:
                            negated = model._negate_literal(fail_lit)
                            routing_parts.append(model._format_literal_text(negated))

            if matched_bucket_idx is None:
                continue

            bucket = chain[matched_bucket_idx]
            clause_bool = [model.apply_clause(features, c) for c in bucket["clauses"]]
            negated_set = {i for i, val in enumerate(clause_bool) if not val}

            target_literals = defaultdict(list)
            for l in bucket["lookalikes"]:
                for condition in bucket["lookalikes"][l]:
                    if all(c_idx in negated_set for c_idx in condition):
                        global_idx = bucket["members"][int(l)]
                        target_val = str(model.targets[global_idx])
                        and_parts = []
                        seen = set()
                        for c_idx in condition:
                            if c_idx < len(bucket["clauses"]):
                                for lit in bucket["clauses"][c_idx]:
                                    negated = model._negate_literal(lit)
                                    desc = model._format_literal_text(negated)
                                    if desc not in seen:
                                        seen.add(desc)
                                        and_parts.append(desc)
                        target_literals[target_val].append(and_parts)

            for target_val, all_parts in target_literals.items():
                combined = tuple(sorted(set(routing_parts)))
                literal_counts = defaultdict(int)
                for parts in all_parts:
                    for p in parts:
                        literal_counts[p] += 1
                frequent = tuple(sorted(p for p, c in literal_counts.items()
                                        if c >= max(1, len(all_parts) * 0.5)))
                key = (target_val, combined + frequent)
                rule_bank[key]["hits"] += 1
                rule_bank[key]["target_val"] = target_val
                if actual == target_val:
                    rule_bank[key]["correct"] += 1

    rules = []
    n_records = len(records)
    for (target_val, cond_tuple), stats in rule_bank.items():
        hits = stats["hits"]
        correct = stats["correct"]
        if hits < 2:
            continue
        accuracy = correct / hits if hits > 0 else 0.0
        lift = accuracy / base_rate if base_rate > 0 else 1.0
        p_value = _binomial_p_value(correct, hits, base_rate)
        conditions = list(cond_tuple)
        if not conditions:
            continue
        rules.append(Rule(
            target_col=target_col,
            target_val=target_val,
            conditions=conditions,
            coverage=hits,
            accuracy=accuracy,
            support=n_records,
            p_value=p_value,
            lift=lift,
            layer_idx=0,
            bucket_idx=0,
        ))

    rules.sort(key=lambda r: (r.p_value, -r.lift))
    return rules[:top_n]


def extract_regression_formulas(model, records, target_col, top_n=12):
    """Extract formulas for regression: routing conditions → candle distribution."""
    # Compute overall stats
    all_values = []
    for r in records:
        try:
            all_values.append(float(r[target_col]))
        except (ValueError, TypeError, KeyError):
            pass

    if not all_values:
        return []

    overall_mean = sum(all_values) / len(all_values)
    overall_std = math.sqrt(sum((v - overall_mean) ** 2 for v in all_values) / len(all_values))
    if overall_std == 0:
        return []

    # Group records by bucket per layer, compute candle stats per group
    bucket_groups = defaultdict(list)

    for record in records:
        features = {k: v for k, v in record.items() if k != target_col}
        features = model._normalize_features(model._as_single(features))
        try:
            actual = float(record[target_col])
        except (ValueError, TypeError):
            continue

        for layer_idx, chain in enumerate(model.layers):
            routing_parts = []
            matched_bucket_idx = None

            for b_idx, entry in enumerate(chain):
                cond = entry["condition"]
                if cond is None:
                    matched_bucket_idx = b_idx
                else:
                    if all(model.apply_literal(features, lit) for lit in cond):
                        matched_bucket_idx = b_idx
                        for lit in cond:
                            routing_parts.append(model._format_literal_text(lit))
                        break
                    else:
                        fail_lit = model._first_failing_literal(features, cond)
                        if fail_lit is not None:
                            negated = model._negate_literal(fail_lit)
                            routing_parts.append(model._format_literal_text(negated))

            if matched_bucket_idx is not None:
                key = (layer_idx, matched_bucket_idx, tuple(sorted(set(routing_parts))))
                bucket_groups[key].append(actual)

    rules = []
    for (layer_idx, bucket_idx, cond_tuple), values in bucket_groups.items():
        if len(values) < 3 or not cond_tuple:
            continue

        values_sorted = sorted(values)
        n = len(values_sorted)
        median = values_sorted[n // 2]
        q1 = values_sorted[n // 4]
        q3 = values_sorted[3 * n // 4]
        group_mean = sum(values) / n
        group_std = math.sqrt(sum((v - group_mean) ** 2 for v in values) / n)

        # Lift = how much tighter this group is vs overall
        lift = overall_std / group_std if group_std > 0 else overall_std * 10

        # P-value: is this group's mean significantly different from overall?
        se = overall_std / math.sqrt(n)
        z = abs(group_mean - overall_mean) / se if se > 0 else 0
        p_value = 2 * 0.5 * math.erfc(z / math.sqrt(2))
        p_value = max(p_value, 1e-300)

        rules.append(Rule(
            target_col=target_col,
            target_val=f"≈{median:.2f}",
            conditions=list(cond_tuple),
            coverage=n,
            accuracy=0.0,
            support=len(records),
            p_value=p_value,
            lift=lift,
            layer_idx=layer_idx,
            bucket_idx=bucket_idx,
            is_regression=True,
            candle_stats={"median": median, "q1": q1, "q3": q3,
                          "mean": group_mean, "std": group_std, "n": n},
        ))

    rules.sort(key=lambda r: (-r.lift, r.p_value))
    return rules[:top_n]


def format_formulas_md(rules, top_n=12):
    """Format rules as a markdown report."""
    if not rules:
        return "_No significant formulas discovered._"

    lines = []
    lines.append(f"| # | Formula | Lift | Evidence | Sig |")
    lines.append(f"|---|---------|------|----------|-----|")

    for i, rule in enumerate(rules[:top_n]):
        cond_str = " AND ".join(rule.conditions[:4])
        if len(rule.conditions) > 4:
            cond_str += f" (+{len(rule.conditions)-4})"

        if rule.is_regression:
            cs = rule.candle_stats
            formula = f"IF {cond_str} → **{rule.target_col}** ≈ {cs['median']:.2f}"
            evidence = f"n={rule.coverage}, IQR [{cs['q1']:.2f}, {cs['q3']:.2f}]"
        else:
            formula = f"IF {cond_str} → **{rule.target_col}** = {rule.target_val}"
            evidence = f"acc={rule.accuracy:.0%}, n={rule.coverage}/{rule.support}"

        lines.append(f"| {i+1} | {formula} | {rule.lift:.1f}x | {evidence} | {rule.significance} |")

    return "\n".join(lines)
