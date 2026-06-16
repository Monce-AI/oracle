"""
Oracle — question your data.
from monce import Oracle
oracle = Oracle(df)
oracle.predict("Survived", row)   # instant, progressive accuracy
oracle.formula()
"""

import os
import math
import random
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from algorithmeai import Snake
from .formula import (
    extract_formulas_for_row,
    extract_formulas_for_dataset,
    extract_regression_formulas,
    format_formulas_md,
)


_TIERS = [
    {"n_layers": 10, "bucket": 50},
    {"n_layers": 20, "bucket": 150},
    {"n_layers": 40, "bucket": 300},
    {"n_layers": 80, "bucket": 500},
]


class Oracle:
    def __init__(self, data, noise=0.25, workers=1, budget=None,
                 n_layers=None, bucket=None, columns=None, target=None):
        if hasattr(data, "to_dict"):
            self._records = data.to_dict(orient="records")
            self._columns = list(data.columns)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            self._records = data
            self._columns = list(data[0].keys())
        else:
            raise ValueError("Oracle accepts a DataFrame or list[dict]")

        if columns:
            self._columns = [c for c in columns if c in self._columns]
            self._records = [{k: r[k] for k in self._columns if k in r} for r in self._records]

        self._max_columns = 50
        self._target = target
        self._noise = noise
        self._workers = workers
        self._budget = budget
        self._errors = []

        if n_layers or bucket:
            self._tiers = [{"n_layers": n_layers or 20, "bucket": bucket or 150}]
        else:
            self._tiers = _TIERS

        self._tier = 0
        self._max_tier = len(self._tiers) if budget is None else min(budget, len(self._tiers))

        self.models = {}
        self._lock = threading.Lock()
        self._ready_count = 0
        self._total = len(self._columns)
        self._col_types = {}
        self._col_stats = {}
        self._training = True
        self._tier_complete = threading.Event()

        self._compute_stats()

        if len(self._columns) > self._max_columns:
            self._columns = self._mi_filter_columns(self._max_columns)

        self._total = len(self._columns)

        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()
        self._tier_complete.wait()

    def _compute_stats(self):
        for col in self._columns:
            values = [r[col] for r in self._records if col in r and r[col] != ""]
            if not values:
                continue
            str_values = [str(v) for v in values]
            n_unique = len(set(str_values))
            n_total = len(values)

            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    pass

            is_numeric = len(numeric_values) > n_total * 0.8
            is_regression = is_numeric and n_unique > min(20, n_total * 0.5)

            stats = {
                "n_total": n_total,
                "n_unique": n_unique,
                "n_missing": len(self._records) - n_total,
                "type": "regression" if is_regression else "classification",
            }

            if is_numeric and numeric_values:
                stats["min"] = min(numeric_values)
                stats["max"] = max(numeric_values)
                stats["mean"] = statistics.mean(numeric_values)
                stats["median"] = statistics.median(numeric_values)
                if len(numeric_values) > 1:
                    stats["std"] = statistics.stdev(numeric_values)
                else:
                    stats["std"] = 0.0
            else:
                counter = Counter(str_values)
                stats["top_values"] = counter.most_common(10)
                stats["base_rate"] = counter.most_common(1)[0][1] / n_total if n_total else 0

            self._col_stats[col] = stats
            self._col_types[col] = stats["type"]

    def _mi_filter_columns(self, max_cols):
        """Shannon MI column filter. Computes pairwise MI(col; target) directly
        from the joint histogram, with bias correction for cardinality.
        Runs in O(n * k^2) — fast, no model training needed."""
        n = min(len(self._records), 500)
        col_mi_scores = defaultdict(float)

        # Pick scout targets: the target col + up to 9 low-cardinality classification cols
        scout_targets = []
        if self._target and self._target in self._columns:
            scout_targets.append(self._target)
        for c in self._columns:
            if c not in scout_targets:
                stats = self._col_stats.get(c, {})
                if stats.get("type") == "classification" and stats.get("n_unique", 999) <= 20:
                    scout_targets.append(c)
            if len(scout_targets) >= 10:
                break
        if not scout_targets:
            scout_targets = self._columns[:5]

        records = self._records[:n]

        for target_col in scout_targets:
            target_vals = [str(r.get(target_col, "")) for r in records]
            target_counts = Counter(target_vals)
            k_target = len(target_counts)

            for feat_col in self._columns:
                if feat_col == target_col:
                    continue

                feat_vals = [str(r.get(feat_col, "")) for r in records]
                feat_counts = Counter(feat_vals)
                k_feat = len(feat_counts)

                # Joint histogram
                joint = Counter(zip(feat_vals, target_vals))

                # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
                mi = 0.0
                for (fv, tv), count in joint.items():
                    p_xy = count / n
                    p_x = feat_counts[fv] / n
                    p_y = target_counts[tv] / n
                    if p_xy > 0 and p_x > 0 and p_y > 0:
                        mi += p_xy * math.log2(p_xy / (p_x * p_y))

                # Bias correction: E[MI] under independence ~ (k_feat-1)(k_target-1) / (2*n*ln2)
                bias = (k_feat - 1) * (k_target - 1) / (2.0 * n * math.log(2))
                mi_corrected = max(0.0, mi - bias)

                col_mi_scores[feat_col] += mi_corrected
                col_mi_scores[target_col] += mi_corrected

        ranked = sorted(self._columns, key=lambda c: col_mi_scores.get(c, 0.0), reverse=True)

        if self._target and self._target in self._columns:
            if self._target not in ranked[:max_cols]:
                ranked = [self._target] + [c for c in ranked if c != self._target]

        kept = ranked[:max_cols]
        self._mi_scores = {c: col_mi_scores.get(c, 0.0) for c in kept}
        return kept

    def _detect_col_type(self, col):
        return self._col_types.get(col, "classification")

    def _sample_for_tier(self, tier_idx):
        tier = self._tiers[tier_idx]
        if tier_idx == 0 and len(self._tiers) > 1:
            max_rows = 100
            cols = self._select_columns(0)
            source = self._records[:max_rows]
            return [{k: r[k] for k in cols if k in r} for r in source], cols
        else:
            max_rows = min(len(self._records), tier["bucket"] * 4)
        cols = self._select_columns(tier_idx)
        if len(self._records) <= max_rows:
            return self._records, cols
        step = len(self._records) // max_rows
        return self._records[::step][:max_rows], cols

    def _select_columns(self, tier_idx):
        if tier_idx == 0 and len(self._tiers) > 1:
            fast_cols = []
            for col in self._columns:
                vals = [str(r.get(col, "")) for r in self._records[:100]]
                n_unique = len(set(vals))
                max_len = max((len(v) for v in vals), default=0)
                if n_unique <= 20 and max_len <= 10:
                    fast_cols.append(col)
            return fast_cols[:10] if fast_cols else self._columns[:6]
        return self._columns

    def _train_tier(self, tier_idx):
        tier = self._tiers[tier_idx]
        records, columns = self._sample_for_tier(tier_idx)

        def train_col(col):
            col_type = self._detect_col_type(col)
            model = Snake(
                records,
                target_index=col,
                n_layers=tier["n_layers"],
                bucket=tier["bucket"],
                noise=self._noise,
                workers=self._workers,
            )
            with self._lock:
                self.models[col] = model
                self._col_types[col] = col_type
                self._ready_count = len(self.models)
            return col

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            futures = {pool.submit(train_col, col): col for col in columns}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    col = futures[f]
                    with self._lock:
                        self._errors.append((tier_idx, col, str(e)))

    def _train_loop(self):
        try:
            for tier_idx in range(self._max_tier):
                self._train_tier(tier_idx)
                self._tier = tier_idx + 1
                self._tier_complete.set()
        except Exception as e:
            with self._lock:
                self._errors.append((-1, "loop", str(e)))
            self._tier_complete.set()
        finally:
            self._training = False

    def _require_model(self, col):
        if col not in self.models:
            self._tier_complete.wait()
        if col not in self.models:
            raise KeyError(f"No model for column '{col}'")
        return self.models[col]

    def _features(self, row, col):
        return {k: v for k, v in row.items() if k != col}

    # ─── Properties ───

    @property
    def ready(self):
        return self._ready_count

    @property
    def total(self):
        return self._total

    @property
    def tier(self):
        return self._tier

    @property
    def training(self):
        return self._training

    # ─── Snake method wrappers ───

    def _resolve_col(self, col):
        if col is None:
            if self._target:
                return self._target
            raise ValueError("No target column set. Pass col= or set target= in constructor.")
        return col

    def predict(self, col=None, features=None, **kwargs):
        col = self._resolve_col(col)
        if features is None:
            features = kwargs
        return self._require_model(col).get_prediction(features)

    def probability(self, col=None, features=None, **kwargs):
        col = self._resolve_col(col)
        if features is None:
            features = kwargs
        return self._require_model(col).get_probability(features)

    def regression(self, col=None, features=None, **kwargs):
        col = self._resolve_col(col)
        if features is None:
            features = kwargs
        return self._require_model(col).get_regression(features)

    def candle(self, col=None, features=None, **kwargs):
        col = self._resolve_col(col)
        if features is None:
            features = kwargs
        return self._require_model(col).get_candle(features)

    def audit(self, col=None, features=None, **kwargs):
        col = self._resolve_col(col)
        if features is None:
            features = kwargs
        return self._require_model(col).get_audit(features)

    def lookalikes(self, col=None, features=None, **kwargs):
        col = self._resolve_col(col)
        if features is None:
            features = kwargs
        return self._require_model(col).get_lookalikes(features)

    def lookalikes_labeled(self, col=None, features=None, **kwargs):
        col = self._resolve_col(col)
        if features is None:
            features = kwargs
        return self._require_model(col).get_lookalikes_labeled(features)

    def augmented(self, col=None, features=None, **kwargs):
        col = self._resolve_col(col)
        if features is None:
            features = kwargs
        return self._require_model(col).get_augmented(features)

    # ─── Intelligence methods ───

    def anomaly_score(self, row):
        scores = {}
        for col, model in self.models.items():
            features = self._features(row, col)
            prob = model.get_probability(features)
            actual = str(row.get(col, ""))
            confidence = prob.get(actual, 0.0)
            scores[col] = confidence
        avg = sum(scores.values()) / len(scores) if scores else 0.0
        return {"per_column": scores, "overall": avg}

    def fill(self, row, col):
        features = self._features(row, col)
        if self._col_types.get(col) == "regression":
            return self.regression(col, features)
        return self.predict(col, features)

    def correlations(self):
        pairs = []
        for col, model in self.models.items():
            if self._col_types.get(col) == "regression":
                values = [float(r[col]) for r in self._records if col in r]
                mean_val = sum(values) / len(values) if values else 0
                ss_tot = sum((v - mean_val) ** 2 for v in values)
                ss_res = 0
                for record in self._records:
                    features = self._features(record, col)
                    pred = model.get_regression(features)
                    actual = float(record[col])
                    ss_res += (actual - pred) ** 2
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                pairs.append((col, r2, "regression"))
            else:
                correct = 0
                for record in self._records:
                    features = self._features(record, col)
                    pred = model.get_prediction(features)
                    if str(pred) == str(record[col]):
                        correct += 1
                acc = correct / len(self._records) if self._records else 0
                pairs.append((col, acc, "classification"))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def ask(self, question):
        q = question.lower()
        best_col = None
        best_score = -1
        for col in self._columns:
            col_lower = col.lower()
            if col_lower in q:
                score = len(col_lower)
                if score > best_score:
                    best_score = score
                    best_col = col
        if best_col and best_col in self.models:
            model = self.models[best_col]
            col_type = self._col_types.get(best_col, "classification")
            if col_type == "regression":
                return {
                    "target": best_col,
                    "type": "regression",
                    "insight": f"Snake can regress '{best_col}' from the other columns. Use oracle.regression('{best_col}', features) or oracle.candle('{best_col}', features).",
                }
            acc = 0
            for record in self._records[:50]:
                features = self._features(record, best_col)
                pred = model.get_prediction(features)
                if str(pred) == str(record[best_col]):
                    acc += 1
            acc = acc / min(50, len(self._records))
            return {
                "target": best_col,
                "type": "classification",
                "accuracy": f"{acc:.0%}",
                "insight": f"Snake can predict '{best_col}' with {acc:.0%} accuracy from the other columns.",
            }
        return {
            "target": None,
            "insight": f"Available columns: {', '.join(self._columns)}. Try asking about one of them.",
        }

    # ─── Formula ───

    def formula(self, X_or_df=None, col=None, top_n=12):
        """
        .formula(X)  — rules that fire for one row
        .formula(df) — top rules across dataset, ranked by lift × p-value
        .formula()   — top rules across training data for all models
        """
        if X_or_df is None:
            return self._formula_all(top_n, col)

        if hasattr(X_or_df, "to_dict"):
            X_or_df = X_or_df.to_dict(orient="records")

        if isinstance(X_or_df, list):
            return self._formula_dataset(X_or_df, col, top_n)

        if isinstance(X_or_df, dict):
            return self._formula_row(X_or_df, col)

        return {}

    def _formula_all(self, top_n, col=None):
        all_rules = []
        targets = [col] if col else list(self.models.keys())
        for c in targets:
            if c not in self.models:
                continue
            model = self.models[c]
            if self._col_types.get(c) == "regression":
                rules = extract_regression_formulas(model, self._records, c, top_n=top_n)
            else:
                rules = extract_formulas_for_dataset(model, self._records, c, top_n=top_n)
            all_rules.extend(rules)
        all_rules.sort(key=lambda r: (r.p_value, -r.lift))
        return format_formulas_md(all_rules[:top_n])

    def _formula_dataset(self, records, col, top_n):
        targets = [col] if col else list(self.models.keys())
        all_rules = []
        for c in targets:
            if c not in self.models:
                continue
            if self._col_types.get(c) == "regression":
                rules = extract_regression_formulas(self.models[c], records, c, top_n=top_n)
            else:
                rules = extract_formulas_for_dataset(self.models[c], records, c, top_n=top_n)
            all_rules.extend(rules)
        all_rules.sort(key=lambda r: (r.p_value, -r.lift))
        return format_formulas_md(all_rules[:top_n])

    def _formula_row(self, X, col):
        targets = [col] if col else list(self.models.keys())
        all_rules = []
        for c in targets:
            if c not in self.models:
                continue
            features = self._features(X, c)
            rules = extract_formulas_for_row(self.models[c], features, c)
            all_rules.extend(rules)
        if not all_rules:
            return "_No formulas fired for this row._"
        lines = []
        for r in all_rules:
            lines.append(f"- {r.formula_text}  \n  `coverage={r.coverage}`")
        return "\n".join(lines)

    # ─── Score ───

    def score(self, col=None):
        col = self._resolve_col(col) if col else None
        targets = [col] if col else list(self.models.keys())
        results = {}
        for c in targets:
            if c not in self.models:
                continue
            model = self.models[c]
            if self._col_types.get(c) == "regression":
                values = []
                preds = []
                for record in self._records:
                    try:
                        actual = float(record[c])
                    except (ValueError, TypeError, KeyError):
                        continue
                    features = self._features(record, c)
                    pred = model.get_regression(features)
                    values.append(actual)
                    preds.append(pred)
                if len(values) < 2:
                    results[c] = {"type": "regression", "r2": 0.0, "n": 0}
                    continue
                mean_val = sum(values) / len(values)
                ss_tot = sum((v - mean_val) ** 2 for v in values)
                ss_res = sum((a - p) ** 2 for a, p in zip(values, preds))
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                results[c] = {"type": "regression", "r2": round(r2, 4), "n": len(values)}
            else:
                correct = 0
                total = 0
                for record in self._records:
                    if c not in record:
                        continue
                    features = self._features(record, c)
                    pred = model.get_prediction(features)
                    if str(pred) == str(record[c]):
                        correct += 1
                    total += 1
                acc = correct / total if total else 0
                results[c] = {"type": "classification", "accuracy": round(acc, 4), "n": total}
        if col and col in results:
            return results[col]
        return results

    # ─── Context ───

    def context(self, col=None):
        lines = []
        lines.append("## Dataset Context")
        lines.append(f"**{len(self._records)} rows × {len(self._columns)} columns**")
        lines.append("")

        # Column descriptions
        lines.append("### Columns")
        for c in self._columns:
            stats = self._col_stats.get(c, {})
            ct = stats.get("type", "unknown")
            n_unique = stats.get("n_unique", "?")
            n_missing = stats.get("n_missing", 0)
            desc = f"- **{c}** ({ct})"
            if ct == "regression":
                mn = stats.get("min", "?")
                mx = stats.get("max", "?")
                mean = stats.get("mean", "?")
                desc += f" — range [{mn}, {mx}], mean={mean:.1f}" if isinstance(mean, float) else ""
            else:
                top = stats.get("top_values", [])
                if top:
                    top_str = ", ".join(f"{v}({n})" for v, n in top[:5])
                    desc += f" — {n_unique} unique: {top_str}"
            if n_missing > 0:
                desc += f" [{n_missing} missing]"
            lines.append(desc)
        lines.append("")

        # Sample rows
        lines.append("### Sample Rows")
        n_sample = min(5, len(self._records))
        step = max(1, len(self._records) // n_sample)
        sample = self._records[::step][:n_sample]
        if sample:
            headers = list(sample[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in sample:
                vals = [str(row.get(h, ""))[:20] for h in headers]
                lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

        # Predictability
        lines.append("### Predictability")
        scores = self.score()
        if isinstance(scores, dict) and "type" not in scores:
            for c, s in sorted(scores.items(), key=lambda x: -list(x[1].values())[1] if len(x[1]) > 1 else 0):
                if s["type"] == "regression":
                    lines.append(f"- **{c}**: R²={s['r2']:.3f}")
                else:
                    lines.append(f"- **{c}**: accuracy={s['accuracy']:.1%}")
        lines.append("")

        # Formulas
        lines.append("### Discovered Rules")
        formula_md = self.formula(col=col, top_n=8)
        if formula_md and "No significant" not in formula_md:
            lines.append(formula_md)
        else:
            lines.append("_No significant formulas discovered._")
        lines.append("")

        # Human summary
        lines.append("### Summary")
        lines.append(self._human_summary(col))

        return "\n".join(lines)

    def _human_summary(self, col=None):
        parts = []
        n = len(self._records)
        parts.append(f"This dataset has {n} records across {len(self._columns)} fields.")

        # Strongest predictors
        scores = self.score()
        if isinstance(scores, dict) and "type" not in scores:
            strong = [(c, s) for c, s in scores.items()
                      if (s["type"] == "classification" and s.get("accuracy", 0) > 0.8)
                      or (s["type"] == "regression" and s.get("r2", 0) > 0.7)]
            if strong:
                descs = []
                for c, s in strong[:3]:
                    if s["type"] == "classification":
                        descs.append(f"'{c}' is predictable at {s['accuracy']:.0%}")
                    else:
                        descs.append(f"'{c}' is explained (R²={s['r2']:.2f})")
                parts.append(" ".join(descs) + ".")

        # Key relationships
        for c in self._columns:
            stats = self._col_stats.get(c, {})
            if stats.get("type") == "classification":
                top = stats.get("top_values", [])
                if top and len(top) == 2:
                    v1, n1 = top[0]
                    v2, n2 = top[1]
                    parts.append(f"'{c}' is binary: {v1} ({n1}/{n}), {v2} ({n2}/{n}).")
                    break

        return " ".join(parts)

    def __repr__(self):
        status = "training" if self._training else "ready"
        return f"Oracle(tier={self._tier}/{self._max_tier} {status}, {len(self._records)} rows)"
