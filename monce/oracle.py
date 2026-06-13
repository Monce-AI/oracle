"""
Oracle — question your data.
from monce import Oracle
oracle = Oracle(df)
oracle.formula()
oracle.context()
"""

import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from algorithmeai import Snake
from .formula import (
    extract_formulas_for_row,
    extract_formulas_for_dataset,
    extract_regression_formulas,
    format_formulas_md,
)


class Oracle:
    def __init__(self, data, n_layers=5, bucket=250, noise=0.25, workers=1):
        if hasattr(data, "to_dict"):
            self._records = data.to_dict(orient="records")
            self._columns = list(data.columns)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            self._records = data
            self._columns = list(data[0].keys())
        else:
            raise ValueError("Oracle accepts a DataFrame or list[dict]")

        self._config = {
            "n_layers": n_layers,
            "bucket": bucket,
            "noise": noise,
            "workers": workers,
        }

        self.models = {}
        self._lock = threading.Lock()
        self._ready_count = 0
        self._total = len(self._columns)
        self._col_types = {}

        self._train_all()

    def _detect_col_type(self, col):
        """Detect if column is classification or regression (continuous)."""
        values = [r[col] for r in self._records if col in r]
        if not values:
            return "classification"
        n_unique = len(set(str(v) for v in values))
        if n_unique > min(20, len(values) * 0.5):
            try:
                [float(v) for v in values[:20]]
                return "regression"
            except (ValueError, TypeError):
                pass
        return "classification"

    def _train_one(self, col):
        col_type = self._detect_col_type(col)
        model = Snake(
            self._records,
            target_index=col,
            n_layers=self._config["n_layers"],
            bucket=self._config["bucket"],
            noise=self._config["noise"],
            workers=self._config["workers"],
        )
        with self._lock:
            self.models[col] = model
            self._col_types[col] = col_type
            self._ready_count += 1
        return col

    def _train_all(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            futures = {pool.submit(self._train_one, col): col for col in self._columns}
            for future in as_completed(futures):
                future.result()

    def _require_model(self, col):
        if col not in self.models:
            raise KeyError(f"No model for column '{col}' yet")
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

    # ─── Snake method wrappers ───

    def predict(self, col, features):
        return self._require_model(col).get_prediction(features)

    def probability(self, col, features):
        return self._require_model(col).get_probability(features)

    def regression(self, col, features):
        return self._require_model(col).get_regression(features)

    def candle(self, col, features):
        return self._require_model(col).get_candle(features)

    def audit(self, col, features):
        return self._require_model(col).get_audit(features)

    def lookalikes(self, col, features):
        return self._require_model(col).get_lookalikes(features)

    def lookalikes_labeled(self, col, features):
        return self._require_model(col).get_lookalikes_labeled(features)

    def augmented(self, col, features):
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
                # R² approximation via candle
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

    # ─── Context ───

    def context(self, col=None):
        """
        LLM-ready snippet: sample rows + discovered formulas.
        Designed to be injected into an LLM prompt for data understanding.
        """
        lines = []
        lines.append("## Dataset Context")
        lines.append(f"**{len(self._records)} rows × {len(self._columns)} columns**")
        lines.append(f"Columns: {', '.join(self._columns)}")
        lines.append("")

        # Sample rows
        lines.append("### Sample Rows")
        sample_indices = random.sample(range(len(self._records)), min(5, len(self._records)))
        sample = [self._records[i] for i in sample_indices]
        if sample:
            headers = list(sample[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in sample:
                vals = [str(row.get(h, ""))[:20] for h in headers]
                lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

        # Column types
        lines.append("### Column Types")
        for c in self._columns:
            ct = self._col_types.get(c, "unknown")
            lines.append(f"- **{c}**: {ct}")
        lines.append("")

        # Top formulas
        lines.append("### Discovered Formulas")
        formula_md = self.formula(col=col, top_n=8)
        if formula_md:
            lines.append(formula_md)
        else:
            lines.append("_No significant formulas discovered._")

        return "\n".join(lines)

    def __repr__(self):
        return f"Oracle({self._ready_count}/{self._total} models ready, {len(self._records)} rows)"
