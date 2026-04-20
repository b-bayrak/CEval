"""
CEval — Counterfactual Explanation Evaluator
=============================================

A lightweight, dependency-minimal Python package for evaluating the quality of
counterfactual explanations produced by any XAI explainer.

Supported metrics
-----------------
| Metric              | Description                                              |
|---------------------|----------------------------------------------------------|
| validity            | Fraction of CFs that actually flip the model's label     |
| proximity           | Average feature-space distance to the original instance  |
| proximity_gower     | Same as proximity but using the Gower mixed-type metric  |
| sparsity            | Fraction of features changed on average                  |
| count               | Average number of CFs generated per instance             |
| diversity           | Determinant-based spread of the CF set                   |
| diversity_lcc       | Diversity weighted by label-class coverage               |
| yNN                 | Label consistency among k nearest neighbours of the CF   |
| feasibility         | Average kNN distance of CFs to the training set          |
| kNLN_dist           | Distance of CF to its nearest like-labelled neighbour    |
| relative_dist       | dist(x, CF) / dist(x, NUN)                               |
| redundancy          | Average number of unnecessary feature changes            |
| plausibility        | dist(CF, NLN) / dist(NLN, NUN(NLN))                     |
| constraint_violation| Fraction of CFs that break user-defined constraints      |

References
----------
Bayrak, B., & Bach, K. (2024). Evaluation of Instance-Based Explanations: An
In-Depth Analysis of Counterfactual Evaluation Metrics, Challenges, and the CEval
Toolkit. *IEEE Access*.
https://ieeexplore.ieee.org/document/10550910
"""

from __future__ import annotations

from typing import List, Optional, Union

import gower
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.neighbors import NearestNeighbors
import category_encoders as ce

__version__ = "1.1.0"
__author__  = "Betül Bayrak"
__email__   = "betul.bayrak@ntnu.no"
__all__     = ["CEval"]

from .wrappers import (
    BaseWrapper,
    XGBoostWrapper,
    LightGBMWrapper,
    CatBoostWrapper,
    TorchWrapper,
    KerasWrapper,
    GenericWrapper,
    wrap_model,
)

# ---------------------------------------------------------------------------
# Feature-availability table
# ---------------------------------------------------------------------------
_FEATURES_DICT = {
    "generated":      {"validity":"x","proximity":"x","proximity_gower":"x","sparsity":"x","count":"x","diversity":"x","diversity_lcc":"x","yNN":"x","feasibility":"x","kNLN_dist":"x","relative_dist":"x","redundancy":"x","plausibility":"x","constraint_violation":"x"},
    "existed":        {"validity":"x","proximity":"x","proximity_gower":"x","sparsity":"x","count":"x","diversity":"x","diversity_lcc":"x","yNN":"x","feasibility":"x","kNLN_dist":"-","relative_dist":"-","redundancy":"x","plausibility":"-","constraint_violation":"x"},
    "factual":        {"validity":"-","proximity":"x","proximity_gower":"x","sparsity":"-","count":"x","diversity":"x","diversity_lcc":"-","yNN":"x","feasibility":"x","kNLN_dist":"x","relative_dist":"-","redundancy":"-","plausibility":"x","constraint_violation":"x"},
    "counterfactual": {"validity":"x","proximity":"x","proximity_gower":"x","sparsity":"x","count":"x","diversity":"x","diversity_lcc":"x","yNN":"x","feasibility":"x","kNLN_dist":"x","relative_dist":"x","redundancy":"x","plausibility":"x","constraint_violation":"x"},
    "1to1":           {"validity":"x","proximity":"x","proximity_gower":"x","sparsity":"x","count":"-","diversity":"-","diversity_lcc":"-","yNN":"x","feasibility":"x","kNLN_dist":"x","relative_dist":"x","redundancy":"x","plausibility":"x","constraint_violation":"x"},
    "1toN":           {"validity":"x","proximity":"x","proximity_gower":"x","sparsity":"x","count":"x","diversity":"x","diversity_lcc":"x","yNN":"x","feasibility":"x","kNLN_dist":"x","relative_dist":"x","redundancy":"x","plausibility":"x","constraint_violation":"x"},
}

_VALID_DISTANCES = {
    "braycurtis","canberra","chebyshev","jaccard","hamming",
    "cosine","sqeuclidean","cityblock","minkowski","euclidean",
}

_VALID_EXP_TYPES = [
    "existed-cf", "existed-factual",
    "generated-cf", "generated-factual",
]


class CEval:
    """Evaluate counterfactual explanations across multiple quality metrics.

    Parameters
    ----------
    samples : pd.DataFrame
        The instances to be explained.  Must contain the target ``label`` column.
    label : str
        Name of the target / class column in ``samples`` (and ``data``).
    data : pd.DataFrame, optional
        Full background dataset (features + label).  Required for metrics that
        compare against the training distribution (feasibility, kNLN_dist, etc.).
    model : object, optional
        A fitted classifier.  Accepts **any** of the following:

        * A **scikit-learn** compatible model (anything with ``predict`` +
          ``predict_proba``) — pass directly.
        * An **XGBoost**, **LightGBM**, or **CatBoost** model — wrap with
          :class:`~ceval.wrappers.XGBoostWrapper`,
          :class:`~ceval.wrappers.LightGBMWrapper`, or
          :class:`~ceval.wrappers.CatBoostWrapper`.
        * A **PyTorch** ``nn.Module`` — wrap with
          :class:`~ceval.wrappers.TorchWrapper`.
        * A **Keras / TensorFlow** model — wrap with
          :class:`~ceval.wrappers.KerasWrapper`.
        * Anything else — wrap with
          :class:`~ceval.wrappers.GenericWrapper` by supplying callables.

        Required for validity, yNN, redundancy, and plausibility metrics.
    k_nn : int, default 5
        Number of neighbours used in kNN-based metrics.
    explainer_name : str, optional
        Convenience argument to register the first explainer immediately.
    explanations : pd.DataFrame, optional
        Convenience argument paired with ``explainer_name``.
    explanation_type : str, optional
        One of ``"generated-cf"``, ``"generated-factual"``, ``"existed-cf"``,
        ``"existed-factual"``.
    explanation_mode : {"1to1", "1toN"}, default "1to1"
        ``"1to1"`` — one explanation per sample (rows must align).
        ``"1toN"`` — multiple explanations per sample; the DataFrame must have
        an ``"instance"`` column with the index of the source sample.
    encoder : str, optional
        Name of a ``category_encoders`` encoder class for categorical features.
        Defaults to ``OrdinalEncoder`` when ``None``.
    distance : str, optional
        A ``scipy.spatial.distance.cdist`` metric name for proximity/sparsity
        calculations.  When ``None`` a built-in mixed-type metric is used.
    constraints : list of str, optional
        Feature names that must not change in a valid counterfactual.

    Examples
    --------
    **scikit-learn model (no wrapper needed):**

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from ceval import CEval
    >>>
    >>> clf = RandomForestClassifier().fit(X_train, y_train)
    >>> evaluator = CEval(samples=test_df, label="income",
    ...                   data=train_df, model=clf)
    >>> evaluator.add_explainer("MyExplainer", counterfactuals, "generated-cf")
    >>> print(evaluator.comparison_table)

    **PyTorch model:**

    >>> from ceval.wrappers import TorchWrapper
    >>> wrapped = TorchWrapper(my_net, num_classes=2, device="cpu")
    >>> evaluator = CEval(samples=test_df, label="income",
    ...                   data=train_df, model=wrapped)

    **XGBoost model:**

    >>> from ceval.wrappers import XGBoostWrapper
    >>> wrapped = XGBoostWrapper(xgb_clf)
    >>> evaluator = CEval(samples=test_df, label="income",
    ...                   data=train_df, model=wrapped)

    **Custom / unsupported framework:**

    >>> from ceval.wrappers import GenericWrapper
    >>> wrapped = GenericWrapper(
    ...     predict_fn       = lambda X: my_model.infer(X).argmax(1),
    ...     predict_proba_fn = lambda X: my_model.infer(X),
    ... )
    >>> evaluator = CEval(samples=test_df, label="income",
    ...                   data=train_df, model=wrapped)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        samples: pd.DataFrame,
        label: str,
        data: Optional[pd.DataFrame] = None,
        model=None,
        k_nn: int = 5,
        explainer_name: Optional[str] = None,
        explanations: Optional[pd.DataFrame] = None,
        explanation_type: Optional[str] = None,
        explanation_mode: str = "1to1",
        encoder: Optional[str] = None,
        distance: Optional[str] = None,
        constraints: Optional[List[str]] = None,
    ) -> None:
        if not isinstance(samples, pd.DataFrame):
            raise TypeError("'samples' must be a pandas DataFrame.")
        if len(samples) == 0:
            raise ValueError("'samples' must contain at least one instance.")
        if label not in samples.columns:
            raise ValueError(f"Label column '{label}' not found in samples.")

        self.label       = label
        self.knn         = k_nn
        self.data        = data
        self.model       = wrap_model(model)   # normalise to sklearn interface
        self.constraints = constraints or []

        self.explainer_names: List[str] = []
        self.metric_features = pd.DataFrame(_FEATURES_DICT)
        self.metrics         = self.metric_features.index.tolist()
        self.comparison_table = pd.DataFrame(columns=self.metrics)

        # Samples
        self.s_count       = len(samples)
        self.column_names  = samples.columns.tolist()
        self.samples       = samples
        self.samples_X     = samples.drop(columns=[label])
        self.numeric_names = self.samples_X.select_dtypes(include=np.number).columns.tolist()
        self.column_names_X = self.samples_X.columns.tolist()
        self.categorical_names = self.samples_X.select_dtypes(include=object).columns.tolist()

        self._set_distance(distance)

        # Background data (optional)
        if self.data is not None:
            if label not in data.columns:
                raise ValueError(f"Label column '{label}' not found in data.")
            self.data_X   = data.drop(columns=[label])
            self.data_y   = data[label]
            self.num_label = len(np.unique(self.data_y))
            self._set_encoder(encoder)
            self.data_X_enc  = self._encode(self.data_X)
            self.data_enc    = self.data_X_enc.copy()
            self.data_enc[label] = self.data_y.values
            self.kNN_model   = self._knn_fit(self.knn, self.data_X_enc)

        # Optional shortcut: add first explainer during construction
        if explainer_name is not None and explanations is not None and explanation_type is not None:
            self.add_explainer(explainer_name, explanations, explanation_type, explanation_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_explainer(
        self,
        name: str,
        explanations: pd.DataFrame,
        exp_type: str,
        mode: str = "1to1",
    ) -> None:
        """Register an explainer and compute all applicable metrics.

        Parameters
        ----------
        name : str
            Human-readable name for the explainer (used as the row label in
            ``comparison_table``).
        explanations : pd.DataFrame
            The counterfactual (or factual) explanations produced by the explainer.
            Must include the ``label`` column and, in ``"1toN"`` mode, an
            ``"instance"`` column.
        exp_type : str
            One of ``"generated-cf"``, ``"generated-factual"``,
            ``"existed-cf"``, ``"existed-factual"``.
        mode : {"1to1", "1toN"}, default "1to1"
            Explanation multiplicity mode.

        Returns
        -------
        None
            Results are appended to ``self.comparison_table``.
        """
        if not isinstance(explanations, pd.DataFrame):
            raise TypeError("'explanations' must be a pandas DataFrame.")
        if exp_type not in _VALID_EXP_TYPES:
            raise ValueError(
                f"exp_type must be one of {_VALID_EXP_TYPES}, got '{exp_type}'."
            )
        if mode not in ("1to1", "1toN"):
            raise ValueError("mode must be '1to1' or '1toN'.")
        if mode == "1to1" and len(explanations) != self.s_count:
            raise ValueError(
                f"In '1to1' mode explanations must have {self.s_count} rows "
                f"(one per sample), got {len(explanations)}."
            )
        if mode == "1toN" and "instance" not in explanations.columns:
            raise ValueError(
                "In '1toN' mode the explanations DataFrame must contain "
                "an 'instance' column indicating the source sample index."
            )

        c1 = "existed" if "existed" in exp_type else "generated"
        c2 = "counterfactual" if "cf" in exp_type else "factual"
        c3 = mode

        valid_metrics = set(
            self.metric_features[
                self.metric_features[[c1, c2, c3]].ne("-").all(axis=1)
            ].index.tolist()
        )

        metrics_map = {
            "validity":            self._calc_validity,
            "proximity":           self._calc_proximity,
            "proximity_gower":     self._calc_proximity_gower,
            "sparsity":            self._calc_sparsity,
            "count":               self._calc_count,
            "diversity":           self._calc_diversity,
            "diversity_lcc":       self._calc_diversity_lcc,
            "yNN":                 self._calc_yNN,
            "feasibility":         self._calc_feasibility,
            "kNLN_dist":           self._calc_kNLN_distance,
            "relative_dist":       self._calc_relative_dist,
            "redundancy":          self._calc_redundancy,
            "plausibility":        self._calc_plausibility,
            "constraint_violation":self._calc_constraint_violation,
        }

        results = [
            metrics_map[m](explanations, mode) if m in valid_metrics else "-"
            for m in self.metrics
        ]

        self.explainer_names.append(name)
        new_row = pd.DataFrame([results], columns=self.metrics, index=[name])
        self.comparison_table = pd.concat([self.comparison_table, new_row])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_distance(self, dist: Optional[str]) -> None:
        if dist is not None and dist not in _VALID_DISTANCES:
            raise ValueError(
                f"distance must be one of {sorted(_VALID_DISTANCES)} or None, "
                f"got '{dist}'."
            )
        self.dist = dist

    def _set_encoder(self, enc: Optional[str]) -> None:
        _encoders = {
            "BackwardDifferenceEncoder": ce.BackwardDifferenceEncoder,
            "BaseNEncoder":              ce.BaseNEncoder,
            "BinaryEncoder":             ce.BinaryEncoder,
            "CatBoostEncoder":           ce.CatBoostEncoder,
            "CountEncoder":              ce.CountEncoder,
            "GLMMEncoder":               ce.GLMMEncoder,
            "GrayEncoder":               ce.GrayEncoder,
            "HelmertEncoder":            ce.HelmertEncoder,
            "JamesSteinEncoder":         ce.JamesSteinEncoder,
            "LeaveOneOutEncoder":        ce.LeaveOneOutEncoder,
            "MEstimateEncoder":          ce.MEstimateEncoder,
            "OneHotEncoder":             ce.OneHotEncoder,
            "OrdinalEncoder":            ce.OrdinalEncoder,
            "PolynomialEncoder":         ce.PolynomialEncoder,
            "QuantileEncoder":           ce.QuantileEncoder,
            "RankHotEncoder":            ce.RankHotEncoder,
            "SumEncoder":                ce.SumEncoder,
            "TargetEncoder":             ce.TargetEncoder,
            "WOEEncoder":                ce.WOEEncoder,
        }
        if enc is None:
            self.encoder = ce.OrdinalEncoder(self.categorical_names).fit(
                self.data_X, self.data_y
            )
        elif enc in _encoders:
            self.encoder = _encoders[enc](self.categorical_names).fit(
                self.data_X, self.data_y
            )
        else:
            raise ValueError(
                f"encoder must be one of {list(_encoders.keys())} or None."
            )

    def _encode(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(x)

    def _knn_fit(self, k: int, data: pd.DataFrame) -> NearestNeighbors:
        return NearestNeighbors(n_neighbors=k).fit(data)

    def _knn_query(
        self, model: NearestNeighbors, row: pd.Series, k: int, return_distance: bool = False
    ):
        """Query a fitted NearestNeighbors model for a single row.

        Returns
        -------
        np.ndarray
            Shape ``(k,)`` — distances when ``return_distance=True``,
            neighbour indices when ``return_distance=False``.
        """
        if return_distance:
            dists, inds = model.kneighbors(row.values.reshape(1, -1), k,
                                           return_distance=True)
            return dists[0]     # shape (k,)
        else:
            inds = model.kneighbors(row.values.reshape(1, -1), k,
                                    return_distance=False)
            return inds[0]      # shape (k,)

    def _distance(self, s: pd.Series, e: pd.Series) -> float:
        """Mixed-type distance between two feature vectors (no label column)."""
        if self.dist is not None:
            s_enc = self._encode(s.to_frame().T).values
            e_enc = self._encode(e.to_frame().T).values
            return float(scipy_cdist(s_enc, e_enc, metric=self.dist)[0][0])
        num_dist = float(np.linalg.norm(
            s[self.numeric_names].values.astype(float)
            - e[self.numeric_names].values.astype(float)
        ))
        cat_dist = sum(1 for f in self.categorical_names if s[f] != e[f])
        n_total = len(self.column_names_X)
        return (
            num_dist * (len(self.numeric_names) / n_total)
            + cat_dist * (len(self.categorical_names) / n_total)
        )

    @staticmethod
    def _to_float(df: pd.DataFrame) -> pd.DataFrame:
        """Cast all numeric columns to float64 (required by gower)."""
        out = df.copy()
        num_cols = out.select_dtypes(include=np.number).columns
        out[num_cols] = out[num_cols].astype(float)
        return out

    def _gower_distance(self, s: pd.DataFrame, e: pd.DataFrame) -> float:
        """Gower distance between two single-row DataFrames."""
        combined = pd.concat(
            [s.reset_index(drop=True), e.reset_index(drop=True),
             self.data_X.reset_index(drop=True)],
            axis=0, ignore_index=True,
        )
        mat = gower.gower_matrix(self._to_float(combined))
        return float(mat[0, 1])

    def _find_NLN(self, s: pd.Series) -> pd.Series:
        """Nearest Like Neighbour of *s* (same label, encoded)."""
        mask   = self.data[self.label] == s[self.label]
        temp_X = self.data_X_enc[mask.values]
        model  = self._knn_fit(1, temp_X)
        s_enc  = self._encode(s.drop(self.label).to_frame().T)
        idx    = self._knn_query(model, s_enc.iloc[0], 1, False)[0]
        return temp_X.iloc[idx]

    def _find_NUN(self, s: pd.Series) -> pd.Series:
        """Nearest Unlike Neighbour of *s* (different label, encoded)."""
        mask   = self.data[self.label] != s[self.label]
        temp_X = self.data_X_enc[mask.values]
        model  = self._knn_fit(1, temp_X)
        s_enc  = self._encode(s.drop(self.label).to_frame().T)
        idx    = self._knn_query(model, s_enc.iloc[0], 1, False)[0]
        return temp_X.iloc[idx]

    # ------------------------------------------------------------------
    # Metric calculations
    # ------------------------------------------------------------------

    def _calc_validity(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if self.model is None or self.data is None:
            return "-"

        samples_enc = self._encode(self.samples_X)
        samples_enc[self.label] = self.samples[self.label].values

        def _valid(s: pd.Series, e: pd.Series) -> int:
            pred = self.model.predict([e.drop(self.label).values])[0]
            try:
                return int(s[self.label] != e[self.label] and pred == e[self.label])
            except TypeError:
                return int(int(s[self.label]) != int(e[self.label]) and int(pred) == int(e[self.label]))

        val = 0
        if mode == "1to1":
            exp_enc = self._encode(exp_df.drop(columns=[self.label]))
            exp_enc[self.label] = exp_df[self.label].values
            for i in range(self.s_count):
                val += _valid(samples_enc.iloc[i], exp_enc.iloc[i])
        else:
            exp_enc = self._encode(exp_df.drop(columns=["instance", self.label]))
            exp_enc[self.label] = exp_df[self.label].values
            for i in range(self.s_count):
                sub = exp_enc[exp_df["instance"] == i]
                val += sum(_valid(samples_enc.iloc[i], sub.iloc[j]) for j in range(len(sub)))

        return round(val / len(exp_df), 3)

    def _calc_proximity(self, exp_df: pd.DataFrame, mode: str) -> float:
        total = 0.0
        exp_X = exp_df.drop(columns=[self.label])
        if mode == "1to1":
            for i in range(self.s_count):
                total += self._distance(self.samples_X.iloc[i], exp_X.iloc[i])
        else:
            exp_X = exp_X.drop(columns=["instance"])
            for i in range(self.s_count):
                sub = exp_X[exp_df["instance"] == i]
                total += sum(self._distance(self.samples_X.iloc[i], sub.iloc[j]) for j in range(len(sub)))
        return round(total / len(exp_df), 3)

    def _calc_proximity_gower(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if self.data is None:
            return "-"
        total = 0.0
        exp_X = exp_df.drop(columns=[self.label])
        if mode == "1to1":
            for i in range(self.s_count):
                total += self._gower_distance(
                    self.samples_X.iloc[i:i+1], exp_X.iloc[i:i+1]
                )
        else:
            exp_X = exp_X.drop(columns=["instance"])
            for i in range(self.s_count):
                sub = exp_X[exp_df["instance"] == i]
                total += sum(
                    self._gower_distance(self.samples_X.iloc[i:i+1], sub.iloc[j:j+1])
                    for j in range(len(sub))
                )
        return round(total / len(exp_df), 3)

    def _calc_sparsity(self, exp_df: pd.DataFrame, mode: str) -> float:
        total = 0.0
        exp_X = exp_df.drop(columns=[self.label])

        def _sparsity(s: pd.Series, e: pd.Series) -> float:
            return sum(1 for f in self.column_names_X if s[f] != e[f]) / len(self.column_names_X)

        if mode == "1to1":
            for i in range(self.s_count):
                total += _sparsity(self.samples_X.iloc[i], exp_X.iloc[i])
        else:
            exp_X = exp_X.drop(columns=["instance"])
            for i in range(self.s_count):
                sub = exp_X[exp_df["instance"] == i]
                total += sum(_sparsity(self.samples_X.iloc[i], sub.iloc[j]) for j in range(len(sub)))
        return round(total / len(exp_df), 3)

    def _calc_count(self, exp_df: pd.DataFrame, mode: str) -> float:
        return round(len(exp_df) / self.s_count, 3)

    def _calc_diversity(self, exp_df: pd.DataFrame, mode: str) -> float:
        diversity = 0.0
        for i in range(self.s_count):
            sub = exp_df[exp_df["instance"] == i].drop(columns=["instance", self.label])
            if len(sub) > 1:
                diversity += abs(np.linalg.det(gower.gower_matrix(self._to_float(sub))))
        return round(diversity / self.s_count, 3)

    def _calc_diversity_lcc(self, exp_df: pd.DataFrame, mode: str) -> float:
        diversity = self._calc_diversity(exp_df, mode)
        lcc = 0.0
        for i in range(self.s_count):
            sub = exp_df[exp_df["instance"] == i]
            lcc += len(np.unique(sub[self.label].values))
        lcc /= (self.num_label - 1) * self.s_count
        return round(lcc * diversity, 3)

    def _calc_yNN(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if self.model is None or self.data is None:
            return "-"
        df = exp_df.drop(columns=["instance"]) if mode == "1toN" else exp_df
        enc = self._encode(df.drop(columns=[self.label]))
        scores = []
        for i in range(len(df)):
            indices = self._knn_query(self.kNN_model, enc.iloc[i], self.knn, False)
            scores += [
                int(self.model.predict(self.data_X_enc.iloc[k].values.reshape(1, -1))[0] == df[self.label].iloc[i])
                for k in indices
            ]
        return round(float(np.mean(scores)), 3)

    def _calc_feasibility(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if self.data is None:
            return "-"
        df = exp_df.drop(columns=["instance"]) if mode == "1toN" else exp_df
        enc = self._encode(df.drop(columns=[self.label]))
        dists = []
        for i in range(len(df)):
            dists.extend(self._knn_query(self.kNN_model, enc.iloc[i], self.knn, True))
        return round(float(np.mean(dists)), 3)

    def _calc_kNLN_distance(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if self.data is None:
            return "-"
        df = exp_df.drop(columns=["instance"]) if mode == "1toN" else exp_df
        enc = self._encode(df.drop(columns=[self.label]))
        dists = []
        for i in range(len(df)):
            same_label = self.data_enc[self.data[self.label] == df[self.label].iloc[i]].drop(columns=[self.label])
            model = self._knn_fit(self.knn, same_label)
            dists.extend(self._knn_query(model, enc.iloc[i], self.knn, True))
        return round(float(np.mean(dists)), 3)

    def _calc_relative_dist(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if self.data is None:
            return "-"
        total = 0.0
        exp_X = exp_df.drop(columns=[self.label])
        if mode == "1to1":
            for i in range(self.s_count):
                nun = self._find_NUN(self.samples.iloc[i])
                s   = self.samples_X.iloc[i]
                d_se = self._distance(s, exp_X.iloc[i])
                d_sn = self._distance(s, nun)
                total += (1.0 if d_sn == 0 else d_se / d_sn)
        else:
            exp_X = exp_X.drop(columns=["instance"])
            for i in range(self.s_count):
                nun = self._find_NUN(self.samples.iloc[i])
                s   = self.samples_X.iloc[i]
                sub = exp_X[exp_df["instance"] == i]
                for j in range(len(sub)):
                    d_se = self._distance(s, sub.iloc[j])
                    d_sn = self._distance(s, nun)
                    total += (1.0 if d_sn == 0 else d_se / d_sn)
        return round(total / len(exp_df), 3)

    def _calc_redundancy(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if self.model is None or self.data is None:
            return "-"

        samples_enc = self._encode(self.samples_X)
        samples_enc[self.label] = self.samples[self.label].values

        def _redundancy(s: pd.Series, e: pd.Series) -> int:
            red = 0
            e_label = e[self.label]
            s_vals, e_vals = s.drop(self.label).values, e.drop(self.label).values
            for j in range(len(e_vals)):
                if s_vals[j] != e_vals[j]:
                    tmp = e_vals.copy()
                    tmp[j] = s_vals[j]
                    if np.argmax(self.model.predict_proba(tmp.reshape(1, -1))) == e_label:
                        red += 1
            return red

        scores = []
        if mode == "1to1":
            exp_enc = self._encode(exp_df.drop(columns=[self.label]))
            exp_enc[self.label] = exp_df[self.label].values
            for i in range(self.s_count):
                scores.append(_redundancy(samples_enc.iloc[i], exp_enc.iloc[i]))
        else:
            exp_enc = self._encode(exp_df.drop(columns=["instance", self.label]))
            exp_enc[self.label] = exp_df[self.label].values
            for i in range(self.s_count):
                sub = exp_enc[exp_df["instance"] == i]
                scores += [_redundancy(samples_enc.iloc[i], sub.iloc[j]) for j in range(len(sub))]
        return round(float(np.mean(scores)), 3)

    def _calc_plausibility(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if self.model is None or self.data is None:
            return "-"

        def _plausibility(e: pd.Series) -> float:
            nln_e = self._find_NLN(e)
            e_enc = self._encode(e.drop(self.label).to_frame().T).iloc[0]
            nln_copy = nln_e.copy()
            nln_copy[self.label] = self.model.predict([nln_e.values])[0]
            a = self._distance(e_enc, nln_e)
            b = self._distance(nln_e, self._find_NUN(nln_copy))
            return 1.0 if b == 0 else a / b

        total = 0.0
        if mode == "1to1":
            for i in range(self.s_count):
                total += _plausibility(exp_df.iloc[i])
        else:
            for i in range(self.s_count):
                sub = exp_df[exp_df["instance"] == i].drop(columns=["instance"])
                total += sum(_plausibility(sub.iloc[j]) for j in range(len(sub)))
        return round(total / len(exp_df), 3)

    def _calc_constraint_violation(self, exp_df: pd.DataFrame, mode: str) -> Union[float, str]:
        if not self.constraints:
            return "-"
        total = 0.0
        exp_X = exp_df.drop(columns=[self.label])
        if mode == "1to1":
            for i in range(self.s_count):
                total += int(any(self.samples_X.iloc[i][f] != exp_X.iloc[i][f] for f in self.constraints))
        else:
            exp_X = exp_X.drop(columns=["instance"])
            for i in range(self.s_count):
                sub = exp_X[exp_df["instance"] == i]
                total += sum(
                    int(any(self.samples_X.iloc[i][f] != sub.iloc[j][f] for f in self.constraints))
                    for j in range(len(sub))
                )
        return round(total / len(exp_df), 3)
