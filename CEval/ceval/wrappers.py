"""
ceval.wrappers
==============

Drop-in adapters that give any model a scikit-learn-compatible
``predict`` / ``predict_proba`` interface so it can be passed directly
to :class:`~ceval.CEval`.

Supported frameworks
--------------------
* **scikit-learn** — no wrapper needed; pass the model directly.
* :class:`XGBoostWrapper`  — ``xgboost.XGBClassifier`` and ``xgboost.Booster``
* :class:`LightGBMWrapper` — ``lightgbm.LGBMClassifier`` and ``lightgbm.Booster``
* :class:`CatBoostWrapper` — ``catboost.CatBoostClassifier``
* :class:`TorchWrapper`    — ``torch.nn.Module``
* :class:`KerasWrapper`    — ``tf.keras.Model`` / ``keras.Model``
* :class:`GenericWrapper`  — any model; you supply callable functions

Quick start
-----------
>>> from ceval.wrappers import TorchWrapper
>>> wrapped = TorchWrapper(my_net, num_classes=2)
>>> evaluator = CEval(samples=df, label="y", model=wrapped, ...)

All wrappers expose the same two methods used internally by CEval:

``predict(X) -> np.ndarray``
    Returns a 1-D integer array of predicted class labels.

``predict_proba(X) -> np.ndarray``
    Returns a 2-D float array of shape ``(n_samples, n_classes)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import numpy as np

__all__ = [
    "BaseWrapper",
    "XGBoostWrapper",
    "LightGBMWrapper",
    "CatBoostWrapper",
    "TorchWrapper",
    "KerasWrapper",
    "GenericWrapper",
    "wrap_model",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseWrapper(ABC):
    """Abstract base class for all CEval model wrappers.

    Subclass this if you need to support a framework not listed above.
    You only need to implement :meth:`predict_proba`; ``predict`` is
    derived from it automatically.
    """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (argmax of probabilities).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)  — integer class indices
        """
        return np.argmax(self.predict_proba(np.atleast_2d(X)), axis=1)

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        """Safely convert X to a float64 numpy array."""
        if hasattr(X, "values"):       # pandas DataFrame / Series
            X = X.values
        if hasattr(X, "numpy"):        # torch / tf tensor
            X = X.numpy()
        return np.asarray(X, dtype=np.float64)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostWrapper(BaseWrapper):
    """Wrap an XGBoost classifier.

    Works with both the scikit-learn API (``XGBClassifier``) and the
    native API (``xgboost.Booster``).

    Parameters
    ----------
    model : XGBClassifier | xgboost.Booster
        A fitted XGBoost model.
    num_classes : int, optional
        Required only when using the native ``Booster`` API and the
        number of classes cannot be inferred automatically.

    Examples
    --------
    >>> import xgboost as xgb
    >>> from ceval.wrappers import XGBoostWrapper
    >>> bst = xgb.XGBClassifier().fit(X_train, y_train)
    >>> wrapped = XGBoostWrapper(bst)
    >>> evaluator = CEval(samples=test_df, label="y", model=wrapped, ...)
    """

    def __init__(self, model, num_classes: Optional[int] = None):
        try:
            import xgboost as xgb  # noqa: F401
        except ImportError:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            )
        self.model       = model
        self.num_classes = num_classes
        # Detect API style
        self._sklearn_api = hasattr(model, "predict_proba")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._to_numpy(X)
        if self._sklearn_api:
            return self.model.predict_proba(X)
        # Native Booster API
        import xgboost as xgb
        dmat = xgb.DMatrix(X)
        raw  = self.model.predict(dmat)
        if raw.ndim == 1:
            # Binary: raw = P(class=1)
            return np.column_stack([1 - raw, raw])
        return raw  # Multiclass: already (n, k)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMWrapper(BaseWrapper):
    """Wrap a LightGBM classifier.

    Works with both ``LGBMClassifier`` (sklearn API) and
    ``lightgbm.Booster`` (native API).

    Parameters
    ----------
    model : LGBMClassifier | lightgbm.Booster
        A fitted LightGBM model.
    num_classes : int, optional
        Required only for the native Booster when classes cannot be
        inferred automatically.

    Examples
    --------
    >>> import lightgbm as lgb
    >>> from ceval.wrappers import LightGBMWrapper
    >>> clf = lgb.LGBMClassifier().fit(X_train, y_train)
    >>> wrapped = LightGBMWrapper(clf)
    """

    def __init__(self, model, num_classes: Optional[int] = None):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            raise ImportError(
                "lightgbm is not installed. Run: pip install lightgbm"
            )
        self.model       = model
        self.num_classes = num_classes
        self._sklearn_api = hasattr(model, "predict_proba")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._to_numpy(X)
        if self._sklearn_api:
            return self.model.predict_proba(X)
        raw = self.model.predict(X)
        if raw.ndim == 1:
            return np.column_stack([1 - raw, raw])
        return raw


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatBoostWrapper(BaseWrapper):
    """Wrap a CatBoost classifier.

    Parameters
    ----------
    model : CatBoostClassifier
        A fitted CatBoost model.

    Examples
    --------
    >>> from catboost import CatBoostClassifier
    >>> from ceval.wrappers import CatBoostWrapper
    >>> clf = CatBoostClassifier(verbose=0).fit(X_train, y_train)
    >>> wrapped = CatBoostWrapper(clf)
    """

    def __init__(self, model):
        try:
            import catboost  # noqa: F401
        except ImportError:
            raise ImportError(
                "catboost is not installed. Run: pip install catboost"
            )
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._to_numpy(X)
        return self.model.predict_proba(X)


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

class TorchWrapper(BaseWrapper):
    """Wrap a PyTorch ``nn.Module`` classifier.

    The model must output raw logits (pre-softmax). For binary
    classification, outputting a single logit (shape ``(n, 1)`` or
    ``(n,)``) is also supported.

    Parameters
    ----------
    model : torch.nn.Module
        A fitted PyTorch model in eval mode.
    num_classes : int
        Number of output classes.
    device : str, default "cpu"
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).

    Examples
    --------
    >>> import torch
    >>> from ceval.wrappers import TorchWrapper
    >>> net = MyNet(); net.load_state_dict(torch.load("model.pt"))
    >>> net.eval()
    >>> wrapped = TorchWrapper(net, num_classes=2, device="cpu")
    >>> evaluator = CEval(samples=test_df, label="y", model=wrapped, ...)
    """

    def __init__(self, model, num_classes: int, device: str = "cpu"):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch is not installed. Run: pip install torch"
            )
        self.model       = model
        self.num_classes = num_classes
        self.device      = device
        self.model.eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        X = self._to_numpy(X)
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        logits = logits.cpu()

        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            # Binary with single logit output → sigmoid
            probs_pos = torch.sigmoid(logits.squeeze()).unsqueeze(-1).numpy()
            return np.hstack([1 - probs_pos, probs_pos])
        else:
            # Multi-class → softmax
            return F.softmax(logits, dim=1).numpy()


# ---------------------------------------------------------------------------
# Keras / TensorFlow
# ---------------------------------------------------------------------------

class KerasWrapper(BaseWrapper):
    """Wrap a Keras / TensorFlow model.

    The model's final layer must output probabilities (i.e. use
    ``'softmax'`` or ``'sigmoid'`` activation).  If it outputs raw
    logits, pass ``from_logits=True``.

    Parameters
    ----------
    model : tf.keras.Model
        A compiled and fitted Keras model.
    num_classes : int
        Number of output classes.
    from_logits : bool, default False
        Set to ``True`` if the model outputs raw logits instead of
        probabilities.

    Examples
    --------
    >>> from tensorflow import keras
    >>> from ceval.wrappers import KerasWrapper
    >>> model = keras.models.load_model("model.h5")
    >>> wrapped = KerasWrapper(model, num_classes=2)
    >>> evaluator = CEval(samples=test_df, label="y", model=wrapped, ...)
    """

    def __init__(self, model, num_classes: int, from_logits: bool = False):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            try:
                import keras  # noqa: F401  (standalone keras)
            except ImportError:
                raise ImportError(
                    "Neither tensorflow nor keras is installed. "
                    "Run: pip install tensorflow  or  pip install keras"
                )
        self.model       = model
        self.num_classes = num_classes
        self.from_logits = from_logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X    = self._to_numpy(X)
        raw  = self.model.predict(X, verbose=0)

        if self.from_logits:
            # Apply softmax manually
            exp  = np.exp(raw - raw.max(axis=1, keepdims=True))
            raw  = exp / exp.sum(axis=1, keepdims=True)

        if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 1):
            # Binary sigmoid output
            p = raw.squeeze()[:, None]
            return np.hstack([1 - p, p])

        return raw


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

class GenericWrapper(BaseWrapper):
    """Wrap any model by supplying plain Python callables.

    Use this when your framework is not listed above, or when you want
    full control over how predictions are made.

    Parameters
    ----------
    predict_fn : callable
        A function ``predict_fn(X: np.ndarray) -> np.ndarray`` that
        returns a 1-D integer array of class labels.
    predict_proba_fn : callable
        A function ``predict_proba_fn(X: np.ndarray) -> np.ndarray``
        that returns a 2-D float array of shape ``(n_samples, n_classes)``.

    Examples
    --------
    >>> from ceval.wrappers import GenericWrapper
    >>> wrapped = GenericWrapper(
    ...     predict_fn       = lambda X: my_model.infer_labels(X),
    ...     predict_proba_fn = lambda X: my_model.infer_proba(X),
    ... )
    >>> evaluator = CEval(samples=test_df, label="y", model=wrapped, ...)
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    ):
        if not callable(predict_fn) or not callable(predict_proba_fn):
            raise TypeError(
                "Both predict_fn and predict_proba_fn must be callable."
            )
        self._predict_fn       = predict_fn
        self._predict_proba_fn = predict_proba_fn

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict_fn(self._to_numpy(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict_proba_fn(self._to_numpy(X)))


# ---------------------------------------------------------------------------
# Auto-detect helper
# ---------------------------------------------------------------------------

def wrap_model(model) -> Union[BaseWrapper, object]:
    """Auto-detect the model type and return a wrapped version if needed.

    Scikit-learn models (anything already exposing ``predict`` and
    ``predict_proba``) are returned as-is.  Everything else raises a
    helpful error that points to the right wrapper class.

    Parameters
    ----------
    model : any fitted classifier

    Returns
    -------
    The original model (sklearn-compatible) or a :class:`BaseWrapper` subclass.

    Raises
    ------
    TypeError
        If the model is not sklearn-compatible and its type is not
        automatically recognised.
    """
    if model is None:
        return None

    # Already sklearn-compatible
    if hasattr(model, "predict") and hasattr(model, "predict_proba"):
        return model

    type_name = type(model).__module__ + "." + type(model).__name__

    suggestions = {
        "xgboost":   "XGBoostWrapper",
        "lightgbm":  "LightGBMWrapper",
        "catboost":  "CatBoostWrapper",
        "torch":     "TorchWrapper",
        "tensorflow":"KerasWrapper",
        "keras":     "KerasWrapper",
    }

    for key, wrapper in suggestions.items():
        if key in type_name.lower():
            raise TypeError(
                f"Model of type '{type_name}' is not directly sklearn-compatible. "
                f"Wrap it first:\n\n"
                f"    from ceval.wrappers import {wrapper}\n"
                f"    model = {wrapper}(your_model, ...)\n"
            )

    raise TypeError(
        f"Model of type '{type_name}' is not sklearn-compatible and was not "
        f"automatically recognised.  Use GenericWrapper to adapt it:\n\n"
        f"    from ceval.wrappers import GenericWrapper\n"
        f"    model = GenericWrapper(\n"
        f"        predict_fn       = lambda X: your_model.predict(X),\n"
        f"        predict_proba_fn = lambda X: your_model.predict_proba(X),\n"
        f"    )\n"
    )
