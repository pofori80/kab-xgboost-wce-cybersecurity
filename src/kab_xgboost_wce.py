"""
kab_xgboost_wce.py — KAB-XGBoost-WCE (R04, corrected)
=======================================================
Cost-sensitive XGBoost variant grounded in the Knowledge-Attitude-Behaviour
(KAB) model, proposed for predicting cybersecurity behavioural readiness.

Key correction (R04 vs earlier versions):
  The original build_kab_weights() function contained an algebraic error:
    combined = ((1 - k/total) + (1 - a/total)) / 2
  reduces to 0.5 for ALL respondents, making w_i constant.

  Corrected weight function (Equation 2 in manuscript):
    w_i = 1 + (|K_i - A_i| / (K_i + A_i)) * (lambda - 1)
  This is genuinely non-constant: KNUST training partition at lambda=1.5
  produced 122 unique weight values (range 1.0000–1.2600, mean 1.0653).

Author  : Prince Ofori (21990878), KNUST Ghana
Supervisor: Dr. Eric Opoku Osei
Date    : 2026-08-30
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix)
from scipy.stats import t as t_dist

SEED = 42


def cronbach_alpha(items_df):
    """Compute Cronbach's alpha for a DataFrame of Likert items."""
    items_df = items_df.dropna()
    k = items_df.shape[1]
    if k < 2:
        return float('nan')
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float('nan')
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


DEFAULT_XGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    eval_metric='logloss',
    verbosity=0,
)


def make_baseline(depth=5, xgb_params=None):
    """Standard XGBoost classifier with uniform instance weights."""
    p = dict(DEFAULT_XGB_PARAMS if xgb_params is None else xgb_params)
    return XGBClassifier(max_depth=depth, **p)


def make_wce(depth=5, xgb_params=None):
    """XGBoost classifier configured for KAB-weighted training."""
    p = dict(DEFAULT_XGB_PARAMS if xgb_params is None else xgb_params)
    return XGBClassifier(max_depth=depth, **p)


def build_kab_weights(X, y, k_idx, a_idx, multiplier=1.5):
    """
    Compute per-instance KAB asymmetry weights (CORRECTED — R04).

    For Low-readiness respondents (y_i = 0):
        w_i = 1 + (|K_i - A_i| / (K_i + A_i)) * (lambda - 1)

    For High-readiness respondents (y_i = 1):
        w_i = 1  (uniform, no uplift)

    Parameters
    ----------
    X         : np.ndarray, shape (n, d)
    y         : np.ndarray, shape (n,), binary {0, 1}
    k_idx     : int, column index of Knowledge aggregate K_i in X
    a_idx     : int, column index of Attitude aggregate A_i in X
    multiplier: float (lambda), scalar >= 1.0; controls maximum uplift

    Returns
    -------
    w : np.ndarray, shape (n,), per-instance weights

    Notes
    -----
    The asymmetry term |K_i - A_i| / (K_i + A_i) lies in [0, 1):
      - When K_i == A_i: asymmetry = 0, w_i = 1.0 (no uplift)
      - As |K_i - A_i| increases: w_i increases toward 1 + (lambda - 1)
    Verification: KNUST training partition (n=3,296) at lambda=1.5 produced
    122 unique weight values, range [1.0000, 1.2600], mean 1.0653.
    """
    w = np.ones(len(y), dtype=float)
    for i in range(len(y)):
        if y[i] == 0:
            k = float(X[i, k_idx])
            a = float(X[i, a_idx])
            total = k + a
            if total > 0:
                asymmetry = abs(k - a) / total
            else:
                asymmetry = 0.0
            w[i] = 1.0 + asymmetry * (multiplier - 1.0)
    return w


def fit_wce(X_train, y_train, k_idx, a_idx, depth=5, multiplier=1.5,
            xgb_params=None):
    """Fit a KAB-XGBoost-WCE model on training data."""
    model = make_wce(depth=depth, xgb_params=xgb_params)
    w = build_kab_weights(X_train, y_train, k_idx, a_idx, multiplier)
    model.fit(X_train, y_train, sample_weight=w)
    return model


def run_model(model_fn, X, y, label, use_weights=False, k_idx=None,
              a_idx=None, multiplier=1.5, n_splits=10, n_repeats=5,
              seed=SEED, verbose=True):
    """
    Evaluate a model via repeated stratified k-fold cross-validation.

    Parameters
    ----------
    model_fn    : callable returning an unfitted sklearn-compatible estimator
    X           : np.ndarray, feature matrix
    y           : np.ndarray, binary labels
    label       : str, display name for logging
    use_weights : bool, if True apply KAB instance weights (WCE mode)
    k_idx       : int, K_i column index (required when use_weights=True)
    a_idx       : int, A_i column index (required when use_weights=True)
    multiplier  : float, lambda for weight function
    n_splits    : int, folds per repetition (default 10)
    n_repeats   : int, independent repetitions (default 5; 5x10 = 50 folds)
    seed        : int, base random seed (rep i uses seed + i*1000)
    verbose     : bool

    Returns
    -------
    res      : dict with keys 'acc', 'f1', 'auc_roc', 'auc_pr', 'time'
    cms      : list of confusion matrices
    all_preds: np.ndarray of fold predictions
    all_true : np.ndarray of true labels
    """
    if use_weights and (k_idx is None or a_idx is None):
        raise ValueError('use_weights=True requires k_idx and a_idx')

    import time as _time
    res = {'acc': [], 'f1': [], 'auc_roc': [], 'auc_pr': [], 'time': []}
    all_preds, all_true, cms = [], [], []

    for rep in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + rep * 1000)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            model = model_fn()
            t0 = _time.time()
            if use_weights:
                w = build_kab_weights(X_tr, y_tr, k_idx, a_idx, multiplier)
                model.fit(X_tr, y_tr, sample_weight=w)
            else:
                model.fit(X_tr, y_tr)
            res['time'].append(_time.time() - t0)
            preds = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]
            res['acc'].append(accuracy_score(y_te, preds))
            res['f1'].append(f1_score(y_te, preds, average='macro'))
            res['auc_roc'].append(roc_auc_score(y_te, proba))
            res['auc_pr'].append(average_precision_score(y_te, proba))
            cms.append(confusion_matrix(y_te, preds))
            all_preds.extend(preds.tolist())
            all_true.extend(y_te.tolist())

    if verbose:
        n_folds = len(res['f1'])
        print(f'  {label}: {n_repeats}x{n_splits}-fold = {n_folds} paired '
              f'samples | F1={np.mean(res["f1"]):.4f} '
              f'+/-{np.std(res["f1"], ddof=1):.4f}')
    return res, cms, np.array(all_preds), np.array(all_true)


def ci95(vals):
    """95% t-interval on a list of fold-level estimates."""
    vals = np.asarray(vals)
    n = len(vals)
    h = vals.std(ddof=1) / np.sqrt(n) * t_dist.ppf(0.975, n - 1)
    return float(vals.mean() - h), float(vals.mean() + h)


def verify_weight_non_constant(X_train, y_train, k_idx, a_idx,
                                multiplier=1.5, verbose=True):
    """
    Diagnostic: confirm the corrected weight function is non-constant.
    Prints unique value count, min, max, mean for Low-readiness instances.
    Returns True if weight vector has more than 1 unique value.
    """
    w = build_kab_weights(X_train, y_train, k_idx, a_idx, multiplier)
    low_mask = y_train == 0
    low_weights = w[low_mask]
    n_unique = len(np.unique(low_weights))
    if verbose:
        print(f'Weight verification (lambda={multiplier}):')
        print(f'  Low-readiness instances: {low_mask.sum()}')
        print(f'  Unique weight values: {n_unique}')
        print(f'  Min={low_weights.min():.4f} '
              f'Max={low_weights.max():.4f} '
              f'Mean={low_weights.mean():.4f}')
        if n_unique == 1:
            print('  WARNING: Weight function is CONSTANT — check implementation')
        else:
            print('  OK: Weight function is non-constant')
    return n_unique > 1


def diagnose_worst_subscale(fitted_model, X_test, y_test, k_idx, a_idx,
                            verbose=True):
    """
    Diagnose which subscale (K or A) shows larger mean difference between
    correctly and incorrectly classified Low-readiness respondents.
    """
    preds = fitted_model.predict(X_test)
    low = y_test == 0
    wrong = low & (preds != y_test)
    right = low & (preds == y_test)
    diffs = {}
    for sub, idx in [('Knowledge', k_idx), ('Attitude', a_idx)]:
        cm_ = X_test[right, idx].mean() if right.sum() else np.nan
        mm_ = X_test[wrong, idx].mean() if wrong.sum() else cm_
        diffs[sub] = abs(float(cm_) - float(mm_)) if not np.isnan(cm_) else 0.0
        if verbose:
            print(f'  {sub:<10} correct-mean={cm_:.3f} '
                  f'wrong-mean={mm_:.3f} diff={diffs[sub]:.3f}')
    worst = max(diffs, key=diffs.get)
    if verbose:
        print(f'  Worst subscale: {worst}')
    return diffs, worst
