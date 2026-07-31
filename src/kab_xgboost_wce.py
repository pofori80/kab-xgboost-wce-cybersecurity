import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix)
from scipy.stats import t as t_dist

SEED = 42

def cronbach_alpha(items_df):
    items_df = items_df.dropna()
    k = items_df.shape[1]
    if k < 2:
        return float('nan')
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float('nan')
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


DEFAULT_XGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                          reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
                          eval_metric='logloss')

def make_baseline(depth=5, xgb_params=None):
    p = dict(DEFAULT_XGB_PARAMS if xgb_params is None else xgb_params)
    return XGBClassifier(max_depth=depth, **p)

def make_wce(depth=4, xgb_params=None):
    p = dict(DEFAULT_XGB_PARAMS if xgb_params is None else xgb_params)
    return XGBClassifier(max_depth=depth, **p)

def build_kab_weights(X, y, k_idx, a_idx, multiplier=2.0):
    w = np.ones(len(y))
    for i, lab in enumerate(y):
        if lab == 0:
            k, a = X[i, k_idx], X[i, a_idx]
            total = k + a
            if total <= 0:
                combined = 0.5
            else:
                combined = ((1 - k / total) + (1 - a / total)) / 2.0
            w[i] = 1.0 + combined * (multiplier - 1.0)
    return w

def fit_wce(X_train, y_train, k_idx, a_idx, depth=4, multiplier=2.0, xgb_params=None):
    model = make_wce(depth=depth, xgb_params=xgb_params)
    w = build_kab_weights(X_train, y_train, k_idx, a_idx, multiplier)
    model.fit(X_train, y_train, sample_weight=w)
    return model

def run_model(model_fn, X, y, label, use_weights=False, k_idx=None, a_idx=None,
             multiplier=2.0, n_splits=10, n_repeats=1, seed=SEED, verbose=True):
    if use_weights and (k_idx is None or a_idx is None):
        raise ValueError('use_weights=True requires k_idx and a_idx')
    import time as _time
    res = {'acc': [], 'f1': [], 'auc_roc': [], 'auc_pr': [], 'time': []}
    all_preds, all_true, cms = [], [], []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + rep * 1000)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
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
            if verbose and n_repeats == 1:
                print(f'  {label} F{fold+1:02d}: Acc={res["acc"][-1]:.4f} '
                      f'F1={res["f1"][-1]:.4f} AUC={res["auc_roc"][-1]:.4f}')
    if verbose and n_repeats > 1:
        print(f'  {label}: {n_repeats} x {n_splits}-fold = {len(res["f1"])} paired samples | '
              f'F1={np.mean(res["f1"]):.4f}+/-{np.std(res["f1"]):.4f}')
    return res, cms, np.array(all_preds), np.array(all_true)

def ci95(vals):
    vals = np.asarray(vals)
    n = len(vals)
    h = vals.std(ddof=1) / np.sqrt(n) * t_dist.ppf(0.975, n - 1)
    return vals.mean() - h, vals.mean() + h

def diagnose_worst_subscale(fitted_model, X_test, y_test, k_idx, a_idx, verbose=True):
    preds = fitted_model.predict(X_test)
    low = (y_test == 0)
    wrong = low & (preds != y_test)
    right = low & (preds == y_test)
    diffs = {}
    for sub, idx in [('Knowledge', k_idx), ('Attitude', a_idx)]:
        cm_ = X_test[right, idx].mean() if right.sum() else np.nan
        mm_ = X_test[wrong, idx].mean() if wrong.sum() else cm_
        diffs[sub] = abs(cm_ - mm_)
        if verbose:
            print(f'  {sub:<10} correct-mean={cm_:.3f} wrong-mean={mm_:.3f} '
                  f'diff={diffs[sub]:.3f}')
    worst = max(diffs, key=diffs.get)
    if verbose:
        print(f'  Worst subscale: {worst}')
    return diffs, worst

def grid_search_wce(X_train, y_train, k_idx, a_idx, depths=(2, 3, 4, 5),
                    multipliers=(1.1, 1.2, 1.3, 1.5, 1.75, 2.0),
                    xgb_params=None, n_splits=5, n_repeats=1, seed=SEED, verbose=True):
    baseline_f1 = np.mean(run_model(
        lambda: make_baseline(depth=5, xgb_params=xgb_params),
        X_train, y_train, 'baseline(inner)', n_splits=n_splits,
        n_repeats=n_repeats, seed=seed, verbose=False)[0]['f1'])
    best = None
    grid = []
    for depth in depths:
        for mult in multipliers:
            f1s = run_model(
                lambda d=depth: make_wce(depth=d, xgb_params=xgb_params),
                X_train, y_train, 'wce(inner)', use_weights=True,
                k_idx=k_idx, a_idx=a_idx, multiplier=mult,
                n_splits=n_splits, n_repeats=n_repeats, seed=seed,
                verbose=False)[0]['f1']
            mean_f1 = np.mean(f1s)
            delta = mean_f1 - baseline_f1
            grid.append({'depth': depth, 'multiplier': mult,
                        'inner_f1': round(mean_f1, 4), 'delta': round(delta, 4)})
            if verbose:
                flag = ' <-- beats baseline' if delta > 0 else ''
                print(f'  depth={depth} mult={mult}: F1={mean_f1:.4f} '
                      f'(Delta={delta:+.4f}){flag}')
            if best is None or mean_f1 > best[0]:
                best = (mean_f1, depth, mult)
    if verbose:
        print(f'  BEST: depth={best[1]} mult={best[2]} '
              f'(inner F1={best[0]:.4f} vs baseline {baseline_f1:.4f})')
    return best[1], best[2], baseline_f1, grid
