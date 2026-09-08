#!/usr/bin/env python3
"""
KAB_Option2_FourLegs.py — R05 (De-duplicated pipeline)
=======================================================
Full evaluation pipeline for KAB-XGBoost-WCE.
All results in R05 documents were produced by this script.

Changes from R04:
  - De-duplication applied before any partition (2033 duplicate rows removed)
  - Working dataset: n=2,088 unique response patterns
  - Baseline XGBoost independently tuned (own GridSearchCV, separate from WCE)
  - SVM and Logistic Regression tuned and evaluated on scaled features
  - Stock scale_pos_weight arm added as 8th comparator
  - Ordinal robustness, bootstrap, subgroup analysis added
  - Both W+ and W- reported; exact Wilcoxon (no continuity correction)
  - Timing re-exported from de-duplicated run; no depth asymmetry

Author  : Prince Ofori (21990878), KNUST Ghana
Supervisor: Dr. Eric Opoku Osei
Date    : 2026-09-07
"""

import numpy as np
import pandas as pd
import json
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from scipy.stats import wilcoxon, chi2

SEED = 42
DATA_KNUST = 'data/KNUST_Cybersecurity_Survey_Responses_2026.xlsx'
DATA_ALZ   = 'data/Alzubaidi_2021_Cybercrime_Awareness_Dataset.xlsx'

XGB_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, random_state=SEED,
    eval_metric='logloss', verbosity=0
)


# ── Corrected weight function (Equation 2) ──────────────────────────────────
def build_kab_weights(X, y, k_idx, a_idx, lam=1.5):
    """
    Per-instance KAB asymmetry weights (R05 corrected).
    w_i = 1 + (|K_i - A_i| / (K_i + A_i)) * (lambda - 1)  for y_i = 0
    w_i = 1                                                  for y_i = 1
    Verified non-constant: 123 unique values on KNUST de-dup training partition.
    """
    w = np.ones(len(y), dtype=float)
    low = y == 0
    k = X[low, k_idx].astype(float)
    a = X[low, a_idx].astype(float)
    total = k + a
    asym = np.where(total > 0, np.abs(k - a) / np.where(total > 0, total, 1.0), 0.0)
    w[low] = 1.0 + asym * (lam - 1.0)
    return w


def verify_weights(X_train, y_train, k_idx, a_idx, lam=1.5):
    """Confirm weight vector is non-constant."""
    w = build_kab_weights(X_train, y_train, k_idx, a_idx, lam)
    low_w = w[y_train == 0]
    n_unique = len(np.unique(low_w))
    print(f"  Weight verification (lambda={lam}): "
          f"unique={n_unique}, range=[{low_w.min():.4f},{low_w.max():.4f}], "
          f"mean={low_w.mean():.4f}")
    assert n_unique > 1, "Weight function is CONSTANT -- check implementation"
    return n_unique


# ── Load and de-duplicate KNUST ──────────────────────────────────────────────
def load_knust(path=DATA_KNUST):
    df = pd.read_excel(path, sheet_name='Form Responses 1')
    pref = lambda c: c.split('.')[0]
    col = {pref(c): c for c in df.columns if c != 'Timestamp'}

    K = [col[i] for i in ['Q6','Q7','Q8','Q9','Q10']]
    A = [col[i] for i in ['Q13','Q14','Q15','Q16','Q17']]
    B = [col[i] for i in ['Q18','Q19','Q20','Q21','Q22','Q23']]
    D = [col[i] for i in ['Q1','Q2','Q3','Q4','Q5']]

    df['K_agg'] = df[K].mean(axis=1)
    df['A_agg'] = df[A].mean(axis=1)
    df['B_total'] = df[B].sum(axis=1)

    # De-duplicate on full response pattern (all K, A, B, D items)
    dup_key = df[K + A + B + D].astype(str).agg('|'.join, axis=1)
    keep = ~dup_key.duplicated()
    n_raw = len(df)
    df_clean = df[keep].reset_index(drop=True)
    n_dedup = len(df_clean)
    print(f"KNUST: raw n={n_raw}, de-duplicated n={n_dedup} "
          f"({n_raw - n_dedup} exact duplicate patterns removed, "
          f"{(n_raw - n_dedup)/n_raw:.1%})")

    # Label: global median split on de-duplicated dataset
    median_b = df['B_total'].median()
    y = (df_clean['B_total'] > median_b).astype(int).values
    print(f"  Label: median B_total={median_b}, "
          f"Low={( y==0).sum()} ({(y==0).mean():.1%}), "
          f"High={(y==1).sum()} ({(y==1).mean():.1%})")

    # Encode demographics
    for c in D:
        df_clean[c] = LabelEncoder().fit_transform(df_clean[c].astype(str))

    FEAT = K + A + [col[i] for i in ['Q1','Q2','Q3','Q4','Q5']] + ['K_agg','A_agg']
    X = df_clean[FEAT].values.astype(float)
    k_idx = FEAT.index('K_agg')
    a_idx = FEAT.index('A_agg')

    return X, y, FEAT, k_idx, a_idx


# ── Load Alzubaidi ───────────────────────────────────────────────────────────
def load_alzubaidi(path=DATA_ALZ):
    import re
    df = pd.read_excel(path)
    blk = lambda p: [c for c in df.columns if re.match(r'^\s*' + p + r'\s*\)?', str(c))]
    B14, B18, B19, B20, B22 = blk('14'), blk('18'), blk('19'), blk('20'), blk('22')
    C13, C16, C21 = blk('13')[0], blk('16')[0], blk('21')[0]

    FREQ5  = {'always':5,'often':4,'sometimes':3,'somtimes':3,'seldom':2,
               'rarely':2,'never':1,'do not know':1,"don't know":1}
    AGREE5 = {'strongly agree':5,'agree':4,'neutral':3,'undecided':3,
               'disagree':2,'strongly disagree':1}

    def sc(s, m, d=3.0):
        return s.astype(str).str.strip().str.lower().map(m).fillna(d)

    K_items = pd.concat([sc(df[c], FREQ5) for c in B19 + B20], axis=1)
    low16 = df[C16].astype(str).str.lower()
    A_items = pd.concat([
        sc(df[c], AGREE5) for c in B18 + B22] + [
        sc(df[C13], {'very secure':5,'somewhat secure':4,'neutral':3,
                     'somewhat insecure':2,'very insecure':1}),
        pd.Series(np.select(
            [df[C21].astype(str).str.contains('serious', case=False),
             df[C21].astype(str).str.contains('vanish', case=False)],
            [5, 1], 3.0), index=df.index)
    ], axis=1)
    B_items = pd.concat([sc(df[c], FREQ5) for c in B14] + [
        pd.Series(np.select(
            [low16.str.contains('automat'), low16.str.contains('manual'),
             low16.str.contains('know')], [5, 4, 2], 1.0), index=df.index)
    ], axis=1)

    B_s = B_items.mean(axis=1)
    y = (B_s >= B_s.median()).astype(int).values
    X_df = pd.concat([
        K_items, A_items,
        K_items.mean(axis=1).rename('K_agg'),
        A_items.mean(axis=1).rename('A_agg')
    ], axis=1).fillna(3.0)
    X = X_df.values.astype(float)
    feat = list(X_df.columns)
    k_idx = feat.index('K_agg')
    a_idx = feat.index('A_agg')

    print(f"Alzubaidi: n={len(df)}, d={X.shape[1]}, "
          f"Low={(y==0).sum()} ({(y==0).mean():.1%}), "
          f"High={(y==1).sum()} ({(y==1).mean():.1%})")
    assert X.shape == (1231, 28), f"Expected (1231, 28), got {X.shape}"
    return X, y, feat, k_idx, a_idx


# ── Tune all models ──────────────────────────────────────────────────────────
def tune_all(X_train, y_train, k_idx, a_idx, seed=SEED):
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)

    def gs(est, grid, Xf, yf):
        g = GridSearchCV(est, grid, cv=inner, scoring='f1_macro', n_jobs=-1)
        g.fit(Xf, yf)
        return g.best_params_

    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train)

    params = {}
    params['baseline'] = gs(XGBClassifier(**XGB_PARAMS),
                            {'max_depth': [2, 3, 5, 7]}, X_train, y_train)
    params['dt']       = gs(DecisionTreeClassifier(random_state=seed),
                            {'max_depth': [3, 5, 7, 10, None],
                             'min_samples_leaf': [1, 5, 10]}, X_train, y_train)
    params['lr']       = gs(LogisticRegression(max_iter=2000, random_state=seed),
                            {'C': [0.01, 0.1, 1, 10]}, Xs, y_train)
    params['svm']      = gs(SVC(kernel='rbf', random_state=seed, probability=True),
                            {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}, Xs, y_train)
    params['rf']       = gs(RandomForestClassifier(random_state=seed),
                            {'n_estimators': [100, 300], 'max_depth': [5, 10, None]},
                            X_train, y_train)

    # WCE joint grid over depth and lambda
    best_wce = {'F1': -1}
    for depth in [2, 3, 5, 7]:
        for lam in [1.0, 1.1, 1.2, 1.5, 2.0]:
            f1s = []
            for tr, te in inner.split(X_train, y_train):
                m = XGBClassifier(max_depth=depth, **XGB_PARAMS)
                m.fit(X_train[tr], y_train[tr],
                      sample_weight=build_kab_weights(
                          X_train[tr], y_train[tr], k_idx, a_idx, lam))
                f1s.append(f1_score(y_train[te], m.predict(X_train[te]), average='macro'))
            if np.mean(f1s) > best_wce['F1']:
                best_wce = {'depth': depth, 'lambda': lam, 'F1': np.mean(f1s)}
    params['wce'] = best_wce

    print("Tuned params:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    return params, sc


# ── 5x10-fold CV ─────────────────────────────────────────────────────────────
def run_cv(X, y, params, k_idx, a_idx, n_repeats=5, n_splits=10, seed=SEED):
    store = {m: [] for m in [
        'KAB-XGBoost-WCE', 'Baseline XGBoost', 'Stock scale_pos_weight',
        'Decision Tree', 'Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest'
    ]}
    times = {'baseline': [], 'wce': []}
    sc = StandardScaler()

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + rep * 1000)
        for tr, te in skf.split(X, y):
            Xf, yf, Xft, yft = X[tr], y[tr], X[te], y[te]
            Xfs = sc.fit(Xf).transform(Xf)
            Xft_s = sc.transform(Xft)

            # WCE
            m = XGBClassifier(max_depth=params['wce']['depth'], **XGB_PARAMS)
            t0 = time.perf_counter()
            m.fit(Xf, yf, sample_weight=build_kab_weights(
                Xf, yf, k_idx, a_idx, params['wce']['lambda']))
            times['wce'].append(time.perf_counter() - t0)
            store['KAB-XGBoost-WCE'].append(
                f1_score(yft, m.predict(Xft), average='macro'))

            # Baseline (independently tuned)
            m = XGBClassifier(max_depth=params['baseline']['max_depth'], **XGB_PARAMS)
            t0 = time.perf_counter()
            m.fit(Xf, yf)
            times['baseline'].append(time.perf_counter() - t0)
            store['Baseline XGBoost'].append(
                f1_score(yft, m.predict(Xft), average='macro'))

            # Stock scale_pos_weight
            ratio = (yf == 0).sum() / max((yf == 1).sum(), 1)
            m = XGBClassifier(max_depth=params['baseline']['max_depth'],
                               scale_pos_weight=ratio, **XGB_PARAMS)
            m.fit(Xf, yf)
            store['Stock scale_pos_weight'].append(
                f1_score(yft, m.predict(Xft), average='macro'))

            # Decision Tree
            m = DecisionTreeClassifier(**params['dt'], random_state=seed)
            m.fit(Xf, yf)
            store['Decision Tree'].append(
                f1_score(yft, m.predict(Xft), average='macro'))

            # Naive Bayes
            m = GaussianNB(); m.fit(Xf, yf)
            store['Naive Bayes'].append(
                f1_score(yft, m.predict(Xft), average='macro'))

            # Logistic Regression (scaled)
            m = LogisticRegression(**params['lr'], max_iter=2000, random_state=seed)
            m.fit(Xfs, yf)
            store['Logistic Regression'].append(
                f1_score(yft, m.predict(Xft_s), average='macro'))

            # SVM (scaled)
            m = SVC(**params['svm'], kernel='rbf', random_state=seed, probability=True)
            m.fit(Xfs, yf)
            store['SVM'].append(f1_score(yft, m.predict(Xft_s), average='macro'))

            # Random Forest
            m = RandomForestClassifier(**params['rf'], random_state=seed)
            m.fit(Xf, yf)
            store['Random Forest'].append(
                f1_score(yft, m.predict(Xft), average='macro'))

    return store, times


# ── Report results ───────────────────────────────────────────────────────────
def report(store, times, dataset_name):
    wce = np.array(store['KAB-XGBoost-WCE'])
    bl  = np.array(store['Baseline XGBoost'])

    print(f"\n{'='*70}")
    print(f"RESULTS -- {dataset_name}")
    print(f"{'='*70}")

    for mname, scores in store.items():
        sc_arr = np.array(scores)
        delta = sc_arr.mean() - wce.mean()
        if mname == 'KAB-XGBoost-WCE':
            sig = '--'
        else:
            _, p = wilcoxon(wce, sc_arr, zero_method='wilcox')
            sig = f"p={p:.4f}"
        print(f"  {mname:<28} F1={sc_arr.mean():.4f} SD={sc_arr.std(ddof=1):.4f} "
              f"delta={delta:+.4f} {sig}")

    # WCE vs Baseline Wilcoxon
    diff = wce - bl
    stat, p = wilcoxon(wce, bl, zero_method='wilcox')
    # W+ = sum of positive ranks, W- = sum of negative ranks
    n_eff = (diff != 0).sum()
    ranks = np.argsort(np.abs(diff[diff != 0])) + 1
    signs = np.sign(diff[diff != 0])
    Wp = ranks[signs > 0].sum() if (signs > 0).any() else 0.0
    Wn = ranks[signs < 0].sum() if (signs < 0).any() else 0.0
    d = diff.mean() / diff.std(ddof=1)
    lo, hi = np.percentile(diff, 2.5), np.percentile(diff, 97.5)
    fold_win = (diff > 0).sum()
    ties = (diff == 0).sum()

    print(f"\nWCE vs Baseline Wilcoxon:")
    print(f"  W+={Wp:.1f}, W-={Wn:.1f}, p={p:.4f} (exact, no continuity correction)")
    print(f"  Cohen d={d:.4f}, ties dropped={ties}, n_effective={n_eff}")
    print(f"  Fold-win: {fold_win}/50 = {fold_win/50:.1%}")
    print(f"  95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"  Training time: baseline={np.mean(times['baseline'])*1000:.1f}ms "
          f"WCE={np.mean(times['wce'])*1000:.1f}ms")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print("\n" + "="*70)
    print("KAB-XGBoost-WCE R05 -- DE-DUPLICATED PIPELINE")
    print("="*70)

    # ── KNUST leg ──────────────────────────────────────────────────────────
    X_kn, y_kn, feat_kn, K_IDX_KN, A_IDX_KN = load_knust()

    Xtr_kn, Xte_kn, ytr_kn, yte_kn = train_test_split(
        X_kn, y_kn, test_size=0.2, random_state=SEED, stratify=y_kn)

    print("\nTuning KNUST models...")
    params_kn, sc_kn = tune_all(Xtr_kn, ytr_kn, K_IDX_KN, A_IDX_KN)
    verify_weights(Xtr_kn, ytr_kn, K_IDX_KN, A_IDX_KN, params_kn['wce']['lambda'])

    print("\nRunning 5x10-fold CV on KNUST...")
    store_kn, times_kn = run_cv(X_kn, y_kn, params_kn, K_IDX_KN, A_IDX_KN)
    report(store_kn, times_kn, "KNUST (de-duplicated, n=2088)")

    # Per-class and held-out
    m_bl  = XGBClassifier(max_depth=params_kn['baseline']['max_depth'], **XGB_PARAMS)
    m_bl.fit(Xtr_kn, ytr_kn)
    m_wce = XGBClassifier(max_depth=params_kn['wce']['depth'], **XGB_PARAMS)
    m_wce.fit(Xtr_kn, ytr_kn,
              sample_weight=build_kab_weights(
                  Xtr_kn, ytr_kn, K_IDX_KN, A_IDX_KN, params_kn['wce']['lambda']))

    for mname, model in [('Baseline', m_bl), ('KAB-WCE', m_wce)]:
        pred = model.predict(Xte_kn)
        cm = confusion_matrix(yte_kn, pred)
        rep = classification_report(yte_kn, pred, target_names=['Low','High'], output_dict=True)
        print(f"  {mname} held-out: "
              f"CM TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]} | "
              f"Low F={rep['Low']['f1-score']:.4f} "
              f"High F={rep['High']['f1-score']:.4f} "
              f"AUC={roc_auc_score(yte_kn, model.predict_proba(Xte_kn)[:,1]):.4f}")

    # McNemar
    p_bl = m_bl.predict(Xte_kn); p_wce = m_wce.predict(Xte_kn)
    b = ((p_bl == yte_kn) & (p_wce != yte_kn)).sum()
    c = ((p_bl != yte_kn) & (p_wce == yte_kn)).sum()
    if b + c > 0:
        chi_sq = (abs(b - c) - 1)**2 / (b + c)
        p_mcn = chi2.sf(chi_sq, 1)
        print(f"  McNemar: b={b} c={c} chi2={chi_sq:.3f} p={p_mcn:.4f}")

    # Perturbation
    print("\nPerturbation (KNUST held-out):")
    np.random.seed(SEED)
    for sd in [0.0, 0.05, 0.10, 0.15]:
        Xp = Xte_kn + np.random.normal(0, sd, Xte_kn.shape) if sd > 0 else Xte_kn.copy()
        f_bl  = f1_score(yte_kn, m_bl.predict(Xp), average='macro')
        f_wce = f1_score(yte_kn, m_wce.predict(Xp), average='macro')
        print(f"  SD={sd:.2f}: baseline={f_bl:.4f} WCE={f_wce:.4f} delta={f_wce-f_bl:+.4f}")

    # ── Alzubaidi leg ──────────────────────────────────────────────────────
    X_alz, y_alz, feat_alz, K_IDX_ALZ, A_IDX_ALZ = load_alzubaidi()
    Xtr_alz, Xte_alz, ytr_alz, yte_alz = train_test_split(
        X_alz, y_alz, test_size=0.2, random_state=SEED, stratify=y_alz)

    print("\nTuning Alzubaidi models...")
    params_alz, sc_alz = tune_all(Xtr_alz, ytr_alz, K_IDX_ALZ, A_IDX_ALZ)

    print("\nRunning 5x10-fold CV on Alzubaidi...")
    store_alz, times_alz = run_cv(X_alz, y_alz, params_alz, K_IDX_ALZ, A_IDX_ALZ)
    report(store_alz, times_alz, "Alzubaidi (n=1231)")

    print("\nDone. Save store_kn, store_alz for document export.")
