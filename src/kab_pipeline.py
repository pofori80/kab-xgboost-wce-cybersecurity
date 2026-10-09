"""
KAB-XGBoost-WCE : Complete reproducible pipeline (KNUST + Alzubaidi)
====================================================================
Single file that produces EVERY number in Methods and Results, for both
datasets, plus SHAP, robustness (ordinal, bootstrap, subgroup), and the
artificial-skew imbalance test.

Exports:
  results/fold_level_results_knust.csv       50 per-fold macro-F1 per model (KNUST)
  results/fold_level_results_alzubaidi.csv   50 per-fold macro-F1 per model (Alzubaidi)
  results/summary_results.json               every headline number, one run
  results/shap_importance.json               SHAP mean|value| rankings, both datasets
  results/seed_sensitivity.json              WCE vs Baseline across 5 seeds

Run:  python src/kab_pipeline.py   (from repo root)
Deps: see requirements.txt
"""

import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42

# ── Data paths (relative to repo root) ─────────────────────────────────────
KNUST_PATH = "data/KNUST_Cybersecurity_Survey_Responses_2026.xlsx"
ALZ_PATH   = "data/Alzubaidi_2021_Cybercrime_Awareness_Dataset.xlsx"

# ── Output directory ─────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

XGB = dict(n_estimators=50, learning_rate=0.15, subsample=0.9,
           colsample_bytree=0.9, random_state=SEED, n_jobs=-1,
           eval_metric="logloss", verbosity=0)


# ── WCE weight function (R10: median rule) ───────────────────────────────────
def kab_weights(X, y, k_idx, a_idx, lam):
    """
    Per-instance weight = lambda if K_agg < median(K_agg) OR A_agg < median(A_agg),
    else 1.0. Applies to all instances regardless of class label.
    Lambda is tuned per fold via 3-fold inner CV.
    """
    k_col = X[:, k_idx]
    a_col = X[:, a_idx]
    k_med = np.median(k_col)
    a_med = np.median(a_col)
    flag = (k_col < k_med) | (a_col < a_med)
    w = np.where(flag, lam, 1.0).astype(float)
    return w


# ── KNUST loading and preprocessing ─────────────────────────────────────────
def load_knust():
    df = pd.read_excel(KNUST_PATH, sheet_name="Form Responses 1")
    q = {f"Q{i}": df.columns[i] for i in range(1, 24)}
    return df, q


def dedup_knust(df, q):
    answered = ([q[f"Q{i}"] for i in range(1, 6)]
                + [q[f"Q{i}"] for i in range(6, 11)]
                + [q[f"Q{i}"] for i in range(13, 18)]
                + [q[f"Q{i}"] for i in range(18, 24)])
    key = df[answered].astype(str).agg("|".join, axis=1)
    keep = ~key.duplicated(keep="first")
    return df[keep].reset_index(drop=True), len(df), int((~keep).sum())


def features_knust(df, q):
    know = [q[f"Q{i}"] for i in range(6, 11)]
    att  = [q[f"Q{i}"] for i in range(13, 18)]
    demo = [q[f"Q{i}"] for i in range(1, 6)]
    work = df.copy()
    for c in demo:
        work[c] = work[c].astype("category").cat.codes
    work["K_agg"] = df[know].mean(axis=1)
    work["A_agg"] = df[att].mean(axis=1)
    cols = know + att + demo + ["K_agg", "A_agg"]
    printable = ([f"Q{i}" for i in range(6, 11)] + [f"Q{i}" for i in range(13, 18)]
                 + [f"Q{i}" for i in range(1, 6)] + ["K_agg", "A_agg"])
    return work[cols].to_numpy(float), printable, know, att, demo


def label_knust(df, q):
    beh = [q[f"Q{i}"] for i in range(18, 24)]
    bt  = df[beh].sum(axis=1)
    med = bt.median()
    return (bt > med).astype(int).to_numpy(), bt.to_numpy(), float(med)


def cronbach_alpha(frame):
    items     = frame.to_numpy(float)
    k         = items.shape[1]
    if k < 2:
        return float("nan")
    item_var  = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_var.sum() / total_var))


# ── Alzubaidi loading and preprocessing ─────────────────────────────────────
def load_alzubaidi():
    df = pd.read_excel(ALZ_PATH)

    def blk(p):
        return [c for c in df.columns if re.match(r"^\s*" + p + r"\s*\)?", str(c))]

    B13, B14 = blk("13")[0], blk("14")
    B18, B19, B20 = blk("18"), blk("19"), blk("20")
    B21, B22 = blk("21")[0], blk("22")

    FREQ  = {"always": 5, "often": 4, "sometimes": 3, "somtimes": 3,
             "seldom": 2, "rarely": 2, "never": 1, "do not know": 1, "don't know": 1}
    AGREE = {"strongly agree": 5, "agree": 4, "neutral": 3, "undecided": 3,
              "disagree": 2, "strongly disagree": 1}

    def sc(s, m, d=3.0):
        return s.astype(str).str.strip().str.lower().map(m).fillna(d)

    K = pd.concat([sc(df[c], FREQ) for c in B19 + B20], axis=1)
    A = pd.concat(
        [sc(df[c], AGREE) for c in B18 + B22]
        + [sc(df[B13], {"very secure": 5, "somewhat secure": 4, "neutral": 3,
                        "somewhat insecure": 2, "very insecure": 1}),
           pd.Series(np.select(
               [df[B21].astype(str).str.contains("serious", case=False),
                df[B21].astype(str).str.contains("vanish", case=False)],
               [5, 1], 3.0), index=df.index)],
        axis=1)
    B = pd.concat([sc(df[c], FREQ) for c in B14], axis=1)

    # Alzubaidi duplicate removal (17 exact duplicates)
    key = pd.concat([K, A, B], axis=1).astype(str).agg("|".join, axis=1)
    keep = ~key.duplicated(keep="first")
    n_raw = len(df)
    n_removed = int((~keep).sum())
    K, A, B = K[keep].reset_index(drop=True), A[keep].reset_index(drop=True), B[keep].reset_index(drop=True)
    print(f"  Alzubaidi: raw={n_raw}, removed={n_removed}, unique={len(K)}")

    # Label: use > median (same rule as KNUST, N4 fix)
    bmean = B.mean(axis=1)
    med_b = bmean.median()
    y = (bmean > med_b).astype(int).to_numpy()

    Xdf = pd.concat([K, A, K.mean(axis=1).rename("K_agg"),
                     A.mean(axis=1).rename("A_agg")], axis=1).fillna(3.0)
    printable = ([f"B19_{i}" for i in range(len(B19))]
                 + [f"B20_{i}" for i in range(len(B20))]
                 + [f"B18_{i}" for i in range(len(B18))]
                 + [f"B22_{i}" for i in range(len(B22))]
                 + ["B13", "B21", "K_agg", "A_agg"])
    kidx = list(Xdf.columns).index("K_agg")
    aidx = list(Xdf.columns).index("A_agg")
    return Xdf.to_numpy(float), y, printable, kidx, aidx, n_raw, n_removed


# ── Tuning ───────────────────────────────────────────────────────────────────
def tune_depth(Xtr, ytr, grid=(2, 3, 5, 7)):
    gs = GridSearchCV(XGBClassifier(**XGB), {"max_depth": list(grid)},
                      cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                      scoring="f1_macro", n_jobs=1)
    gs.fit(Xtr, ytr)
    return gs.best_params_["max_depth"]


def tune_wce(Xtr, ytr, kidx, aidx, depths=(2, 3, 5), lambdas=(1.25, 1.5)):
    """WCE tuning grid: depth in {2,3,5} x lambda in {1.25,1.5} (N7 fix: depth 2 added)."""
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    best, best_s = (2, 1.25), -1.0
    for d in depths:
        for lam in lambdas:
            s = []
            for tr, va in inner.split(Xtr, ytr):
                m = XGBClassifier(max_depth=d, **XGB)
                m.fit(Xtr[tr], ytr[tr],
                      sample_weight=kab_weights(Xtr[tr], ytr[tr], kidx, aidx, lam))
                s.append(f1_score(ytr[va], m.predict(Xtr[va]), average="macro"))
            if np.mean(s) > best_s:
                best, best_s = (d, lam), np.mean(s)
    return best


# ── Cross-validation ─────────────────────────────────────────────────────────
def repeated_cv(X, y, kidx, aidx, n_rep=5, n_fold=10, seed=SEED):
    names = ["KAB-XGBoost-WCE", "Baseline XGBoost", "Stock scale_pos_weight",
             "Decision Tree", "Naive Bayes", "Logistic Regression", "SVM", "Random Forest"]
    sc = {n: [] for n in names}
    tb, tw = [], []
    tuned_depths_base, tuned_depths_wce, tuned_lambdas = [], [], []
    for rep in range(n_rep):
        skf = StratifiedKFold(n_fold, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
            d_base      = tune_depth(Xtr, ytr)
            d_wce, lam  = tune_wce(Xtr, ytr, kidx, aidx)
            tuned_depths_base.append(d_base)
            tuned_depths_wce.append(d_wce)
            tuned_lambdas.append(lam)

            t0 = time.perf_counter()
            m = XGBClassifier(max_depth=d_base, **XGB).fit(Xtr, ytr)
            tb.append(time.perf_counter() - t0)
            sc["Baseline XGBoost"].append(f1_score(yte, m.predict(Xte), average="macro"))

            t0 = time.perf_counter()
            w = kab_weights(Xtr, ytr, kidx, aidx, lam)
            m = XGBClassifier(max_depth=d_wce, **XGB).fit(Xtr, ytr, sample_weight=w)
            tw.append(time.perf_counter() - t0)
            sc["KAB-XGBoost-WCE"].append(f1_score(yte, m.predict(Xte), average="macro"))

            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            m = XGBClassifier(max_depth=d_base, scale_pos_weight=spw, **XGB).fit(Xtr, ytr)
            sc["Stock scale_pos_weight"].append(f1_score(yte, m.predict(Xte), average="macro"))

            best_dt, best_s = 3, -1
            for dd in (3, 5, 7):
                ss = []
                for itr, iva in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xtr, ytr):
                    mm = DecisionTreeClassifier(max_depth=dd, random_state=SEED).fit(Xtr[itr], ytr[itr])
                    ss.append(f1_score(ytr[iva], mm.predict(Xtr[iva]), average="macro"))
                if np.mean(ss) > best_s:
                    best_dt, best_s = dd, np.mean(ss)
            m = DecisionTreeClassifier(max_depth=best_dt, random_state=SEED).fit(Xtr, ytr)
            sc["Decision Tree"].append(f1_score(yte, m.predict(Xte), average="macro"))

            sc["Naive Bayes"].append(f1_score(yte, GaussianNB().fit(Xtr, ytr).predict(Xte), average="macro"))
            m = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr_s, ytr)
            sc["Logistic Regression"].append(f1_score(yte, m.predict(Xte_s), average="macro"))
            m = SVC(kernel="rbf", random_state=SEED).fit(Xtr_s, ytr)
            sc["SVM"].append(f1_score(yte, m.predict(Xte_s), average="macro"))
            m = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
            sc["Random Forest"].append(f1_score(yte, m.predict(Xte), average="macro"))
    return sc, tb, tw, tuned_depths_base, tuned_depths_wce, tuned_lambdas


# ── Ablation ─────────────────────────────────────────────────────────────────
def ablation(X, y, kidx, aidx, n_rep=5, n_fold=10):
    """
    Four arms with FIXED hyperparameters (E1/E2 fix):
      A: depth=5, no weights
      B: depth=3, no weights
      C: depth=5, WCE lambda=1.5 (fixed, not tuned)
      D: depth=3, WCE lambda=1.5 (fixed, not tuned)
    Per-arm SDs and Wilcoxon tests are exported in run_dataset().
    Note: Arm D uses fixed lambda=1.5; the main WCE row in Table 3a uses
    per-fold tuned lambda and depth, which is why they differ.
    """
    arms = {"A": [], "B": [], "C": [], "D": []}
    for rep in range(n_rep):
        skf = StratifiedKFold(n_fold, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            arms["A"].append(f1_score(yte, XGBClassifier(max_depth=5, **XGB).fit(Xtr, ytr).predict(Xte), average="macro"))
            arms["B"].append(f1_score(yte, XGBClassifier(max_depth=3, **XGB).fit(Xtr, ytr).predict(Xte), average="macro"))
            w = kab_weights(Xtr, ytr, kidx, aidx, 1.5)
            arms["C"].append(f1_score(yte, XGBClassifier(max_depth=5, **XGB).fit(Xtr, ytr, sample_weight=w).predict(Xte), average="macro"))
            arms["D"].append(f1_score(yte, XGBClassifier(max_depth=3, **XGB).fit(Xtr, ytr, sample_weight=w).predict(Xte), average="macro"))
    return arms


def dt_curve(X, y, n_rep=5, n_fold=10):
    out = {}
    for depth in [1, 2, 3, None]:
        s = []
        for rep in range(n_rep):
            skf = StratifiedKFold(n_fold, shuffle=True, random_state=SEED + rep)
            for tr, te in skf.split(X, y):
                m = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X[tr], y[tr])
                s.append(f1_score(y[te], m.predict(X[te]), average="macro"))
        out["unlimited" if depth is None else f"depth{depth}"] = [round(float(np.mean(s)), 4), round(float(np.std(s)), 4)]
    return out


# ── Held-out ─────────────────────────────────────────────────────────────────
def held_out(X, y, kidx, aidx, d_base, d_wce, lam):
    """Tune on training split only (N2 fix: no leakage into held-out)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    # Retune on training split only
    d_base_ho = tune_depth(Xtr, ytr)
    d_wce_ho, lam_ho = tune_wce(Xtr, ytr, kidx, aidx)
    mb = XGBClassifier(max_depth=d_base_ho, **XGB).fit(Xtr, ytr)
    w  = kab_weights(Xtr, ytr, kidx, aidx, lam_ho)
    mw = XGBClassifier(max_depth=d_wce_ho, **XGB).fit(Xtr, ytr, sample_weight=w)
    pb, pw = mb.predict(Xte), mw.predict(Xte)
    b = int(np.sum((pb == yte) & (pw != yte)))
    c = int(np.sum((pb != yte) & (pw == yte)))
    chisq = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) else 0.0
    wce_ho_f1   = round(float(f1_score(yte, pw, average="macro")), 4)
    base_ho_f1  = round(float(f1_score(yte, pb, average="macro")), 4)
    return dict(
        n_test=int(len(yte)),
        cm_baseline=confusion_matrix(yte, pb).tolist(),
        cm_wce=confusion_matrix(yte, pw).tolist(),
        mcnemar_b=b, mcnemar_c=c,
        mcnemar_chi2=round(float(chisq), 3),
        mcnemar_p=round(float(1 - chi2.cdf(chisq, 1)), 4),
        auc_baseline=round(float(roc_auc_score(yte, mb.predict_proba(Xte)[:, 1])), 4),
        auc_wce=round(float(roc_auc_score(yte, mw.predict_proba(Xte)[:, 1])), 4),
        wce_heldout_macro_f1=wce_ho_f1,
        baseline_heldout_macro_f1=base_ho_f1,
        heldout_tuned_base_depth=int(d_base_ho),
        heldout_tuned_wce_depth=int(d_wce_ho),
        heldout_tuned_lambda=float(lam_ho),
    )


# ── Robustness ───────────────────────────────────────────────────────────────
def robustness(X, y, kidx, aidx, d_wce, lam, n_likert):
    """
    Ordinal robustness: perturb Likert items, then recompute K_agg and A_agg
    from perturbed items before predicting (Q8/N8 fix).
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    # Retune on training split only
    d_wce_ho, lam_ho = tune_wce(Xtr, ytr, kidx, aidx)
    w = kab_weights(Xtr, ytr, kidx, aidx, lam_ho)
    m = XGBClassifier(max_depth=d_wce_ho, **XGB).fit(Xtr, ytr, sample_weight=w)
    base_pred = m.predict(Xte)
    base_f1   = f1_score(yte, base_pred, average="macro")
    rng    = np.random.RandomState(SEED)
    likert = list(range(n_likert))
    jit    = []
    for _ in range(100):
        Xj = Xte.copy()
        for c in likert:
            mask  = rng.rand(Xj.shape[0]) < 0.10
            noise = rng.choice([-1, 1], size=int(mask.sum()))
            Xj[mask, c] = np.clip(Xj[mask, c] + noise, 1, 5)
        # Recompute K_agg and A_agg from perturbed items
        k_cols = list(range(0, 5))   # first 5 cols = Q6-Q10 = Knowledge items
        a_cols = list(range(5, 10))  # next 5 cols = Q13-Q17 = Attitude items
        Xj[:, kidx] = Xj[:, k_cols].mean(axis=1)
        Xj[:, aidx] = Xj[:, a_cols].mean(axis=1)
        jit.append(f1_score(yte, m.predict(Xj), average="macro"))
    boots = []
    for _ in range(1000):
        idx = rng.randint(0, len(yte), len(yte))
        boots.append(f1_score(yte[idx], base_pred[idx], average="macro"))
    perturbed_mean = float(np.mean(jit))
    return dict(
        ordinal_base=round(float(base_f1), 4),
        ordinal_mean=round(float(perturbed_mean), 4),
        ordinal_sd=round(float(np.std(jit)), 4),
        ordinal_change=round(float(perturbed_mean - base_f1), 4),  # perturbed - base; negative = degradation
        bootstrap_mean=round(float(np.mean(boots)), 4),
        bootstrap_lo=round(float(np.percentile(boots, 2.5)), 4),
        bootstrap_hi=round(float(np.percentile(boots, 97.5)), 4),
        bootstrap_iters=1000,
    )


def subgroups(X, y, kidx, aidx, demo_codes):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        d_wce, lam = tune_wce(X[tr], y[tr], kidx, aidx)
        w = kab_weights(X[tr], y[tr], kidx, aidx, lam)
        m = XGBClassifier(max_depth=d_wce, **XGB).fit(X[tr], y[tr], sample_weight=w)
        oof[te] = m.predict(X[te])
    out = {}
    for label, codes in demo_codes.items():
        grp = {}
        for v in sorted(np.unique(codes)):
            mask = codes == v
            if mask.sum() > 30:
                grp[str(int(v))] = dict(n=int(mask.sum()),
                                        f1=round(float(f1_score(y[mask], oof[mask], average="macro")), 4))
        out[label] = grp
    return out


def skew_test(X, y, kidx, aidx, minority_frac=0.25):
    """5x10 folds with per-fold tuned lambda; Wilcoxon WCE vs scale_pos_weight."""
    rng   = np.random.RandomState(SEED)
    low   = np.where(y == 0)[0]
    high  = np.where(y == 1)[0]
    n_high = int(len(low) * minority_frac / (1 - minority_frac))
    n_high = min(n_high, len(high))
    keep  = np.concatenate([low, rng.choice(high, n_high, replace=False)])
    Xs, ys = X[keep], y[keep]
    base, wce_s, spw = [], [], []
    for rep in range(5):
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED + rep).split(Xs, ys):
            d_b = tune_depth(Xs[tr], ys[tr])
            d_w, lam = tune_wce(Xs[tr], ys[tr], kidx, aidx)
            base.append(f1_score(ys[te], XGBClassifier(max_depth=d_b, **XGB).fit(Xs[tr], ys[tr]).predict(Xs[te]), average="macro"))
            w = kab_weights(Xs[tr], ys[tr], kidx, aidx, lam)
            wce_s.append(f1_score(ys[te], XGBClassifier(max_depth=d_w, **XGB).fit(Xs[tr], ys[tr], sample_weight=w).predict(Xs[te]), average="macro"))
            r = (ys[tr] == 0).sum() / max((ys[tr] == 1).sum(), 1)
            spw.append(f1_score(ys[te], XGBClassifier(max_depth=d_b, scale_pos_weight=r, **XGB).fit(Xs[tr], ys[tr]).predict(Xs[te]), average="macro"))
    b_arr, wce_arr, sp_arr = np.array(base), np.array(wce_s), np.array(spw)
    _, p_wce_vs_spw = wilcoxon(wce_arr, sp_arr)
    _, p_wce_vs_base = wilcoxon(wce_arr, b_arr)
    return dict(
        minority_frac=minority_frac, n=int(len(ys)),
        baseline=round(float(b_arr.mean()), 4), baseline_sd=round(float(b_arr.std(ddof=1)), 4),
        wce=round(float(wce_arr.mean()), 4), wce_sd=round(float(wce_arr.std(ddof=1)), 4),
        scale_pos_weight=round(float(sp_arr.mean()), 4), spw_sd=round(float(sp_arr.std(ddof=1)), 4),
        wce_gain=round(float(wce_arr.mean() - b_arr.mean()), 4),
        spw_gain=round(float(sp_arr.mean() - b_arr.mean()), 4),
        wilcoxon_wce_vs_spw_p=round(float(p_wce_vs_spw), 4),
        wilcoxon_wce_vs_base_p=round(float(p_wce_vs_base), 4),
    )


def shap_importance(X, y, kidx, aidx, feat_names):
    """Tune SHAP model on 80% training split only (N2 fix)."""
    try:
        import shap
    except ImportError:
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    d_wce, lam = tune_wce(Xtr, ytr, kidx, aidx)
    w = kab_weights(Xtr, ytr, kidx, aidx, lam)
    m = XGBClassifier(max_depth=d_wce, **XGB).fit(Xtr, ytr, sample_weight=w)
    sv = shap.TreeExplainer(m).shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    ranking = sorted(zip(feat_names, mean_abs.tolist()), key=lambda t: -t[1])
    return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in ranking]


def wilcoxon_stats(a, b):
    """Return W+, W-, ties, p, effect r for paired comparison."""
    diff = np.array(a) - np.array(b)
    nz   = diff[diff != 0]
    ties = int((diff == 0).sum())
    if len(nz) < 2:
        return dict(w_plus=0.0, w_minus=0.0, ties=ties, n_nontied=0, p=1.0, r=0.0)
    _, p = wilcoxon(np.array(a), np.array(b))
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    wplus  = float(ranks[nz > 0].sum())
    wminus = float(ranks[nz < 0].sum())
    z = (wplus - wminus) / np.sqrt(len(nz) * (len(nz) + 1) * (2 * len(nz) + 1) / 6 + 1e-9)
    r = abs(z) / np.sqrt(len(nz))
    return dict(w_plus=wplus, w_minus=wminus, ties=ties,
                n_nontied=int(len(nz)), p=round(float(p), 4), r=round(float(r), 4))


# ── Main dataset runner ──────────────────────────────────────────────────────
def run_dataset(name, X, y, kidx, aidx, feat_names, do_subgroups=None, seed=SEED):
    print(f"\n{'='*60}\n{name}  X.shape={X.shape}  Low={int((y==0).sum())} High={int((y==1).sum())}")
    d_base = tune_depth(X, y)
    d_wce, lam = tune_wce(X, y, kidx, aidx)
    print(f"  tuned baseline depth={d_base}  WCE depth={d_wce} lambda={lam}")
    sc, tb, tw, depths_base, depths_wce, lambdas = repeated_cv(X, y, kidx, aidx, seed=seed)
    fold_df = pd.DataFrame(sc)
    fold_df.insert(0, "fold", range(1, len(fold_df) + 1))
    fold_df.to_csv(os.path.join(RESULTS_DIR, f"fold_level_results_{name.lower()}.csv"), index=False)

    wce  = np.array(sc["KAB-XGBoost-WCE"])
    base = np.array(sc["Baseline XGBoost"])
    diff = wce - base
    wvb  = wilcoxon_stats(wce, base)
    print(f"  WCE {round(float(wce.mean()),4)}  Baseline {round(float(base.mean()),4)}  p={wvb['p']}")

    # Ablation with per-arm Wilcoxon tests
    arms = ablation(X, y, kidx, aidx)
    A, B, C, Dd = (np.array(arms[k]) for k in "ABCD")
    _, p_ba = wilcoxon(B, A)
    _, p_ca = wilcoxon(C, A)
    _, p_db = wilcoxon(Dd, B)

    curve = dt_curve(X, y)
    ho    = held_out(X, y, kidx, aidx, d_base, d_wce, lam)
    rob   = robustness(X, y, kidx, aidx, d_wce, lam, n_likert=10)
    skew  = skew_test(X, y, kidx, aidx)
    shp   = shap_importance(X, y, kidx, aidx, feat_names)
    subg  = subgroups(X, y, kidx, aidx, do_subgroups) if do_subgroups else None

    # All-comparator Wilcoxon vs WCE
    comp_tests = {}
    for cname, cscores in sc.items():
        if cname == "KAB-XGBoost-WCE":
            continue
        stats = wilcoxon_stats(np.array(cscores), wce)
        comp_tests[cname] = dict(
            mean=round(float(np.mean(cscores)), 4),
            sd=round(float(np.std(cscores, ddof=1)), 4),
            delta_vs_wce=round(float(np.mean(cscores) - wce.mean()), 4),
            **stats
        )

    print(f"  R11 base_f1={rob['ordinal_base']}  held-out WCE F1={ho['wce_heldout_macro_f1']}")
    return dict(
        X_shape=list(X.shape), feature_order=feat_names,
        class_low=int((y == 0).sum()), class_high=int((y == 1).sum()),
        tuned_baseline_depth=int(d_base), tuned_wce_depth=int(d_wce), tuned_wce_lambda=float(lam),
        avg_fold_baseline_depth=round(float(np.mean(depths_base)), 3),
        avg_fold_wce_depth=round(float(np.mean(depths_wce)), 3),
        avg_fold_wce_lambda=round(float(np.mean(lambdas)), 3),
        cv={m: dict(mean=round(float(np.mean(v)), 4), sd=round(float(np.std(v, ddof=1)), 4)) for m, v in sc.items()},
        wce_vs_baseline=dict(delta=round(float(diff.mean()), 4), **wvb),
        comparator_tests=comp_tests,
        ablation=dict(
            A=dict(mean=round(float(A.mean()), 4), sd=round(float(A.std(ddof=1)), 4)),
            B=dict(mean=round(float(B.mean()), 4), sd=round(float(B.std(ddof=1)), 4)),
            C=dict(mean=round(float(C.mean()), 4), sd=round(float(C.std(ddof=1)), 4)),
            D=dict(mean=round(float(Dd.mean()), 4), sd=round(float(Dd.std(ddof=1)), 4)),
            depth_contribution=round(float(B.mean() - A.mean()), 4),
            weight_at_depth5=round(float(C.mean() - A.mean()), 4),
            weight_at_depth3=round(float(Dd.mean() - B.mean()), 4),
            p_B_vs_A=round(float(p_ba), 4),
            p_C_vs_A=round(float(p_ca), 4),
            p_D_vs_B=round(float(p_db), 4),
        ),
        dt_depth_curve={k: dict(mean=v[0], sd=v[1]) for k, v in curve.items()},
        held_out=ho,
        timing=dict(baseline_ms=round(float(np.mean(tb) * 1000), 1),
                    wce_ms=round(float(np.mean(tw) * 1000), 1)),
        robustness=rob, skew_test=skew, subgroups=subg, shap=shp,
    )


# ── Seed sensitivity ─────────────────────────────────────────────────────────
def seed_sensitivity(X, y, kidx, aidx, seeds=(0, 7, 13, 21, 42)):
    """WCE vs Baseline across multiple XGBoost random seeds."""
    results = {}
    for s in seeds:
        print(f"  seed={s}...")
        global XGB
        XGB_orig = XGB.copy()
        XGB = {**XGB, "random_state": s}
        sc, _, _, _, _, _ = repeated_cv(X, y, kidx, aidx, seed=s)
        wce  = np.array(sc["KAB-XGBoost-WCE"])
        base = np.array(sc["Baseline XGBoost"])
        stats = wilcoxon_stats(wce, base)
        results[str(s)] = dict(
            wce_mean=round(float(wce.mean()), 4),
            base_mean=round(float(base.mean()), 4),
            delta=round(float((wce - base).mean()), 4),
            **stats
        )
        XGB = XGB_orig
    return results


# ── Straight-line detection ──────────────────────────────────────────────────
def straight_line_screen(df, q):
    """Count rows where all Likert items are identical (Q11 fix)."""
    likert_cols = ([q[f"Q{i}"] for i in range(6, 11)]
                   + [q[f"Q{i}"] for i in range(13, 18)]
                   + [q[f"Q{i}"] for i in range(18, 24)])
    sub = df[likert_cols]
    straight = (sub.nunique(axis=1) == 1)
    return int(straight.sum())


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("KAB-XGBoost-WCE pipeline R10")
    print(f"SEED={SEED}  KNUST={KNUST_PATH}  ALZ={ALZ_PATH}")

    # ── KNUST ────────────────────────────────────────────────────────────────
    df, q = load_knust()
    dfc, n_raw, n_removed = dedup_knust(df, q)
    sl_count = straight_line_screen(dfc, q)
    print(f"KNUST: raw={n_raw} removed={n_removed} unique={len(dfc)} straight-line={sl_count}")

    Xk, feat_k, know, att, demo = features_knust(dfc, q)
    yk, bt, med = label_knust(dfc, q)
    kidx, aidx = feat_k.index("K_agg"), feat_k.index("A_agg")

    alpha_know = cronbach_alpha(dfc[know])
    alpha_att  = cronbach_alpha(dfc[att])
    alpha_beh  = cronbach_alpha(dfc[[q[f"Q{i}"] for i in range(18, 24)]])

    vals, counts = np.unique(bt, return_counts=True)
    dist = {int(v): int(c) for v, c in zip(vals, counts)}

    # Threshold sensitivity: both WCE and Baseline at each cut (Q17 fix)
    thr = {}
    for name_t, cut in [("median", med),
                         ("tertile_top", float(np.quantile(bt, 2 / 3))),
                         ("quartile_top", float(np.quantile(bt, 0.75)))]:
        yt = (bt > cut).astype(int)
        if len(np.unique(yt)) == 2:
            base_s, wce_s2 = [], []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xk, yt):
                d_b = tune_depth(Xk[tr], yt[tr])
                d_w, lam_t = tune_wce(Xk[tr], yt[tr], kidx, aidx)
                m_b = XGBClassifier(max_depth=d_b, **XGB).fit(Xk[tr], yt[tr])
                base_s.append(f1_score(yt[te], m_b.predict(Xk[te]), average="macro"))
                w = kab_weights(Xk[tr], yt[tr], kidx, aidx, lam_t)
                m_w = XGBClassifier(max_depth=d_w, **XGB).fit(Xk[tr], yt[tr], sample_weight=w)
                wce_s2.append(f1_score(yt[te], m_w.predict(Xk[te]), average="macro"))
            _, p_thr = wilcoxon(np.array(wce_s2), np.array(base_s))
            thr[name_t] = dict(
                cut=float(cut),
                low_frac=round(float((yt == 0).mean()), 3),
                baseline_f1=round(float(np.mean(base_s)), 4),
                wce_f1=round(float(np.mean(wce_s2)), 4),
                wilcoxon_p=round(float(p_thr), 4),
            )

    demo_codes = dict(
        year=dfc[q["Q2"]].astype("category").cat.codes.to_numpy(),
        it_course=dfc[q["Q3"]].astype("category").cat.codes.to_numpy(),
    )
    knust = run_dataset("KNUST", Xk, yk, kidx, aidx, feat_k, do_subgroups=demo_codes)
    knust.update(dict(
        n_raw=n_raw, n_removed=n_removed, n_unique=len(dfc),
        straight_line_removed=sl_count,
        label_median=med,
        cronbach=dict(knowledge=round(alpha_know, 3),
                      attitude=round(alpha_att, 3),
                      behaviour=round(alpha_beh, 3)),
        behaviour_distribution=dist,
        threshold_sensitivity=thr,
        sampling_frame=dict(registered_est=3200, raw=n_raw, unique=len(dfc),
                            response_rate_raw=round(n_raw / 3200, 3),
                            response_rate_unique=round(len(dfc) / 3200, 3)),
    ))

    # ── Alzubaidi ────────────────────────────────────────────────────────────
    Xa, ya, feat_a, kidx_a, aidx_a, n_raw_a, n_rem_a = load_alzubaidi()
    alz = run_dataset("ALZUBAIDI", Xa, ya, kidx_a, aidx_a, feat_a)
    alz.update(dict(n_raw=n_raw_a, n_removed=n_rem_a, n_unique=len(ya)))

    # ── Seed sensitivity (KNUST) ─────────────────────────────────────────────
    print("\nSeed sensitivity (KNUST)...")
    seeds_out = seed_sensitivity(Xk, yk, kidx, aidx)

    # ── Write outputs ─────────────────────────────────────────────────────────
    summary = dict(knust=knust, alzubaidi=alz)
    json.dump(summary,
              open(os.path.join(RESULTS_DIR, "summary_results.json"), "w"), indent=2)
    json.dump({"knust": knust.get("shap"), "alzubaidi": alz.get("shap")},
              open(os.path.join(RESULTS_DIR, "shap_importance.json"), "w"), indent=2)
    json.dump(seeds_out,
              open(os.path.join(RESULTS_DIR, "seed_sensitivity.json"), "w"), indent=2)
    print(f"\nWrote results to {RESULTS_DIR}/")
    print("  summary_results.json")
    print("  shap_importance.json")
    print("  seed_sensitivity.json")
    print("  fold_level_results_knust.csv")
    print("  fold_level_results_alzubaidi.csv")


if __name__ == "__main__":
    main()
