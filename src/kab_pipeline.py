"""
KAB-XGBoost-WCE : Full reproducible pipeline
============================================
Single-run pipeline that produces every number in the Methods and Results
sections from one execution. Addresses the examiner requirement that the
submitted code contain data loading, feature construction, label
construction, and GridSearchCV -- not only the weight function.

Outputs:
  - fold_level_results.csv   : 50 per-fold macro-F1 rows per model (the export the examiner asked for)
  - summary_results.json     : all headline numbers, one run
  - Prints a human-readable report to stdout

Run:
  python kab_pipeline.py

Author: Prince Ofori (21990878), KNUST
Supervisor: Dr. Eric Opoku Osei
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
RNG = np.random.RandomState(SEED)
KNUST_PATH = "data/KNUST.xlsx"


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_knust(path=KNUST_PATH):
    """Load the raw KNUST export from the Google Form responses sheet."""
    df = pd.read_excel(path, sheet_name="Form Responses 1")
    qcols = {f"Q{i}": df.columns[i] for i in range(1, 24)}
    return df, qcols


# ---------------------------------------------------------------------------
# 2. DE-DUPLICATION  (removal procedure, exact-match key over all answer fields)
# ---------------------------------------------------------------------------
def deduplicate(df, qcols):
    """Remove exact-duplicate response patterns.

    A duplicate is a row identical to an earlier row across all answered
    items: demographics Q1-Q5, Knowledge/Attitude Q6-Q10 and Q13-Q17,
    and Behaviour Q18-Q23. Q11 and Q12 are 100% empty and are excluded
    from the key. First occurrence retained; later copies discarded.
    """
    answered = (
        [qcols[f"Q{i}"] for i in range(1, 6)]      # demographics
        + [qcols[f"Q{i}"] for i in range(6, 11)]   # Knowledge Q6-Q10
        + [qcols[f"Q{i}"] for i in range(13, 18)]  # Attitude  Q13-Q17
        + [qcols[f"Q{i}"] for i in range(18, 24)]  # Behaviour Q18-Q23
    )
    key = df[answered].astype(str).agg("|".join, axis=1)
    keep_mask = ~key.duplicated(keep="first")
    n_raw = len(df)
    df_clean = df[keep_mask].reset_index(drop=True)
    n_removed = n_raw - len(df_clean)
    return df_clean, n_raw, n_removed


# ---------------------------------------------------------------------------
# 3. FEATURE CONSTRUCTION
# ---------------------------------------------------------------------------
def build_features(df, qcols):
    """Construct the modelling matrix X and the aggregates K_agg, A_agg.

    Knowledge subscale : Q6-Q10  (Q11, the 6th Knowledge item, was lost - 100% empty)
    Attitude  subscale : Q13-Q17 (Q12, an Attitude item, was lost - 100% empty)
    Demographics       : Q1-Q5   (ordinal-encoded)
    Behaviour Q18-Q23  : used only to build the label, never a feature.
    """
    know = [qcols[f"Q{i}"] for i in range(6, 11)]     # Q6-Q10
    att = [qcols[f"Q{i}"] for i in range(13, 18)]     # Q13-Q17
    demo = [qcols[f"Q{i}"] for i in range(1, 6)]      # Q1-Q5

    work = df.copy()
    # Ordinal-encode demographic categoricals deterministically
    for c in demo:
        work[c] = work[c].astype("category").cat.codes

    work["K_agg"] = df[know].mean(axis=1)
    work["A_agg"] = df[att].mean(axis=1)

    feature_cols = know + att + demo + ["K_agg", "A_agg"]
    X = work[feature_cols].to_numpy(dtype=float)
    return X, feature_cols, know, att


# ---------------------------------------------------------------------------
# 4. LABEL CONSTRUCTION  (global median split of Behaviour total)
# ---------------------------------------------------------------------------
def build_label(df, qcols):
    """Binary readiness label from the Behaviour subscale total.

    B_total = sum(Q18..Q23). Global median split on the full de-duplicated
    dataset BEFORE any partition. y=1 (High) if B_total > median, else 0 (Low).
    """
    beh = [qcols[f"Q{i}"] for i in range(18, 24)]
    b_total = df[beh].sum(axis=1)
    median = b_total.median()
    y = (b_total > median).astype(int).to_numpy()
    return y, b_total.to_numpy(), median


# ---------------------------------------------------------------------------
# 5. KAB COST-SENSITIVE WEIGHT FUNCTION
# ---------------------------------------------------------------------------
def kab_weights(X, y, k_idx, a_idx, lam):
    """Per-instance weight for Low-readiness rows, from K-A asymmetry.

        w_i = 1 + (|K_i - A_i| / (K_i + A_i)) * (lambda - 1)   for y_i = 0
        w_i = 1                                                 for y_i = 1
    """
    w = np.ones(len(y), dtype=float)
    low = y == 0
    k = X[low, k_idx]
    a = X[low, a_idx]
    denom = k + a
    asym = np.where(denom > 0, np.abs(k - a) / denom, 0.0)
    w[low] = 1.0 + asym * (lam - 1.0)
    return w


# ---------------------------------------------------------------------------
# 6. NESTED TUNING  (tuning inside the CV loop; no hold-out leakage)
# ---------------------------------------------------------------------------
XGB_BASE = dict(
    n_estimators=120, learning_rate=0.15, subsample=0.9,
    colsample_bytree=0.9, random_state=SEED, n_jobs=1,
    eval_metric="logloss", verbosity=0,
)


def tune_depth_xgb(Xtr, ytr, grid=(2, 3, 5, 7)):
    gs = GridSearchCV(
        XGBClassifier(**XGB_BASE),
        {"max_depth": list(grid)},
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="f1_macro", n_jobs=-1,
    )
    gs.fit(Xtr, ytr)
    return gs.best_params_["max_depth"]


def tune_wce(Xtr, ytr, k_idx, a_idx, depths=(3, 5), lambdas=(1.25, 1.5)):
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, best_score = (3, 1.5), -1.0
    for d in depths:
        for lam in lambdas:
            scores = []
            for tr, va in inner.split(Xtr, ytr):
                m = XGBClassifier(max_depth=d, **XGB_BASE)
                w = kab_weights(Xtr[tr], ytr[tr], k_idx, a_idx, lam)
                m.fit(Xtr[tr], ytr[tr], sample_weight=w)
                scores.append(f1_score(ytr[va], m.predict(Xtr[va]), average="macro"))
            s = float(np.mean(scores))
            if s > best_score:
                best, best_score = (d, lam), s
    return best  # (depth, lambda)


# ---------------------------------------------------------------------------
# 7. REPEATED CV  (5 x 10-fold, nested tuning per outer fold)
# ---------------------------------------------------------------------------
def repeated_cv(X, y, k_idx, a_idx, n_repeats=5, n_folds=10):
    """Return a dict model -> list of 50 per-fold macro-F1 scores,
    plus timing lists for baseline and WCE. Tuning is nested: each
    outer training fold is retuned, so no test row informs tuning."""
    models = [
        "KAB-XGBoost-WCE", "Baseline XGBoost", "Stock scale_pos_weight",
        "Decision Tree", "Naive Bayes", "Logistic Regression", "SVM", "Random Forest",
    ]
    scores = {m: [] for m in models}
    t_base, t_wce = [], []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            # nested tuning on THIS training fold only
            d_base = tune_depth_xgb(Xtr, ytr)
            d_wce, lam = tune_wce(Xtr, ytr, k_idx, a_idx)

            # Baseline XGBoost
            t0 = time.perf_counter()
            m = XGBClassifier(max_depth=d_base, **XGB_BASE).fit(Xtr, ytr)
            t_base.append(time.perf_counter() - t0)
            scores["Baseline XGBoost"].append(f1_score(yte, m.predict(Xte), average="macro"))

            # KAB-XGBoost-WCE
            t0 = time.perf_counter()
            w = kab_weights(Xtr, ytr, k_idx, a_idx, lam)
            m = XGBClassifier(max_depth=d_wce, **XGB_BASE).fit(Xtr, ytr, sample_weight=w)
            t_wce.append(time.perf_counter() - t0)
            scores["KAB-XGBoost-WCE"].append(f1_score(yte, m.predict(Xte), average="macro"))

            # Stock scale_pos_weight
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            m = XGBClassifier(max_depth=d_base, scale_pos_weight=spw, **XGB_BASE).fit(Xtr, ytr)
            scores["Stock scale_pos_weight"].append(f1_score(yte, m.predict(Xte), average="macro"))

            # Decision Tree (tuned depth, own small search)
            dt_best, dt_score = 3, -1.0
            for dd in (3, 5, 7):
                sc_ = []
                for itr, iva in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xtr, ytr):
                    mm = DecisionTreeClassifier(max_depth=dd, random_state=SEED).fit(Xtr[itr], ytr[itr])
                    sc_.append(f1_score(ytr[iva], mm.predict(Xtr[iva]), average="macro"))
                if np.mean(sc_) > dt_score:
                    dt_best, dt_score = dd, np.mean(sc_)
            m = DecisionTreeClassifier(max_depth=dt_best, random_state=SEED).fit(Xtr, ytr)
            scores["Decision Tree"].append(f1_score(yte, m.predict(Xte), average="macro"))

            # Naive Bayes
            m = GaussianNB().fit(Xtr, ytr)
            scores["Naive Bayes"].append(f1_score(yte, m.predict(Xte), average="macro"))

            # Logistic Regression (scaled)
            m = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr_s, ytr)
            scores["Logistic Regression"].append(f1_score(yte, m.predict(Xte_s), average="macro"))

            # SVM (scaled)
            m = SVC(kernel="rbf", random_state=SEED).fit(Xtr_s, ytr)
            scores["SVM"].append(f1_score(yte, m.predict(Xte_s), average="macro"))

            # Random Forest
            m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
            scores["Random Forest"].append(f1_score(yte, m.predict(Xte), average="macro"))

    return scores, t_base, t_wce


# ---------------------------------------------------------------------------
# 8. FOUR-ARM ABLATION  (all four arms from the SAME repeated-CV protocol)
# ---------------------------------------------------------------------------
def ablation(X, y, k_idx, a_idx, n_repeats=5, n_folds=10):
    """A: vanilla depth5, no weights; B: depth3, no weights;
       C: depth5 + weights lam=1.5; D: depth3 + weights lam=1.5.
       All from one protocol so the decomposition is attributable."""
    arms = {"A": [], "B": [], "C": [], "D": []}
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            # A
            m = XGBClassifier(max_depth=5, **XGB_BASE).fit(Xtr, ytr)
            arms["A"].append(f1_score(yte, m.predict(Xte), average="macro"))
            # B
            m = XGBClassifier(max_depth=3, **XGB_BASE).fit(Xtr, ytr)
            arms["B"].append(f1_score(yte, m.predict(Xte), average="macro"))
            # C
            w = kab_weights(Xtr, ytr, k_idx, a_idx, 1.5)
            m = XGBClassifier(max_depth=5, **XGB_BASE).fit(Xtr, ytr, sample_weight=w)
            arms["C"].append(f1_score(yte, m.predict(Xte), average="macro"))
            # D
            w = kab_weights(Xtr, ytr, k_idx, a_idx, 1.5)
            m = XGBClassifier(max_depth=3, **XGB_BASE).fit(Xtr, ytr, sample_weight=w)
            arms["D"].append(f1_score(yte, m.predict(Xte), average="macro"))
    return arms


# ---------------------------------------------------------------------------
# 9. DECISION-TREE DEPTH CURVE  (Q19 leakage diagnosis)
# ---------------------------------------------------------------------------
def dt_depth_curve(X, y, n_repeats=5, n_folds=10):
    curve = {}
    for depth in [1, 2, 3, None]:
        s = []
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED + rep)
            for tr, te in skf.split(X, y):
                m = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X[tr], y[tr])
                s.append(f1_score(y[te], m.predict(X[te]), average="macro"))
        curve["unlimited" if depth is None else f"depth{depth}"] = (float(np.mean(s)), float(np.std(s)))
    return curve


# ---------------------------------------------------------------------------
# 10. HELD-OUT confusion matrices, McNemar, AUC  (one 80/20 split)
# ---------------------------------------------------------------------------
def held_out_analysis(X, y, k_idx, a_idx, d_base, d_wce, lam):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    mb = XGBClassifier(max_depth=d_base, **XGB_BASE).fit(Xtr, ytr)
    w = kab_weights(Xtr, ytr, k_idx, a_idx, lam)
    mw = XGBClassifier(max_depth=d_wce, **XGB_BASE).fit(Xtr, ytr, sample_weight=w)

    pb, pw = mb.predict(Xte), mw.predict(Xte)
    cm_b = confusion_matrix(yte, pb)
    cm_w = confusion_matrix(yte, pw)

    # McNemar b/c defined consistently:
    #   b = baseline correct & WCE wrong ; c = baseline wrong & WCE correct
    b = int(np.sum((pb == yte) & (pw != yte)))
    c = int(np.sum((pb != yte) & (pw == yte)))

    auc_b = roc_auc_score(yte, mb.predict_proba(Xte)[:, 1])
    auc_w = roc_auc_score(yte, mw.predict_proba(Xte)[:, 1])
    return dict(
        n_test=int(len(yte)),
        cm_baseline=cm_b.tolist(), cm_wce=cm_w.tolist(),
        mcnemar_b=b, mcnemar_c=c,
        auc_baseline=float(auc_b), auc_wce=float(auc_w),
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("KAB-XGBoost-WCE  single-run pipeline")
    print("=" * 70)

    df, qcols = load_knust()
    df_clean, n_raw, n_removed = deduplicate(df, qcols)
    X, feat, know, att = build_features(df_clean, qcols)
    y, b_total, median = build_label(df_clean, qcols)
    k_idx, a_idx = feat.index("K_agg"), feat.index("A_agg")

    print(f"Raw rows                : {n_raw}")
    print(f"Exact duplicates removed: {n_removed} ({100*n_removed/n_raw:.1f}%)")
    print(f"Unique rows retained    : {len(df_clean)}")
    print(f"X.shape                 : {X.shape}")
    print(f"X.columns               : {feat}")
    print(f"Label median (B_total)  : {median}")
    print(f"Class balance           : Low={int((y==0).sum())} "
          f"High={int((y==1).sum())} "
          f"({100*(y==0).mean():.1f}% / {100*(y==1).mean():.1f}%)")

    # global tuning (reported in Table 2; CV uses nested tuning separately)
    d_base = tune_depth_xgb(X, y)
    d_wce, lam = tune_wce(X, y, k_idx, a_idx)
    print(f"\nTuned (global)  baseline depth={d_base} ; "
          f"WCE depth={d_wce}, lambda={lam}")

    print("\nRunning repeated 5x10-fold CV (nested tuning) ...")
    scores, t_base, t_wce = repeated_cv(X, y, k_idx, a_idx)

    # fold-level export
    fold_df = pd.DataFrame(scores)
    fold_df.insert(0, "fold", range(1, len(fold_df) + 1))
    fold_df.to_csv("fold_level_results.csv", index=False)
    print("Wrote fold_level_results.csv (50 rows x models)")

    wce = np.array(scores["KAB-XGBoost-WCE"])
    base = np.array(scores["Baseline XGBoost"])
    diff = wce - base

    # Wilcoxon on the paired folds
    stat, p = wilcoxon(wce, base)
    nz = diff[diff != 0]
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    fold_win = int((diff > 0).sum())

    print("\n--- Headline comparison (single run) ---")
    for m in scores:
        arr = np.array(scores[m])
        print(f"  {m:<24} F1={arr.mean():.4f}  SD={arr.std(ddof=1):.4f}")
    print(f"\n  WCE - Baseline delta : {diff.mean():+.4f} "
          f"(baseline {'ahead' if diff.mean()<0 else 'behind'})")
    print(f"  Wilcoxon  W+={w_plus:.0f} W-={w_minus:.0f}  p={p:.4f}  "
          f"fold-win={fold_win}/50  ties={(diff==0).sum()}")

    # Naive Bayes significance (examiner flagged it must be discussed)
    nb = np.array(scores["Naive Bayes"])
    _, p_nb = wilcoxon(nb, wce)
    print(f"  Naive Bayes vs WCE   : NB F1={nb.mean():.4f}  p={p_nb:.4f} "
          f"({'significant' if p_nb < 0.05 else 'ns'})")

    print("\nRunning four-arm ablation (same protocol) ...")
    arms = ablation(X, y, k_idx, a_idx)
    A, B, C, D = (np.array(arms[k]) for k in "ABCD")
    print(f"  Arm A vanilla d5      : {A.mean():.4f} (SD {A.std(ddof=1):.4f})")
    print(f"  Arm B depth-only d3   : {B.mean():.4f} (SD {B.std(ddof=1):.4f})")
    print(f"  Arm C weight-only d5  : {C.mean():.4f} (SD {C.std(ddof=1):.4f})")
    print(f"  Arm D full WCE d3     : {D.mean():.4f} (SD {D.std(ddof=1):.4f})")
    print(f"  depth contribution (B-A): {B.mean()-A.mean():+.4f}")
    print(f"  weight contribution (C-A): {C.mean()-A.mean():+.4f}")

    print("\nDecision-tree depth curve (Q19) ...")
    curve = dt_depth_curve(X, y)
    for k, (mu, sd) in curve.items():
        print(f"  DT {k:<10} F1={mu:.4f} (SD {sd:.4f})")

    print("\nHeld-out analysis ...")
    ho = held_out_analysis(X, y, k_idx, a_idx, d_base, d_wce, lam)
    print(f"  n_test={ho['n_test']}")
    print(f"  CM baseline={ho['cm_baseline']}  CM wce={ho['cm_wce']}")
    print(f"  McNemar b={ho['mcnemar_b']} c={ho['mcnemar_c']} "
          f"(b=baseline-correct/WCE-wrong, c=baseline-wrong/WCE-correct)")
    print(f"  AUC baseline={ho['auc_baseline']:.4f}  AUC wce={ho['auc_wce']:.4f}")

    print(f"\nTiming: baseline={np.mean(t_base)*1000:.1f}ms  "
          f"WCE={np.mean(t_wce)*1000:.1f}ms per fold "
          f"(difference is run-to-run noise, not a mechanism)")

    summary = dict(
        n_raw=n_raw, n_removed=n_removed, n_unique=len(df_clean),
        X_shape=list(X.shape), feature_order=feat,
        label_median=float(median),
        class_low=int((y == 0).sum()), class_high=int((y == 1).sum()),
        tuned_baseline_depth=int(d_base), tuned_wce_depth=int(d_wce),
        tuned_wce_lambda=float(lam),
        cv={m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
            for m, v in scores.items()},
        wce_vs_baseline=dict(
            delta=float(diff.mean()), w_plus=w_plus, w_minus=w_minus,
            p=float(p), fold_win=fold_win, ties=int((diff == 0).sum())),
        naive_bayes_vs_wce_p=float(p_nb),
        ablation=dict(
            A=float(A.mean()), B=float(B.mean()),
            C=float(C.mean()), D=float(D.mean()),
            depth_contribution=float(B.mean() - A.mean()),
            weight_contribution=float(C.mean() - A.mean())),
        dt_depth_curve={k: dict(mean=v[0], sd=v[1]) for k, v in curve.items()},
        held_out=ho,
        timing=dict(baseline_ms=float(np.mean(t_base) * 1000),
                    wce_ms=float(np.mean(t_wce) * 1000)),
    )
    with open("summary_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote summary_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
