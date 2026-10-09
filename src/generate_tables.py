"""
generate_tables.py — Build all manuscript tables from results/summary_results.json
===================================================================================
Produces console output (copy-paste into Word) AND a Word-friendly text file.

Run:  python src/generate_tables.py   (from repo root)

Tables produced:
  Table 2   : Demographics encoding
  Table 3a  : KNUST 5x10 CV results (all 8 models)
  Table 3b  : Alzubaidi 5x10 CV results (all 8 models)
  Table 4   : Ablation (KNUST) — A/B/C/D arms
  Table 5   : Held-out confusion matrices + McNemar (KNUST)
  Table 6a  : SHAP top-10 feature importance (KNUST)
  Table 6b  : SHAP top-10 feature importance (Alzubaidi)
  Table 7   : Subgroup F1 (KNUST)
  Table R10a: Raw vs deduplicated tree (Q19 diagnosis)
  Table S1  : Seed sensitivity (KNUST + Alzubaidi, 5 seeds)
  Table S2  : Threshold sensitivity (3 cuts, KNUST)
  Table S3  : Alzubaidi ablation
"""

import json
import os

RESULTS_DIR = "results"
OUT_FILE    = os.path.join(RESULTS_DIR, "manuscript_tables.txt")


def fmt_p(p):
    """Format p-value for prose — '<0.001' if below threshold, else 4 dp."""
    if isinstance(p, str):
        return p  # already formatted (e.g. '<0.001')
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def rule(n=72):
    return "-" * n


def header(title):
    return f"\n{'=' * 72}\n{title}\n{'=' * 72}"


def main():
    with open(os.path.join(RESULTS_DIR, "summary_results.json")) as f:
        sr = json.load(f)

    with open(os.path.join(RESULTS_DIR, "seed_sensitivity.json")) as f:
        seeds = json.load(f)

    k = sr["knust"]
    a = sr["alzubaidi"]

    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    # ── Table 2: Demographics encoding ───────────────────────────────────────
    p(header("Table M7a / Table 2 — Demographics encoding"))
    p(f"{'Item':<6} {'Description':<30} {'Encoding':<35} {'Range'}")
    p(rule())
    rows = [
        ("Q1", "Gender",                "Binary: Male=1, Female=0",          "0, 1"),
        ("",   "",                       "(Prefer not to say merged with 0)", ""),
        ("Q2", "Year of study",          "Ordinal: Year 2=2, Year 3=3, Year 4=4", "2, 3, 4"),
        ("Q3", "Prior IT course",        "Binary: Yes=1, No=0",               "0, 1"),
        ("Q4", "Years of internet use",  "Ordinal: <2yr=1, 2-5yr=2, >5yr=3", "1, 2, 3"),
        ("Q5", "Digital tool frequency", "Ordinal: Rarely=1, Sometimes=2, Often=3, Very often=4", "1, 2, 3, 4"),
    ]
    for row in rows:
        p(f"{row[0]:<6} {row[1]:<30} {row[2]:<35} {row[3]}")

    # ── Table 3a: KNUST CV ───────────────────────────────────────────────────
    p(header("Table 3a — KNUST 5x10 CV Results (50 folds)"))
    p(f"{'Model':<30} {'Mean F1':>9} {'SD':>8} {'Delta vs WCE':>14} {'W+':>7} {'W-':>7} {'Ties':>5} {'p':>10} {'r':>7}")
    p(rule())
    model_order = [
        "KAB-XGBoost-WCE",
        "Baseline XGBoost",
        "Stock scale_pos_weight",
        "Decision Tree",
        "Naive Bayes",
        "Logistic Regression",
        "SVM",
        "Random Forest",
    ]
    wce_mean_k = k["cv"]["KAB-XGBoost-WCE"]["mean"]
    for m in model_order:
        cv = k["cv"][m]
        mn, sd = cv["mean"], cv["sd"]
        if m == "KAB-XGBoost-WCE":
            wvb = k["wce_vs_baseline"]
            p(f"{'KAB-XGBoost-WCE (proposed)':<30} {mn:>9.4f} {sd:>8.4f} {'—':>14} {'—':>7} {'—':>7} {'—':>5} {'—':>10} {'—':>7}")
        else:
            ct = k["comparator_tests"].get(m, {})
            delta  = ct.get("delta_vs_wce", mn - wce_mean_k)
            wp     = ct.get("w_plus", "—")
            wm     = ct.get("w_minus", "—")
            ties   = ct.get("ties", "—")
            pv     = fmt_p(ct.get("p", "—")) if ct else "—"
            r_val  = ct.get("r", "—")
            p(f"{m:<30} {mn:>9.4f} {sd:>8.4f} {delta:>+14.4f} {str(wp):>7} {str(wm):>7} {str(ties):>5} {pv:>10} {str(r_val):>7}")
    wvb = k["wce_vs_baseline"]
    p(f"\nWCE vs Baseline: delta={wvb['delta']:+.4f}  W+={wvb['w_plus']}  W-={wvb['w_minus']}  ties={wvb['ties']}  p={fmt_p(wvb['p'])}  r={wvb['r']:.4f}")

    # ── Table 3b: Alzubaidi CV ───────────────────────────────────────────────
    p(header("Table 3b — Alzubaidi 5x10 CV Results (50 folds)"))
    p(f"{'Model':<30} {'Mean F1':>9} {'SD':>8} {'Delta vs WCE':>14} {'W+':>7} {'W-':>7} {'Ties':>5} {'p':>10} {'r':>7}")
    p(rule())
    wce_mean_a = a["cv"]["KAB-XGBoost-WCE"]["mean"]
    for m in model_order:
        cv = a["cv"][m]
        mn, sd = cv["mean"], cv["sd"]
        if m == "KAB-XGBoost-WCE":
            p(f"{'KAB-XGBoost-WCE (proposed)':<30} {mn:>9.4f} {sd:>8.4f} {'—':>14} {'—':>7} {'—':>7} {'—':>5} {'—':>10} {'—':>7}")
        else:
            ct = a["comparator_tests"].get(m, {})
            delta  = ct.get("delta_vs_wce", mn - wce_mean_a)
            wp     = ct.get("w_plus", "—")
            wm     = ct.get("w_minus", "—")
            ties   = ct.get("ties", "—")
            pv     = fmt_p(ct.get("p", "—")) if ct else "—"
            r_val  = ct.get("r", "—")
            p(f"{m:<30} {mn:>9.4f} {sd:>8.4f} {delta:>+14.4f} {str(wp):>7} {str(wm):>7} {str(ties):>5} {pv:>10} {str(r_val):>7}")
    wvb_a = a["wce_vs_baseline"]
    p(f"\nWCE vs Baseline: delta={wvb_a['delta']:+.4f}  W+={wvb_a['w_plus']}  W-={wvb_a['w_minus']}  ties={wvb_a['ties']}  p={fmt_p(wvb_a['p'])}  r={wvb_a['r']:.4f}")

    # ── Table 4: Ablation KNUST ──────────────────────────────────────────────
    p(header("Table 4 — Ablation Study, KNUST (50 folds each)"))
    p(f"{'Arm':<6} {'Depth':>6} {'Lambda':>8} {'Weights':>8} {'Mean F1':>9} {'SD':>8}")
    p(rule())
    ka = k["ablation"]
    arm_rows = [
        ("A", 5, "—",   "No",  ka["A"]["mean"], ka["A"]["sd"]),
        ("B", 3, "—",   "No",  ka["B"]["mean"], ka["B"]["sd"]),
        ("C", 5, "1.5", "Yes", ka["C"]["mean"], ka["C"]["sd"]),
        ("D", 3, "1.5", "Yes", ka["D"]["mean"], ka["D"]["sd"]),
    ]
    for row in arm_rows:
        p(f"{row[0]:<6} {row[1]:>6} {row[2]:>8} {row[3]:>8} {row[4]:>9.4f} {row[5]:>8.4f}")
    p(rule())
    p(f"  Depth contribution  (B-A): {ka['depth_contribution']:+.4f}   Wilcoxon B vs A p={fmt_p(ka['p_B_vs_A'])}")
    p(f"  Weight at depth 5   (C-A): {ka['weight_at_depth5']:+.4f}   Wilcoxon C vs A p={fmt_p(ka['p_C_vs_A'])}")
    p(f"  Weight at depth 3   (D-B): {ka['weight_at_depth3']:+.4f}   Wilcoxon D vs B p={fmt_p(ka['p_D_vs_B'])}")

    # ── Table 4 Alzubaidi (supplement) ──────────────────────────────────────
    p(header("Table S3 — Ablation Study, Alzubaidi (50 folds each)"))
    p(f"{'Arm':<6} {'Depth':>6} {'Lambda':>8} {'Weights':>8} {'Mean F1':>9} {'SD':>8}")
    p(rule())
    aa = a["ablation"]
    arm_rows_a = [
        ("A", 5, "—",   "No",  aa["A"]["mean"], aa["A"]["sd"]),
        ("B", 3, "—",   "No",  aa["B"]["mean"], aa["B"]["sd"]),
        ("C", 5, "1.5", "Yes", aa["C"]["mean"], aa["C"]["sd"]),
        ("D", 3, "1.5", "Yes", aa["D"]["mean"], aa["D"]["sd"]),
    ]
    for row in arm_rows_a:
        p(f"{row[0]:<6} {row[1]:>6} {row[2]:>8} {row[3]:>8} {row[4]:>9.4f} {row[5]:>8.4f}")
    p(rule())
    p(f"  Depth contribution  (B-A): {aa['depth_contribution']:+.4f}   Wilcoxon B vs A p={fmt_p(aa['p_B_vs_A'])}")
    p(f"  Weight at depth 5   (C-A): {aa['weight_at_depth5']:+.4f}   Wilcoxon C vs A p={fmt_p(aa['p_C_vs_A'])}")
    p(f"  Weight at depth 3   (D-B): {aa['weight_at_depth3']:+.4f}   Wilcoxon D vs B p={fmt_p(aa['p_D_vs_B'])}")

    # ── Table 5: Held-out KNUST ──────────────────────────────────────────────
    p(header("Table 5 — Held-out 80/20 Split Results, KNUST"))
    ho = k["held_out"]
    cm_b = ho["cm_baseline"]
    cm_w = ho["cm_wce"]
    p(f"  n_test = {ho['n_test']}  (stratified 80/20, SEED=42)")
    p()
    p("  BASELINE XGBoost confusion matrix:")
    p(f"    Predicted Low   Predicted High")
    p(f"  True Low:   {cm_b[0][0]:>4}          {cm_b[0][1]:>4}")
    p(f"  True High:  {cm_b[1][0]:>4}          {cm_b[1][1]:>4}")
    p(f"  Macro F1 = {ho['baseline_heldout_macro_f1']:.4f}   AUC = {ho['auc_baseline']:.4f}")
    p()
    p("  KAB-XGBoost-WCE confusion matrix:")
    p(f"    Predicted Low   Predicted High")
    p(f"  True Low:   {cm_w[0][0]:>4}          {cm_w[0][1]:>4}")
    p(f"  True High:  {cm_w[1][0]:>4}          {cm_w[1][1]:>4}")
    p(f"  Macro F1 = {ho['wce_heldout_macro_f1']:.4f}   AUC = {ho['auc_wce']:.4f}")
    p()
    p(f"  McNemar: b={ho['mcnemar_b']}  c={ho['mcnemar_c']}  chi2={ho['mcnemar_chi2']:.3f}  p={fmt_p(ho['mcnemar_p'])}")
    p(f"  (b = baseline correct & WCE wrong; c = baseline wrong & WCE correct)")
    p()
    p("  Interpretation:")
    p(f"    WCE raised FP from {cm_b[0][1]} to {cm_w[0][1]} (more Low predicted as High).")
    p(f"    WCE did not reduce FN: {cm_b[1][0]} vs {cm_w[1][0]}.")
    p(f"    Baseline F1 ({ho['baseline_heldout_macro_f1']:.4f}) > WCE F1 ({ho['wce_heldout_macro_f1']:.4f}) on held-out split.")
    p(f"    McNemar p={fmt_p(ho['mcnemar_p'])} — no significant disagreement.")

    # ── Table 5b: Held-out Alzubaidi ─────────────────────────────────────────
    p(header("Table 5b — Held-out 80/20 Split Results, Alzubaidi"))
    ho_a = a["held_out"]
    cm_ba = ho_a["cm_baseline"]
    cm_wa = ho_a["cm_wce"]
    p(f"  n_test = {ho_a['n_test']}")
    p(f"  Baseline F1={ho_a['baseline_heldout_macro_f1']:.4f}  WCE F1={ho_a['wce_heldout_macro_f1']:.4f}")
    p(f"  Baseline AUC={ho_a['auc_baseline']:.4f}  WCE AUC={ho_a['auc_wce']:.4f}")
    p(f"  McNemar b={ho_a['mcnemar_b']} c={ho_a['mcnemar_c']} chi2={ho_a['mcnemar_chi2']:.3f} p={fmt_p(ho_a['mcnemar_p'])}")

    # ── Table 6a: SHAP KNUST ─────────────────────────────────────────────────
    p(header("Table 6a — SHAP Feature Importance, KNUST (top 10)"))
    p(f"{'Rank':<6} {'Feature':<20} {'Mean |SHAP|':>12}")
    p(rule(40))
    shap_k_data = sr["knust"].get("shap", [])
    for i, row in enumerate(shap_k_data[:10], 1):
        p(f"{i:<6} {row['feature']:<20} {row['mean_abs_shap']:>12.4f}")

    # ── Table 6b: SHAP Alzubaidi ─────────────────────────────────────────────
    p(header("Table 6b — SHAP Feature Importance, Alzubaidi (top 10)"))
    p(f"{'Rank':<6} {'Feature':<20} {'Mean |SHAP|':>12}")
    p(rule(40))
    shap_a_data = sr["alzubaidi"].get("shap", [])
    for i, row in enumerate(shap_a_data[:10], 1):
        p(f"{i:<6} {row['feature']:<20} {row['mean_abs_shap']:>12.4f}")

    # ── Table 7: Subgroups KNUST ─────────────────────────────────────────────
    p(header("Table 7 — Subgroup F1, KNUST (WCE vs Baseline, OOF 5x10)"))
    subg = k.get("subgroups", {})
    if subg:
        for grp_name, grp_vals in subg.items():
            p(f"\n  {grp_name}:")
            p(f"  {'Group':<15} {'n':>6} {'WCE F1':>9} {'Base F1':>9}")
            p(f"  {rule(42)}")
            for label, vals in grp_vals.items():
                p(f"  {label:<15} {vals['n']:>6} {vals['wce_f1']:>9.4f} {vals['base_f1']:>9.4f}")
    else:
        p("  [subgroups not available]")

    # ── Table R10a: Raw vs deduplicated ──────────────────────────────────────
    p(header("Table R10a — Raw vs Deduplicated Tree Comparison (Q19 diagnosis)"))
    rd = k.get("raw_vs_dedup_tree", {})
    p(f"{'Dataset':<20} {'DT Mean':>9} {'DT SD':>8} {'RF Mean':>9} {'RF SD':>8} {'Mean test twins':>16}")
    p(rule())
    for label, vals in rd.items():
        p(f"{label:<20} {vals['dt_mean']:>9.4f} {vals['dt_sd']:>8.4f} {vals['rf_mean']:>9.4f} {vals['rf_sd']:>8.4f} {vals['mean_test_rows_with_train_twin']:>16.1f}")

    # ── Table S1: Seed sensitivity ────────────────────────────────────────────
    p(header("Table S1 — Seed Sensitivity (5 seeds, KNUST + Alzubaidi)"))
    p(f"{'Seed':<6} {'KNUST WCE':>11} {'KNUST Base':>12} {'KNUST Δ':>10} {'KNUST p':>10} {'ALZ WCE':>9} {'ALZ Base':>10} {'ALZ Δ':>8} {'ALZ p':>10}")
    p(rule())
    for s_key in sorted(seeds.keys(), key=int):
        sv = seeds[s_key]
        # Handle both old (flat) and new (nested) seed sensitivity format
        if "knust" in sv:
            kv = sv["knust"]
            av = sv["alzubaidi"]
        else:
            kv = sv
            av = {"wce_mean": "—", "base_mean": "—", "delta": "—", "p": "—"}
        p(f"{s_key:<6} {str(kv.get('wce_mean','—')):>11} {str(kv.get('base_mean','—')):>12} "
          f"{str(kv.get('delta','—')):>10} {fmt_p(kv.get('p','—')) if kv.get('p') not in ('—',None) else '—':>10} "
          f"{str(av.get('wce_mean','—')):>9} {str(av.get('base_mean','—')):>10} "
          f"{str(av.get('delta','—')):>8} {fmt_p(av.get('p','—')) if av.get('p') not in ('—',None) else '—':>10}")

    # ── Table S2: Threshold sensitivity ─────────────────────────────────────
    p(header("Table S2 — Threshold Sensitivity, KNUST (5x10 folds)"))
    p(f"{'Cut':<16} {'Cutpoint':>10} {'Low frac':>10} {'Base F1':>9} {'WCE F1':>9} {'p':>10}")
    p(rule())
    thr = k.get("threshold_sensitivity", {})
    for tname, tv in thr.items():
        p(f"{tname:<16} {tv['cut']:>10.1f} {tv['low_frac']:>10.3f} {tv['baseline_f1']:>9.4f} {tv['wce_f1']:>9.4f} {fmt_p(tv['wilcoxon_p']):>10}")

    # ── Key numbers for prose ─────────────────────────────────────────────────
    p(header("KEY NUMBERS FOR PROSE (Methods and Results)"))
    p("\n--- DATASET ---")
    p(f"KNUST: raw={k['n_raw']}, removed={k['n_removed']}, unique={k['n_unique']}")
    p(f"  class Low={k['class_low']} ({100*k['class_low']/k['n_unique']:.1f}%), High={k['class_high']} ({100*k['class_high']/k['n_unique']:.1f}%)")
    p(f"  avg tuned depth: baseline={k['avg_fold_baseline_depth']}, WCE={k['avg_fold_wce_depth']}, avg lambda={k['avg_fold_wce_lambda']}")
    p(f"  straight_line removed={k['straight_line_removed']}")
    p(f"  Cronbach: K={k['cronbach']['knowledge']}, A={k['cronbach']['attitude']}, B={k['cronbach']['behaviour']}")

    p(f"\nAlzubaidi: raw={a['n_raw']}, removed={a['n_removed']}, unique={a['n_unique']}")
    p(f"  class Low={a['class_low']} ({100*a['class_low']/a['n_unique']:.1f}%), High={a['class_high']} ({100*a['class_high']/a['n_unique']:.1f}%)")
    p(f"  avg tuned depth: baseline={a['avg_fold_baseline_depth']}, WCE={a['avg_fold_wce_depth']}, avg lambda={a['avg_fold_wce_lambda']}")

    p("\n--- TIMING (depth-matched) ---")
    p(f"KNUST: baseline={k['timing']['baseline_matched_ms']} ms, WCE={k['timing']['wce_matched_ms']} ms")
    p(f"Alzubaidi: baseline={a['timing']['baseline_matched_ms']} ms, WCE={a['timing']['wce_matched_ms']} ms")

    p("\n--- ROBUSTNESS ---")
    rob_k = k["robustness"]
    p(f"KNUST ordinal: base={rob_k['ordinal_base']}, mean={rob_k['ordinal_mean']} (SD={rob_k['ordinal_sd']}), change={rob_k['ordinal_change']}")
    p(f"  bootstrap 95% CI: [{rob_k['bootstrap_lo']}, {rob_k['bootstrap_hi']}]")
    rob_a = a["robustness"]
    p(f"Alzubaidi ordinal: base={rob_a['ordinal_base']}, mean={rob_a['ordinal_mean']} (SD={rob_a['ordinal_sd']}), change={rob_a['ordinal_change']}")
    p(f"  bootstrap 95% CI: [{rob_a['bootstrap_lo']}, {rob_a['bootstrap_hi']}]")

    p("\n--- SKEW TEST ---")
    st_k = k["skew_test"]
    p(f"KNUST (n={st_k['n']}): baseline={st_k['baseline']}, WCE={st_k['wce']} (gain {st_k['wce_gain']:+.4f}), spw={st_k['scale_pos_weight']} (gain {st_k['spw_gain']:+.4f})")
    p(f"  WCE vs spw p={fmt_p(st_k['wilcoxon_wce_vs_spw_p'])}, WCE vs baseline p={fmt_p(st_k['wilcoxon_wce_vs_base_p'])}")
    st_a = a["skew_test"]
    p(f"Alzubaidi (n={st_a['n']}): baseline={st_a['baseline']}, WCE={st_a['wce']} (gain {st_a['wce_gain']:+.4f}), spw={st_a['scale_pos_weight']} (gain {st_a['spw_gain']:+.4f})")
    p(f"  WCE vs spw p={fmt_p(st_a['wilcoxon_wce_vs_spw_p'])}, WCE vs baseline p={fmt_p(st_a['wilcoxon_wce_vs_base_p'])}")

    # ── Write to file ─────────────────────────────────────────────────────────
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\n\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
