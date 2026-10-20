# kab-xgboost-wce-cybersecurity

A cost-sensitive XGBoost variant (KAB-XGBoost-WCE) for predicting cybersecurity behaviour readiness, grounded in the Knowledge-Attitude-Behaviour model. Includes SHAP explainability on KNUST SCM student field data and a public dataset (Alzubaidi 2021).

## Key finding

WCE weighting produces no measurable advantage over the baseline on either dataset. The null result holds across five random seeds (0, 7, 13, 21, 42) on both datasets.

## Reproducibility

Run from repo root:

```bash
pip install -r requirements.txt
python src/kab_pipeline.py
```

Outputs are written to `results/`. See `results/run_log.txt` for the Python/OS/CPU environment and `results/summary_results.json` for all headline numbers.

### Environment (primary run)

Recorded in `results/run_log.txt`:

| Item | Value |
|---|---|
| Python | 3.13.16 (main, Oct 1 2026) [GCC 13.3.0] |
| OS | Linux 6.18.44-fc-v114 x86_64 (glibc 2.39) |
| CPU | x86_64, 2 logical cores |
| XGBoost | 3.4.1 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.5 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| shap | 0.52.0 |
| openpyxl | 3.1.5 |
| matplotlib | 3.10.3 |
| Primary seed | 42 |
| Run timestamp | 2026-10-09T23:11:49Z |

## Data

- `data/KNUST_Cybersecurity_Survey_Responses_2026.xlsx` — KNUST Supply Chain Management student survey (2,088 unique responses after deduplication; 3,200 sampling frame)
- `data/Alzubaidi_2021_Cybercrime_Awareness_Dataset.xlsx` — public dataset (1,187 unique responses after deduplication; doi:10.17632/fbs9mgmh4y.3)

## Results files

| File | Description |
|---|---|
| `results/fold_level_results_knust.csv` | 50 per-fold macro-F1 for 8 models, KNUST (seed=42) |
| `results/fold_level_results_alzubaidi.csv` | 50 per-fold macro-F1 for 8 models, Alzubaidi (seed=42) |
| `results/fold_depth_lambda_knust.csv` | 50 per-fold tuned depth and lambda, KNUST |
| `results/fold_depth_lambda_alzubaidi.csv` | 50 per-fold tuned depth and lambda, Alzubaidi |
| `results/fold_fit_times_knust.csv` | 50 per-fold fit times (ms), KNUST |
| `results/fold_fit_times_alzubaidi.csv` | 50 per-fold fit times (ms), Alzubaidi |
| `results/summary_results.json` | Every headline number from a single pipeline run |
| `results/shap_importance.json` | SHAP mean absolute value rankings, both datasets |
| `results/seed_sensitivity.json` | WCE vs Baseline Wilcoxon and per-seed ablation deltas across seeds 0, 7, 13, 21, 42 |
| `results/run_log.txt` | Python, OS, CPU, package versions and run timestamp |
| `results/figures/` | Pipeline-generated figures (SHAP, confusion, stability boxplots) |

## Documents

- `docs/Ofori_Prince_Phase1_Methods_R13.docx` — Methods chapter (R13)
- `docs/Ofori_Prince_Phase1_Results_R13.docx` — Results chapter (R13)

## Source

- `src/kab_pipeline.py` — complete reproducible pipeline (KNUST + Alzubaidi)
- `src/generate_tables.py` — table formatter used for manuscript_tables.txt
- `src/KAB_Option2_FourLegs.py` — exploratory four-leg design (not used in final analysis)

## Ethics

KNUST data collected under HuSSREC clearance HuSSREC/AP/639/VOL.5 (28 September 2026 – 28 September 2027). The dataset is restricted and not shared in this repository.
