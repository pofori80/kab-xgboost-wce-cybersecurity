# kab-xgboost-wce-cybersecurity

A cost-sensitive XGBoost variant (KAB-XGBoost-WCE) for predicting cybersecurity behaviour readiness, grounded in the Knowledge-Attitude-Behaviour model. Includes SHAP explainability on KNUST SCM student field data and a public dataset (Alzubaidi 2021).

## Key finding

WCE weighting produces no measurable advantage over the baseline on either dataset. The null result holds across five random seeds on KNUST and on the Alzubaidi validation set.

## Reproducibility

Run from repo root:

```bash
pip install -r requirements.txt
python src/kab_pipeline.py
```

Outputs are written to `results/`. See `results/pipeline_run_log.txt` for the console log and `results/summary_results.json` for all headline numbers.

## Data

- `data/KNUST_Cybersecurity_Survey_Responses_2026.xlsx` — KNUST Supply Chain Management student survey (2088 unique responses after deduplication)
- `data/Alzubaidi_2021_Cybercrime_Awareness_Dataset.xlsx` — public dataset (1187 unique responses after deduplication)

## Documents

- `docs/Ofori_Prince_Phase1_Methods_R11.docx` — Methods chapter (R11)
- `docs/Ofori_Prince_Phase1_Results_R11.docx` — Results chapter (R11)
