# Blind validation (n ≈ 600)

Earlier screening metrics (precision 0.993, recall 0.985) compared the LLMs with labels that came mostly from reviewing LLM **disagreements**. Records where all models agreed were never checked by a person. This validation checks all kinds of records, blind, with two reviewers.

## Design

- **Strata:** region bloc (SSA / Maghreb / other) × model agreement (agree / disagree) × verdict (include / exclude)
- **Quotas:** 200 per bloc, so Africa gets equal attention; 20% of each bloc from disagreements
- **Weights:** each record carries `stratum population / stratum sample`, so results generalize to the whole corpus
- **Blind:** reviewers see title, abstract, year, venue, language, DOI only, each in a different order
- **Two questions:** Q1 tick-focused (ticks or tick material studied) and Q2 climate effect (validates the climate flag)
- **Metrics:** Cohen's kappa between reviewers; weighted sensitivity, specificity, precision and F1 per screener with 95% bootstrap CIs, overall and per bloc

## Steps

1. **Draw the sample** (in Colab, Drive mounted):
   ```bash
   pip install pandas pyarrow openpyxl
   python validation/draw_validation_sample.py \
       --input "/content/drive/MyDrive/ticks/preprocessing_data/tickclass2/screened_llm_v3.parquet" \
       --out   "/content/drive/MyDrive/ticks/validation_v1"
   ```
   Check the printed "columns used". If a column is wrong, pass it explicitly (`--final-col llm_label`, `--model-cols gpt_label,deepseek_label,gemini_label`, `--bloc-col _bloc`).
2. **Send** `reviewer_A.xlsx` to reviewer A (you) and `reviewer_B.xlsx` to reviewer B (a partner) with `REVIEWER_GUIDE.md`. Keep `sample_key.csv` private: it holds the verdicts.
3. **Score:**
   ```bash
   python validation/compute_validation.py --dir "/content/drive/MyDrive/ticks/validation_v1"
   ```
   The first run writes `adjudication.csv` with the records where A and B disagree. Fill the `adjudicated` column, then run again.
4. **Report** `validation_metrics.csv` in P0 (data descriptor) and P9 (methods paper).

The sample files hold titles and abstracts, so they stay in Drive and are not committed here (see `.gitignore`).
