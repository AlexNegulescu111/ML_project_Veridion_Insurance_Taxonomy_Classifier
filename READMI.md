# Veridion Insurance Taxonomy Classifier

A lightweight, reproducible pipeline that assigns one or more insurance taxonomy labels to companies using TF‑IDF + cosine similarity, with an optional SBERT fallback for low‑confidence cases. The project also includes a simple manual validation workflow and metric reporting.

---

## 1) Quick start

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or see minimal deps below

# Place input CSV files under ./data/
#  - data/ml_insurance_challenge.csv
#  - data/insurance_taxonomy.csv

# Run the TF‑IDF baseline
python pipeline_tf_idf_simple.py

# (Optional) Alternative pipelines / ablations
python pipeline_split_tax.py       # split taxonomy into field/what
python pipeline_split_lemma.py     # lemmatized variant

# Unify, evaluate, and export final annotated list
python pipeline_unified.py
```

**Outputs (default paths):**

- `companies_lab_simple.csv` / `companies_lab_split.csv` / `companies_lab_split_lemma.csv` — raw top‑k predictions per approach.
- `manual_validation_sample.csv` — sample to be labeled by hand (column `manual_lab`).
- `data/metrics.csv` — aggregate metrics (overall and by confidence bands).
- `company_list_solution.csv` — final deliverable with `insurance_label` and `confidence`.

---

## 2) What the pipelines do

**Baseline (TF‑IDF simple)**

- Concatenate key company fields (tags, sector, category, niche) → TF‑IDF.
- Vectorize taxonomy labels → TF‑IDF.
- Compute cosine similarities and take top‑k labels per company.

**Split‑Tax / Lemma (optional)**

- Experimental variants that split labels into *field* vs *what*, and/or apply lemmatization.
- Keep results for analysis/ablations; the baseline tends to perform best.

**Unified step**

- Prepares a manual validation sample.
- Computes metrics (overall and for similarity bands `<0.3`, `0.3–0.4`, `>0.4`).
- Creates `insurance_label` by joining non‑empty top‑k labels (after dropping those with low similarity).
- Maps `confidence` from `sim_val_1` to `50%` / `75%` / `90%`.

---

## 3) Manual validation

1. Run `pipeline_unified.py` once — this generates `manual_validation_sample.csv`.
2. Open that CSV and fill in the `manual_lab` column.
3. Run `pipeline_unified.py` again to compute metrics and export the final list.

`data/metrics.csv` will include weighted accuracy/precision/recall/F1 across all samples and per similarity band.

---

## 4) Final deliverable format

`company_list_solution.csv` contains (at minimum):

- `description`, `business_tags`, `category`, `sector`, `niche`
- `insurance_label` — one or more taxonomy labels (comma‑separated) or `Unknown` if none pass the threshold
- `confidence` — derived from `sim_val_1`:
  - `< 0.3` → `50%`
  - `0.3 – 0.4` → `75%`
  - `> 0.4` → `90%`

---

## 5) Configuration & thresholds

- Default similarity threshold for dropping labels from the final list: `0.3`.
- Confidence bands are based on `sim_val_1` (top‑1 similarity).
- You can change thresholds directly in `pipeline_unified.py`.

---

## 6) Minimal dependencies

If you don’t use the lemmatized pipeline or plots, only the following are required:

```
pandas
numpy
scikit-learn
sentence-transformers   # only if you enable embedding fallback
```

Optional:

```
matplotlib, seaborn     # plots
spacy, en_core_web_sm   # for the lemma pipeline
pandasgui               # interactive inspection (optional)
```

---

## 7) Repository structure (indicative)

```
./data/
  ml_insurance_challenge.csv
  insurance_taxonomy.csv

pipeline_tf_idf_simple.py
pipeline_split_tax.py
pipeline_split_lemma.py
pipeline_unified.py
src/
  cleaning.py
  processing.py
  embedding.py
  metrics.py
  manual_label.py
  utils.py
  views.py
```

---

## 8) Notes & limitations

- Taxonomy coverage and class imbalance can impact per‑class F1.
- For reproducibility and speed, consider caching embeddings if you enable the SBERT fallback.

