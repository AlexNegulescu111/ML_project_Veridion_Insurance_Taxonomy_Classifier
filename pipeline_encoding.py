import pandas as pd
import numpy as np
from src.utils import load_data
from src.cleaning import clean_data
from src.embedding import embed_fallback
from src.processing import save_sim_values_csv
from pandasgui import show
simple_stats = load_data("simple_values")
split_tax_stats = load_data("split_tax_values")
split_lemma_stats = load_data("split_lemma_values")

print(simple_stats)
print(split_tax_stats)
print(split_lemma_stats)

#--------> by simple observation we see that simple tf-idf vectorization gets the highest scores
### load data

df_comp = pd.read_csv("companies_lab_simple.csv")
df_tax = load_data("insurance_taxonomy")

df_comp_clean = clean_data(df_comp.copy())
df_tax_clean = clean_data(df_tax.copy())

df_comp_clean["row_id"] = np.arange(len(df_comp_clean))########################################

df_comp_clean["text"] = (
    df_comp_clean["description"].fillna("") + " " +
    df_comp_clean["business_tags"].fillna("") + " " +
    df_comp_clean["sector"].fillna("") + " " +
    df_comp_clean["category"].fillna("")  + " " +
    df_comp_clean["niche"].fillna("")
    )
companies_text = df_comp_clean["text"].str.strip().to_list()
########################################################################################
tfidf = df_comp_clean["sim_val_1"].copy()
tfidf_top1 = tfidf.to_numpy()

fallback = embed_fallback(companies_text, tfidf_top1, df_tax_clean, thresh=0.3)

# building arrays from initioal 
pred_labels = df_comp["label_pred1"].to_numpy().copy()
pred_scores = df_comp["sim_val_1"].to_numpy().copy()
pred_method = np.full(len(df_comp), "tfidf", dtype=object)


# putting the labels in place
for idx, (label, score) in fallback.items():
    pred_labels[idx] = label
    pred_scores[idx] = score
    pred_method[idx] = "embed"

df_comp["label_pred1"] = pred_labels
df_comp["sim_val_1"] = pred_scores
df_comp["method"] = pred_method

stats = save_sim_values_csv(pd.DataFrame(df_comp["sim_val_1"]), out_path="./data/encoding_values.csv")

df_comp.to_csv("companies_lab_encoding.csv")