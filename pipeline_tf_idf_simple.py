# === pipeline.py ===
from src.utils import load_data
from src.utils import save_model
from src.cleaning import clean_data
from src.processing import company_vectorizing, taxonomy_vectorizing, vector_similarity, save_sim_values_csv
import pandas as pd


# Loading data
df_companies = load_data("ml_insurance_challenge")
df_taxonomy = load_data("insurance_taxonomy")

# Cleaning data
df_companies = df_companies.drop_duplicates()
df_companies_clean = clean_data(df_companies.copy())
df_taxonomy_clean = clean_data(df_taxonomy.copy())


### TF-IDF Vectorizing

# Learn vocabulary from company data and create the vector
vectorizer, company_vec = company_vectorizing(df_companies_clean)

taxonomy_vec = taxonomy_vectorizing(df_taxonomy_clean["label"], vectorizer)

# Check vectors similarity and generate the matrix with the top n (default 3) similarity scores for each company and a matrix with their positions

best_index, best_values = vector_similarity(company_vec, taxonomy_vec)

df_best_values = pd.DataFrame(best_values)
df_best_index = pd.DataFrame(best_index)

labels = df_taxonomy["label"].to_numpy()
for i in range(len(df_best_values.columns)):
    df_companies[f"label_pred{i+1}"] = labels[df_best_index[i].to_numpy()]
    df_companies[f"sim_val_{i+1}"] = df_best_values[i].to_numpy()

stats = save_sim_values_csv(df_best_values, out_path="./data/simple_values.csv")

df_companies.to_csv("companies_lab_simple.csv")
