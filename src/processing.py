import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

def company_vectorizing(data):
    """
    For every row combine the features from companies and create string with the most important features.
    Create the vectorizer to learn the vocabulary from the company data

    """
    data["doc"] = (
    # data["description"].fillna("") + " " +
    data["business_tags"].fillna("") + " " +
    data["sector"].fillna("") + " " +
    data["category"].fillna("")  + " " +
    data["niche"].fillna("")
    ).str.strip()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    company_vectorised = vectorizer.fit_transform(data["doc"])
    return vectorizer, company_vectorised

def taxonomy_vectorizing(data, vectorizer):
    """
    Use the vocabulary learned from company data to vectorize the taxonomy data
    """
    taxonomy_vectorised = vectorizer.transform(data)
    return taxonomy_vectorised

def vector_similarity(company_vec, taxonomy_vec, top_k = 3):
    """
    Use cosine similarity to compare the vectors of companies and taxonomy.
    Return top 3 (default) scores and indices of the most similar vectors in taxonomy.
    """
    sims = cosine_similarity(company_vec, taxonomy_vec)
    top_indices = np.argsort(-sims, axis=1)[:, :top_k]
    top_values = np.take_along_axis(sims, top_indices, axis=1)
    return top_indices, top_values

def vector_similarity_ponder(company_vec, taxonomy_field_vec, taxonomy_what_vec, top_k = 3):
    """
    Use cosine similarity to compare the vectors of companies and taxonomy - (split in 2) 'field', 'what'.
    Return top 3 (default) scores and indices of the most similar vectors in taxonomy.
    """
    sims_field = cosine_similarity(company_vec, taxonomy_field_vec)
    sims_what = cosine_similarity(company_vec, taxonomy_what_vec)
    sims = sims_field*0.7 + sims_what*0.3
    top_indices = np.argsort(-sims, axis=1)[:, :top_k]
    top_values = np.take_along_axis(sims, top_indices, axis=1)
    return top_indices, top_values

def save_sim_values_csv(df, out_path="sim_stats.csv", thresh=0.25):
    """
    Designed to be used on cosine similarity values
    Return a DataFrame with the most important data in the series
    """
    rows = []
    for col in df.columns:
        arr = df[col].to_numpy()
        arr = arr[arr > thresh]
        if arr.size > 0:
            rows.append({
                "column": col,
                "count": int(arr.size),
                "mean": np.mean(arr),
                "std": np.std(arr),
                "min": np.min(arr),
                "max": np.max(arr),
            })
        else:
            rows.append({
                "column": col,
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_path, index=False)
    return stats_df



