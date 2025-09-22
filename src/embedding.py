from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

def embed_fallback(company_texts, tfidf_scores, df_taxonomy_clean, thresh = 0.3):
    """
    For companies with TF-IDF < `thresh`, find the closest taxonomy label using SBERT.
    Returns a dict: {company_idx: (label, cosine_similarity)}
    Minimal, safe changes applied to prevent shape/empty crashes.
    """
    # --- 1) Ensure numeric scores and select indices under threshold ---
    # Convert to numeric; invalid values become NaN
    scores = pd.to_numeric(pd.Series(tfidf_scores), errors="coerce").to_numpy()
    # Treat NaN as +inf so they never qualify as < thresh
    comp_idx = np.where(np.nan_to_num(scores, nan=np.inf) < thresh)[0]
    assert len(company_texts) == len(scores), "texts vs scores length mismatch"

    # Collect the corresponding company texts for the selected indices
    # (No additional filtering here, as you said there are no empty rows)
    company_texts_low = [str(company_texts[i]) for i in comp_idx]

    # --- 2) Prepare taxonomy texts (trim just to be safe) ---
    tax_texts = df_taxonomy_clean["label"].astype(str).str.strip().to_list()

    # --- 3) Encode with the same model and normalized embeddings ---
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tax_emb = model.encode(tax_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
    comp_emb = model.encode(company_texts_low, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)

    # Sanity checks: if these fail, inputs are effectively empty
    assert comp_emb.shape[0] > 0, "Encoded 0 fallback companies (check the score filter and text extraction)."
    assert tax_emb.shape[0]  > 0, "Taxonomy is empty after preprocessing."

    # --- 4) Cosine similarity (companies x taxonomy) and top-1 per company ---
    # IMPORTANT: order is (companies, taxonomy) -> shape (C x T)
    sims = util.cos_sim(comp_emb, tax_emb)
    best_tax_val, best_tax_idx = sims.max(dim=1)  # top taxonomy for each company

    # --- 5) Build the result mapping back to original company indices ---
    labels_full = df_taxonomy_clean["label"].to_numpy()
    results = {
        int(comp_idx[i]): (labels_full[int(best_tax_idx[i].item())], float(best_tax_val[i].item()))
        for i in range(len(comp_idx))
    }
    return results
