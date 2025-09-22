from sentence_transformers import SentenceTransformer, util
import numpy as np

def embed_fallback(company_texts, tfidf_scores, df_taxonomy_clean, thresh = 0.3):
    """
    Embedding the values from companies with tfidf_scores < threshold
    Embedding the taxonomy
    Calculate the cosine similarity between the 2 vect
    return a dictionary {comp_idx: (label, value)  
    """
    comp_idx = np.where(tfidf_scores<thresh)[0]
    company_texts_low = [company_texts[i] for i in comp_idx]

    tax_texts = df_taxonomy_clean["label"].astype(str).to_list()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    tax_emb = model.encode(tax_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
    comp_emb = model.encode(company_texts_low, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)


    assert comp_emb.shape[0] > 0, "Encoded 0 fallback companies (verifică filtrarea înainte de encode)."
    assert tax_emb.shape[0]  > 0, "Taxonomy e goală după preprocesare."
    
    # cosine similarity
    sims = util.cos_sim(comp_emb, tax_emb)
    best_tax_val, best_tax_idx = sims.max(dim=1)

    labels_full = df_taxonomy_clean["label"].to_numpy()
    results = {
    comp_idx[i]: (labels_full[best_tax_idx[i].item()], best_tax_val[i].item())
    for i in range(len(comp_idx))
    }
    return results
