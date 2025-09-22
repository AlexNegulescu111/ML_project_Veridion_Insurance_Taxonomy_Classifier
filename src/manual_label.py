import pandas as pd
import os

def generate_for_manual_lab(df):
    # ensure similarity scores are numeric (avoid string issues)
    df["sim_val_1"] = pd.to_numeric(df["sim_val_1"], errors="coerce")

    # sort by confidence score, highest first
    df_sorted = df.sort_values("sim_val_1", ascending=False).reset_index(drop=True)

    top30 = df_sorted.head(30)              # 30 best predictions
    bottom20 = df_sorted.tail(20)           # 20 weakest predictions

    # pick 40 random samples from the middle range (40–60%)
    n = len(df_sorted)
    mid_section = df_sorted.iloc[int(0.4*n): int(0.6*n)]
    mid50 = mid_section.sample(50, random_state=42)

    # merge the three groups into one validation set
    sample_df = pd.concat([top30, mid50, bottom20]).reset_index(drop=True)

    # keep only the columns useful for manual labeling
    cols_to_review = ["company_name", "description", "business_tags", "sector", "category", "niche",
                    "label_predicted_1", "sim_val_1"]
    available = [c for c in cols_to_review if c in sample_df.columns]
    out_file = "manual_validation_sample.csv"

    if not os.path.exists(out_file):
        sample_df[available].to_csv("manual_validation_sample.csv", index=False)