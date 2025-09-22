import pandas as pd
import numpy as np
from pandasgui import show
from src.manual_label import generate_for_manual_lab
from src.metrics import performance_label_metrics

df = pd.read_csv("companies_lab_encoding.csv")
df = df.rename(columns={
          "label_pred1":"label_predicted_1",
          "label_pred2":"label_predicted_2",
          "label_pred3":"label_predicted_3"
})
generate_for_manual_lab(df.copy()) 

#TODO: manual labelling for the generated csv. Name the column "manual_lab"

df_manual = pd.read_csv("manual_validation_sample.csv")
if "manual_lab" in df_manual.columns:
    df_manual["text"] = (
        df_manual["description"].fillna("") + " " +
        df_manual["business_tags"].fillna("") + " " +
        df_manual["sector"].fillna("") + " " +
        df_manual["category"].fillna("")  + " " +
        df_manual["niche"].fillna("")
    )
    df["text"] = (
        df["description"].fillna("") + " " +
        df["business_tags"].fillna("") + " " +
        df["sector"].fillna("") + " " +
        df["category"].fillna("")  + " " +
        df["niche"].fillna("")
    )
    df_pred = df[df["text"].isin(df_manual["text"])]
    y_pred = df_pred.copy()
    y_true = df_manual.copy()
    y_pred.sort_values(by="sim_val_1", axis=0, ascending=True, inplace=True)
    y_true.sort_values(by="sim_val_1", axis=0, ascending=True, inplace=True)

    y_pred["label_predicted_1"] = y_pred["label_predicted_1"].astype(str).str.strip().str.lower()
    y_true["manual_lab"] = y_true["manual_lab"].astype(str).str.strip().str.lower()

    y_pred_low = y_pred[y_pred["sim_val_1"]<0.3]
    y_true_low = y_true[y_true["sim_val_1"]<0.3]

    y_pred_medium = y_pred[(y_pred["sim_val_1"]>=0.3)&(y_pred["sim_val_1"]<=0.4)]
    y_true_medium = y_true[(y_true["sim_val_1"]>=0.3)&(y_true["sim_val_1"]<=0.4)]

    y_pred_high = y_pred[y_pred["sim_val_1"]>0.4]
    y_true_high = y_true[y_true["sim_val_1"]>0.4]

    metrics_all_sample = performance_label_metrics(y_true["manual_lab"], y_pred["label_predicted_1"], "total")
    metrics_low = performance_label_metrics(y_true_low["manual_lab"], y_pred_low["label_predicted_1"], "under_0.3")
        
    metrics_medium = performance_label_metrics(y_true_medium["manual_lab"], y_pred_medium["label_predicted_1"], "0.3-0.4")

    metrics_high = performance_label_metrics(y_true_high["manual_lab"], y_pred_high["label_predicted_1"], "over_0.4")

    combined_metrics = pd.DataFrame([metrics_all_sample, metrics_low, metrics_medium, metrics_high], index=["total", "under_0.3", "0.3-0.4", "over_0.4"])
    print(combined_metrics)
    combined_metrics.to_csv("data/metrics.csv")

    # process the final dataset
    for k in (2, 3):
        m = pd.to_numeric(df[f"sim_val_{k}"], errors="coerce") < 0.3
        df.loc[m, [f"sim_val_{k}", f"label_predicted_{k}"]] = ""
    df['insurance_label'] = df["label_predicted_1"].fillna("") + ", " + df["label_predicted_2"].fillna("") + ", " + df["label_predicted_3"].fillna("")
    df_final = df[["description", "business_tags", "category", "sector", "niche", "insurance_label", "sim_val_1"]]
    df_final["confidence"] = "50%"
    df_final["sim_val_1"] = df_final["sim_val_1"].replace("", pd.NA).fillna(0.0)
    df_final.loc[df_final["sim_val_1"]>=0.3, "confidence"] = "75%"
    df_final.loc[df_final["sim_val_1"]>=0.4, "confidence"] = "90%"
    df_final.drop(columns=["sim_val_1"], inplace=True)
    df_final['insurance_label'] = df_final['insurance_label'].str.replace(r'[,\s]+$', '', regex=True)
    df_final['insurance_label'] = df_final['insurance_label'].str.lower().str.title()
    df_final.to_csv("company_list_solution.csv")
    show(df_final)
