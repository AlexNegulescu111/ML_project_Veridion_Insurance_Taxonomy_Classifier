import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .cleaning import remove_punctuation

def view_sim_distribution(data, comment, ROOT, col_nr = 0):
    """
    Generate a plot with the distribution of the data (continuous data type) in 10 bins
    """
    plt.figure(figsize=(8, 4), dpi=100)
    sns.histplot(y=data.iloc[:, col_nr].dropna().astype(float), bins=np.linspace(0.0, 1.0, 11), stat="percent")
    plt.title("Distribution of vector similarity" + f"{comment}")
    plt.ylabel("Similarity values of labels")
    plt.xlabel("Frequency(%)")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    title = remove_punctuation(comment)
    plt.savefig(str(ROOT) + f"/plots/similarity_{"_".join(title.strip().split())}.png")
    plt.show()
    