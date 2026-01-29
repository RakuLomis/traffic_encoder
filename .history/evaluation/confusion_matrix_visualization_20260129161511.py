import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix_percentage(
    csv_path: str,
    output_pdf_path: str,
    cmap: str = "Blues",
    figsize=(8, 6),
    font_scale: float = 0.9,
    if_save: bool = False
):
    """
    Plot a row-normalized confusion matrix (percentage-based heatmap),
    with labels sorted by row sample count (descending).

    Parameters
    ----------
    csv_path : str
        Path to confusion matrix CSV file.
        Rows = true labels, columns = predicted labels.
    output_pdf_path : str
        Output path for the PDF figure.
    cmap : str
        Colormap for heatmap (default: 'Blues').
    figsize : tuple
        Figure size.
    font_scale : float
        Seaborn font scale.
    """

    # ----------------------------
    # 1. Load confusion matrix
    # ----------------------------
    cm = pd.read_csv(csv_path, index_col=0)

    # ----------------------------
    # 2. Sort labels by row sum
    # ----------------------------
    row_sums = cm.sum(axis=1)
    sorted_labels = row_sums.sort_values(ascending=False).index
    cm_sorted = cm.loc[sorted_labels, sorted_labels]

    # ----------------------------
    # 3. Row-wise normalization (%)
    # ----------------------------
    cm_percentage = cm_sorted.div(cm_sorted.sum(axis=1), axis=0) * 100

    # ----------------------------
    # 4. Plot
    # ----------------------------
    sns.set(style="white")
    sns.set_context("paper", font_scale=font_scale)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm_percentage,
        cmap=cmap,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Percentage (%)"}
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Row-normalized Confusion Matrix (%)")

    plt.tight_layout()

    # ----------------------------
    # 5. Save as PDF
    # ----------------------------
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    plt.show()
    if if_save: 
        plt.savefig(output_pdf_path, format="pdf")
    plt.close()
