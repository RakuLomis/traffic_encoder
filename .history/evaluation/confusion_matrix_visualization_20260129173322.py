import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix_percentage(
    csv_path: str,
    output_pdf_path: str,
    cmap: str = "Blues",
    figsize=(5.5, 4.8),
    tick_label_size: int = 11,
    annot_font_size: int = 10,
    show_colorbar: bool = True, 
    if_save: bool = False
):
    """
    Plot a row-normalized confusion matrix (percentage-based heatmap),
    with labels sorted by row sample count (descending).

    Parameters
    ----------
    csv_path : str
        Path to confusion matrix CSV file.
    output_pdf_path : str
        Output path for the PDF figure.
    cmap : str
        Colormap for heatmap.
    figsize : tuple
        Figure size (smaller -> denser blocks).
    tick_label_size : int
        Font size for x/y tick labels.
    annot_font_size : int
        Font size for percentage annotations.
    show_colorbar : bool
        Whether to show the colorbar.
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
    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        cm_percentage,
        cmap=cmap,
        vmin=0,
        vmax=100,
        square=True,
        annot=True,
        fmt=".1f",
        annot_kws={"size": annot_font_size},
        linewidths=0.4,
        cbar=show_colorbar,
        cbar_kws={"label": "Percentage (%)"} if show_colorbar else None
    )

    # 去掉标题（你明确要求）
    ax.set_title("")

    # 坐标轴标签
    ax.set_xlabel("Predicted label", fontsize=tick_label_size)
    ax.set_ylabel("True label", fontsize=tick_label_size)

    # 坐标刻度字体大小
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    plt.show()
    if if_save:
        plt.savefig(output_pdf_path, format="pdf")
    plt.close()