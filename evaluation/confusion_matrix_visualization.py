import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

def plot_confusion_matrix_percentage(
    csv_path: str,
    output_pdf_path: str,
    cmap: str = "Blues",
    figsize=(5.5, 4.8),
    tick_label_size: int = 14,
    annot_font_size: int = 14,
    show_colorbar: bool = True, 
    cbar_label_size: int = 14,
    cbar_outline: bool = False,
    cbar_outline_width: float = 0.8, 
    if_save: bool = False,
    label_order: list = None  # 鏂板鍙傛暟
):
    """
    Plot a row-normalized confusion matrix (percentage-based heatmap).

    Parameters
    ----------
    csv_path : str
        Path to confusion matrix CSV file.
    output_pdf_path : str
        Output path for the PDF figure.
    cmap : str
        Colormap for heatmap.
    figsize : tuple
        Figure size.
    tick_label_size : int
        Font size for x/y tick labels.
    annot_font_size : int
        Font size for percentage annotations.
    show_colorbar : bool
        Whether to show the colorbar.
    if_save : bool
        Whether to save the figure.
    label_order : list, optional
        Custom order of labels. If provided, sorts both rows and columns by this order.
        If None (default), sorts by row sample count (descending).
    """

    sns.set(style="white")
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False
    })

    # 1. Load confusion matrix
    cm = pd.read_csv(csv_path, index_col=0)

    # 2. Sort labels
    if label_order is not None:
        # 楠岃瘉鎵€鏈夋寚瀹氱殑鏍囩閮藉瓨鍦ㄤ簬鏁版嵁涓?
        missing_labels = set(label_order) - set(cm.index)
        if missing_labels:
            raise ValueError(f"Specified labels not found in CSV: {missing_labels}")
        
        # 淇濈暀鍘熸暟鎹腑鏈夌殑鏍囩锛屾寜鐓?label_order 鐨勯『搴忔帓鍒?
        available_labels = [label for label in label_order if label in cm.index]
        
        # 濡傛灉杩樻湁鏈湪 label_order 涓寚瀹氱殑鏍囩锛屾坊鍔犲埌鏈熬锛堟垨鎶涘嚭璀﹀憡锛?
        remaining_labels = [label for label in cm.index if label not in label_order]
        if remaining_labels:
            print(f"Warning: The following labels are not in label_order and will be appended: {remaining_labels}")
            available_labels.extend(remaining_labels)
        
        sorted_labels = available_labels
    else:
        # 榛樿锛氭寜琛屾牱鏈暟闄嶅簭鎺掑垪
        row_sums = cm.sum(axis=1)
        sorted_labels = row_sums.sort_values(ascending=False).index

    # 纭繚鍒椾篃鎸夌収鐩稿悓鐨勯『搴忔帓鍒楋紙娣锋穯鐭╅樀搴旇鏄柟闃碉級
    cm_sorted = cm.loc[sorted_labels, sorted_labels]

    # 3. Row-wise normalization (%)
    cm_percentage = cm_sorted.div(cm_sorted.sum(axis=1), axis=0) * 100

    # 4. Plot
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
        # cbar_kws={"label": "Percentage (%)"} if show_colorbar else None
        cbar_kws={}
    )

        # --- Colorbar fine control ---
    if show_colorbar:
        cbar = ax.collections[0].colorbar

        # 1. Colorbar label font size
        cbar.set_label("Percentage (%)", fontsize=cbar_label_size)

        # 2. Tick label size
        cbar.ax.tick_params(labelsize=cbar_label_size - 1)

        # 3. Outline (border)
        if cbar_outline:
            cbar.outline.set_visible(True)
            cbar.outline.set_linewidth(cbar_outline_width)
            
        else:
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(axis='y', length=0)

    # ax.set_title("")
    # ax.set_xlabel("Predicted label", fontsize=tick_label_size)
    # ax.set_ylabel("True label", fontsize=tick_label_size)
    # ax.tick_params(axis="x", labelsize=tick_label_size)
    # ax.tick_params(axis="y", labelsize=tick_label_size)

    # 1. Remove axis titles (redundant in multi-panel figures)
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    ax.set_xlabel("Predicted label", fontsize=tick_label_size+1)
    ax.set_ylabel("True label", fontsize=tick_label_size+1)
    
    # 2. Tick label font size
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    
    # 3. Rotate tick labels for better readability
    plt.setp(
        ax.get_xticklabels(),
        rotation=20,
        ha="center",
        rotation_mode="anchor"
    )
    
    plt.setp(
        ax.get_yticklabels(),
        rotation=0,
        ha="right"
    )

    # --- Add outer border ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

    plt.tight_layout()
    
    if output_pdf_path:
        dir_name = os.path.dirname(output_pdf_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
    
    if if_save:
        plt.savefig(output_pdf_path, format="pdf", bbox_inches='tight')
        print("PDF is saved.")
    
    plt.show()
    plt.close()

def plot_percentage_colorbar_horizontal(
    output_pdf_path: str,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 100.0,
    figsize=(12, 0.3),
    label: str = "Percentage (%)",
    label_font_size: int = 16,
    tick_font_size: int = 14, 
    if_save: bool = False
):
    """
    Plot a standalone horizontal colorbar for percentage-based heatmaps.
    """

    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False
    })

    fig, ax = plt.subplots(figsize=figsize)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label(label, fontsize=label_font_size)
    cbar.ax.tick_params(labelsize=tick_font_size)

    # 鍒犻櫎 colorbar 鐨勮竟妗嗭紙outline锛?
    cbar.outline.set_visible(False)

    # 鍙€夛細鍚屾椂鍒犻櫎 tick 鐨勫埢搴︾嚎锛堝彧淇濈暀鏁板瓧鏍囩锛屾洿绠€娲侊級
    cbar.ax.tick_params(axis='x', length=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    if if_save: 
        plt.savefig(output_pdf_path, format="pdf",  bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()

def plot_confusion_matrix_panel_gamma(
    cm_list,
    titles,
    label_order,
    figsize=(16, 4),
    annot=True,
    annot_fontsize=11,
    tick_fontsize=11,
    title_fontsize=13,
    gamma=0.5,                 # 闈炵嚎鎬у己搴?
    vmin=0,
    vmax=100
):
    # ---- Font ----
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False
    })

    # ---- 鍗曡壊绉戠爺椋庢笎鍙?----
    color_points = [
        "#F7F9FA",   # near white
        "#D6E4EA",   # very light blue
        "#7FB3D5",   # medium blue
        "#154360"    # deep blue
    ]

    cmap = LinearSegmentedColormap.from_list(
        "academic_blue",
        color_points,
        N=256
    )

    # ---- 闈炵嚎鎬у綊涓€鍖?----
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    n = len(cm_list)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, n+1, width_ratios=[1]*n + [0.05])

    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cbar_ax = fig.add_subplot(gs[0, -1])

    for i, ax in enumerate(axes):
        cm = cm_list[i].values.astype(float)

        # row normalization
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm / row_sum * 100

        sns.heatmap(
            cm,
            ax=ax,
            cmap=cmap,
            norm=norm,
            square=True,
            annot=annot,
            fmt=".1f",
            annot_kws={"size": annot_fontsize},
            linewidths=0.4,
            cbar=False
        )

        ax.set_title(titles[i], fontsize=title_fontsize)

        # X axis
        ax.set_xticks(np.arange(len(label_order)) + 0.5)
        ax.set_xticklabels(
            label_order,
            rotation=20,
            ha="right",
            fontsize=tick_fontsize
        )
        ax.set_xlabel("Predicted label", fontsize=tick_fontsize+1)

        # Y axis
        ax.set_yticks(np.arange(len(label_order)) + 0.5)
        if i == 0:
            ax.set_yticklabels(
                label_order,
                rotation=0,
                fontsize=tick_fontsize
            )
            ax.set_ylabel("True label", fontsize=tick_fontsize+1)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        # Outer border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    # ---- Shared colorbar ----
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Percentage (%)", fontsize=tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.outline.set_linewidth(0.8)

    return fig

# def plot_confusion_matrix_panel(
#     cm_list,
#     titles,
#     cmap="Blues",
#     vmin=0,
#     vmax=100,
#     figsize=(16, 4),
#     font_family="Times New Roman",
#     annot=True,
#     annot_fontsize=11,
#     tick_fontsize=11,
#     title_fontsize=13,
#     show_colorbar=True,
#     x_label="Predicted label",
#     y_label="True label",
# ):
#     mpl.rcParams.update({
#         "font.family": font_family,
#         "mathtext.fontset": "stix",
#         "axes.unicode_minus": False
#     })

#     n = len(cm_list)

#     fig = plt.figure(figsize=figsize, constrained_layout=True)

#     # GridSpec: extra column for colorbar
#     gs = fig.add_gridspec(1, n + (1 if show_colorbar else 0),
#                           width_ratios=[1]*n + ([0.05] if show_colorbar else []))

#     axes = []
#     for i in range(n):
#         axes.append(fig.add_subplot(gs[0, i]))

#     if show_colorbar:
#         cbar_ax = fig.add_subplot(gs[0, -1])

#     # Normalize matrices
#     processed = []
#     for cm in cm_list:
#         data = cm.values.astype(float)
#         row_sum = data.sum(axis=1, keepdims=True)
#         row_sum[row_sum == 0] = 1
#         data = data / row_sum * 100
#         processed.append(data)

#     # Plot each heatmap
#     for i, ax in enumerate(axes):
#         sns.heatmap(
#             processed[i],
#             ax=ax,
#             cmap=cmap,
#             vmin=vmin,
#             vmax=vmax,
#             square=True,
#             annot=annot,
#             fmt=".1f",
#             annot_kws={"size": annot_fontsize},
#             linewidths=0.4,
#             cbar=False
#         )

#         ax.set_title(titles[i], fontsize=title_fontsize)

#         ax.set_xlabel(x_label, fontsize=tick_fontsize)

#         if i == 0:
#             ax.set_ylabel(y_label, fontsize=tick_fontsize)
#         else:
#             ax.set_ylabel("")
#             ax.set_yticklabels([])

#         ax.tick_params(axis="x", labelsize=tick_fontsize)
#         ax.tick_params(axis="y", labelsize=tick_fontsize)

#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
#         plt.setp(ax.get_yticklabels(), rotation=0)

#         # outer border
#         for spine in ax.spines.values():
#             spine.set_visible(True)
#             spine.set_linewidth(1.0)

#     # Shared colorbar
#     if show_colorbar:
#         norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#         sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#         sm.set_array([])

#         cbar = fig.colorbar(sm, cax=cbar_ax)
#         cbar.set_label("Percentage (%)", fontsize=tick_fontsize)
#         cbar.ax.tick_params(labelsize=tick_fontsize)
#         cbar.outline.set_linewidth(0.8)

#     return fig

def plot_confusion_matrix_panel(
    cm_list,
    titles,
    label_order,
    annot=True,
    annot_fontsize=11,
    tick_fontsize=11,
    title_fontsize=13,
    gamma=0.5,
    vmin=0,
    vmax=100,
    cell_scale=0.6, 
):
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False
    })

    # 鍗曡壊绉戠爺娓愬彉
    color_points = [
        "#F7F9FA",
        "#D6E4EA",
        "#7FB3D5",
        "#154360"
    ]

    cmap = LinearSegmentedColormap.from_list(
        "academic_blue",
        color_points,
        N=256
    )

    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    n = len(cm_list)
    n_class = len(label_order)

    # 鑷姩鏍规嵁 cell_scale 璁＄畻 figsize
    single_width = n_class * cell_scale
    total_width = single_width * n + 1.0   # + colorbar space
    total_height = n_class * cell_scale

    fig = plt.figure(figsize=(total_width, total_height),
                     constrained_layout=True)

    gs = fig.add_gridspec(1, n+1,
                          width_ratios=[1]*n + [0.05])

    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cbar_ax = fig.add_subplot(gs[0, -1])

    for i, ax in enumerate(axes):
        cm = cm_list[i].values.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm / row_sum * 100

        sns.heatmap(
            cm,
            ax=ax,
            cmap=cmap,
            norm=norm,
            square=True,
            annot=annot,
            fmt=".1f",
            annot_kws={"size": annot_fontsize},
            linewidths=0,        # 鍘绘帀鐧借竟
            cbar=False
        )

        ax.set_title(titles[i], fontsize=title_fontsize)

        ax.set_xticks(np.arange(n_class) + 0.5)
        ax.set_xticklabels(label_order,
                           rotation=20,
                        #    ha="right",
                           ha="center", 
                           fontsize=tick_fontsize)

        ax.set_xlabel("Predicted label", fontsize=tick_fontsize)

        ax.set_yticks(np.arange(n_class) + 0.5)

        if i == 0:
            ax.set_yticklabels(label_order,
                               rotation=0,
                               fontsize=tick_fontsize)
            ax.set_ylabel("True label", fontsize=tick_fontsize)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Percentage (%)", fontsize=tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    return fig

def create_custom_colormap(color_list):
    return LinearSegmentedColormap.from_list(
        "custom_cmap",
        color_list,
        N=256
    )

def plot_confusion_matrix_panel_color(
    cm_list,
    titles,
    label_order,
    color_points,
    vmin=0,
    vmax=100,
    figsize=(16, 4),
    annot=True,
    annot_fontsize=11,
    tick_fontsize=11,
    title_fontsize=13,
):
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False
    })

    cmap = create_custom_colormap(color_points)

    n = len(cm_list)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, n+1, width_ratios=[1]*n + [0.05])

    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cbar_ax = fig.add_subplot(gs[0, -1])

    for i, ax in enumerate(axes):
        cm = cm_list[i].values.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm / row_sum * 100

        sns.heatmap(
            cm,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            annot=annot,
            fmt=".1f",
            annot_kws={"size": annot_fontsize},
            linewidths=0.4,
            cbar=False
        )

        ax.set_title(titles[i], fontsize=title_fontsize)

        # X axis
        ax.set_xticks(np.arange(len(label_order)) + 0.5)
        ax.set_xticklabels(label_order, rotation=20, ha="right", fontsize=tick_fontsize)

        # Y axis
        ax.set_yticks(np.arange(len(label_order)) + 0.5)
        if i == 0:
            ax.set_yticklabels(label_order, rotation=0, fontsize=tick_fontsize)
            ax.set_ylabel("True label", fontsize=tick_fontsize)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.set_xlabel("Predicted label", fontsize=tick_fontsize)

        # Outer border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    # Shared colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Percentage (%)", fontsize=tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    return fig

def plot_confusion_matrix_global(
    cm_list,
    titles,
    label_order,
    tick_fontsize=9,
    title_fontsize=12,
    annot=False,
    annot_fontsize=9,
    gamma=0.5,
    vmin=0,
    vmax=100,
    cell_scale=0.15,  # 鏄捐憲鍑忓皬鍗曞厓鏍肩缉鏀?
    show_yticklabels=True,
    global_xlabel_y=0.03,
    use_fixed_figsize=False,
    fixed_figsize=(8.0, 4.2),
    matrix_only=False,
):
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    import seaborn as sns
    import numpy as np

    # 鍩虹椋庢牸
    mpl.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})

    color_points = ["#F7F9FA", "#D6E4EA", "#7FB3D5", "#154360"]
    cmap = LinearSegmentedColormap.from_list("academic_blue", color_points, N=256)
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    n = len(cm_list)
    n_class = len(label_order)

    # Figure size: dynamic by default, optionally fixed for cross-dataset consistency.
    if use_fixed_figsize:
        total_width, total_height = fixed_figsize
        fig = plt.figure(figsize=(total_width, total_height), constrained_layout=False)
        # Keep margins stable across datasets when fixed size is requested.
        fig.subplots_adjust(left=0.12, right=0.90, bottom=0.16, top=0.88, wspace=0.10)
    else:
        single_width = n_class * cell_scale
        total_width = single_width * n + 0.8
        total_height = n_class * cell_scale + 0.5
        fig = plt.figure(figsize=(total_width, total_height), constrained_layout=True)

    gs = fig.add_gridspec(1, n)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]

    for i, ax in enumerate(axes):
        # 褰掍竴鍖栧鐞?
        cm = cm_list[i].values.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_perc = cm / row_sum * 100

        sns.heatmap(
            cm_perc,
            ax=ax,
            cmap=cmap,
            norm=norm,
            square=True,
            annot=annot,
            fmt=".1f",
            annot_kws={"size": annot_fontsize},
            cbar=False,
            xticklabels=False if matrix_only else False, # keep hidden in global mode
            yticklabels=False if matrix_only else (label_order if (i == 0 and show_yticklabels) else False),
            linewidths=0
        )
        if matrix_only:
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(titles[i], fontsize=title_fontsize)
            ax.set_xlabel("")
            if i == 0 and show_yticklabels:
                ax.set_ylabel("True", fontsize=tick_fontsize)
                ax.tick_params(axis='y', labelsize=tick_fontsize - 2) # 鍩熷悕澶氭椂缂╁皬瀛楀彿
            else:
                ax.set_ylabel("")

        # 鏄惧寲杈规
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    if not matrix_only:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        # Bind colorbar to all heatmap axes so its height follows the matrix panel.
        cbar = fig.colorbar(
            sm,
            ax=axes,
            location='right',
            fraction=0.025,
            pad=0.02
        )
        cbar.set_label("Percentage (%)", fontsize=tick_fontsize)
        # Center global x-label with respect to heatmap panel group (exclude colorbar).
        fig.canvas.draw()
        x0 = min(ax.get_position().x0 for ax in axes)
        x1 = max(ax.get_position().x1 for ax in axes)
        x_center = (x0 + x1) / 2.0
        fig.text(
            x_center,
            global_xlabel_y,
            "Predicted",
            ha="center",
            va="bottom",
            fontsize=tick_fontsize
        )
    
    return fig


def plot_confusion_matrix_global_representative(
    cm_list,
    titles,
    repr_labels,
    tick_fontsize=9,
    title_fontsize=12,
    annot=True,
    annot_fontsize=8,
    gamma=0.5,
    vmin=0,
    vmax=100,
    cell_scale=0.24,
    show_yticklabels=True,
    global_xlabel_y=0.03,
    use_fixed_figsize=False,
    fixed_figsize=(8.0, 4.2),
    matrix_only=False,
):
    """
    Plot representative confusion matrices by selecting a label subset.

    Parameters
    ----------
    cm_list : list[pd.DataFrame]
        List of confusion-matrix DataFrames. Index and columns should be labels.
    titles : list[str]
        Title for each confusion matrix.
    repr_labels : list[str]
        Representative label list used to subset/reorder rows & columns.
        Only labels existing in each matrix are kept.
    """
    if len(cm_list) != len(titles):
        raise ValueError("cm_list and titles must have the same length.")
    if not isinstance(repr_labels, (list, tuple)) or len(repr_labels) == 0:
        raise ValueError("repr_labels must be a non-empty list/tuple of labels.")

    cm_subset_list = []
    effective_labels = None

    for idx, cm in enumerate(cm_list):
        if not isinstance(cm, pd.DataFrame):
            raise TypeError(f"cm_list[{idx}] must be a pandas DataFrame.")

        available_labels = [lbl for lbl in repr_labels if lbl in cm.index and lbl in cm.columns]
        if len(available_labels) == 0:
            raise ValueError(
                f"No representative labels found in cm_list[{idx}]. "
                f"Requested labels: {repr_labels}"
            )

        if effective_labels is None:
            effective_labels = available_labels
        else:
            # Keep a common ordered subset across all panels for strict comparability.
            effective_labels = [lbl for lbl in effective_labels if lbl in available_labels]
            if len(effective_labels) == 0:
                raise ValueError(
                    "No common representative labels across all confusion matrices."
                )

        cm_subset_list.append(cm)

    cm_subset_list = [cm.loc[effective_labels, effective_labels] for cm in cm_subset_list]

    return plot_confusion_matrix_global(
        cm_list=cm_subset_list,
        titles=titles,
        label_order=effective_labels,
        tick_fontsize=tick_fontsize,
        title_fontsize=title_fontsize,
        annot=annot,
        annot_fontsize=annot_fontsize,
        gamma=gamma,
        vmin=vmin,
        vmax=vmax,
        cell_scale=cell_scale,
        show_yticklabels=show_yticklabels,
        global_xlabel_y=global_xlabel_y,
        use_fixed_figsize=use_fixed_figsize,
        fixed_figsize=fixed_figsize,
        matrix_only=matrix_only,
    )

