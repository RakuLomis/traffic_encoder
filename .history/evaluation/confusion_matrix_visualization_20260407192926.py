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
    label_order: list = None  # 新增参数
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
        # 验证所有指定的标签都存在于数据中
        missing_labels = set(label_order) - set(cm.index)
        if missing_labels:
            raise ValueError(f"Specified labels not found in CSV: {missing_labels}")
        
        # 保留原数据中有的标签，按照 label_order 的顺序排列
        available_labels = [label for label in label_order if label in cm.index]
        
        # 如果还有未在 label_order 中指定的标签，添加到末尾（或抛出警告）
        remaining_labels = [label for label in cm.index if label not in label_order]
        if remaining_labels:
            print(f"Warning: The following labels are not in label_order and will be appended: {remaining_labels}")
            available_labels.extend(remaining_labels)
        
        sorted_labels = available_labels
    else:
        # 默认：按行样本数降序排列
        row_sums = cm.sum(axis=1)
        sorted_labels = row_sums.sort_values(ascending=False).index

    # 确保列也按照相同的顺序排列（混淆矩阵应该是方阵）
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

    # 删除 colorbar 的边框（outline）
    cbar.outline.set_visible(False)

    # 可选：同时删除 tick 的刻度线（只保留数字标签，更简洁）
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
    gamma=0.5,                 # 非线性强度
    vmin=0,
    vmax=100
):
    # ---- Font ----
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False
    })

    # ---- 单色科研风渐变 ----
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

    # ---- 非线性归一化 ----
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

    # 单色科研渐变
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

    # 自动根据 cell_scale 计算 figsize
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
            linewidths=0,        # 去掉白边
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
    gamma=0.5,
    vmin=0,
    vmax=100,
    cell_scale=0.15,  # 显著减小单元格缩放
    show_yticklabels=True
):
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    import seaborn as sns
    import numpy as np

    # 基础风格
    mpl.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})

    color_points = ["#F7F9FA", "#D6E4EA", "#7FB3D5", "#154360"]
    cmap = LinearSegmentedColormap.from_list("academic_blue", color_points, N=256)
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    n = len(cm_list)
    n_class = len(label_order)

    # 计算尺寸：全局图不需要容纳文字，可以非常紧凑
    single_width = n_class * cell_scale
    total_width = single_width * n + 0.8
    total_height = n_class * cell_scale + 0.5

    fig = plt.figure(figsize=(total_width, total_height), constrained_layout=True)
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.05])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cbar_ax = fig.add_subplot(gs[0, -1])

    for i, ax in enumerate(axes):
        # 归一化处理
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
            annot=False,    # 核心：去掉数字标注
            cbar=False,
            xticklabels=False, # 核心：去掉X轴域名
            yticklabels=label_order if (i == 0 and show_yticklabels) else False,
            linewidths=0
        )

        ax.set_title(titles[i], fontsize=title_fontsize)
        ax.set_xlabel("Predicted", fontsize=tick_fontsize)
        
        if i == 0 and show_yticklabels:
            ax.set_ylabel("True", fontsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize - 2) # 域名多时缩小字号
        else:
            ax.set_ylabel("")

        # 显化边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Accuracy (%)", fontsize=tick_fontsize)
    
    return fig

def plot_confusion_matrix_global_v2(
    cm_list,
    titles,
    label_order,
    tick_fontsize=10,
    title_fontsize=12,
    gamma=0.5,
    vmin=0,
    vmax=100,
    cell_scale=0.2, 
    show_yticklabels=True
):
    # 1. 风格设置
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "axes.unicode_minus": False
    })

    # 科研深蓝配色
    color_points = ["#F7F9FA", "#D6E4EA", "#7FB3D5", "#154360"]
    cmap = LinearSegmentedColormap.from_list("academic_blue", color_points, N=256)
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    n = len(cm_list)
    n_class = len(label_order)

    # 2. 动态计算画布尺寸
    # 由于是全局图且没有数字标注，cell_scale 可以设得很小
    single_width = n_class * cell_scale
    total_width = single_width * n + 1.2  # 为右侧 colorbar 预留空间
    total_height = n_class * cell_scale + 0.8

    # 使用 subplots 布局，不再手动分配 cbar 轴
    fig, axes = plt.subplots(1, n, figsize=(total_width, total_height), constrained_layout=True)
    if n == 1: axes = [axes]

    mappable = None
    for i, ax in enumerate(axes):
        # 归一化处理（百分比）
        cm = cm_list[i].values.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_perc = cm / row_sum * 100

        # 绘制热力图
        sns_ax = sns.heatmap(
            cm_perc,
            ax=ax,
            cmap=cmap,
            norm=norm,
            square=True,      # 强制正方形单元格
            annot=False,     # 全局图去掉数字
            cbar=False,      # 循环内不画 cbar
            xticklabels=False, # 去掉 X 轴域名
            yticklabels=label_order if (i == 0 and show_yticklabels) else False,
            linewidths=0
        )
        
        # 记录最后一张图的 mappable 用于创建 colorbar
        mappable = sns_ax.get_children()[0]

        ax.set_title(titles[i], fontsize=title_fontsize, fontweight='bold')
        
        # 去掉子图各自的 xlabel
        ax.set_xlabel("")

        if i == 0 and show_yticklabels:
            ax.set_ylabel("True label", fontsize=tick_fontsize, fontweight='bold')
            ax.tick_params(axis='y', labelsize=tick_fontsize - 2)
        else:
            ax.set_ylabel("")

        # 显化边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    # 3. 解决需求 1：全局 X 轴标题
    fig.supxlabel("Predicted label", fontsize=tick_fontsize, fontweight='bold')

    # 4. 解决需求 2：Colorbar 高度对齐
    # 通过 ax=axes 参数，matplotlib 会自动计算所有子图的整体高度并匹配
    cbar = fig.colorbar(mappable, ax=axes, location='right', fraction=0.02, pad=0.02)
    cbar.set_label("Percentage (%)", fontsize=tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize - 1)

    return fig

def plot_confusion_matrix_global_v3(
    cm_list,
    titles,
    label_order,
    tick_fontsize=10,
    title_fontsize=12,
    gamma=0.5,
    vmin=0,
    vmax=100,
    cell_scale=0.2, 
    show_yticklabels=True
):
    mpl.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})

    color_points = ["#F7F9FA", "#D6E4EA", "#7FB3D5", "#154360"]
    cmap = LinearSegmentedColormap.from_list("academic_blue", color_points, N=256)
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    n = len(cm_list)
    n_class = len(label_order)

    # 动态计算尺寸
    single_width = n_class * cell_scale
    total_width = single_width * n + 1.5 
    total_height = n_class * cell_scale + 1.0

    fig, axes = plt.subplots(1, n, figsize=(total_width, total_height), constrained_layout=True)
    if n == 1: axes = [axes]

    for i, ax in enumerate(axes):
        cm = cm_list[i].values.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_perc = cm / row_sum * 100

        # 核心：cbar=False，我们后面手动添加
        sns_ax = sns.heatmap(
            cm_perc, ax=ax, cmap=cmap, norm=norm,
            square=True, annot=False, cbar=False,
            xticklabels=False, 
            yticklabels=label_order if (i == 0 and show_yticklabels) else False,
            linewidths=0
        )

        ax.set_title(titles[i], fontsize=title_fontsize, fontweight='bold', pad=10)
        
        # 显化边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        # 只在最后一个子图右侧添加对齐的 Colorbar
        if i == n - 1:
            divider = make_axes_locatable(ax)
            # size="5%" 代表 colorbar 宽度，pad="5%" 代表间距
            cax = divider.append_axes("right", size="5%", pad=0.1)
            
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Accuracy (%)", fontsize=tick_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize - 1)

    # 全局 X 轴标题
    fig.supxlabel("Predicted label", fontsize=tick_fontsize, fontweight='bold', y=0.05)

    if show_yticklabels:
        axes[0].set_ylabel("True label", fontsize=tick_fontsize, fontweight='bold')
        axes[0].tick_params(axis='y', labelsize=tick_fontsize - 3)

    return fig