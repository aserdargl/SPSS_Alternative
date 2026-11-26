"""
Görselleştirme modülü (Matplotlib).

Bu modül akademik yayın kalitesinde şu grafikleri üretir:
- Model 4 Yol Diyagramı (Path Diagram) — tekli ve çoklu paralel aracılar için dinamik yerleşim
- Model 4 Bootstrap Dolaylı Etki Dağılımı Grafiği
- Model 1 İstatistiksel Yol Diyagramı
- Model 1 Düzenleyicilik Etkileşim Grafiği (Simple Slopes: -1 SD, Mean, +1 SD)
"""

from __future__ import annotations

import io
import textwrap
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from analysis import format_p_clause, format_p_stars

# Headless / sunucu ortamlarında güvenli çalışma için Agg backend
matplotlib.use("Agg")


def _wrap_label(text: str, width: int = 16) -> str:
    """Uzun değişken adlarını kutulara sığacak şekilde satırlara böler."""
    return textwrap.fill(str(text), width=width)


def _draw_node_box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str,
    title_color: str = "#0F172A",
    subtitle_color: str = "#475569",
) -> None:
    """Yuvarlatılmış kenarlı modern yol diyagramı kutusu çizer."""
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(box)

    ax.text(
        cx,
        cy + height * 0.12,
        _wrap_label(title, width=18),
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )
    ax.text(
        cx,
        cy - height * 0.27,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.2,
        color=subtitle_color,
        style="italic",
        zorder=4,
    )


def _draw_path_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    is_significant: bool,
    label_pos: float = 0.5,
    label_offset: tuple[float, float] = (0.0, 0.0),
    rad: float = 0.0,
) -> None:
    """Yol okunu ve üzerinde katsayı rozetini çizer."""
    color = "#1E293B" if is_significant else "#94A3B8"
    linestyle = "-" if is_significant else "--"
    linewidth = 2.0 if is_significant else 1.5

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_length=8,head_width=4.5",
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        shrinkA=2,
        shrinkB=2,
        zorder=2,
    )
    ax.add_patch(arrow)

    lx = start[0] + (end[0] - start[0]) * label_pos + label_offset[0]
    ly = start[1] + (end[1] - start[1]) * label_pos + label_offset[1]

    ax.text(
        lx,
        ly,
        label,
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold" if is_significant else "normal",
        color="#0F172A" if is_significant else "#64748B",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1" if is_significant else "#E2E8F0",
            linewidth=1.0,
            alpha=0.96,
        ),
        zorder=5,
    )


def plot_mediation_path_diagram(results: dict[str, Any]) -> plt.Figure:
    """
    Model 4 (Aracılık) için Matplotlib yol diyagramı oluşturur.
    Tekli ve çoklu paralel aracı değişkenleri otomatik yerleşimle destekler.
    """
    paths = results["paths"]
    x_name = paths["x"]
    y_name = paths["y"]
    mediators = paths["mediators"]
    k = len(mediators)

    fig_height = max(5.2, 2.8 + 1.55 * k)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    box_w = 2.25
    box_h = 0.92
    x_pos = 1.5
    m_pos = 5.25
    y_pos = 9.0

    # Dikey koordinatlar
    if k == 1:
        y_base = 1.35
        m_ys = [3.65]
        y_top = 4.55
    else:
        y_base = 1.15
        m_ys = np.linspace(2.65, 2.65 + (k - 1) * 1.45, k)[::-1].tolist()
        y_top = max(m_ys) + 1.05

    # X ve Y kutuları
    _draw_node_box(
        ax,
        x_pos,
        y_base,
        box_w,
        box_h,
        x_name,
        "Bağımsız (X)",
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
    )
    y_r2 = float(results["y_model"]["summary_df"]["R²"].iloc[0])
    _draw_node_box(
        ax,
        y_pos,
        y_base,
        box_w,
        box_h,
        y_name,
        f"Bağımlı (Y) | R² = {y_r2:.3f}",
        facecolor="#EEF2FF",
        edgecolor="#4F46E5",
    )

    # Aracı kutuları ve a_i, b_i okları
    for i, (m_name, my) in enumerate(zip(mediators, m_ys), start=1):
        m_r2 = float(results["a_models"][m_name]["summary_df"]["R²"].iloc[0])
        m_sub = f"Aracı (M{i}) | R² = {m_r2:.3f}" if k > 1 else f"Aracı (M) | R² = {m_r2:.3f}"
        _draw_node_box(
            ax,
            m_pos,
            my,
            box_w,
            box_h,
            m_name,
            m_sub,
            facecolor="#ECFDF5",
            edgecolor="#059669",
        )

        a_val = paths["a"][m_name]
        a_p = paths["a_p"][m_name]
        b_val = paths["b"][m_name]
        b_p = paths["b_p"][m_name]

        a_lbl = f"a{i} = {a_val:.3f}{format_p_stars(a_p)}" if k > 1 else f"a = {a_val:.3f}{format_p_stars(a_p)}"
        b_lbl = f"b{i} = {b_val:.3f}{format_p_stars(b_p)}" if k > 1 else f"b = {b_val:.3f}{format_p_stars(b_p)}"

        # X -> M_i
        start_a = (x_pos + box_w / 2, y_base + box_h * 0.22)
        end_a = (m_pos - box_w / 2, my)
        _draw_path_arrow(
            ax,
            start_a,
            end_a,
            label=a_lbl,
            is_significant=(a_p < 0.05),
            label_pos=0.52,
            label_offset=(-0.08, 0.14 if k == 1 else 0.08),
        )

        # M_i -> Y
        start_b = (m_pos + box_w / 2, my)
        end_b = (y_pos - box_w / 2, y_base + box_h * 0.22)
        _draw_path_arrow(
            ax,
            start_b,
            end_b,
            label=b_lbl,
            is_significant=(b_p < 0.05),
            label_pos=0.48,
            label_offset=(0.08, 0.14 if k == 1 else 0.08),
        )

    # Doğrudan (c') ve Toplam (c) etki oku: X -> Y
    c_prime = paths["c_prime"]
    c_prime_p = paths["c_prime_p"]
    c_total = paths["c"]
    c_total_p = paths["c_p"]
    direct_label = (
        f"c' = {c_prime:.3f}{format_p_stars(c_prime_p)}   "
        f"(c = {c_total:.3f}{format_p_stars(c_total_p)})"
    )
    _draw_path_arrow(
        ax,
        (x_pos + box_w / 2, y_base),
        (y_pos - box_w / 2, y_base),
        label=direct_label,
        is_significant=(c_prime_p < 0.05),
        label_pos=0.5,
        label_offset=(0.0, -0.32),
    )

    # Başlık ve dipnot
    ax.set_title(
        f"Model 4 Aracılık Yol Diyagramı (N = {results['data_summary']['valid_n']})",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )

    cov_text = (
        f" | Kontrol Değişkenleri: {', '.join(results['covariates'])}"
        if results["covariates"]
        else ""
    )
    footnote = (
        f"Not: Standardize edilmemiş OLS katsayıları (B) gösterilmiştir. "
        f"* p < .05, ** p < .01, *** p < .001 | Düz ok: p < .05, kesikli ok: p ≥ .05{cov_text}"
    )
    ax.text(
        5.25,
        0.18,
        footnote,
        ha="center",
        va="center",
        fontsize=8.2,
        color="#475569",
    )

    ax.set_xlim(0.0, 10.5)
    ax.set_ylim(-0.1, y_top)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_bootstrap_distribution(results: dict[str, Any]) -> plt.Figure:
    """Model 4 için Bootstrap dolaylı etki dağılım(lar)ını ve %95 güven aralığını çizer."""
    m_cols = results["m_cols"]
    k = len(m_cols)
    conf_level = results["conf_level"]
    alpha = (100.0 - conf_level) / 100.0
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)

    fig, axes = plt.subplots(1, k, figsize=(max(6.8, 5.2 * k), 4.0), dpi=150, squeeze=False)
    fig.patch.set_facecolor("#FFFFFF")

    for idx, m_col in enumerate(m_cols):
        ax = axes[0, idx]
        dist = results["boot_distributions"][m_col]
        ab = results["paths"]["a"][m_col] * results["paths"]["b"][m_col]
        llci = float(np.percentile(dist, low_q))
        ulci = float(np.percentile(dist, high_q))
        is_sig = not (llci <= 0.0 <= ulci)

        ax.hist(dist, bins=40, color="#3B82F6", alpha=0.35, edgecolor="#1D4ED8", linewidth=0.5)
        ax.axvspan(llci, ulci, color="#10B981" if is_sig else "#F59E0B", alpha=0.16, label=f"%{conf_level:g} GA")
        ax.axvline(ab, color="#1D4ED8", linewidth=2.0, label=f"a×b = {ab:.4f}")
        ax.axvline(0.0, color="#DC2626", linestyle="--", linewidth=1.8, label="Sıfır (0)")
        ax.axvline(llci, color="#059669" if is_sig else "#D97706", linestyle=":", linewidth=1.5)
        ax.axvline(ulci, color="#059669" if is_sig else "#D97706", linestyle=":", linewidth=1.5)

        ax.set_title(
            f"{m_col}\n[{llci:.4f}, {ulci:.4f}] ({'Anlamlı' if is_sig else 'Anlamlı Değil'})",
            fontsize=10.5,
            fontweight="bold",
            color="#0F172A",
        )
        ax.set_xlabel("Bootstrap Dolaylı Etki (a×b)", fontsize=9.5, color="#334155")
        if idx == 0:
            ax.set_ylabel("Frekans", fontsize=9.5, color="#334155")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")

    fig.suptitle(
        f"Percentile Bootstrap Dolaylı Etki Dağılımı (B = {results['n_bootstrap']:,})",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_moderation_path_diagram(results: dict[str, Any]) -> plt.Figure:
    """Model 1 (Düzenleyicilik) için Hayes istatistiksel yol diyagramı çizer."""
    paths = results["paths"]
    fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    box_w = 2.45
    box_h = 0.88
    left_x = 1.8
    right_x = 8.4

    ys = [4.0, 2.55, 1.1]

    _draw_node_box(
        ax, left_x, ys[0], box_w, box_h, paths["x"], "Bağımsız (X)", facecolor="#EFF6FF", edgecolor="#2563EB"
    )
    _draw_node_box(
        ax, left_x, ys[1], box_w, box_h, paths["w"], "Düzenleyici (W)", facecolor="#ECFDF5", edgecolor="#059669"
    )
    _draw_node_box(
        ax,
        left_x,
        ys[2],
        box_w,
        box_h,
        paths["int_col"],
        f"Etkileşim (ΔR² = {paths['delta_r2']:.4f})",
        facecolor="#FFFBEB",
        edgecolor="#D97706",
    )
    _draw_node_box(
        ax,
        right_x,
        ys[1],
        box_w,
        box_h,
        paths["y"],
        f"Bağımlı (Y) | R² = {paths['r2']:.3f}",
        facecolor="#EEF2FF",
        edgecolor="#4F46E5",
    )

    _draw_path_arrow(
        ax,
        (left_x + box_w / 2, ys[0]),
        (right_x - box_w / 2, ys[1] + 0.22),
        label=f"b₁ = {paths['b1']:.3f}{format_p_stars(paths['b1_p'])}",
        is_significant=(paths["b1_p"] < 0.05),
        label_pos=0.46,
        label_offset=(0.0, 0.14),
    )
    _draw_path_arrow(
        ax,
        (left_x + box_w / 2, ys[1]),
        (right_x - box_w / 2, ys[1]),
        label=f"b₂ = {paths['b2']:.3f}{format_p_stars(paths['b2_p'])}",
        is_significant=(paths["b2_p"] < 0.05),
        label_pos=0.46,
        label_offset=(0.0, 0.0),
    )
    _draw_path_arrow(
        ax,
        (left_x + box_w / 2, ys[2]),
        (right_x - box_w / 2, ys[1] - 0.22),
        label=f"b₃ = {paths['b3']:.3f}{format_p_stars(paths['b3_p'])}",
        is_significant=(paths["b3_p"] < 0.05),
        label_pos=0.46,
        label_offset=(0.0, -0.14),
    )

    center_txt = "Ortalama Merkezileştirilmiş" if results["mean_center"] else "Merkezileştirilmemiş"
    ax.set_title(
        f"Model 1 Düzenleyicilik İstatistiksel Yol Diyagramı (N = {results['data_summary']['valid_n']})",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )

    cov_text = (
        f" | Kontrol Değişkenleri: {', '.join(results['covariates'])}"
        if results["covariates"]
        else ""
    )
    ax.text(
        5.1,
        0.2,
        f"Not: {center_txt} OLS katsayıları. * p < .05, ** p < .01, *** p < .001{cov_text}",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#475569",
    )

    ax.set_xlim(0.0, 10.2)
    ax.set_ylim(-0.05, 4.9)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_moderation_interaction(
    results: dict[str, Any], show_ci: bool = True
) -> plt.Figure:
    """
    Model 1 için -1 SD, Ortalama ve +1 SD düzenleyici düzeylerindeki etkileşim (Simple Slopes) grafiğini çizer.
    """
    x_col = results["x_col"]
    y_col = results["y_col"]
    w_col = results["w_col"]
    plot_df = results["plot_data_df"]
    slopes_df = results["simple_slopes_df"]
    paths = results["paths"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    styles = {
        "Düşük (-1 SD)": {"color": "#2563EB", "ls": "--", "marker": "o"},
        "Ortalama (Mean)": {"color": "#0F766E", "ls": "-.", "marker": "s"},
        "Yüksek (+1 SD)": {"color": "#DC2626", "ls": "-", "marker": "D"},
    }

    x_col_val = f"{x_col} Değeri"
    y_col_pred = f"Tahmin Edilen {y_col}"

    for _, s_row in slopes_df.iterrows():
        w_label = s_row["Düzey (W)"]
        sub = plot_df[plot_df["W Düzeyi"] == w_label].sort_values(x_col_val)
        st = styles.get(w_label, {"color": "#334155", "ls": "-", "marker": "o"})

        w_orig = s_row[f"{w_col} (Orijinal)"]
        slope_b = s_row["Koşullu Etki (Eğim)"]
        stars = s_row["Anlamlılık"]
        legend_lbl = f"{w_label} ({w_col} = {w_orig:.2f}) | b = {slope_b:.3f}{stars}"

        x_vals = sub[x_col_val].to_numpy()
        y_vals = sub[y_col_pred].to_numpy()

        ax.plot(
            x_vals,
            y_vals,
            label=legend_lbl,
            color=st["color"],
            linestyle=st["ls"],
            marker=st["marker"],
            linewidth=2.4,
            markersize=7.2,
            zorder=4,
        )

        if show_ci:
            ax.fill_between(
                x_vals,
                sub["Alt Sınır"].to_numpy(),
                sub["Üst Sınır"].to_numpy(),
                color=st["color"],
                alpha=0.10,
                zorder=2,
            )

    # X ekseni işaretçileri: -1 SD, Mean, +1 SD
    x_mean = results["x_stats"]["mean"]
    x_sd = results["x_stats"]["sd"]
    xticks = [x_mean - x_sd, x_mean, x_mean + x_sd]
    xtick_labels = [
        f"Düşük (-1 SD)\n({x_mean - x_sd:.2f})",
        f"Ortalama (Mean)\n({x_mean:.2f})",
        f"Yüksek (+1 SD)\n({x_mean + x_sd:.2f})",
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=9.5, color="#1E293B")

    ax.set_xlabel(x_col, fontsize=11, fontweight="bold", color="#0F172A", labelpad=8)
    ax.set_ylabel(f"Tahmin Edilen {y_col}", fontsize=11, fontweight="bold", color="#0F172A", labelpad=8)

    b3 = paths["b3"]
    b3_p = paths["b3_p"]
    delta_r2 = paths["delta_r2"]
    ax.set_title(
        f"Düzenleyicilik Etkileşim Grafiği: {x_col} × {w_col} → {y_col}\n"
        f"(Etkileşim B = {b3:.4f}, {format_p_clause(b3_p)}, ΔR² = {delta_r2:.4f})",
        fontsize=11.5,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")

    ax.legend(
        title=f"Düzenleyici: {w_col}",
        title_fontsize=9.5,
        fontsize=9.0,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        loc="best",
    )

    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    """Matplotlib figürünü yüksek çözünürlüklü PNG bayt akışına dönüştürür."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()
