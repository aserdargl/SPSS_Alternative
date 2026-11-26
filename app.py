"""
Hayes PROCESS Macro Alternatifi — Aracılık (Mediation) ve Düzenleyicilik (Moderation) Analiz Uygulaması.

Çalıştırmak için:
    streamlit run app.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis import (
    compute_descriptive_stats,
    compute_missing_summary,
    export_results_to_excel,
    get_excel_sheet_names,
    load_dataset,
    run_mediation_model4,
    run_moderation_model1,
)
from plotting import (
    figure_to_png_bytes,
    plot_bootstrap_distribution,
    plot_mediation_path_diagram,
    plot_moderation_interaction,
    plot_moderation_path_diagram,
)

# ---------------------------------------------------------------------------
# Sayfa Yapılandırması ve Stil
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PROCESS Alternatifi | Aracılık & Düzenleyicilik Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 1.7rem;
        margin: 0 0 0.25rem 0;
        font-weight: 700;
    }
    .main-header p {
        color: #E2E8F0;
        margin: 0;
        font-size: 0.96rem;
    }
    .interp-box-sig {
        background-color: #ECFDF5;
        border-left: 5px solid #059669;
        padding: 0.95rem 1.15rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
        color: #064E3B;
    }
    .interp-box-ns {
        background-color: #FFFBEB;
        border-left: 5px solid #D97706;
        padding: 0.95rem 1.15rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
        color: #78350F;
    }
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.75rem 1rem;
        border-radius: 10px;
    }
    /* Sol menü (sidebar) dikey boşluklarını sıkılaştır */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.6rem !important;
        padding-bottom: 0.8rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding-top: 0.4rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.42rem !important;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.02rem !important;
        padding: 0.15rem 0 0.05rem 0 !important;
        margin: 0 !important;
        line-height: 1.25 !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.35rem 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stAlert"] {
        padding: 0.45rem 0.7rem !important;
        font-size: 0.85rem !important;
    }
    section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] {
        padding: 0.55rem 0.7rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] {
        margin-bottom: 0.15rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Yardımcı Görüntüleme Fonksiyonları
# ---------------------------------------------------------------------------

def _clean_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Dahili '_' önekli yardımcı sütunları gizler ve sayısal biçimlendirme uygular."""
    visible = df[[c for c in df.columns if not str(c).startswith("_")]].copy()
    return visible


def _format_numeric_styler(df: pd.DataFrame):
    """İstatistik tablolarını 4 ondalık basamakla temiz biçimlendirir."""
    visible = _clean_display_df(df)
    fmt: dict[str, Any] = {}
    for col in visible.columns:
        if pd.api.types.is_integer_dtype(visible[col]):
            fmt[col] = "{:,d}"
        elif pd.api.types.is_float_dtype(visible[col]):
            if "(%)" in col:
                fmt[col] = "{:.2f}%"
            elif col in {"df1", "df2"}:
                fmt[col] = "{:.0f}"
            else:
                fmt[col] = "{:.4f}"
    return visible.style.format(fmt, na_rep="—")


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    """Örnek veri seti yüklendiğinde anlamlı varsayılan değişkeni seçer."""
    if preferred in options:
        return options.index(preferred)
    return min(fallback, max(len(options) - 1, 0))


def _md_bold_to_html(text: str) -> str:
    """Markdown **kalın** ifadelerini güvenli HTML <strong> etiketine dönüştürür."""
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-header">
        <h1>📊 Aracılık (Mediation) ve Düzenleyicilik (Moderation) Analiz Uygulaması</h1>
        <p>SPSS Hayes PROCESS Macro Alternatifi — Model 4 (Basit/Çoklu Paralel Aracılık) &amp; Model 1 (Düzenleyicilik) | OLS + Percentile Bootstrap + Sobel Testi</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sol Menü (Sidebar): Veri Yükleme ve Model Ayarları
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Veri Seti")
    uploaded_file = st.file_uploader(
        "Veri dosyası yükleyin (.sav, .xlsx, .csv)",
        type=["sav", "xlsx", "xls", "csv"],
        help="SPSS (.sav), Excel (.xlsx) veya CSV (.csv) dosyanızı sürükleyip bırakabilirsiniz.",
    )

    sample_path = Path(__file__).parent / "sample_data.csv"
    use_sample = False
    if uploaded_file is None and sample_path.exists():
        use_sample = st.checkbox(
            "Örnek veri setini kullan (sample_data.csv)",
            value=True,
            help="Uygulamayı hemen test etmek için hazır sentetik veri setini yükler.",
        )

    df: pd.DataFrame | None = None
    meta_info: dict[str, Any] = {}

    if uploaded_file is not None:
        sheet_name: str | int = 0
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            try:
                sheets = get_excel_sheet_names(uploaded_file)
                if len(sheets) > 1:
                    sheet_name = st.selectbox("Excel Sayfası (Sheet)", sheets)
            except Exception as exc:
                st.error(f"Excel sayfaları okunamadı: {exc}")
        try:
            df, meta_info = load_dataset(uploaded_file, sheet_name=sheet_name)
            st.success(f"Yüklendi: **{meta_info['filename']}** ({len(df)} satır, {len(df.columns)} sütun)")
        except Exception as exc:
            st.error(f"Veri yükleme hatası: {exc}")
    elif use_sample and sample_path.exists():
        try:
            df, meta_info = load_dataset(sample_path)
            st.info(f"Örnek veri aktif: **{meta_info['filename']}** ({len(df)} satır, {len(df.columns)} sütun)")
        except Exception as exc:
            st.error(f"Örnek veri yüklenemedi: {exc}")

    st.divider()
    st.header("2. Model Seçimi")
    model_choice = st.radio(
        "Hayes PROCESS Modeli",
        options=[
            "Model 4 — Aracılık (Mediation)",
            "Model 1 — Düzenleyicilik (Moderation)",
        ],
        help="Model 4: X → M → Y (tek veya çoklu paralel aracı). Model 1: X × W → Y düzenleyici etkileşim.",
    )

    run_clicked = False
    params: dict[str, Any] = {}

    if df is not None and len(df.columns) >= 3:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        all_cols = df.columns.tolist()
        candidate_cols = numeric_cols if len(numeric_cols) >= 3 else all_cols

        st.divider()
        st.header("3. Değişken Ataması")

        if model_choice.startswith("Model 4"):
            x_col = st.selectbox(
                "Bağımsız Değişken (X)",
                options=candidate_cols,
                index=_default_index(candidate_cols, "Is_Stresi", 0),
            )
            remaining_for_y = [c for c in candidate_cols if c != x_col]
            y_col = st.selectbox(
                "Bağımlı Değişken (Y)",
                options=remaining_for_y,
                index=_default_index(remaining_for_y, "Performans", 0),
            )
            remaining_for_m = [c for c in candidate_cols if c not in {x_col, y_col}]
            default_m = [c for c in ["Tukenmislik", "Is_Doyumu"] if c in remaining_for_m]
            if not default_m and remaining_for_m:
                default_m = [remaining_for_m[0]]
            m_cols = st.multiselect(
                "Aracı Değişken(ler) (M)",
                options=remaining_for_m,
                default=default_m,
                help="Birden fazla aracı seçerek paralel çoklu aracılık analizi yapabilirsiniz.",
            )
            remaining_for_cov = [c for c in candidate_cols if c not in {x_col, y_col, *m_cols}]
            covariates = st.multiselect(
                "Kontrol Değişkenleri (Covariates — İsteğe Bağlı)",
                options=remaining_for_cov,
                default=[],
            )

            st.divider()
            st.header("4. Bootstrap Ayarları")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                n_bootstrap = st.number_input(
                    "Tekrar Sayısı",
                    min_value=500,
                    max_value=20000,
                    value=5000,
                    step=500,
                    help="Hayes PROCESS varsayılanı 5000 tekrardır.",
                )
            with b_col2:
                conf_level = st.selectbox("Güven Düzeyi (%)", options=[90.0, 95.0, 99.0], index=1)
            use_seed = st.checkbox("Sabit Random Seed kullan", value=True)
            random_seed = (
                int(st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1))
                if use_seed
                else None
            )

            params = {
                "model": "Model 4",
                "x_col": x_col,
                "y_col": y_col,
                "m_cols": m_cols,
                "covariates": covariates,
                "n_bootstrap": int(n_bootstrap),
                "conf_level": float(conf_level),
                "random_seed": random_seed,
            }
        else:
            x_col = st.selectbox(
                "Bağımsız Değişken (X)",
                options=candidate_cols,
                index=_default_index(candidate_cols, "Is_Stresi", 0),
            )
            remaining_for_y = [c for c in candidate_cols if c != x_col]
            y_col = st.selectbox(
                "Bağımlı Değişken (Y)",
                options=remaining_for_y,
                index=_default_index(remaining_for_y, "Tukenmislik", 0),
            )
            remaining_for_w = [c for c in candidate_cols if c not in {x_col, y_col}]
            w_col = st.selectbox(
                "Düzenleyici Değişken (W)",
                options=remaining_for_w,
                index=_default_index(remaining_for_w, "Sosyal_Destek", 0),
            )
            remaining_for_cov = [c for c in candidate_cols if c not in {x_col, y_col, w_col}]
            covariates = st.multiselect(
                "Kontrol Değişkenleri (Covariates — İsteğe Bağlı)",
                options=remaining_for_cov,
                default=[],
            )

            st.divider()
            st.header("4. Model 1 Seçenekleri")
            mean_center = st.checkbox(
                "Ortalama Merkezileştirme (Mean-Centering)",
                value=True,
                help="Etkileşim terimini (X×W) oluşturmadan önce X ve W değişkenlerini ortalamalarından çıkarır (önerilen).",
            )
            conf_level = st.selectbox("Güven Düzeyi (%)", options=[90.0, 95.0, 99.0], index=1)

            params = {
                "model": "Model 1",
                "x_col": x_col,
                "y_col": y_col,
                "w_col": w_col,
                "covariates": covariates,
                "mean_center": mean_center,
                "conf_level": float(conf_level),
            }

        run_clicked = st.button("🚀 Analizi Çalıştır", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Ana İçerik Alanı
# ---------------------------------------------------------------------------

if df is None:
    st.info(
        "Başlamak için sol menüden bir **.sav (SPSS)**, **.xlsx (Excel)** veya **.csv** veri dosyası yükleyin "
        "ya da örnek veri setini etkinleştirin."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Bölüm 1: Veri Önizleme, Eksik Değer Özeti ve Betimsel İstatistikler
# ---------------------------------------------------------------------------

missing_df = compute_missing_summary(df)
desc_df = compute_descriptive_stats(df)
total_missing_cells = int(df.isna().sum().sum())
complete_cases = int(df.dropna().shape[0])

with st.expander("📂 Veri Seti Özeti, İlk 5 Satır, Eksik Değerler ve Betimsel İstatistikler", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Gözlem (Satır)", f"{len(df):,}")
    c2.metric("Değişken Sayısı (Sütun)", f"{len(df.columns)}")
    c3.metric("Eksik Hücre Sayısı", f"{total_missing_cells:,}")
    c4.metric("Tüm Değişkenlerde Tam Gözlem", f"{complete_cases:,}")

    tab_head, tab_missing, tab_desc = st.tabs(
        ["🔍 İlk 5 Satır", "⚠️ Eksik Değer Özeti", "📊 Betimsel İstatistikler"]
    )
    with tab_head:
        st.dataframe(df.head(5), width="stretch")
    with tab_missing:
        st.dataframe(_format_numeric_styler(missing_df), width="stretch", hide_index=True)
        st.caption(
            "Not: Analiz sırasında yalnızca seçilen değişkenler (X, Y, M/W ve Kontrol) üzerinde "
            "**satır bazlı silme (listwise deletion)** uygulanır."
        )
        if meta_info.get("variable_labels"):
            with st.expander("SPSS Değişken Etiketleri (.sav)"):
                lbl_df = pd.DataFrame(
                    [
                        {"Değişken": k, "SPSS Etiketi": v}
                        for k, v in meta_info["variable_labels"].items()
                    ]
                )
                st.dataframe(lbl_df, width="stretch", hide_index=True)
    with tab_desc:
        if desc_df.empty:
            st.warning("Veri setinde sayısal değişken bulunamadı.")
        else:
            st.dataframe(_format_numeric_styler(desc_df), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Bölüm 2: İstatistiksel Analizin Çalıştırılması ve Saklanması
# ---------------------------------------------------------------------------

if run_clicked:
    try:
        with st.spinner("İstatistiksel analiz ve Bootstrap hesaplamaları çalıştırılıyor..."):
            if params["model"] == "Model 4":
                res = run_mediation_model4(
                    df=df,
                    x_col=params["x_col"],
                    y_col=params["y_col"],
                    m_cols=params["m_cols"],
                    covariates=params["covariates"],
                    n_bootstrap=params["n_bootstrap"],
                    conf_level=params["conf_level"],
                    random_seed=params["random_seed"],
                )
            else:
                res = run_moderation_model1(
                    df=df,
                    x_col=params["x_col"],
                    y_col=params["y_col"],
                    w_col=params["w_col"],
                    covariates=params["covariates"],
                    mean_center=params["mean_center"],
                    conf_level=params["conf_level"],
                )
        st.session_state["last_results"] = res
        st.session_state["last_dataset_name"] = meta_info.get("filename", "veri")
    except Exception as exc:
        st.error(f"Analiz çalıştırılırken hata oluştu: {exc}")

results: dict[str, Any] | None = st.session_state.get("last_results")

if results is None:
    st.info("Sol menüden değişkenlerinizi seçip **🚀 Analizi Çalıştır** butonuna tıklayın.")
    st.stop()


# ---------------------------------------------------------------------------
# Bölüm 3: Sonuçların Sunumu
# ---------------------------------------------------------------------------

st.divider()
ds = results["data_summary"]

header_col, export_col = st.columns([3.2, 1.3])
with header_col:
    st.subheader(f"📌 Analiz Sonuçları — {results['model_type']}")
    st.caption(
        f"Satır bazlı silme (listwise deletion) uygulandı: Başlangıç N = {ds['original_n']} → "
        f"**Analiz N = {ds['valid_n']}** (Silinen: {ds['dropped_n']} satır, %{ds['dropped_pct']})"
    )
    if ds.get("non_numeric_converted"):
        st.warning(
            f"Sayısal olmayan değer içerdiği için eksik (NaN) olarak işlenen sütunlar: "
            f"{', '.join(ds['non_numeric_converted'])}"
        )

with export_col:
    excel_bytes = export_results_to_excel(results, missing_summary_df=missing_df)
    model_slug = "Model4_Aracilik" if results["model_type"] == "Model 4" else "Model1_Duzenleyicilik"
    st.download_button(
        label="📥 Sonuçları Excel (.xlsx) İndir",
        data=excel_bytes,
        file_name=f"PROCESS_{model_slug}_Sonuclar.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width="stretch",
    )

# Akademik Türkçe Yorum Alanı
st.markdown("### 📝 Akademik Yorumlama")
if results["model_type"] == "Model 4":
    for item in results["interpretations"]:
        css_cls = "interp-box-sig" if item["significant"] else "interp-box-ns"
        icon = "✅" if item["significant"] else "ℹ️"
        rendered_text = _md_bold_to_html(item["text"])
        st.markdown(f'<div class="{css_cls}">{icon} {rendered_text}</div>', unsafe_allow_html=True)
else:
    css_cls = "interp-box-sig" if results["interaction_significant"] else "interp-box-ns"
    icon = "✅" if results["interaction_significant"] else "ℹ️"
    slopes_html = "<br>".join(_md_bold_to_html(note) for note in results["slope_notes"])
    interp_html = _md_bold_to_html(results["interpretation"])
    st.markdown(
        f'<div class="{css_cls}">{icon} {interp_html}<br><br>'
        f"<strong>Koşullu Etkiler (Simple Slopes):</strong><br>{slopes_html}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MODEL 4 SEKMELERİ
# ---------------------------------------------------------------------------

if results["model_type"] == "Model 4":
    m4_tab_vis, m4_tab_ind, m4_tab_reg = st.tabs(
        [
            "🗺️ Yol Diyagramı ve Bootstrap Dağılımı",
            "🎯 Dolaylı Etki (Bootstrap) & Sobel Testi",
            "📋 Regresyon Tabloları (a, b, c, c')",
        ]
    )

    with m4_tab_vis:
        fig_path = plot_mediation_path_diagram(results)
        st.pyplot(fig_path, width="stretch")
        st.download_button(
            "🖼️ Yol Diyagramını PNG İndir (300 DPI)",
            data=figure_to_png_bytes(fig_path),
            file_name="model4_yol_diyagrami.png",
            mime="image/png",
        )
        plt.close(fig_path)

        st.markdown("#### Percentile Bootstrap Dolaylı Etki Dağılımı")
        fig_boot = plot_bootstrap_distribution(results)
        st.pyplot(fig_boot, width="stretch")
        st.download_button(
            "🖼️ Bootstrap Grafiğini PNG İndir (300 DPI)",
            data=figure_to_png_bytes(fig_boot),
            file_name="model4_bootstrap_dagilimi.png",
            mime="image/png",
        )
        plt.close(fig_boot)

    with m4_tab_ind:
        st.markdown("#### Toplam Etki (c) ve Doğrudan Etki (c')")
        st.dataframe(_format_numeric_styler(results["total_direct_df"]), width="stretch", hide_index=True)

        st.markdown(
            f"#### Dolaylı Etki(ler) — Percentile Bootstrap "
            f"(Tekrar: {results['n_bootstrap']:,}, Seed: {results['random_seed']})"
        )
        st.dataframe(_format_numeric_styler(results["indirect_df"]), width="stretch", hide_index=True)

        st.markdown("#### Klasik Sobel Testi (Normal Teori Testi)")
        st.dataframe(_format_numeric_styler(results["sobel_df"]), width="stretch", hide_index=True)
        st.caption(
            "Not: Akademik literatürde (Hayes, 2022) dolaylı etkinin anlamlılığı için normallik varsayımı gerektirmeyen "
            "Percentile Bootstrap güven aralığı (BootLLCI – BootULCI) birincil ölçüt olarak önerilmektedir."
        )

    with m4_tab_reg:
        st.markdown("#### Tüm Modellerin Özet Tablosu ($R$, $R^2$, $F$, $p$)")
        st.dataframe(_format_numeric_styler(results["all_summaries_df"]), width="stretch", hide_index=True)

        st.markdown("#### Aracı Değişken Modelleri (a Yolları)")
        for m_col, m_res in results["a_models"].items():
            st.markdown(f"**{m_res['title']}**")
            st.dataframe(_format_numeric_styler(m_res["coef_df"]), width="stretch", hide_index=True)

        st.markdown(f"#### {results['y_model']['title']}")
        st.dataframe(_format_numeric_styler(results["y_model"]["coef_df"]), width="stretch", hide_index=True)

        st.markdown(f"#### {results['total_model']['title']}")
        st.dataframe(_format_numeric_styler(results["total_model"]["coef_df"]), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# MODEL 1 SEKMELERİ
# ---------------------------------------------------------------------------

else:
    m1_tab_vis, m1_tab_reg, m1_tab_slopes = st.tabs(
        [
            "📈 Etkileşim Grafiği & Yol Diyagramı",
            "📋 Regresyon Modeli & Etkileşim (ΔR²)",
            "📐 Koşullu Etkiler (Simple Slopes)",
        ]
    )

    with m1_tab_vis:
        show_ci = st.checkbox("Etkileşim grafiğinde %95 güven bandını göster", value=True)
        fig_int = plot_moderation_interaction(results, show_ci=show_ci)
        st.pyplot(fig_int, width="stretch")
        st.download_button(
            "🖼️ Etkileşim Grafiğini PNG İndir (300 DPI)",
            data=figure_to_png_bytes(fig_int),
            file_name="model1_etkilesim_grafigi.png",
            mime="image/png",
        )
        plt.close(fig_int)

        st.markdown("#### İstatistiksel Yol Diyagramı (Model 1)")
        fig_m1_path = plot_moderation_path_diagram(results)
        st.pyplot(fig_m1_path, width="stretch")
        st.download_button(
            "🖼️ Yol Diyagramını PNG İndir (300 DPI)",
            data=figure_to_png_bytes(fig_m1_path),
            file_name="model1_yol_diyagrami.png",
            mime="image/png",
        )
        plt.close(fig_m1_path)

    with m1_tab_reg:
        st.markdown("#### Model Özeti ($R$, $R^2$, $F$, $p$)")
        st.dataframe(
            _format_numeric_styler(results["full_model"]["summary_df"]),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### OLS Regresyon Katsayıları")
        center_badge = (
            "X ve W değişkenleri etkileşim terimi oluşturulmadan önce **ortalama merkezileştirilmiştir**."
            if results["mean_center"]
            else "Değişkenler **orijinal (merkezileştirilmemiş)** ölçekleriyle analize dahil edilmiştir."
        )
        st.caption(center_badge)
        st.dataframe(
            _format_numeric_styler(results["full_model"]["coef_df"]),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### En Yüksek Dereceli Koşulsuz Etkileşim Testi ($\\Delta R^2$)")
        st.dataframe(
            _format_numeric_styler(results["interaction_test_df"]),
            width="stretch",
            hide_index=True,
        )

    with m1_tab_slopes:
        st.markdown(
            f"#### {results['w_col']} Düzeylerine Göre {results['x_col']} → {results['y_col']} Koşullu Etkileri (Simple Slopes)"
        )
        st.dataframe(
            _format_numeric_styler(results["simple_slopes_df"]),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Etkileşim Grafiği Tahmin Edilen $\\hat{Y}$ Değerleri")
        st.dataframe(
            _format_numeric_styler(results["plot_data_df"]),
            width="stretch",
            hide_index=True,
        )
