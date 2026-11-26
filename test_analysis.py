"""
Kapsamlı doğrulama test paketi (pytest).

Doğrulanan bileşenler:
- .csv, .xlsx ve .sav dosya yükleme
- Eksik değer özeti, betimsel istatistikler ve listwise deletion
- Model 4 (Tekli ve Çoklu Paralel Aracılık): OLS c = c' + sum(a*b), Percentile Bootstrap, Sobel Testi, Türkçe yorum
- Model 1 (Düzenleyicilik): Ortalama merkezileştirme değişmezliği, ΔR², Simple Slopes (Aiken & West yeniden merkezileştirme doğrulaması)
- Matplotlib grafikleri ve PNG dışa aktarım
- Biçimlendirilmiş Excel (.xlsx) dışa aktarım
- Streamlit arayüz uçtan uca duman testi (AppTest)
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import pyreadstat
import pytest
import statsmodels.api as sm

from analysis import (
    compute_descriptive_stats,
    compute_missing_summary,
    export_results_to_excel,
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

SAMPLE_CSV = Path(__file__).parent / "sample_data.csv"


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    df, meta = load_dataset(SAMPLE_CSV)
    assert meta["format"] == ".csv"
    assert len(df) == 250
    return df


def test_file_loaders_csv_xlsx_sav(sample_df: pd.DataFrame, tmp_path: Path):
    # Excel (.xlsx)
    xlsx_path = tmp_path / "test_data.xlsx"
    sample_df.to_excel(xlsx_path, index=False, engine="openpyxl")
    df_xl, meta_xl = load_dataset(xlsx_path)
    assert len(df_xl) == len(sample_df)
    assert list(df_xl.columns) == list(sample_df.columns)
    assert meta_xl["format"] == ".xlsx"

    # SPSS (.sav) + değişken etiketleri
    sav_path = tmp_path / "test_data.sav"
    labels = {"Is_Stresi": "Algılanan İş Stresi", "Performans": "Görev Performansı"}
    pyreadstat.write_sav(sample_df, str(sav_path), column_labels=labels)
    df_sav, meta_sav = load_dataset(sav_path)
    assert len(df_sav) == len(sample_df)
    assert meta_sav["format"] == ".sav"
    assert meta_sav["variable_labels"]["Is_Stresi"] == "Algılanan İş Stresi"

    # Dosya-benzeri UploadedFile simülasyonu (.sav)
    class FakeUpload:
        def __init__(self, name: str, raw: bytes):
            self.name = name
            self._raw = raw

        def getvalue(self) -> bytes:
            return self._raw

    fake_sav = FakeUpload("upload.sav", sav_path.read_bytes())
    df_sav2, meta_sav2 = load_dataset(fake_sav)
    assert len(df_sav2) == len(sample_df)
    assert meta_sav2["variable_labels"]["Performans"] == "Görev Performansı"


def test_missing_and_descriptives(sample_df: pd.DataFrame):
    miss = compute_missing_summary(sample_df)
    assert int(miss.loc[miss["Değişken"] == "Is_Stresi", "Eksik N"].iloc[0]) == 3
    assert int(miss.loc[miss["Değişken"] == "Tukenmislik", "Eksik N"].iloc[0]) == 4

    desc = compute_descriptive_stats(sample_df)
    assert "Ortalama" in desc.columns
    assert "Std. Sapma" in desc.columns
    assert int(desc.loc[desc["Değişken"] == "Is_Stresi", "N"].iloc[0]) == 247


def test_model4_single_mediation_and_identities(sample_df: pd.DataFrame):
    res1 = run_mediation_model4(
        df=sample_df,
        x_col="Is_Stresi",
        y_col="Performans",
        m_cols=["Tukenmislik"],
        covariates=["Yas"],
        n_bootstrap=2000,
        random_seed=42,
    )
    # Tekrarlanabilirlik (aynı seed -> birebir aynı Bootstrap sonuçları)
    res2 = run_mediation_model4(
        df=sample_df,
        x_col="Is_Stresi",
        y_col="Performans",
        m_cols=["Tukenmislik"],
        covariates=["Yas"],
        n_bootstrap=2000,
        random_seed=42,
    )
    pd.testing.assert_frame_equal(res1["indirect_df"], res2["indirect_df"])

    # Listwise deletion kontrolü (Is_Stresi: 3, Tukenmislik: 4, Performans: 2 -> benzersiz 9 satır)
    assert res1["data_summary"]["original_n"] == 250
    assert res1["data_summary"]["dropped_n"] == 9
    assert res1["data_summary"]["valid_n"] == 241

    # OLS temel özdeşliği: c == c' + a * b
    p = res1["paths"]
    ab = p["a"]["Tukenmislik"] * p["b"]["Tukenmislik"]
    assert np.isclose(p["c"], p["c_prime"] + ab, atol=1e-10)

    # Sobel testi formül kontrolü
    sobel_row = res1["sobel_df"].iloc[0]
    expected_se = np.sqrt(
        (sobel_row["b"] ** 2) * (sobel_row["SE(a)"] ** 2)
        + (sobel_row["a"] ** 2) * (sobel_row["SE(b)"] ** 2)
    )
    assert np.isclose(sobel_row["Sobel SE"], expected_se, atol=1e-12)
    assert np.isclose(sobel_row["Sobel z"], ab / expected_se, atol=1e-12)

    # Sentetik veride anlamlı aracılık beklenir
    ind_row = res1["indirect_df"].iloc[0]
    assert ind_row["BootLLCI (%2.5)"] < ind_row["BootULCI (%97.5)"]
    assert bool(ind_row["_significant"]) is True
    assert "Aracılık etkisi istatistiksel olarak anlamlıdır" in res1["interpretations"][0]["text"]


def test_model4_parallel_multiple_mediation(sample_df: pd.DataFrame):
    res = run_mediation_model4(
        df=sample_df,
        x_col="Is_Stresi",
        y_col="Performans",
        m_cols=["Tukenmislik", "Is_Doyumu"],
        covariates=["Yas", "Kidem_Yil"],
        n_bootstrap=2000,
        random_seed=123,
    )
    p = res["paths"]
    total_ab = (
        p["a"]["Tukenmislik"] * p["b"]["Tukenmislik"]
        + p["a"]["Is_Doyumu"] * p["b"]["Is_Doyumu"]
    )
    # Çoklu paralel aracılık özdeşliği: c == c' + a1*b1 + a2*b2
    assert np.isclose(p["c"], p["c_prime"] + total_ab, atol=1e-10)
    # TOPLAM + 2 aracı = 3 satır
    assert len(res["indirect_df"]) == 3
    assert res["indirect_df"].iloc[0]["Aracı Yol"] == "TOPLAM DOLAYLI ETKİ"
    assert len(res["sobel_df"]) == 2


def test_model4_nonsignificant_mediation():
    rng = np.random.default_rng(99)
    n = 150
    df_null = pd.DataFrame(
        {
            "X": rng.normal(0, 1, n),
            "M": rng.normal(0, 1, n),
            "Y": rng.normal(0, 1, n),
        }
    )
    res = run_mediation_model4(
        df=df_null, x_col="X", y_col="Y", m_cols=["M"], n_bootstrap=1000, random_seed=7
    )
    assert res["interpretations"][0]["significant"] is False
    assert "Aracılık etkisi istatistiksel olarak anlamlı değildir" in res["interpretations"][0]["text"]


def test_model1_moderation_and_simple_slopes_verification(sample_df: pd.DataFrame):
    res_centered = run_moderation_model1(
        df=sample_df,
        x_col="Is_Stresi",
        y_col="Tukenmislik",
        w_col="Sosyal_Destek",
        covariates=["Yas"],
        mean_center=True,
    )
    res_raw = run_moderation_model1(
        df=sample_df,
        x_col="Is_Stresi",
        y_col="Tukenmislik",
        w_col="Sosyal_Destek",
        covariates=["Yas"],
        mean_center=False,
    )

    # Etkileşim katsayısı b3, ΔR² ve simple slopes ortalama merkezileştirmeden bağımsız (değişmez) olmalıdır!
    assert np.isclose(res_centered["paths"]["b3"], res_raw["paths"]["b3"], atol=1e-10)
    assert np.isclose(res_centered["paths"]["delta_r2"], res_raw["paths"]["delta_r2"], atol=1e-10)
    assert np.allclose(
        res_centered["simple_slopes_df"]["Koşullu Etki (Eğim)"].to_numpy(),
        res_raw["simple_slopes_df"]["Koşullu Etki (Eğim)"].to_numpy(),
        atol=1e-10,
    )
    assert np.allclose(
        res_centered["simple_slopes_df"]["SE"].to_numpy(),
        res_raw["simple_slopes_df"]["SE"].to_numpy(),
        atol=1e-10,
    )

    # Aiken & West yeniden merkezileştirme yöntemiyle Simple Slopes kesin doğrulaması
    clean = sample_df[["Is_Stresi", "Tukenmislik", "Sosyal_Destek", "Yas"]].dropna()
    w_mean = clean["Sosyal_Destek"].mean()
    w_sd = clean["Sosyal_Destek"].std(ddof=1)
    for idx, shift in enumerate([w_mean - w_sd, w_mean, w_mean + w_sd]):
        w_shifted = clean["Sosyal_Destek"] - shift
        X_check = sm.add_constant(
            pd.DataFrame(
                {
                    "X": clean["Is_Stresi"],
                    "W_s": w_shifted,
                    "XW": clean["Is_Stresi"] * w_shifted,
                    "Yas": clean["Yas"],
                }
            )
        )
        m_check = sm.OLS(clean["Tukenmislik"], X_check).fit()
        row = res_centered["simple_slopes_df"].iloc[idx]
        assert np.isclose(row["Koşullu Etki (Eğim)"], m_check.params["X"], atol=1e-10)
        assert np.isclose(row["SE"], m_check.bse["X"], atol=1e-10)
        assert np.isclose(row["t"], m_check.tvalues["X"], atol=1e-10)
        assert np.isclose(row["p"], m_check.pvalues["X"], atol=1e-10)

    # Sentetik veride düzenleyicilik anlamlı olmalı
    assert res_centered["interaction_significant"] is True
    assert "düzenleyicilik (moderasyon) etkisi istatistiksel olarak anlamlıdır" in res_centered["interpretation"]


def test_plotting_and_excel_exports(sample_df: pd.DataFrame):
    # Model 4 grafikleri ve Excel
    res4 = run_mediation_model4(
        df=sample_df,
        x_col="Is_Stresi",
        y_col="Performans",
        m_cols=["Tukenmislik", "Is_Doyumu"],
        n_bootstrap=500,
        random_seed=42,
    )
    fig4_path = plot_mediation_path_diagram(res4)
    png4_path = figure_to_png_bytes(fig4_path, dpi=120)
    plt.close(fig4_path)
    assert png4_path[:8] == b"\x89PNG\r\n\x1a\n"

    fig4_boot = plot_bootstrap_distribution(res4)
    png4_boot = figure_to_png_bytes(fig4_boot, dpi=120)
    plt.close(fig4_boot)
    assert png4_boot[:8] == b"\x89PNG\r\n\x1a\n"

    xlsx4_bytes = export_results_to_excel(res4, compute_missing_summary(sample_df))
    wb4 = openpyxl.load_workbook(io.BytesIO(xlsx4_bytes))
    assert set(wb4.sheetnames) == {
        "Ozet_ve_Yorum",
        "Betimsel_Istatistikler",
        "Model_Ozetleri",
        "Regresyon_Katsayilari",
        "Dolayli_Etki_Bootstrap",
        "Sobel_Testi",
    }

    # Model 1 grafikleri ve Excel
    res1 = run_moderation_model1(
        df=sample_df,
        x_col="Is_Stresi",
        y_col="Tukenmislik",
        w_col="Sosyal_Destek",
        mean_center=True,
    )
    fig1_int = plot_moderation_interaction(res1, show_ci=True)
    png1_int = figure_to_png_bytes(fig1_int, dpi=120)
    plt.close(fig1_int)
    assert png1_int[:8] == b"\x89PNG\r\n\x1a\n"

    fig1_path = plot_moderation_path_diagram(res1)
    png1_path = figure_to_png_bytes(fig1_path, dpi=120)
    plt.close(fig1_path)
    assert png1_path[:8] == b"\x89PNG\r\n\x1a\n"

    xlsx1_bytes = export_results_to_excel(res1, compute_missing_summary(sample_df))
    wb1 = openpyxl.load_workbook(io.BytesIO(xlsx1_bytes))
    assert set(wb1.sheetnames) == {
        "Ozet_ve_Yorum",
        "Betimsel_Istatistikler",
        "Model_Ozeti_ve_Etkilesim",
        "Regresyon_Katsayilari",
        "Basit_Egimler",
    }


def test_streamlit_app_smoke():
    """Streamlit AppTest ile app.py dosyasının Model 4 ve Model 1 akışlarını hatasız çalıştırdığını doğrular."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).parent / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Varsayılan Model 4 analizini çalıştır
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.session_state["last_results"]["model_type"] == "Model 4"

    # Model 1'e geç ve çalıştır
    at.sidebar.radio[0].set_value("Model 1 — Düzenleyicilik (Moderation)").run()
    assert not at.exception
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.session_state["last_results"]["model_type"] == "Model 1"
