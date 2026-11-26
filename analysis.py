"""
İstatistiksel analiz motoru (Hayes PROCESS Macro Model 4 ve Model 1 alternatifi).

Bu modül şu işlevleri sağlar:
- .sav (SPSS), .xlsx (Excel) ve .csv dosyalarını yükleme
- Eksik değer özeti, betimsel istatistikler ve listwise deletion
- Model 4 (Basit ve Çoklu Paralel Aracılık) OLS regresyonları, Percentile Bootstrap ve Sobel Testi
- Model 1 (Düzenleyicilik) OLS regresyonu, ortalama merkezileştirme, ΔR² testi ve Simple Slopes
- Hayes PROCESS benzeri akademik Türkçe raporlama ve biçimlendirilmiş Excel (.xlsx) dışa aktarımı
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy import stats


# ---------------------------------------------------------------------------
# Yardımcı Biçimlendirme Fonksiyonları
# ---------------------------------------------------------------------------

def format_p_stars(p_value: float) -> str:
    """p değerine göre akademik anlamlılık yıldızlarını döndürür."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_p_text(p_value: float) -> str:
    """p değerini akademik gösterimde biçimlendirir."""
    if pd.isna(p_value):
        return "NaN"
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.4f}"


def format_p_clause(p_value: float) -> str:
    """APA uyumlu p ifadesi döndürür (örn. 'p < .001' veya 'p = 0.0312')."""
    if pd.isna(p_value):
        return "p = NaN"
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.4f}"


# ---------------------------------------------------------------------------
# Veri Yükleme ve Ön İşleme
# ---------------------------------------------------------------------------

def get_excel_sheet_names(file_source: Any) -> list[str]:
    """Excel dosyasındaki sayfa (sheet) adlarını döndürür."""
    if isinstance(file_source, (str, Path)):
        with pd.ExcelFile(file_source, engine="openpyxl") as xl:
            return list(xl.sheet_names)
    data_bytes = file_source.getvalue()
    with pd.ExcelFile(io.BytesIO(data_bytes), engine="openpyxl") as xl:
        return list(xl.sheet_names)


def load_dataset(
    file_source: Any,
    filename: str | None = None,
    sheet_name: str | int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    .sav, .xlsx veya .csv dosyasını DataFrame olarak yükler.

    Parametreler
    ------------
    file_source : str | Path | UploadedFile | BytesIO
        Dosya yolu veya Streamlit yüklenen dosya nesnesi.
    filename : str | None
        Dosya adı (akış nesnesi verildiğinde uzantı tespiti için kullanılır).
    sheet_name : str | int
        Excel dosyaları için okunacak sayfa adı veya indeksi.

    Döndürür
    --------
    tuple[pd.DataFrame, dict]
        Yüklenen veri çerçevesi ve dosya meta bilgileri (örn. SPSS değişken etiketleri).
    """
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        ext = path.suffix.lower()
        name = path.name
        is_path = True
    else:
        name = filename or getattr(file_source, "name", "")
        ext = Path(name).suffix.lower()
        is_path = False

    meta_info: dict[str, Any] = {
        "filename": name,
        "format": ext,
        "variable_labels": {},
        "sheet_names": [],
    }

    if ext == ".sav":
        if is_path:
            df, spss_meta = pyreadstat.read_sav(str(file_source))
        else:
            raw_bytes = file_source.getvalue()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name
                df, spss_meta = pyreadstat.read_sav(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        if getattr(spss_meta, "column_names_to_labels", None):
            meta_info["variable_labels"] = {
                k: v for k, v in spss_meta.column_names_to_labels.items() if v
            }
        return df, meta_info

    if ext in {".xlsx", ".xls"}:
        meta_info["sheet_names"] = get_excel_sheet_names(file_source)
        if is_path:
            df = pd.read_excel(file_source, sheet_name=sheet_name, engine="openpyxl")
        else:
            df = pd.read_excel(
                io.BytesIO(file_source.getvalue()),
                sheet_name=sheet_name,
                engine="openpyxl",
            )
        return df, meta_info

    if ext == ".csv":
        encodings = ["utf-8-sig", "utf-8", "cp1254", "latin-1"]
        last_err: Exception | None = None
        for enc in encodings:
            try:
                if is_path:
                    df = pd.read_csv(file_source, encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(
                        io.BytesIO(file_source.getvalue()),
                        encoding=enc,
                        sep=None,
                        engine="python",
                    )
                return df, meta_info
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise ValueError(f"CSV dosyası okunamadı: {last_err}")

    raise ValueError(
        f"Desteklenmeyen dosya formatı: '{ext}'. Lütfen .sav, .xlsx veya .csv yükleyin."
    )


def compute_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Her değişken için eksik değer sayısı ve yüzdesini hesaplar."""
    total = len(df)
    rows = []
    for col in df.columns:
        missing_n = int(df[col].isna().sum())
        valid_n = total - missing_n
        missing_pct = (missing_n / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "Değişken": col,
                "Veri Tipi": str(df[col].dtype),
                "Geçerli N": valid_n,
                "Eksik N": missing_n,
                "Eksik (%)": round(missing_pct, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_descriptive_stats(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Sayısal değişkenler için temel betimsel istatistik tablosu döndürür."""
    if columns is not None:
        subset = df[columns].apply(pd.to_numeric, errors="coerce")
    else:
        subset = df.select_dtypes(include=[np.number])

    rows = []
    for col in subset.columns:
        s = subset[col].dropna()
        if len(s) == 0:
            continue
        rows.append(
            {
                "Değişken": col,
                "N": int(len(s)),
                "Ortalama": float(s.mean()),
                "Std. Sapma": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "Min": float(s.min()),
                "%25": float(s.quantile(0.25)),
                "Medyan": float(s.median()),
                "%75": float(s.quantile(0.75)),
                "Max": float(s.max()),
                "Çarpıklık": float(s.skew()) if len(s) > 2 else np.nan,
                "Basıklık": float(s.kurt()) if len(s) > 3 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def prepare_analysis_data(
    df: pd.DataFrame, required_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Seçilen değişkenleri sayısala dönüştürür ve satır bazlı silme (listwise deletion) uygular.
    """
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Veri setinde bulunmayan değişkenler: {', '.join(missing_cols)}")

    original_n = len(df)
    work = df[required_cols].copy()
    non_numeric_converted: list[str] = []

    for col in required_cols:
        before_na = int(work[col].isna().sum())
        work[col] = pd.to_numeric(work[col], errors="coerce")
        after_na = int(work[col].isna().sum())
        if after_na > before_na:
            non_numeric_converted.append(col)

    clean_df = work.dropna(axis=0, how="any").reset_index(drop=True)
    valid_n = len(clean_df)
    dropped_n = original_n - valid_n
    dropped_pct = (dropped_n / original_n * 100.0) if original_n > 0 else 0.0

    min_required = len(required_cols) + 3
    if valid_n < min_required:
        raise ValueError(
            f"Satır bazlı silme (listwise deletion) sonrası yeterli gözlem kalmadı "
            f"(Geçerli N = {valid_n}, en az {min_required} gözlem gerekli)."
        )

    for col in required_cols:
        if float(clean_df[col].std(ddof=1)) == 0.0:
            raise ValueError(
                f"'{col}' değişkeninin varyansı sıfır (tüm değerler aynı). Regresyon hesaplanamaz."
            )

    summary = {
        "original_n": original_n,
        "valid_n": valid_n,
        "dropped_n": dropped_n,
        "dropped_pct": round(dropped_pct, 2),
        "non_numeric_converted": non_numeric_converted,
    }
    return clean_df, summary


# ---------------------------------------------------------------------------
# OLS Yardımcıları
# ---------------------------------------------------------------------------

def _fit_ols_table(
    y: pd.Series,
    X_df: pd.DataFrame,
    outcome_name: str,
    model_title: str,
    conf_level: float = 95.0,
) -> dict[str, Any]:
    """statsmodels OLS modelini kurar ve Hayes PROCESS biçiminde özet tablolar döndürür."""
    alpha = (100.0 - conf_level) / 100.0
    X_const = sm.add_constant(X_df, has_constant="add")
    model = sm.OLS(y, X_const).fit()

    ci = model.conf_int(alpha=alpha)
    coef_rows = []
    for var_name in model.params.index:
        display_name = "Sabit (Constant)" if var_name == "const" else var_name
        p_val = float(model.pvalues[var_name])
        coef_rows.append(
            {
                "Değişken": display_name,
                "_raw_name": var_name,
                "B": float(model.params[var_name]),
                "SE": float(model.bse[var_name]),
                "t": float(model.tvalues[var_name]),
                "p": p_val,
                f"LLCI (%{conf_level:g})": float(ci.loc[var_name, 0]),
                f"ULCI (%{conf_level:g})": float(ci.loc[var_name, 1]),
                "Anlamlılık": format_p_stars(p_val),
            }
        )
    coef_df = pd.DataFrame(coef_rows)

    r_sq = float(model.rsquared)
    r_val = float(np.sqrt(max(r_sq, 0.0)))
    mse = float(model.mse_resid)
    f_val = float(model.fvalue) if model.fvalue is not None else np.nan
    df1 = float(model.df_model)
    df2 = float(model.df_resid)
    f_pval = float(model.f_pvalue) if model.f_pvalue is not None else np.nan

    summary_df = pd.DataFrame(
        [
            {
                "Model": model_title,
                "Bağımlı Değişken": outcome_name,
                "R": r_val,
                "R²": r_sq,
                "Düz. R²": float(model.rsquared_adj),
                "MSE": mse,
                "F": f_val,
                "df1": int(round(df1)),
                "df2": int(round(df2)),
                "p": f_pval,
            }
        ]
    )

    return {
        "model": model,
        "title": model_title,
        "outcome": outcome_name,
        "summary_df": summary_df,
        "coef_df": coef_df,
    }


def _fast_ols_solve(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Bootstrap döngüsü için hızlı ve kararlı OLS çözücüsü (normal denklem + lstsq yedeği)."""
    xtx = X.T @ X
    xty = X.T @ Y
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return coef


# ---------------------------------------------------------------------------
# MODEL 4: ARACILIK ANALİZİ (BASİT VE ÇOKLU PARALEL MEDIATION)
# ---------------------------------------------------------------------------

def run_mediation_model4(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    m_cols: list[str],
    covariates: list[str] | None = None,
    n_bootstrap: int = 5000,
    conf_level: float = 95.0,
    random_seed: int | None = 42,
) -> dict[str, Any]:
    """
    Hayes PROCESS Model 4 (Basit ve Çoklu Paralel Aracılık) analizini çalıştırır.

    Yollar:
      - a_i yolları: M_i = i_{M_i} + a_i * X + Covariates + e
      - b_i ve c' yolu: Y = i_Y + c' * X + sum(b_i * M_i) + Covariates + e
      - Toplam etki c yolu: Y = i_{Y*} + c * X + Covariates + e
      - Dolaylı etkiler: a_i * b_i (Percentile Bootstrap + Klasik Sobel Testi)
    """
    covariates = list(covariates) if covariates else []
    m_cols = list(m_cols)

    if not m_cols:
        raise ValueError("En az bir aracı değişken (M) seçmelisiniz.")

    all_vars = [x_col, y_col] + m_cols + covariates
    if len(set(all_vars)) != len(all_vars):
        raise ValueError(
            "X, Y, aracı değişkenler (M) ve kontrol değişkenleri (Covariates) birbirinden farklı olmalıdır."
        )

    if n_bootstrap < 100:
        raise ValueError("Bootstrap tekrar sayısı en az 100 olmalıdır.")

    clean_df, data_summary = prepare_analysis_data(df, all_vars)
    n_obs = len(clean_df)
    k = len(m_cols)
    alpha = (100.0 - conf_level) / 100.0
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)

    # 1) Her aracı değişken M_i için a_i regresyonları
    a_models: dict[str, dict[str, Any]] = {}
    a_coefs: dict[str, float] = {}
    a_ses: dict[str, float] = {}
    a_pvals: dict[str, float] = {}

    X_for_a = clean_df[[x_col] + covariates]
    for idx, m_col in enumerate(m_cols, start=1):
        label = f"Aracı Model (a{idx}: {x_col} → {m_col})" if k > 1 else f"Aracı Model (a yolu: {x_col} → {m_col})"
        res_a = _fit_ols_table(
            y=clean_df[m_col],
            X_df=X_for_a,
            outcome_name=m_col,
            model_title=label,
            conf_level=conf_level,
        )
        a_models[m_col] = res_a
        a_coefs[m_col] = float(res_a["model"].params[x_col])
        a_ses[m_col] = float(res_a["model"].bse[x_col])
        a_pvals[m_col] = float(res_a["model"].pvalues[x_col])

    # 2) Y sonuç modeli (c' ve b_i yolları)
    X_for_y = clean_df[[x_col] + m_cols + covariates]
    y_model_res = _fit_ols_table(
        y=clean_df[y_col],
        X_df=X_for_y,
        outcome_name=y_col,
        model_title=f"Sonuç Modeli (b ve c' yolları: → {y_col})",
        conf_level=conf_level,
    )
    y_ols = y_model_res["model"]
    c_prime = float(y_ols.params[x_col])
    c_prime_se = float(y_ols.bse[x_col])
    c_prime_t = float(y_ols.tvalues[x_col])
    c_prime_p = float(y_ols.pvalues[x_col])
    c_prime_ci = y_ols.conf_int(alpha=alpha).loc[x_col]

    b_coefs = {m: float(y_ols.params[m]) for m in m_cols}
    b_ses = {m: float(y_ols.bse[m]) for m in m_cols}
    b_pvals = {m: float(y_ols.pvalues[m]) for m in m_cols}

    # 3) Toplam etki modeli (c yolu)
    total_model_res = _fit_ols_table(
        y=clean_df[y_col],
        X_df=X_for_a,
        outcome_name=y_col,
        model_title=f"Toplam Etki Modeli (c yolu: {x_col} → {y_col})",
        conf_level=conf_level,
    )
    tot_ols = total_model_res["model"]
    c_total = float(tot_ols.params[x_col])
    c_total_se = float(tot_ols.bse[x_col])
    c_total_t = float(tot_ols.tvalues[x_col])
    c_total_p = float(tot_ols.pvalues[x_col])
    c_total_ci = tot_ols.conf_int(alpha=alpha).loc[x_col]

    # 4) Percentile Bootstrap (NumPy ile hızlandırılmış vektörel çözüm)
    rng = np.random.default_rng(random_seed)
    ones = np.ones((n_obs, 1), dtype=np.float64)
    x_arr = clean_df[[x_col]].to_numpy(dtype=np.float64)
    m_arr = clean_df[m_cols].to_numpy(dtype=np.float64)
    y_arr = clean_df[y_col].to_numpy(dtype=np.float64)

    if covariates:
        cov_arr = clean_df[covariates].to_numpy(dtype=np.float64)
        Xa_mat = np.hstack([ones, x_arr, cov_arr])
        Xy_mat = np.hstack([ones, x_arr, m_arr, cov_arr])
    else:
        Xa_mat = np.hstack([ones, x_arr])
        Xy_mat = np.hstack([ones, x_arr, m_arr])

    boot_indirect = np.empty((n_bootstrap, k), dtype=np.float64)
    for r in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=n_obs)
        # Tüm M_i için a_i katsayılarını tek seferde çöz
        coef_a_mat = _fast_ols_solve(Xa_mat[idx], m_arr[idx])  # şekil: (p_a, k)
        a_boot = coef_a_mat[1, :]  # X sütunu indeks 1
        coef_y_vec = _fast_ols_solve(Xy_mat[idx], y_arr[idx])  # şekil: (p_y,)
        b_boot = coef_y_vec[2 : 2 + k]  # M sütunları indeks 2..2+k
        boot_indirect[r, :] = a_boot * b_boot

    boot_total_indirect = boot_indirect.sum(axis=1)

    # 5) Dolaylı Etki Tablosu (Bootstrap)
    sd_x = float(clean_df[x_col].std(ddof=1))
    sd_y = float(clean_df[y_col].std(ddof=1))
    std_factor = sd_x / sd_y if sd_y > 0 else np.nan

    indirect_rows = []
    sobel_rows = []
    interpretations: list[dict[str, Any]] = []

    if k > 1:
        tot_ind_effect = float(sum(a_coefs[m] * b_coefs[m] for m in m_cols))
        tot_boot_se = float(np.std(boot_total_indirect, ddof=1))
        tot_llci = float(np.percentile(boot_total_indirect, low_q))
        tot_ulci = float(np.percentile(boot_total_indirect, high_q))
        tot_sig = not (tot_llci <= 0.0 <= tot_ulci)
        indirect_rows.append(
            {
                "Aracı Yol": "TOPLAM DOLAYLI ETKİ",
                "Etki (a×b)": tot_ind_effect,
                "BootSE": tot_boot_se,
                f"BootLLCI (%{low_q:g})": tot_llci,
                f"BootULCI (%{high_q:g})": tot_ulci,
                "Tam Std. Etki": tot_ind_effect * std_factor,
                "Anlamlı mı?": "Evet (Anlamlı)" if tot_sig else "Hayır (Anlamlı Değil)",
                "_significant": tot_sig,
            }
        )

    for i, m_col in enumerate(m_cols):
        ab = a_coefs[m_col] * b_coefs[m_col]
        b_se = float(np.std(boot_indirect[:, i], ddof=1))
        b_llci = float(np.percentile(boot_indirect[:, i], low_q))
        b_ulci = float(np.percentile(boot_indirect[:, i], high_q))
        is_sig = not (b_llci <= 0.0 <= b_ulci)

        indirect_rows.append(
            {
                "Aracı Yol": f"{x_col} → {m_col} → {y_col}",
                "Etki (a×b)": ab,
                "BootSE": b_se,
                f"BootLLCI (%{low_q:g})": b_llci,
                f"BootULCI (%{high_q:g})": b_ulci,
                "Tam Std. Etki": ab * std_factor,
                "Anlamlı mı?": "Evet (Anlamlı)" if is_sig else "Hayır (Anlamlı Değil)",
                "_significant": is_sig,
            }
        )

        # Klasik Sobel Testi: SE = sqrt(b^2 * SE_a^2 + a^2 * SE_b^2)
        sobel_se = float(
            np.sqrt((b_coefs[m_col] ** 2) * (a_ses[m_col] ** 2) + (a_coefs[m_col] ** 2) * (b_ses[m_col] ** 2))
        )
        sobel_z = ab / sobel_se if sobel_se > 0 else np.nan
        sobel_p = float(2.0 * stats.norm.sf(abs(sobel_z))) if not np.isnan(sobel_z) else np.nan
        sobel_rows.append(
            {
                "Aracı Değişken (M)": m_col,
                "a": a_coefs[m_col],
                "SE(a)": a_ses[m_col],
                "b": b_coefs[m_col],
                "SE(b)": b_ses[m_col],
                "Dolaylı Etki (a×b)": ab,
                "Sobel SE": sobel_se,
                "Sobel z": sobel_z,
                "p": sobel_p,
                "Anlamlılık": format_p_stars(sobel_p),
            }
        )

        # Akademik Türkçe Yorum
        if is_sig:
            if c_prime_p < 0.05:
                mediation_type = (
                    f"Doğrudan etki de istatistiksel olarak anlamlı olduğundan "
                    f"(c' = {c_prime:.4f}, {format_p_clause(c_prime_p)}), "
                    f"bu bulgu **kısmi aracılık** (partial mediation) yapısına işaret etmektedir."
                )
            else:
                mediation_type = (
                    f"Aracı değişken modele eklendiğinde doğrudan etki istatistiksel olarak anlamlılığını "
                    f"yitirdiğinden (c' = {c_prime:.4f}, {format_p_clause(c_prime_p)}), "
                    f"bu bulgu **tam aracılık** (full/indirect-only mediation) yapısına işaret etmektedir."
                )
            text = (
                f"**{x_col} → {m_col} → {y_col}:** %{conf_level:g} Bootstrap güven aralığı "
                f"[BootLLCI = {b_llci:.4f}, BootULCI = {b_ulci:.4f}] sıfır (0) değerini **içermediği** için "
                f"**Aracılık etkisi istatistiksel olarak anlamlıdır** "
                f"(a×b = {ab:.4f}, BootSE = {b_se:.4f}). {mediation_type}"
            )
        else:
            text = (
                f"**{x_col} → {m_col} → {y_col}:** %{conf_level:g} Bootstrap güven aralığı "
                f"[BootLLCI = {b_llci:.4f}, BootULCI = {b_ulci:.4f}] sıfır (0) değerini **içerdiği** için "
                f"**Aracılık etkisi istatistiksel olarak anlamlı değildir** "
                f"(a×b = {ab:.4f}, BootSE = {b_se:.4f})."
            )
        interpretations.append(
            {
                "mediator": m_col,
                "significant": is_sig,
                "text": text,
            }
        )

    indirect_df = pd.DataFrame(indirect_rows)
    sobel_df = pd.DataFrame(sobel_rows)

    # Toplam ve Doğrudan Etki Özet Tablosu
    total_direct_df = pd.DataFrame(
        [
            {
                "Etki Türü": f"Toplam Etki (c: {x_col} → {y_col})",
                "Etki (B)": c_total,
                "SE": c_total_se,
                "t": c_total_t,
                "p": c_total_p,
                f"LLCI (%{conf_level:g})": float(c_total_ci[0]),
                f"ULCI (%{conf_level:g})": float(c_total_ci[1]),
                "Anlamlılık": format_p_stars(c_total_p),
            },
            {
                "Etki Türü": f"Doğrudan Etki (c': {x_col} → {y_col})",
                "Etki (B)": c_prime,
                "SE": c_prime_se,
                "t": c_prime_t,
                "p": c_prime_p,
                f"LLCI (%{conf_level:g})": float(c_prime_ci[0]),
                f"ULCI (%{conf_level:g})": float(c_prime_ci[1]),
                "Anlamlılık": format_p_stars(c_prime_p),
            },
        ]
    )

    all_summaries_df = pd.concat(
        [a_models[m]["summary_df"] for m in m_cols]
        + [y_model_res["summary_df"], total_model_res["summary_df"]],
        ignore_index=True,
    )

    paths = {
        "x": x_col,
        "y": y_col,
        "mediators": m_cols,
        "a": a_coefs,
        "a_p": a_pvals,
        "b": b_coefs,
        "b_p": b_pvals,
        "c": c_total,
        "c_p": c_total_p,
        "c_prime": c_prime,
        "c_prime_p": c_prime_p,
    }

    return {
        "model_type": "Model 4",
        "x_col": x_col,
        "y_col": y_col,
        "m_cols": m_cols,
        "covariates": covariates,
        "n_bootstrap": n_bootstrap,
        "conf_level": conf_level,
        "random_seed": random_seed,
        "data_summary": data_summary,
        "descriptives_df": compute_descriptive_stats(clean_df, all_vars),
        "a_models": a_models,
        "y_model": y_model_res,
        "total_model": total_model_res,
        "all_summaries_df": all_summaries_df,
        "total_direct_df": total_direct_df,
        "indirect_df": indirect_df,
        "sobel_df": sobel_df,
        "paths": paths,
        "interpretations": interpretations,
        "boot_distributions": {
            m: boot_indirect[:, i] for i, m in enumerate(m_cols)
        },
    }


# ---------------------------------------------------------------------------
# MODEL 1: DÜZENLEYİCİLİK ANALİZİ (MODERATION)
# ---------------------------------------------------------------------------

def run_moderation_model1(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    w_col: str,
    covariates: list[str] | None = None,
    mean_center: bool = True,
    conf_level: float = 95.0,
) -> dict[str, Any]:
    """
    Hayes PROCESS Model 1 (Düzenleyicilik / Moderation) analizini çalıştırır.

    - İsteğe bağlı ortalama merkezileştirme (mean-centering)
    - Y = b0 + b1*X + b2*W + b3*(X*W) + Covariates + e
    - Etkileşim terimi için ΔR² ve F testi
    - Düzenleyici değişkenin -1 SD, Ortalama ve +1 SD düzeylerinde koşullu etkiler (Simple Slopes)
    - Etkileşim grafiği için tahmin edilen Y değerleri
    """
    covariates = list(covariates) if covariates else []
    all_vars = [x_col, y_col, w_col] + covariates
    if len(set(all_vars)) != len(all_vars):
        raise ValueError(
            "X, Y, düzenleyici değişken (W) ve kontrol değişkenleri (Covariates) birbirinden farklı olmalıdır."
        )

    clean_df, data_summary = prepare_analysis_data(df, all_vars)
    alpha = (100.0 - conf_level) / 100.0

    x_mean = float(clean_df[x_col].mean())
    x_sd = float(clean_df[x_col].std(ddof=1))
    w_mean = float(clean_df[w_col].mean())
    w_sd = float(clean_df[w_col].std(ddof=1))

    work = clean_df.copy()
    if mean_center:
        x_used = work[x_col] - x_mean
        w_used = work[w_col] - w_mean
    else:
        x_used = work[x_col].copy()
        w_used = work[w_col].copy()

    int_col = f"{x_col} × {w_col}"
    X_full = pd.DataFrame(
        {
            x_col: x_used,
            w_col: w_used,
            int_col: x_used * w_used,
        }
    )
    for cov in covariates:
        X_full[cov] = work[cov]

    X_reduced = X_full.drop(columns=[int_col])

    full_res = _fit_ols_table(
        y=work[y_col],
        X_df=X_full,
        outcome_name=y_col,
        model_title=f"Düzenleyicilik Modeli (Model 1: {x_col} × {w_col} → {y_col})",
        conf_level=conf_level,
    )
    reduced_res = _fit_ols_table(
        y=work[y_col],
        X_df=X_reduced,
        outcome_name=y_col,
        model_title="Etkileşimsiz Model (Temel Etkiler)",
        conf_level=conf_level,
    )

    ols_full = full_res["model"]
    ols_red = reduced_res["model"]

    b0 = float(ols_full.params["const"])
    b1 = float(ols_full.params[x_col])
    b2 = float(ols_full.params[w_col])
    b3 = float(ols_full.params[int_col])
    b3_se = float(ols_full.bse[int_col])
    b3_t = float(ols_full.tvalues[int_col])
    b3_p = float(ols_full.pvalues[int_col])
    b3_ci = ols_full.conf_int(alpha=alpha).loc[int_col]
    b3_llci = float(b3_ci[0])
    b3_ulci = float(b3_ci[1])

    # Etkileşim teriminin ΔR² (R2-chng) testi
    delta_r2 = float(ols_full.rsquared - ols_red.rsquared)
    int_f = float(b3_t ** 2)
    df1 = 1
    df2 = int(round(ols_full.df_resid))

    interaction_test_df = pd.DataFrame(
        [
            {
                "Etkileşim Terimi": int_col,
                "ΔR² (R²-chng)": delta_r2,
                "F": int_f,
                "df1": df1,
                "df2": df2,
                "p": b3_p,
                "Anlamlılık": format_p_stars(b3_p),
            }
        ]
    )

    # Koşullu Etkiler (Simple Slopes): W = -1 SD, Mean, +1 SD
    cov_mat = ols_full.cov_params()
    var_b1 = float(cov_mat.loc[x_col, x_col])
    var_b3 = float(cov_mat.loc[int_col, int_col])
    cov_b1_b3 = float(cov_mat.loc[x_col, int_col])
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=ols_full.df_resid))

    w_levels = [
        ("Düşük (-1 SD)", w_mean - w_sd, -w_sd if mean_center else (w_mean - w_sd)),
        ("Ortalama (Mean)", w_mean, 0.0 if mean_center else w_mean),
        ("Yüksek (+1 SD)", w_mean + w_sd, +w_sd if mean_center else (w_mean + w_sd)),
    ]

    slopes_rows = []
    for label, w_orig, w_val in w_levels:
        slope = b1 + b3 * w_val
        var_slope = var_b1 + (w_val ** 2) * var_b3 + 2.0 * w_val * cov_b1_b3
        se_slope = float(np.sqrt(max(var_slope, 0.0)))
        t_val = slope / se_slope if se_slope > 0 else np.nan
        p_val = float(2.0 * stats.t.sf(abs(t_val), df=ols_full.df_resid)) if not np.isnan(t_val) else np.nan
        llci = slope - t_crit * se_slope
        ulci = slope + t_crit * se_slope

        slopes_rows.append(
            {
                "Düzey (W)": label,
                f"{w_col} (Orijinal)": w_orig,
                f"{w_col} (Model)": w_val,
                "Koşullu Etki (Eğim)": slope,
                "SE": se_slope,
                "t": t_val,
                "p": p_val,
                f"LLCI (%{conf_level:g})": llci,
                f"ULCI (%{conf_level:g})": ulci,
                "Anlamlılık": format_p_stars(p_val),
            }
        )
    simple_slopes_df = pd.DataFrame(slopes_rows)

    # Etkileşim Grafiği için Tahmin Edilen Y Değerleri (Kontrol değişkenleri ortalamada sabit)
    cov_means = {cov: float(work[cov].mean()) for cov in covariates}
    x_levels = [
        ("Düşük (-1 SD)", x_mean - x_sd, -x_sd if mean_center else (x_mean - x_sd)),
        ("Ortalama (Mean)", x_mean, 0.0 if mean_center else x_mean),
        ("Yüksek (+1 SD)", x_mean + x_sd, +x_sd if mean_center else (x_mean + x_sd)),
    ]

    pred_rows = []
    for w_label, w_orig, w_val in w_levels:
        for x_label, x_orig, x_val in x_levels:
            row_dict: dict[str, float] = {
                "const": 1.0,
                x_col: x_val,
                w_col: w_val,
                int_col: x_val * w_val,
            }
            row_dict.update(cov_means)
            exog = pd.DataFrame([row_dict])[ols_full.params.index]
            pred_summary = ols_full.get_prediction(exog).summary_frame(alpha=alpha)
            pred_rows.append(
                {
                    "W Düzeyi": w_label,
                    f"{w_col} Değeri": w_orig,
                    "X Düzeyi": x_label,
                    f"{x_col} Değeri": x_orig,
                    f"Tahmin Edilen {y_col}": float(pred_summary["mean"].iloc[0]),
                    "SE (Tahmin)": float(pred_summary["mean_se"].iloc[0]),
                    "Alt Sınır": float(pred_summary["mean_ci_lower"].iloc[0]),
                    "Üst Sınır": float(pred_summary["mean_ci_upper"].iloc[0]),
                }
            )
    plot_data_df = pd.DataFrame(pred_rows)

    # Akademik Türkçe Yorum
    interaction_sig = (b3_p < alpha) and not (b3_llci <= 0.0 <= b3_ulci)
    if interaction_sig:
        main_interp = (
            f"**{int_col}** etkileşim terimi istatistiksel olarak anlamlıdır "
            f"(B = {b3:.4f}, SE = {b3_se:.4f}, t = {b3_t:.3f}, {format_p_clause(b3_p)}, "
            f"%{conf_level:g} GA [{b3_llci:.4f}, {b3_ulci:.4f}], ΔR² = {delta_r2:.4f}). "
            f"Güven aralığı sıfır (0) değerini içermediğinden **{w_col} değişkeninin düzenleyicilik (moderasyon) "
            f"etkisi istatistiksel olarak anlamlıdır**."
        )
    else:
        main_interp = (
            f"**{int_col}** etkileşim terimi istatistiksel olarak anlamlı değildir "
            f"(B = {b3:.4f}, SE = {b3_se:.4f}, t = {b3_t:.3f}, {format_p_clause(b3_p)}, "
            f"%{conf_level:g} GA [{b3_llci:.4f}, {b3_ulci:.4f}], ΔR² = {delta_r2:.4f}). "
            f"Güven aralığı sıfır (0) değerini içerdiğinden **{w_col} değişkeninin düzenleyicilik (moderasyon) "
            f"etkisi istatistiksel olarak anlamlı değildir**."
        )

    slope_notes = []
    for _, row in simple_slopes_df.iterrows():
        sig_txt = "anlamlıdır" if row["p"] < alpha else "anlamlı değildir"
        slope_notes.append(
            f"- **{row['Düzey (W)']}** ({w_col} = {row[f'{w_col} (Orijinal)']:.2f}): "
            f"Eğim B = {row['Koşullu Etki (Eğim)']:.4f}, SE = {row['SE']:.4f}, "
            f"t = {row['t']:.3f}, {format_p_clause(row['p'])} ({sig_txt})."
        )

    paths = {
        "x": x_col,
        "y": y_col,
        "w": w_col,
        "int_col": int_col,
        "b1": b1,
        "b1_p": float(ols_full.pvalues[x_col]),
        "b2": b2,
        "b2_p": float(ols_full.pvalues[w_col]),
        "b3": b3,
        "b3_p": b3_p,
        "r2": float(ols_full.rsquared),
        "delta_r2": delta_r2,
    }

    return {
        "model_type": "Model 1",
        "x_col": x_col,
        "y_col": y_col,
        "w_col": w_col,
        "int_col": int_col,
        "covariates": covariates,
        "mean_center": mean_center,
        "conf_level": conf_level,
        "data_summary": data_summary,
        "descriptives_df": compute_descriptive_stats(clean_df, all_vars),
        "full_model": full_res,
        "reduced_model": reduced_res,
        "interaction_test_df": interaction_test_df,
        "simple_slopes_df": simple_slopes_df,
        "plot_data_df": plot_data_df,
        "paths": paths,
        "interaction_significant": interaction_sig,
        "interpretation": main_interp,
        "slope_notes": slope_notes,
        "x_stats": {"mean": x_mean, "sd": x_sd},
        "w_stats": {"mean": w_mean, "sd": w_sd},
        "intercept": b0,
    }


# ---------------------------------------------------------------------------
# EXCEL (.xlsx) RAPOR DIŞA AKTARIMI
# ---------------------------------------------------------------------------

def _write_df_to_sheet(
    ws,
    df: pd.DataFrame,
    start_row: int = 1,
    title: str | None = None,
) -> int:
    """DataFrame'i biçimlendirilmiş tablo olarak openpyxl sayfasına yazar."""
    header_fill = PatternFill(start_color="1E3A5F", end_color="1E3A5F", fill_type="solid")
    zebra_fill = PatternFill(start_color="F4F7FA", end_color="F4F7FA", fill_type="solid")
    title_font = Font(name="Calibri", size=12, bold=True, color="1E3A5F")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=10, color="1F2937")
    thin_side = Side(style="thin", color="D1D5DB")
    border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    current_row = start_row
    if title:
        cell = ws.cell(row=current_row, column=1, value=title)
        cell.font = title_font
        current_row += 1

    clean_df = df[[c for c in df.columns if not str(c).startswith("_")]].copy()

    # Başlık satırı
    for col_idx, col_name in enumerate(clean_df.columns, start=1):
        c = ws.cell(row=current_row, column=col_idx, value=str(col_name))
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border
    ws.row_dimensions[current_row].height = 24
    current_row += 1

    # Veri satırları
    for r_i, (_, row) in enumerate(clean_df.iterrows()):
        for col_idx, val in enumerate(row, start=1):
            cell_val = None if pd.isna(val) else val
            c = ws.cell(row=current_row, column=col_idx, value=cell_val)
            c.font = body_font
            c.border = border
            if r_i % 2 == 1:
                c.fill = zebra_fill
            if isinstance(cell_val, (int, np.integer)):
                c.number_format = "#,##0"
                c.alignment = Alignment(horizontal="right", vertical="center")
            elif isinstance(cell_val, (float, np.floating)):
                c.number_format = "0.0000"
                c.alignment = Alignment(horizontal="right", vertical="center")
            else:
                c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        current_row += 1

    return current_row + 1


def _autofit_columns(ws) -> None:
    """Excel sütun genişliklerini içeriğe göre otomatik ayarlar."""
    for col_cells in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            val_str = "" if cell.value is None else str(cell.value)
            # Çok uzun paragraf hücrelerinin sütunu aşırı genişletmesini önle
            line_len = max((len(line) for line in val_str.splitlines()), default=0)
            max_len = max(max_len, min(line_len, 55))
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)


def export_results_to_excel(
    results: dict[str, Any],
    missing_summary_df: pd.DataFrame | None = None,
) -> bytes:
    """Analiz sonuçlarını çok sekmeli, biçimlendirilmiş Excel (.xlsx) bayt akışına dönüştürür."""
    wb = Workbook()
    ws_overview = wb.active
    ws_overview.title = "Ozet_ve_Yorum"

    ds = results["data_summary"]
    meta_rows = [
        {"Parametre": "Analiz Modeli", "Değer": results["model_type"]},
        {"Parametre": "Bağımsız Değişken (X)", "Değer": results["x_col"]},
        {"Parametre": "Bağımlı Değişken (Y)", "Değer": results["y_col"]},
    ]
    if results["model_type"] == "Model 4":
        meta_rows.append({"Parametre": "Aracı Değişken(ler) (M)", "Değer": ", ".join(results["m_cols"])})
        meta_rows.append({"Parametre": "Bootstrap Tekrar Sayısı", "Değer": results["n_bootstrap"]})
        meta_rows.append({"Parametre": "Rastgele Tohum (Seed)", "Değer": str(results["random_seed"])})
    else:
        meta_rows.append({"Parametre": "Düzenleyici Değişken (W)", "Değer": results["w_col"]})
        meta_rows.append(
            {
                "Parametre": "Ortalama Merkezileştirme",
                "Değer": "Evet (Mean-Centered)" if results["mean_center"] else "Hayır",
            }
        )

    meta_rows.extend(
        [
            {
                "Parametre": "Kontrol Değişkenleri",
                "Değer": ", ".join(results["covariates"]) if results["covariates"] else "Yok",
            },
            {"Parametre": "Güven Düzeyi (%)", "Değer": results["conf_level"]},
            {"Parametre": "Başlangıç Gözlem Sayısı", "Değer": ds["original_n"]},
            {"Parametre": "Analiz Örneklem Sayısı (N)", "Değer": ds["valid_n"]},
            {"Parametre": "Silinen Satır (Listwise)", "Değer": f"{ds['dropped_n']} (%{ds['dropped_pct']})"},
        ]
    )

    next_row = _write_df_to_sheet(
        ws_overview, pd.DataFrame(meta_rows), start_row=1, title="Analiz Parametreleri ve Örneklem Özeti"
    )

    if results["model_type"] == "Model 4":
        interp_df = pd.DataFrame(
            [
                {
                    "Aracı Değişken": item["mediator"],
                    "Sonuç": "Anlamlı" if item["significant"] else "Anlamlı Değil",
                    "Akademik Yorum": item["text"].replace("**", ""),
                }
                for item in results["interpretations"]
            ]
        )
        _write_df_to_sheet(ws_overview, interp_df, start_row=next_row, title="Akademik Yorumlama")
    else:
        interp_rows = [
            {
                "Bölüm": "Düzenleyicilik (Etkileşim) Testi",
                "Akademik Yorum": results["interpretation"].replace("**", ""),
            }
        ]
        for note in results["slope_notes"]:
            interp_rows.append(
                {
                    "Bölüm": "Basit Eğim (Simple Slope)",
                    "Akademik Yorum": note.replace("**", "").lstrip("- "),
                }
            )
        _write_df_to_sheet(ws_overview, pd.DataFrame(interp_rows), start_row=next_row, title="Akademik Yorumlama")
    _autofit_columns(ws_overview)

    # Betimsel istatistikler ve Eksik değerler
    ws_desc = wb.create_sheet("Betimsel_Istatistikler")
    r_desc = _write_df_to_sheet(
        ws_desc,
        results["descriptives_df"],
        start_row=1,
        title="Analiz Değişkenleri Betimsel İstatistikleri (Listwise Sonrası)",
    )
    if missing_summary_df is not None:
        _write_df_to_sheet(ws_desc, missing_summary_df, start_row=r_desc, title="Ham Veri Eksik Değer Özeti")
    _autofit_columns(ws_desc)

    if results["model_type"] == "Model 4":
        ws_mod = wb.create_sheet("Model_Ozetleri")
        r_m = _write_df_to_sheet(ws_mod, results["all_summaries_df"], start_row=1, title="OLS Model Özetleri")
        _write_df_to_sheet(
            ws_mod, results["total_direct_df"], start_row=r_m, title="Toplam ve Doğrudan Etki Özeti"
        )
        _autofit_columns(ws_mod)

        ws_reg = wb.create_sheet("Regresyon_Katsayilari")
        curr = 1
        for m_col, m_res in results["a_models"].items():
            curr = _write_df_to_sheet(ws_reg, m_res["coef_df"], start_row=curr, title=m_res["title"])
        curr = _write_df_to_sheet(
            ws_reg, results["y_model"]["coef_df"], start_row=curr, title=results["y_model"]["title"]
        )
        _write_df_to_sheet(
            ws_reg, results["total_model"]["coef_df"], start_row=curr, title=results["total_model"]["title"]
        )
        _autofit_columns(ws_reg)

        ws_boot = wb.create_sheet("Dolayli_Etki_Bootstrap")
        _write_df_to_sheet(
            ws_boot,
            results["indirect_df"],
            start_row=1,
            title=f"Dolaylı Etki(ler) - Percentile Bootstrap (B = {results['n_bootstrap']})",
        )
        _autofit_columns(ws_boot)

        ws_sobel = wb.create_sheet("Sobel_Testi")
        _write_df_to_sheet(ws_sobel, results["sobel_df"], start_row=1, title="Klasik Sobel Testi Sonuçları")
        _autofit_columns(ws_sobel)
    else:
        ws_mod = wb.create_sheet("Model_Ozeti_ve_Etkilesim")
        r_m = _write_df_to_sheet(
            ws_mod, results["full_model"]["summary_df"], start_row=1, title="Düzenleyicilik Modeli Özeti"
        )
        _write_df_to_sheet(
            ws_mod,
            results["interaction_test_df"],
            start_row=r_m,
            title="En Yüksek Dereceli Koşulsuz Etkileşim Testi (ΔR²)",
        )
        _autofit_columns(ws_mod)

        ws_reg = wb.create_sheet("Regresyon_Katsayilari")
        _write_df_to_sheet(
            ws_reg, results["full_model"]["coef_df"], start_row=1, title=results["full_model"]["title"]
        )
        _autofit_columns(ws_reg)

        ws_slopes = wb.create_sheet("Basit_Egimler")
        r_s = _write_df_to_sheet(
            ws_slopes,
            results["simple_slopes_df"],
            start_row=1,
            title=f"{results['w_col']} Düzeylerine Göre {results['x_col']} → {results['y_col']} Koşullu Etkileri (Simple Slopes)",
        )
        _write_df_to_sheet(
            ws_slopes,
            results["plot_data_df"],
            start_row=r_s,
            title="Etkileşim Grafiği Tahmin Değerleri",
        )
        _autofit_columns(ws_slopes)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()
