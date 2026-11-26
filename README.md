# 📊 Aracılık (Mediation) ve Düzenleyicilik (Moderation) Analiz Uygulaması

SPSS'teki **Hayes PROCESS Macro**'ya açık kaynaklı, modern ve kullanıcı dostu bir Python alternatifi. **Streamlit**, **statsmodels**, **NumPy**, **Matplotlib** ve **pyreadstat** kullanılarak geliştirilmiştir.

---

## ✨ Temel Özellikler

- **Çoklu Veri Formatı Desteği:** `.sav` (SPSS — `pyreadstat` ile değişken etiketleri dahil), `.xlsx` (Excel — çoklu sayfa seçimi) ve `.csv` (otomatik ayırıcı ve kodlama algılama) dosyalarını sürükle-bırak ile yükleme.
- **Veri Keşfi ve Kalite Kontrolü:** İlk 5 satır önizlemesi, değişken bazlı eksik değer özeti, betimsel istatistikler (N, Ortalama, SD, çeyrekler, çarpıklık, basıklık) ve analiz değişkenleri üzerinde otomatik **satır bazlı silme (listwise deletion)**.
- **Model 4 — Aracılık (Mediation) Analizi:**
  - Tekli ve **çoklu paralel aracılık** ($M_1, M_2, \dots, M_k$) ile isteğe bağlı kontrol değişkenleri (Covariates).
  - Tüm yollar ($a$, $b$, $c$, $c'$) için OLS regresyon katsayıları ($B$), $SE$, $t$, $p$, güven aralıkları, $R^2$ ve $F$ istatistikleri.
  - Yapılandırılabilir tekrar sayılı (varsayılan 5000) ve sabit random seed destekli **Percentile Bootstrap** (`BootSE`, `BootLLCI`, `BootULCI`) + klasik **Sobel Testi** ($z$ ve $p$).
- **Model 1 — Düzenleyicilik (Moderation) Analizi:**
  - Bağımsız ($X$), Bağımlı ($Y$), Düzenleyici ($W$) ve isteğe bağlı Kontrol değişkenleri.
  - Otomatik **ortalama merkezileştirme (mean-centering)** seçeneği.
  - Etkileşim terimi ($X \times W$) için $\Delta R^2$ ve $F$ testi.
  - Düzenleyicinin **-1 SD (Düşük)**, **Ortalama (Mean)** ve **+1 SD (Yüksek)** düzeylerindeki koşullu etkiler (**Simple Slopes**).
- **Yayın Kalitesinde Görselleştirme ve Raporlama:**
  - Katsayılar ve anlamlılık yıldızları (`* p < .05`, `** p < .01`, `*** p < .001`) ile **Yol Diyagramı (Path Diagram)**.
  - Model 4 için **Bootstrap Dağılım Grafiği**, Model 1 için **Etkileşim (Simple Slopes) Grafiği**.
  - Güven aralığının 0'ı içerip içermemesine dayalı otomatik **akademik Türkçe yorumlama**.
  - Tüm tabloları tek tıkla biçimlendirilmiş **Excel (`.xlsx`)** dosyası olarak indirme ve grafikleri 300 DPI PNG olarak dışa aktarma.

---

## 📁 Proje Yapısı

```text
SPSS_Alternative/
├── app.py              # Streamlit kullanıcı arayüzü
├── analysis.py         # İstatistiksel motor (OLS, Bootstrap, Sobel, Simple Slopes, Excel dışa aktarım)
├── plotting.py         # Matplotlib yol diyagramları, Bootstrap ve etkileşim grafikleri
├── sample_data.csv     # Test ve deneme için sentetik örnek veri seti (N = 250)
├── test_analysis.py    # Otomatik doğrulama test paketi (pytest)
├── requirements.txt    # Python bağımlılıkları
└── README.md           # Kurulum ve kullanım kılavuzu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Ortamı Hazırlama (Önerilen: Sanal Ortam)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Gerekli Kütüphaneleri Kurma

```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatma

```bash
streamlit run app.py
```

Komut çalıştırıldıktan sonra uygulama tarayıcınızda varsayılan olarak `http://localhost:8501` adresinde açılacaktır.

---

## 🧪 Örnek Veri Seti ile Hızlı Deneme

Uygulama açıldığında sol menüdeki **"Örnek veri setini kullan (`sample_data.csv`)"** seçeneği varsayılan olarak aktiftir:

- **Model 4 (Aracılık) Örneği:**
  - `X`: `Is_Stresi`, `Y`: `Performans`, `M`: `Tukenmislik` ve/veya `Is_Doyumu`, Kontrol: `Yas`, `Kidem_Yil`
- **Model 1 (Düzenleyicilik) Örneği:**
  - `X`: `Is_Stresi`, `Y`: `Tukenmislik`, `W`: `Sosyal_Destek`, Kontrol: `Yas`

### Otomatik Testleri Çalıştırma

```bash
pytest -v
```
