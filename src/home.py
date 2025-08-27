# home.py
# Streamlit Home page untuk proyek Analisis Sentimen 3-Kelas
# DISET untuk hanya menggunakan dataset: amazon_reviews.csv
# Selaras dengan notebook & eda.py: mapping label 0=negative, 1=neutral, 2=positive

import os
import pandas as pd
import streamlit as st

ID2SENT = {0: "negative", 1: "neutral", 2: "positive"}

SUGGEST_TEXT_COLS = ["text", "review", "content", "comment", "body", "sentence", "message", "tweet", "reviews"]
SUGGEST_LABEL_COLS = ["label", "labels", "sentiment", "target", "y", "y_true", "true", "category"]

DATA_FILENAME = "amazon_reviews.csv"  # <- satu-satunya dataset yang digunakan

def _read_amazon_reviews() -> pd.DataFrame:
    """
    Membaca amazon_reviews.csv dari root repo.
    """
    if os.path.exists(DATA_FILENAME):
        try:
            return pd.read_csv(DATA_FILENAME)
        except Exception:
            pass
    return pd.DataFrame()

def _first_present(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None

def _detect_text_col(df: pd.DataFrame):
    col = _first_present(df.columns.tolist(), [c.lower() for c in SUGGEST_TEXT_COLS])
    if col:
        return col
    # fallback: pilih kolom object dengan rata2 panjang teks terbesar
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        lens = {c: df[c].dropna().astype(str).str.len().mean() for c in obj_cols}
        return max(lens, key=lens.get) if lens else obj_cols[0]
    return None

def _detect_label_col(df: pd.DataFrame):
    col = _first_present(df.columns.tolist(), [c.lower() for c in SUGGEST_LABEL_COLS])
    if col:
        return col
    small = [c for c in df.columns if df[c].nunique(dropna=True) <= 5]
    return small[0] if small else None

def home():
    # Judul
    st.title("Analisis Sentimen 3-Kelas pada Amazon Reviews")
    st.caption("Skema label: 0 = negative, 1 = neutral, 2 = positive")
    st.markdown("---")

    # Latar belakang
    st.markdown("## Latar Belakang")
    st.markdown(
        """
Ulasan pelanggan di marketplace/aplikasi mengandung opini **negatif**, **netral**, dan **positif**.
Kelas **netral** sering ambigu (ada pujian + keluhan) sehingga rentan salah prediksi.
Proyek ini membangun **model klasifikasi sentimen 3-kelas** untuk memahami persepsi pengguna
dan memandu prioritas perbaikan produk/layanan.
        """
    )

    # Tujuan
    st.markdown("## Tujuan & Ruang Lingkup")
    st.markdown(
        """
- Mengklasifikasikan teks ulasan ke dalam **negative**, **neutral**, atau **positive**.  
- **Fokus kualitas**: meminimalkan kesalahan pada kelas **netral** (FN/FP) sesuai insight di notebook.  
- Menyediakan antarmuka **Streamlit** untuk EDA, evaluasi, dan inference.
        """
    )

    st.markdown("---")

    # Dataset (Preview khusus amazon_reviews.csv)
    st.markdown("## Dataset (Preview)")
    st.caption(f"Sumber data: **{DATA_FILENAME}** (letakkan di root repo).")
    df = _read_amazon_reviews()
    if not df.empty:
        text_col = _detect_text_col(df)
        label_col = _detect_label_col(df)

        st.dataframe(df.head(8), use_container_width=True)
        st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns)}")
        if text_col:
            st.caption(f"Teridentifikasi kolom teks: **{text_col}**")
        if label_col:
            st.caption(f"Teridentifikasi kolom label: **{label_col}**")
    else:
        st.info(
            f"Tidak menemukan file `{DATA_FILENAME}` di root project. "
            f"Tambahkan file tersebut agar preview dataset tampil."
        )

    st.markdown("---")

    # Data Overview (skema kolom utama yang dipakai komponen lain)
    st.markdown("## Data Overview (Kolom Kunci)", unsafe_allow_html=True)
    st.markdown(
        """
<table>
<thead>
<tr><th>Kolom</th><th>Tipe</th><th>Wajib</th><th>Penjelasan</th></tr>
</thead>
<tbody>
<tr><td><code>text</code></td><td>string</td><td>Ya</td><td>Teks ulasan yang akan dianalisis.</td></tr>
<tr><td><code>y_true</code> / <code>label</code></td><td>int / string</td><td>Tidak (direkomendasikan)</td><td>Label ground truth (0=neg, 1=neu, 2=pos atau nama kelas).</td></tr>
<tr><td><code>y_pred</code></td><td>int / string</td><td>Tidak</td><td>Prediksi model (jika sudah tersedia).</td></tr>
<tr><td><code>p_neg</code>, <code>p_neu</code>, <code>p_pos</code></td><td>float</td><td>Tidak</td><td>Probabilitas per kelas (softmax) untuk analisis kesalahan (terutama kelas netral).</td></tr>
</tbody>
</table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Model ringkas
    st.markdown("## Ringkasan Model & Evaluasi")
    st.markdown(
        """
- **Arsitektur**: Model urutan (TensorFlow) dengan output **3 kelas (softmax)**.  
- **Pelabelan**: `0 -> negative`, `1 -> neutral`, `2 -> positive` (selaras notebook).  
- **Output**: Prediksi kelas dan probabilitas `p_neg`, `p_neu`, `p_pos`.  
- **Evaluasi**: Confusion Matrix + analisis **FN/FP kelas netral** (tersedia di halaman EDA).
        """
    )
    st.info(
        "Buka tab **EDA** untuk distribusi kelas, panjang teks, top words/bigrams, "
        "wordcloud per kelas, dan *error analysis* fokus kelas netral."
    )

    st.markdown("---")

    # Cara pakai
    st.markdown("## Cara Menggunakan Aplikasi")
    st.markdown(
        """
1) Tab **EDA** -> upload/preview `amazon_reviews.csv`, cek distribusi & error netral.  
2) Tab **Prediction** -> input teks atau unggah batch untuk diprediksi.  
3) Unduh hasil (pred_label + p_neg/p_neu/p_pos) bila diperlukan.
        """
    )

    st.success("Home page sudah dikunci untuk dataset amazon_reviews.csv dan konsisten dengan notebook & eda.py.")

if __name__ == "__main__":
    home()