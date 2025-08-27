# eda.py — EDA sederhana khusus Amazon Reviews
# Menampilkan: (1) Distribusi Label (score), (2) Panjang Teks, (3) Boxplot per skor, (4) WordCloud
# Dataset & kolom yang dipakai: amazon_reviews.csv, kolom wajib: score, content

import os
import re
import math
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_OK = True
except Exception:
    WORDCLOUD_OK = False

DATA_FILE = "amazon_reviews.csv"
REQ_COLS = ["score", "content"]

# ---------- Helpers ----------
def _load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"Tidak menemukan `{DATA_FILE}` di root project.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        st.error(f"Gagal membaca `{DATA_FILE}`: {e}")
        return pd.DataFrame()
    return df

def _ensure_cols(df: pd.DataFrame) -> bool:
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {missing}. Pastikan ada kolom `score` dan `content`.")
        return False
    return True

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # pastikan tipe
    df["score"] = pd.to_numeric(df["score"], errors="coerce").astype("Int64")
    df["content"] = df["content"].astype(str)
    # panjang teks (jumlah kata)
    df["text_len"] = df["content"].str.split().apply(len)
    return df

def _barh_counts(counts: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    # urutkan index agar 1..5
    counts = counts.sort_index()
    ax.barh(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel("count")
    ax.set_ylabel("score")
    for y, v in enumerate(counts.values):
        ax.text(v, y, f" {int(v):,}", va="center")
    st.pyplot(fig, clear_figure=True)

def _hist_text_len(series: pd.Series, bins: int = 40):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(series.values, bins=bins)
    ax.set_title("Distribusi Panjang Ulasan (Jumlah Kata)")
    ax.set_xlabel("Jumlah Kata")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig, clear_figure=True)

def _boxplot_series(series: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.boxplot(series.values, vert=False, showfliers=True)
    ax.set_title(title)
    ax.set_xlabel("Jumlah Kata")
    st.pyplot(fig, clear_figure=True)

def _boxplot_by_score(df: pd.DataFrame):
    # boxplot text_len untuk tiap score
    groups = [df.loc[df["score"] == s, "text_len"].values for s in sorted(df["score"].dropna().unique())]
    labels = [str(s) for s in sorted(df["score"].dropna().unique())]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(groups, labels=labels, showfliers=True)
    ax.set_title("Panjang Ulasan berdasarkan Skor")
    ax.set_xlabel("Skor")
    ax.set_ylabel("Jumlah Kata")
    st.pyplot(fig, clear_figure=True)

def _wordcloud_all(texts: List[str], title: str):
    if not WORDCLOUD_OK:
        st.info("Paket `wordcloud` belum terpasang di environment.")
        return
    text_all = " ".join(texts)
    wc = WordCloud(width=1000, height=420, background_color="white", stopwords=set(STOPWORDS))
    img = wc.generate(text_all).to_image()
    st.image(img, caption=title, use_container_width=True)

def _wordcloud_per_score(df: pd.DataFrame):
    if not WORDCLOUD_OK:
        st.info("Paket `wordcloud` belum terpasang di environment.")
        return
    for s in sorted(df["score"].dropna().unique()):
        subset = df.loc[df["score"] == s, "content"].astype(str).str.lower()
        if subset.empty:
            continue
        _wordcloud_all(subset.tolist(), title=f"WordCloud - Skor {s}")

# ---------- Page ----------
def eda():
    st.title("Exploratory Data Analysis")
    st.markdown("Di bawah ini hanya menampilkan komponen EDA sesuai notebook: Distribusi Label, Panjang Teks, Boxplot per Skor, dan WordCloud.")

    df = _load_data()
    if df.empty or not _ensure_cols(df):
        return
    df = _clean_df(df)

    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head(12), use_container_width=True)
        st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns)}")

    st.markdown("## Distribusi Label")
    counts = df["score"].value_counts(dropna=False)
    _barh_counts(counts, title="Distribusi Label")
    st.write("##### Tabel Count")
    st.dataframe(counts.rename("count"))
    st.write("##### Persentase per Kelas (%)")
    pct = (df["score"].value_counts(normalize=True) * 100).round(2)
    st.dataframe(pct.rename("proportion"))
    st.markdown("Dari grafik terlihat bahwa:")
    st.markdown("1. Skor 1 memiliki 47,34% data dan mendominasi hampir setengah dari seluruh dataset.")
    st.markdown("2. Skor 5 memiliki 26,26% data dan menjadi kelas kedua terbanyak.")
    st.markdown("3. Skor 2 memiliki 11,23% data, Skor 3 memiliki 8,65% data, dan Skor 4 memiliki 6,53% data sehingga jumlahnya jauh lebih sedikit dibandingkan kelas 1 dan 5.")
    st.markdown("Hal ini menunjukkan bahwa dataset bersifat imbalanced, di mana beberapa kelas memiliki data jauh lebih banyak dibanding yang lain. Kondisi ini perlu dipertimbangkan karena model bisa saja cenderung memprediksi kelas mayoritas.")

    st.markdown("---")
    st.markdown("## Panjang Teks")
    st.caption("Panjang teks dihitung sebagai jumlah kata pada kolom `content`.")
    st.write("##### Ringkasan Statistik `text_len`")
    st.dataframe(df["text_len"].describe().to_frame().T)
    _hist_text_len(df["text_len"], bins=40)
    _boxplot_series(df["text_len"], title="Boxplot Panjang Ulasan")
    st.markdown("""Dari grafik terlihat bahwa:

1. Sebagian besar ulasan memiliki panjang kurang dari 100 kata, dengan puncak di sekitar 20-30 kata.
2. Boxplot menunjukkan adanya beberapa outlier di atas 100 kata, bahkan ada yang mencapai 300 kata.
3. Sebagian besar data berada di rentang 16-52 kata, sesuai dengan ringkasan statistik sebelumnya.

Kesimpulannya, mayoritas teks relatif pendek hingga sedang, sedangkan teks yang sangat panjang hanya sedikit jumlahnya.""")

    st.markdown("---")
    st.markdown("## Panjang Teks per Skor")
    _boxplot_by_score(df)
    st.markdown("""Dari grafik terlihat bahwa:

1. Panjang ulasan di semua skor relatif mirip, mayoritas berada di bawah 100 kata.
2. Skor 1 dan 2 memiliki beberapa outlier dengan ulasan sangat panjang hingga 300 kata.
3. Skor 5 cenderung memiliki ulasan yang sedikit lebih pendek dibanding skor lain.

Kesimpulannya, panjang ulasan tidak berbeda jauh antar skor, tapi skor rendah kadang berisi ulasan yang lebih panjang dan detail.""")

    st.markdown("---")
    st.markdown("## WordCloud")
    st.caption("WordCloud keseluruhan dan per skor (1-5). Kata umum/stopwords dihapus.")
    _wordcloud_all(df["content"].astype(str).str.lower().tolist(), title="WordCloud — Semua Data")
    _wordcloud_per_score(df)

    st.success("Selesai.")

# Debug manual
if __name__ == "__main__":
    eda()