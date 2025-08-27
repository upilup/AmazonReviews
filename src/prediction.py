# deployment/prediction.py — Sentiment 3-Class (negative=0, neutral=1, positive=2)
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ====== Lokasi artefak ======
ARTIFACT_DIR = Path("artifacts_textcnn_v1")
MODEL_PATH   = ARTIFACT_DIR / "model.keras"
META_PATH    = ARTIFACT_DIR / "meta.json"

ID2SENT_DEFAULT = {0: "negative", 1: "neutral", 2: "positive"}
TEXT_COL_CANDIDATES = ["text", "review", "content", "comment", "body", "sentence", "message", "reviews"]

# ====== Utilities ======
def _detect_text_col(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in TEXT_COL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return None
    means = {c: df[c].dropna().astype(str).str.len().mean() for c in obj_cols}
    return max(means, key=means.get) if means else obj_cols[0]

def _apply_alias(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "pred_sent": "sentiment",
        "p_neg": "prob_negatif",
        "p_neu": "prob_netral",
        "p_pos": "prob_positif",
    })

def _download_csv(df: pd.DataFrame, filename: str = "sentiment_predictions.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name=filename, mime="text/csv")

# ====== Model Loader ======
def _setup_environment():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    
    # Import setelah set environment
    import keras
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return keras

def _load_meta() -> Dict[str, Any]:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

@st.cache_resource(show_spinner=False)
def _load_model_and_meta():
    keras = _setup_environment()
    
    if not MODEL_PATH.exists():
        st.error(f"Model tidak ditemukan di {MODEL_PATH}.")
        st.stop()

    try:
        # Approach untuk Keras 3
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
        
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.info("""
        **Troubleshooting:**
        1. Pastikan model dibuat dengan Keras 3
        2. Coba convert model ke format lain jika perlu
        """)
        st.stop()

    # Validasi model
    try:
        # Test prediction sederhana
        test_texts = ["test prediction"]
        _ = model.predict(np.array(test_texts, dtype=object), verbose=0)
    except Exception as e:
        st.warning(f"Test prediction warning: {e}")

    meta = _load_meta()
    id2sent = {int(k): str(v) for k, v in meta.get("label_map", ID2SENT_DEFAULT).items()}
    return model, id2sent, meta

# ====== Inference ======
def _predict_texts(model, texts: List[str], id2sent: Dict[int, str]) -> pd.DataFrame:
    import numpy as np
    
    # Convert ke numpy array dengan dtype object untuk string
    arr = np.array([str(t) for t in texts], dtype=object)
    
    try:
        probs = model.predict(arr, verbose=0)
    except Exception as e:
        st.error(f"Error selama prediksi: {e}")
        return pd.DataFrame({
            "pred_sent": ["error"] * len(texts),
            "p_neg": [0.0] * len(texts),
            "p_neu": [0.0] * len(texts),
            "p_pos": [0.0] * len(texts),
        })
    
    # Handle output shape
    if probs.shape[1] != 3:
        st.warning(f"Model output shape: {probs.shape}, expected 3 classes")
        yhat = probs.argmax(axis=1)
        pred_sent = [f"class_{int(i)}" for i in yhat]
        
        # Create probability columns based on actual output shape
        prob_cols = {}
        for i in range(min(probs.shape[1], 3)):
            prob_cols[f"p_{i}"] = probs[:, i]
        
        # Fill missing columns with zeros
        for col in ["p_neg", "p_neu", "p_pos"]:
            if col not in prob_cols:
                prob_cols[col] = [0.0] * len(texts)
                
        return pd.DataFrame({
            "pred_sent": pred_sent,
            **prob_cols
        })
    else:
        yhat = probs.argmax(axis=1)
        pred_sent = [id2sent.get(int(i), f"class_{int(i)}") for i in yhat]
        
        return pd.DataFrame({
            "pred_sent": pred_sent,
            "p_neg": probs[:, 0],
            "p_neu": probs[:, 1],
            "p_pos": probs[:, 2],
        })

# ====== Streamlit Page ======
def run():
    st.header("Prediction — Sentiment 3-Class")
    
    try:
        model, id2sent, meta = _load_model_and_meta()
    except Exception as e:
        st.error(f"Error inisialisasi: {e}")
        return

    tabs = st.tabs(["Single Text", "Batch CSV"])

    # ---- Single Text ----
    with tabs[0]:
        st.subheader("Single Text Inference")
        txt = st.text_area("Masukkan teks ulasan", height=140,
                           placeholder="e.g., 'love the app but it keeps crashing after the last update'")
        use_alias = st.checkbox("Tampilkan nama kolom ramah", value=True)
        if st.button("Predict", type="primary"):
            if not txt.strip():
                st.warning("Teks masih kosong.")
            else:
                with st.spinner("Memproses..."):
                    res = _predict_texts(model, [txt.strip()], id2sent)
                    out = res.copy()
                    out.insert(0, "text", [txt.strip()])
                    if use_alias:
                        out = _apply_alias(out)
                    st.dataframe(out, use_container_width=True)

    # ---- Batch CSV ----
    with tabs[1]:
        st.subheader("Batch Inference via CSV")
        file = st.file_uploader("Upload CSV", type=["csv"])
        use_alias2 = st.checkbox("Tampilkan nama kolom ramah", value=True, key="alias2")
        if file is not None:
            try:
                df = pd.read_csv(file)
            except Exception as e:
                st.error(f"Gagal membaca CSV: {e}")
                return
            if df.empty:
                st.warning("CSV kosong.")
                return
            text_col = _detect_text_col(df)
            if not text_col:
                st.error("Tidak menemukan kolom teks.")
                return
            st.caption(f"Kolom teks terdeteksi: **{text_col}**")
            if st.button("Predict CSV", type="primary"):
                with st.spinner("Memproses batch..."):
                    texts = df[text_col].astype(str).fillna("").tolist()
                    res = _predict_texts(model, texts, id2sent)
                    out = pd.concat([df.reset_index(drop=True), res], axis=1)
                    if use_alias2:
                        out = _apply_alias(out)
                    st.success(f"Selesai memprediksi {len(out):,} baris.")
                    st.dataframe(out.head(25), use_container_width=True)
                    _download_csv(out)

if __name__ == "__main__":
    st.set_page_config(page_title="Sentiment Prediction", layout="wide")
    run()