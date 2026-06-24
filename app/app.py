"""
FASE 4 — Interfaz Streamlit de recomendación musical
Modos:
  1. Por canción semilla  → top-N canciones similares
  2. Por preferencias     → sliders de audio features + filtro género
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# ── Carga de recursos (cacheados) ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(PROC / "tracks_unique_clustered.csv")
    return df

@st.cache_resource
def load_models():
    knn    = joblib.load(MODELS / "knn_recommender.pkl")
    scaler = joblib.load(MODELS / "scaler_cluster.pkl")
    kmeans = joblib.load(MODELS / "kmeans.pkl")
    X_sc   = np.load(MODELS / "X_scaled.npy")
    return knn, scaler, kmeans, X_sc

CLUSTER_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

FEATURE_RANGES = {
    "danceability":      (0.0, 1.0, 0.5),
    "energy":            (0.0, 1.0, 0.5),
    "valence":           (0.0, 1.0, 0.5),
    "acousticness":      (0.0, 1.0, 0.3),
    "tempo":             (50.0, 220.0, 120.0),
    "speechiness":       (0.0, 1.0, 0.1),
    "instrumentalness":  (0.0, 1.0, 0.05),
    "liveness":          (0.0, 1.0, 0.15),
    "loudness":          (-40.0, 0.0, -8.0),
}

# ── Funciones de recomendación ────────────────────────────────────────────────
def recommend_knn(seed_idx, df, X_sc, knn, n=10):
    dists, indices = knn.kneighbors(X_sc[seed_idx].reshape(1, -1), n_neighbors=n + 1)
    recs = indices[0][1:]
    return df.iloc[recs][["track_name", "artists", "track_genre", "popularity"]].copy()

def recommend_hybrid(seed_idx, df, X_sc, knn, n=10):
    seed_genre = df.iloc[seed_idx]["track_genre"]
    pool = df[df["track_genre"] == seed_genre].drop(index=seed_idx, errors="ignore")
    pool_idx = pool.index.tolist()
    if len(pool_idx) < n:
        return recommend_knn(seed_idx, df, X_sc, knn, n)
    pool_scaled = X_sc[pool_idx]
    dists = np.linalg.norm(pool_scaled - X_sc[seed_idx], axis=1)
    top_n = np.argsort(dists)[:n]
    return pool.iloc[top_n][["track_name", "artists", "track_genre", "popularity"]].copy()

def recommend_by_profile(profile_vec, df, X_sc, scaler, n=10, genre_filter=None):
    query_scaled = scaler.transform([profile_vec])
    if genre_filter and genre_filter != "Todos":
        mask = df["track_genre"] == genre_filter
        sub_df  = df[mask].reset_index(drop=True)
        sub_X   = X_sc[df[mask].index]
        if len(sub_df) == 0:
            sub_df = df; sub_X = X_sc
    else:
        sub_df = df.reset_index(drop=True)
        sub_X  = X_sc
    dists = np.linalg.norm(sub_X - query_scaled, axis=1)
    top_n = np.argsort(dists)[:n]
    return sub_df.iloc[top_n][["track_name", "artists", "track_genre", "popularity"]].copy()

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spotify Recommender — TFG", page_icon="🎵", layout="wide")
st.title("🎵 Recomendador Musical — TFG Spotify")
st.caption("Sistema basado en contenido (audio features) · Datos: Spotify Tracks Dataset")

df      = load_data()
knn, scaler, kmeans, X_sc = load_models()

genres = sorted(df["track_genre"].dropna().unique().tolist())

tab1, tab2 = st.tabs(["🔍 Por canción semilla", "🎛️ Por preferencias"])

# ── Tab 1: Por canción semilla ────────────────────────────────────────────────
with tab1:
    st.subheader("Busca una canción y recibe recomendaciones similares")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        search_query = st.text_input("Buscar por nombre de canción o artista", "")
    with col_b:
        method = st.selectbox("Método", ["Híbrido (género+KNN)", "KNN global"])
        n_recs = st.slider("Nº recomendaciones", 5, 20, 10)

    if search_query:
        matches = df[
            df["track_name"].str.contains(search_query, case=False, na=False) |
            df["artists"].str.contains(search_query, case=False, na=False)
        ].head(50)

        if len(matches) == 0:
            st.warning("No se encontraron canciones. Prueba otro término.")
        else:
            options = [
                f"{row['track_name']} — {row['artists']} ({row['track_genre']})"
                for _, row in matches.iterrows()
            ]
            selected = st.selectbox("Selecciona la canción", options)
            sel_idx  = matches.index[options.index(selected)]

            seed_row = df.iloc[sel_idx]
            st.markdown("---")
            st.markdown(f"**Canción seleccionada:** {seed_row['track_name']}  \n"
                        f"**Artista:** {seed_row['artists']}  \n"
                        f"**Género:** {seed_row['track_genre']}  \n"
                        f"**Popularity:** {seed_row['popularity']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Danceability", f"{seed_row['danceability']:.2f}")
            col2.metric("Energy",       f"{seed_row['energy']:.2f}")
            col3.metric("Valence",      f"{seed_row['valence']:.2f}")

            if st.button("🎵 Recomendar"):
                with st.spinner("Buscando canciones similares..."):
                    if method == "Híbrido (género+KNN)":
                        recs = recommend_hybrid(sel_idx, df, X_sc, knn, n_recs)
                    else:
                        recs = recommend_knn(sel_idx, df, X_sc, knn, n_recs)

                st.markdown("### Recomendaciones")
                recs_display = recs.reset_index(drop=True)
                recs_display.index += 1
                recs_display.columns = ["Canción", "Artista", "Género", "Popularity"]
                st.dataframe(recs_display, use_container_width=True)

# ── Tab 2: Por preferencias ───────────────────────────────────────────────────
with tab2:
    st.subheader("Define tu perfil musical y encuentra canciones que encajen")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**Ajusta las características de audio:**")
        col1, col2 = st.columns(2)

        with col1:
            dance = st.slider("Danceability",    0.0, 1.0, 0.5, 0.01)
            energy= st.slider("Energy",          0.0, 1.0, 0.5, 0.01)
            valence=st.slider("Valence (alegría)",0.0, 1.0, 0.5, 0.01)
            acoustic=st.slider("Acousticness",   0.0, 1.0, 0.3, 0.01)
            speech = st.slider("Speechiness",    0.0, 1.0, 0.1, 0.01)

        with col2:
            tempo   = st.slider("Tempo (BPM)",       50.0, 220.0, 120.0, 1.0)
            loudness= st.slider("Loudness (dB)",     -40.0, 0.0,  -8.0,  0.5)
            instrum = st.slider("Instrumentalness",  0.0, 1.0, 0.05, 0.01)
            liveness= st.slider("Liveness",          0.0, 1.0, 0.15, 0.01)

    with col_right:
        st.markdown("**Filtros adicionales:**")
        genre_filter = st.selectbox("Filtrar por género", ["Todos"] + genres)
        n_recs2 = st.slider("Nº recomendaciones ", 5, 20, 10)

        st.markdown("---")
        st.markdown("**Tu perfil:**")
        st.metric("Danceability", f"{dance:.2f}")
        st.metric("Energy",       f"{energy:.2f}")
        st.metric("Valence",      f"{valence:.2f}")
        st.metric("Tempo",        f"{tempo:.0f} BPM")

    profile = [dance, energy, loudness, speech, acoustic, instrum, liveness, valence, tempo]

    if st.button("🎛️ Encontrar canciones"):
        with st.spinner("Buscando canciones que encajen con tu perfil..."):
            recs2 = recommend_by_profile(profile, df, X_sc, scaler, n_recs2, genre_filter)

        st.markdown("### Canciones recomendadas")
        recs2_display = recs2.reset_index(drop=True)
        recs2_display.index += 1
        recs2_display.columns = ["Canción", "Artista", "Género", "Popularity"]
        st.dataframe(recs2_display, use_container_width=True)

st.markdown("---")
st.caption(
    f"Dataset: {len(df):,} canciones únicas · "
    f"{df['track_genre'].nunique()} géneros · "
    "TFG — Matemáticas/Estadística"
)
