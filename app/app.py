"""
Recomendador Musical — TFG Spotify
Tres modos en una sola app:
  1. Por canción semilla   → KNN coseno (global o híbrido por macro-género)
  2. Por preferencias      → sliders de audio features + filtro macro-género
  3. Mi perfil de usuario  → historial ponderado + radar chart
Dataset: data/processed/tracks_clean_final.csv (75 710 canciones limpias)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

from recommendation_system.recommender import PersonalizedRecommender

# ── Configuración ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Recommender · TFG",
    page_icon="🎵",
    layout="wide"
)

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

GENRE_COLORS = {
    "latino":       "🟠", "pop":          "🔵", "rock":         "🔴",
    "electronica":  "🟣", "hip-hop":      "🟡", "clasica":      "⚪",
    "jazz-blues":   "🟤", "metal":        "⚫", "folk-acustico":"🟢", "otros": "🔘"
}


# ── Carga única del recomendador ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando dataset y modelos…")
def load_recommender():
    return PersonalizedRecommender()

rec = load_recommender()
df   = rec.df          # DataFrame limpio (75 710 filas)
X_sc = rec.X_scaled    # Matriz escalada (mismo orden que df)
macro_genres = sorted(df["macro_genre"].dropna().unique().tolist())


# ── Funciones de búsqueda ─────────────────────────────────────────────────────
def knn_global(seed_idx: int, n: int = 10) -> pd.DataFrame:
    sims = cosine_similarity(X_sc[seed_idx].reshape(1, -1), X_sc)[0]
    sims[seed_idx] = -1
    top = np.argsort(sims)[::-1][:n]
    result = df.iloc[top][["track_name", "artists", "macro_genre", "popularity"]].copy()
    result.insert(0, "similitud", (sims[top] * 100).round(1).astype(str) + "%")
    return result.reset_index(drop=True)


def knn_hybrid(seed_idx: int, n: int = 10) -> pd.DataFrame:
    genre = df.iloc[seed_idx]["macro_genre"]
    mask  = (df["macro_genre"] == genre)
    mask.iloc[seed_idx] = False
    genre_idx = np.where(mask)[0]
    if len(genre_idx) < n:
        return knn_global(seed_idx, n)
    sims_genre = cosine_similarity(X_sc[seed_idx].reshape(1, -1), X_sc[genre_idx])[0]
    top_local  = np.argsort(sims_genre)[::-1][:n]
    top_global = genre_idx[top_local]
    result = df.iloc[top_global][["track_name", "artists", "macro_genre", "popularity"]].copy()
    result.insert(0, "similitud", (sims_genre[top_local] * 100).round(1).astype(str) + "%")
    return result.reset_index(drop=True)


def knn_by_profile(profile_vec: list, n: int = 10, genre_filter: str = "Todos") -> pd.DataFrame:
    query_sc = rec.scaler.transform(np.array(profile_vec).reshape(1, -1))
    if genre_filter and genre_filter != "Todos":
        mask   = df["macro_genre"] == genre_filter
        idx    = np.where(mask)[0]
        if len(idx) == 0:
            idx = np.arange(len(df))
    else:
        idx = np.arange(len(df))
    sims   = cosine_similarity(query_sc, X_sc[idx])[0]
    top    = np.argsort(sims)[::-1][:n]
    result = df.iloc[idx[top]][["track_name", "artists", "macro_genre", "popularity"]].copy()
    result.insert(0, "similitud", (sims[top] * 100).round(1).astype(str) + "%")
    return result.reset_index(drop=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 Music Recommender")
    st.caption("TFG · Análisis Estadístico aplicado a la Industria Musical")
    st.markdown(
        f"**Dataset:** {len(df):,} canciones  \n"
        f"**Géneros:** {df['macro_genre'].nunique()} macro-categorías"
    )
    st.divider()
    st.markdown(
        "#### Cómo funciona\n"
        "- **Canción semilla**: KNN coseno sobre 9 audio features\n"
        "- **Preferencias**: busca canciones que encajen con tu perfil ideal\n"
        "- **Mi perfil**: historial ponderado (última × 1.5) + radar chart"
    )


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Por canción semilla",
    "🎛️ Por preferencias",
    "👤 Mi perfil de usuario",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Por canción semilla
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Busca una canción y recibe recomendaciones similares")

    col_search, col_opts = st.columns([4, 2])
    with col_search:
        query1 = st.text_input("🔍 Artista o título", key="q1")
    with col_opts:
        method  = st.selectbox("Método", ["Híbrido (género + KNN)", "KNN global"])
        n_recs1 = st.slider("Nº recomendaciones", 5, 20, 10, key="n1")

    if query1:
        matches = rec.search_songs(query1, n_results=50)
        if matches.empty:
            st.warning("No se encontraron canciones. Prueba otro término.")
        else:
            options = [
                f"{row['track_name']} — {row['artists']}  [{row['macro_genre']}]"
                for _, row in matches.iterrows()
            ]
            selected = st.selectbox("Selecciona la canción", options, key="sel1")
            sel_pos  = options.index(selected)
            sel_tid  = matches.iloc[sel_pos]["track_id"]

            # Buscar índice en df completo
            df_match = df[df["track_id"] == sel_tid]
            if df_match.empty:
                st.error("Canción no encontrada en el dataset escalado.")
            else:
                seed_idx = df_match.index[0]
                seed_row = df.iloc[seed_idx]

                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🎵 Canción", seed_row["track_name"][:30])
                c2.metric("👤 Artista", seed_row["artists"][:25])
                c3.metric("🏷️ Género",  seed_row.get("macro_genre", "—"))
                c4.metric("⭐ Popularity", int(seed_row["popularity"]))

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Danceability", f"{seed_row['danceability']:.2f}")
                col_m2.metric("Energy",       f"{seed_row['energy']:.2f}")
                col_m3.metric("Valence",      f"{seed_row['valence']:.2f}")

                if st.button("🎵 Recomendar", key="btn1"):
                    with st.spinner("Calculando similitudes…"):
                        recs = (knn_hybrid(seed_idx, n_recs1)
                                if method == "Híbrido (género + KNN)"
                                else knn_global(seed_idx, n_recs1))

                    st.markdown("### Recomendaciones")
                    recs.index += 1
                    recs.columns = ["Similitud", "Canción", "Artista", "Género", "Popularity"]
                    st.dataframe(recs, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Por preferencias
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Define tu perfil musical y encuentra canciones que encajen")

    col_sliders, col_opts2 = st.columns([3, 1])

    with col_sliders:
        st.markdown("**Características de audio:**")
        s1, s2 = st.columns(2)
        with s1:
            dance    = st.slider("Danceability",       0.0,  1.0,  0.5,  0.01)
            energy   = st.slider("Energy",             0.0,  1.0,  0.5,  0.01)
            valence  = st.slider("Valence (alegría)",  0.0,  1.0,  0.5,  0.01)
            acoustic = st.slider("Acousticness",       0.0,  1.0,  0.3,  0.01)
            speech   = st.slider("Speechiness",        0.0,  1.0,  0.1,  0.01)
        with s2:
            tempo    = st.slider("Tempo (BPM)",       50.0, 220.0, 120.0, 1.0)
            loudness = st.slider("Loudness (dB)",    -40.0,   0.0,  -8.0, 0.5)
            instrum  = st.slider("Instrumentalness",   0.0,   1.0,  0.05, 0.01)
            liveness = st.slider("Liveness",           0.0,   1.0,  0.15, 0.01)

    with col_opts2:
        st.markdown("**Filtros:**")
        genre_filter = st.selectbox("Macro-género", ["Todos"] + macro_genres)
        n_recs2      = st.slider("Nº recomendaciones", 5, 20, 10, key="n2")
        st.markdown("---")
        st.markdown("**Tu perfil actual:**")
        st.metric("Danceability", f"{dance:.2f}")
        st.metric("Energy",       f"{energy:.2f}")
        st.metric("Valence",      f"{valence:.2f}")
        st.metric("Tempo",        f"{tempo:.0f} BPM")

    # Orden: danceability, energy, loudness, speechiness,
    #        acousticness, instrumentalness, liveness, valence, tempo
    profile_vec = [dance, energy, loudness, speech, acoustic, instrum, liveness, valence, tempo]

    if st.button("🎛️ Encontrar canciones", key="btn2"):
        with st.spinner("Buscando canciones que encajen…"):
            recs2 = knn_by_profile(profile_vec, n_recs2, genre_filter)

        st.markdown("### Canciones recomendadas")
        recs2.index += 1
        recs2.columns = ["Similitud", "Canción", "Artista", "Género", "Popularity"]
        st.dataframe(recs2, use_container_width=True)

        # Mini radar del perfil buscado vs media del género
        features_radar = ["danceability", "energy", "valence",
                          "acousticness", "speechiness", "instrumentalness"]
        user_vals = [dance, energy, valence, acoustic, speech, instrum]

        if genre_filter != "Todos":
            genre_mean = df[df["macro_genre"] == genre_filter][features_radar].mean()
        else:
            genre_mean = df[features_radar].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=user_vals, theta=features_radar, fill="toself",
            name="Tu perfil", line_color="#1DB954", fillcolor="rgba(29,185,84,0.2)"
        ))
        fig.add_trace(go.Scatterpolar(
            r=genre_mean.values, theta=features_radar, fill="toself",
            name=f"Media {genre_filter}", line_color="#FF6B35",
            fillcolor="rgba(255,107,53,0.15)"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=380,
            title="Tu perfil vs media del género seleccionado"
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Mi perfil de usuario
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Tu perfil musical personalizado")

    # Selector de usuario
    existing_users = rec.get_all_users()
    col_u1, col_u2 = st.columns([2, 3])
    with col_u1:
        username_text = st.text_input("Nombre de usuario", placeholder="Ej: carlos_rdz", key="uname")
    with col_u2:
        if existing_users:
            sel_user = st.selectbox(
                "O elige uno existente",
                options=[""] + existing_users,
                key="sel_user"
            )
            if sel_user:
                username_text = sel_user

    username = username_text.strip()

    if not username:
        st.info("Introduce un nombre de usuario o selecciona uno existente para empezar.")
    else:
        rec.get_or_create_user(username)
        stats = rec.get_user_stats(username)

        # Estadísticas rápidas
        if stats["total_songs"] > 0:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("🎵 Canciones", stats["total_songs"])
            m2.metric("🏷️ Género favorito", stats.get("genero_favorito", "—"))
            m3.metric("💃 Dance",  f"{stats.get('danceability_media', 0):.2f}")
            m4.metric("⚡ Energy", f"{stats.get('energy_media', 0):.2f}")
            m5.metric("😊 Mood",   f"{stats.get('valence_media', 0):.2f}")
        st.divider()

        result     = rec.recommend(username)
        last_songs = result["last_songs"]
        recs_main  = result["recommendations_main"]
        recs_sec   = result["recommendations_secondary"]

        col_left3, col_right3 = st.columns([4, 6])

        # ── Columna izquierda: historial + buscador ───────────────────────────
        with col_left3:
            st.markdown("#### 🎧 Últimas escuchadas")

            if last_songs.empty:
                st.info("Aún no has escuchado ninguna canción. Busca y añade una abajo.")
            else:
                for i, (_, row) in enumerate(last_songs.iterrows()):
                    peso_label = "⭐ **ÚLTIMA** · Peso 1.5×" if i == 0 else "☆ Peso 1.0×"
                    genre_icon = GENRE_COLORS.get(row.get("macro_genre", ""), "⚪")
                    st.markdown(
                        f"**{i+1}. {row['track_name']}**  \n"
                        f"{row['artists']} · {genre_icon} {row.get('macro_genre', '—')}  \n"
                        f"{peso_label}"
                    )
                    c1, c2 = st.columns(2)
                    c1.progress(float(row.get("danceability", 0)),
                                text=f"Dance {row.get('danceability', 0):.2f}")
                    c2.progress(float(row.get("energy", 0)),
                                text=f"Energy {row.get('energy', 0):.2f}")
                    st.divider()

            st.markdown("#### ➕ Añadir canción escuchada")
            search_q = st.text_input("🔍 Buscar por artista o título", key="search3")
            if search_q:
                search_res = rec.search_songs(search_q, n_results=8)
                if search_res.empty:
                    st.warning("No se encontraron resultados.")
                else:
                    for _, srow in search_res.iterrows():
                        genre_icon = GENRE_COLORS.get(srow.get("macro_genre", ""), "⚪")
                        c_name, c_btn = st.columns([5, 2])
                        c_name.write(
                            f"**{srow['track_name']}** — {srow['artists']} "
                            f"({genre_icon} {srow['macro_genre']})"
                        )
                        if c_btn.button("+ Añadir", key=f"add3_{srow['track_id']}"):
                            added = rec.add_song_to_history(username, srow["track_id"])
                            if added:
                                st.success(f"✅ Añadida: {srow['track_name']}")
                                st.rerun()
                            else:
                                st.info("Ya era tu última canción.")

        # ── Columna derecha: recomendaciones ──────────────────────────────────
        with col_right3:
            st.markdown("#### ✨ Recomendadas para ti")

            if last_songs.empty:
                st.info("Añade canciones para recibir recomendaciones personalizadas.")
            else:
                st.caption(
                    f"Perfil sonoro → género dominante: **{result['profile_genre']}**"
                )
                with st.expander("💡 ¿Cómo funciona esta recomendación?"):
                    st.markdown(
                        f"Calculamos tu **perfil sonoro** como la media ponderada de tus "
                        f"últimas {len(last_songs)} canciones. Tu canción más reciente tiene "
                        f"peso **1.5×** (influye un 50% más que las demás). Luego buscamos "
                        f"las canciones más similares usando **similitud coseno** sobre 9 "
                        f"audio features de Spotify, dentro del género dominante "
                        f"(**{result['profile_genre']}**)."
                    )

                if not recs_main.empty:
                    st.markdown(f"##### Porque te gusta el {result['profile_genre']}")
                    d = recs_main[["track_name", "artists", "popularity", "similarity"]].copy()
                    d["similarity"] = (d["similarity"] * 100).round(1).astype(str) + "%"
                    d.columns = ["Canción", "Artista", "Popularity", "Similitud"]
                    st.dataframe(d, use_container_width=True, hide_index=True)

                if not recs_sec.empty:
                    st.markdown(f"##### Puede que también te guste · {result['secondary_genre']}")
                    d2 = recs_sec[["track_name", "artists", "popularity", "similarity"]].copy()
                    d2["similarity"] = (d2["similarity"] * 100).round(1).astype(str) + "%"
                    d2.columns = ["Canción", "Artista", "Popularity", "Similitud"]
                    st.dataframe(d2, use_container_width=True, hide_index=True)

        # ── Radar chart ───────────────────────────────────────────────────────
        if not last_songs.empty:
            st.divider()
            st.markdown("#### 📊 Tu perfil sonoro")

            profile_raw    = rec.compute_user_profile_raw(last_songs)
            features_radar = ["danceability", "energy", "valence",
                              "acousticness", "speechiness", "instrumentalness"]
            genre_mean = (
                df[df["macro_genre"] == result["profile_genre"]][features_radar]
                .mean().to_dict()
            )

            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(
                r=[profile_raw.get(f, 0) for f in features_radar],
                theta=features_radar, fill="toself",
                name=f"Tu perfil ({username})",
                line_color="#1DB954", fillcolor="rgba(29,185,84,0.2)"
            ))
            fig3.add_trace(go.Scatterpolar(
                r=[genre_mean.get(f, 0) for f in features_radar],
                theta=features_radar, fill="toself",
                name=f"Media {result['profile_genre']}",
                line_color="#FF6B35", fillcolor="rgba(255,107,53,0.15)"
            ))
            fig3.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=420,
                title=f"Perfil sonoro vs media del género {result['profile_genre']}"
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(
                "Verde = tu perfil ponderado (la última canción pesa 1.5×). "
                "Naranja = media del género dominante. "
                "La diferencia muestra en qué aspectos tu gusto se desvía del género."
            )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Dataset: {len(df):,} canciones únicas · "
    f"{df['macro_genre'].nunique()} macro-géneros · "
    "TFG — Grado en Estadística · Universidad de Sevilla"
)
