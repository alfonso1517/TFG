import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, '.')
from recommendation_system.recommender import PersonalizedRecommender
import plotly.graph_objects as go

st.set_page_config(
    page_title="Music Recommender · TFG",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    return PersonalizedRecommender()

rec = load_recommender()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 Music Recommender")
    st.caption("TFG · Análisis Estadístico aplicado a la Industria Musical")
    st.divider()

    existing_users = rec.get_all_users()
    username_input = st.text_input("Nombre de usuario", placeholder="Ej: carlos_rdz")

    if existing_users:
        selected_existing = st.selectbox(
            "O elige un usuario existente",
            options=[""] + existing_users,
            index=0
        )
        if selected_existing:
            username_input = selected_existing

    username = username_input.strip()
    if not username:
        st.info("Introduce un nombre de usuario para empezar.")
        st.stop()

    rec.get_or_create_user(username)
    stats = rec.get_user_stats(username)

    st.divider()
    st.markdown(f"**Usuario activo:** `{username}`")
    if stats['total_songs'] > 0:
        st.metric("Canciones escuchadas", stats['total_songs'])
        st.metric("Género favorito", stats.get('genero_favorito', '—'))
        col1, col2, col3 = st.columns(3)
        col1.metric("Dance", f"{stats.get('danceability_media',0):.2f}")
        col2.metric("Energy", f"{stats.get('energy_media',0):.2f}")
        col3.metric("Mood", f"{stats.get('valence_media',0):.2f}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
result = rec.recommend(username)
last_songs = result['last_songs']
recs_main = result['recommendations_main']
recs_secondary = result['recommendations_secondary']

col_left, col_right = st.columns([4, 6])

# ─── COLUMNA IZQUIERDA ───────────────────────────────────────────────────────
with col_left:
    st.subheader("🎧 Últimas escuchadas")

    if last_songs.empty:
        st.info("Aún no has escuchado ninguna canción. Busca y añade una abajo.")
    else:
        for i, (_, row) in enumerate(last_songs.iterrows()):
            with st.container():
                peso_label = "⭐ **ÚLTIMA** · Peso 1.5×" if i == 0 else f"☆ Peso 1.0×"
                genre_color = {
                    'latino': '🟠', 'pop': '🔵', 'rock': '🔴',
                    'electronica': '🟣', 'hip-hop': '🟡', 'clasica': '⚪',
                    'jazz-blues': '🟤', 'metal': '⚫', 'folk-acustico': '🟢',
                }.get(row.get('macro_genre', ''), '⚪')
                st.markdown(
                    f"**{i+1}. {row['track_name']}**  \n"
                    f"{row['artists']} · {genre_color} {row.get('macro_genre','—')}  \n"
                    f"{peso_label}"
                )
                c1, c2 = st.columns(2)
                c1.progress(float(row.get('danceability', 0)), text=f"Dance {row.get('danceability',0):.2f}")
                c2.progress(float(row.get('energy', 0)), text=f"Energy {row.get('energy',0):.2f}")
                st.divider()

    st.subheader("➕ Añadir canción escuchada")
    search_query = st.text_input("🔍 Buscar por artista o título", key="search")
    if search_query:
        search_results = rec.search_songs(search_query, n_results=8)
        if search_results.empty:
            st.warning("No se encontraron resultados.")
        else:
            for _, srow in search_results.iterrows():
                cols = st.columns([5, 2])
                cols[0].write(f"**{srow['track_name']}** — {srow['artists']} ({srow['macro_genre']})")
                if cols[1].button("+ Añadir", key=f"add_{srow['track_id']}"):
                    added = rec.add_song_to_history(username, srow['track_id'])
                    if added:
                        st.success(f"✅ Añadida: {srow['track_name']}")
                        st.rerun()
                    else:
                        st.info("Ya era tu última canción.")

# ─── COLUMNA DERECHA ─────────────────────────────────────────────────────────
with col_right:
    st.subheader("✨ Recomendadas para ti")

    if last_songs.empty:
        st.info("Añade canciones para recibir recomendaciones personalizadas.")
    else:
        st.caption(f"Perfil sonoro → género dominante: **{result['profile_genre']}**")

        with st.expander("💡 ¿Cómo funciona esta recomendación?"):
            st.markdown(
                f"Calculamos tu **perfil sonoro** como la media ponderada de tus "
                f"últimas {len(last_songs)} canciones. Tu canción más reciente tiene "
                f"peso **1.5×** (influye un 50% más que las demás). Luego buscamos "
                f"las canciones más similares a ese perfil usando **similitud coseno** "
                f"en el espacio de 9 audio features de Spotify, dentro del género "
                f"dominante (**{result['profile_genre']}**)."
            )

        if not recs_main.empty:
            st.markdown(f"#### Porque te gusta el {result['profile_genre']}")
            display_cols = ['track_name', 'artists', 'popularity', 'similarity']
            display_df = recs_main[display_cols].copy()
            display_df['similarity'] = (display_df['similarity'] * 100).round(1).astype(str) + '%'
            display_df.columns = ['Canción', 'Artista', 'Popularidad', 'Similitud']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        if not recs_secondary.empty:
            st.markdown(f"#### Puede que también te guste · {result['secondary_genre']}")
            display_df2 = recs_secondary[display_cols].copy()
            display_df2['similarity'] = (display_df2['similarity'] * 100).round(1).astype(str) + '%'
            display_df2.columns = ['Canción', 'Artista', 'Popularidad', 'Similitud']
            st.dataframe(display_df2, use_container_width=True, hide_index=True)

# ─── RADAR ───────────────────────────────────────────────────────────────────
if not last_songs.empty:
    st.divider()
    st.subheader("📊 Tu perfil sonoro")

    profile_raw = rec.compute_user_profile_raw(last_songs)
    features_radar = ['danceability', 'energy', 'valence',
                      'acousticness', 'speechiness', 'instrumentalness']

    genre_mean = (rec.df[rec.df['macro_genre'] == result['profile_genre']]
                  [features_radar].mean().to_dict())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[profile_raw.get(f, 0) for f in features_radar],
        theta=features_radar,
        fill='toself',
        name=f'Tu perfil ({username})',
        line_color='#1DB954',
        fillcolor='rgba(29,185,84,0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[genre_mean.get(f, 0) for f in features_radar],
        theta=features_radar,
        fill='toself',
        name=f'Media {result["profile_genre"]}',
        line_color='#FF6B35',
        fillcolor='rgba(255,107,53,0.15)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f'Perfil sonoro vs media del género {result["profile_genre"]}',
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Verde = tu perfil ponderado (la última canción pesa 1.5×). "
        "Naranja = media del género dominante. "
        "La diferencia entre ambas muestra en qué aspectos tu gusto se desvía de la media del género."
    )
