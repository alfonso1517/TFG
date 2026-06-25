"""
Recomendador Musical — TFG Spotify
Tres modos:
  1. Por cancion semilla   — KNN coseno (global o hibrido por macro-genero)
  2. Por preferencias      — sliders de audio features + filtro macro-genero
  3. Mi perfil de usuario  — historial ponderado + radar chart
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
from pathlib import Path

from recommendation_system.recommender import PersonalizedRecommender

# ── Rutas de assets ────────────────────────────────────────────────────────────
ASSETS = Path(__file__).parent / "assets"
import base64

def asset(name):
    p = ASSETS / name
    return str(p) if p.exists() else None

def show_icon(name, width=44, radius=10, padding=8, bg="#FFFFFF"):
    """Muestra una imagen de assets dentro de un cuadrado blanco (para iconos oscuros)."""
    p = ASSETS / name
    if not p.exists():
        return
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = p.suffix.lstrip(".")
    st.markdown(
        f'<div style="display:inline-block; background:{bg}; '
        f'padding:{padding}px; border-radius:{radius}px; line-height:0;">'
        f'<img src="data:image/{ext};base64,{data}" width="{width}"></div>',
        unsafe_allow_html=True
    )

# ── Configuracion ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Recommender · TFG",
    page_icon=asset("spotify_logo.png") or ":material/music_note:",
    layout="wide"
)

# ── Tema Spotify ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo y tipografia general */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Circular', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #000000;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #B3B3B3;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #121212;
        border-bottom: 1px solid #282828;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #B3B3B3;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 14px 24px;
        border-radius: 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background-color: #1a1a1a;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF;
        border-bottom: 2px solid #1DB954;
        background-color: transparent;
    }

    /* Botones primarios */
    .stButton > button {
        background-color: #1DB954;
        color: #000000;
        border: none;
        border-radius: 500px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 11px 32px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #1ed760;
        color: #000000;
        border: none;
        transform: scale(1.02);
    }
    .stButton > button:active {
        background-color: #169c46;
    }

    /* Inputs de texto */
    .stTextInput > div > div > input {
        background-color: #282828;
        color: #FFFFFF;
        border: 1px solid #404040;
        border-radius: 4px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1DB954;
        box-shadow: 0 0 0 1px #1DB954;
    }
    .stTextInput label { color: #B3B3B3 !important; }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #282828;
        border: 1px solid #404040;
        border-radius: 4px;
        color: #FFFFFF;
    }
    .stSelectbox label { color: #B3B3B3 !important; }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #1DB954 !important;
    }
    .stSlider label { color: #B3B3B3 !important; }

    /* Metricas */
    [data-testid="metric-container"] {
        background-color: #282828;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #333333;
    }
    [data-testid="metric-container"] label {
        color: #B3B3B3 !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* Barras de progreso */
    .stProgress > div > div > div {
        background-color: #282828;
        border-radius: 4px;
    }
    .stProgress > div > div > div > div {
        background-color: #1DB954 !important;
        border-radius: 4px;
    }

    /* Dataframes */
    .stDataFrame {
        background-color: #282828;
        border-radius: 8px;
    }
    .stDataFrame th {
        background-color: #1a1a1a !important;
        color: #B3B3B3 !important;
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .stDataFrame td { color: #FFFFFF !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #282828;
        border-radius: 6px;
        color: #B3B3B3 !important;
    }
    .streamlit-expanderContent {
        background-color: #1e1e1e;
        border-radius: 0 0 6px 6px;
    }

    /* Alertas / info */
    .stAlert {
        background-color: #282828;
        border-radius: 8px;
        border-left: 4px solid #1DB954;
    }

    /* Divisores */
    hr { border-color: #282828 !important; }

    /* Titulos */
    h1 { color: #FFFFFF; font-weight: 900; letter-spacing: -1px; }
    h2, h3, h4 { color: #FFFFFF; font-weight: 700; }
    h5, h6 { color: #FFFFFF; }

    /* Caption / texto secundario */
    .stCaption, caption { color: #B3B3B3 !important; }

    /* Footer de Streamlit */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

FEATURE_LABELS = {
    "danceability":     "Bailabilidad",
    "energy":           "Energía",
    "valence":          "Positividad",
    "acousticness":     "Acústica",
    "speechiness":      "Contenido vocal",
    "tempo":            "Tempo (BPM)",
    "loudness":         "Volumen (dB)",
    "instrumentalness": "Instrumentalidad",
    "liveness":         "En directo",
}

GENRE_COLORS = {
    "latino":       "#FF6B35",
    "pop":          "#1DB954",
    "rock":         "#E91429",
    "electronica":  "#9B59B6",
    "hip-hop":      "#F39C12",
    "clasica":      "#BDC3C7",
    "jazz-blues":   "#8B7355",
    "metal":        "#555555",
    "folk-acustico":"#27AE60",
    "otros":        "#7F8C8D",
}

PLOTLY_DARK = dict(
    plot_bgcolor="#121212",
    paper_bgcolor="#121212",
    font_color="#FFFFFF",
)


# ── Carga del recomendador ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando dataset…")
def load_recommender():
    return PersonalizedRecommender()

rec  = load_recommender()
df   = rec.df
X_sc = rec.X_scaled
macro_genres = sorted(df["macro_genre"].dropna().unique().tolist())


# ── Funciones de busqueda ──────────────────────────────────────────────────────
def knn_global(seed_idx: int, n: int = 10) -> pd.DataFrame:
    sims = cosine_similarity(X_sc[seed_idx].reshape(1, -1), X_sc)[0]
    sims[seed_idx] = -1
    top = np.argsort(sims)[::-1][:n]
    result = df.iloc[top][["track_name", "artists", "macro_genre", "popularity"]].copy()
    result.insert(0, "similitud", (sims[top] * 100).round(1).astype(str) + " %")
    return result.reset_index(drop=True)

def knn_hybrid(seed_idx: int, n: int = 10) -> pd.DataFrame:
    genre = df.iloc[seed_idx]["macro_genre"]
    mask  = df["macro_genre"] == genre
    mask.iloc[seed_idx] = False
    genre_idx = np.where(mask)[0]
    if len(genre_idx) < n:
        return knn_global(seed_idx, n)
    sims_g = cosine_similarity(X_sc[seed_idx].reshape(1, -1), X_sc[genre_idx])[0]
    top_l  = np.argsort(sims_g)[::-1][:n]
    top_g  = genre_idx[top_l]
    result = df.iloc[top_g][["track_name", "artists", "macro_genre", "popularity"]].copy()
    result.insert(0, "similitud", (sims_g[top_l] * 100).round(1).astype(str) + " %")
    return result.reset_index(drop=True)

def knn_by_profile(profile_vec: list, n: int = 10, genre_filter: str = "Todos") -> pd.DataFrame:
    query_sc = rec.scaler.transform(np.array(profile_vec).reshape(1, -1))
    if genre_filter and genre_filter != "Todos":
        idx = np.where(df["macro_genre"] == genre_filter)[0]
        if len(idx) == 0:
            idx = np.arange(len(df))
    else:
        idx = np.arange(len(df))
    sims  = cosine_similarity(query_sc, X_sc[idx])[0]
    top   = np.argsort(sims)[::-1][:n]
    result = df.iloc[idx[top]][["track_name", "artists", "macro_genre", "popularity"]].copy()
    result.insert(0, "similitud", (sims[top] * 100).round(1).astype(str) + " %")
    return result.reset_index(drop=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    show_icon("spotify_logo.png", width=48, radius=12, padding=6)

    st.markdown(
        "<h2 style='color:#FFFFFF; margin-top:8px; margin-bottom:2px;'>"
        "Music Recommender</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#B3B3B3; font-size:12px; margin-top:0;'>"
        "TFG — Analisis Estadistico aplicado a la Industria Musical</p>",
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown(
        f"<p style='color:#B3B3B3; font-size:13px;'>"
        f"<span style='color:#1DB954; font-weight:700;'>{len(df):,}</span> canciones &nbsp;·&nbsp; "
        f"<span style='color:#1DB954; font-weight:700;'>{df['macro_genre'].nunique()}</span> generos"
        f"</p>",
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown(
        "<p style='color:#B3B3B3; font-size:12px; line-height:1.7;'>"
        "<b style='color:#FFFFFF;'>Cancion semilla</b><br>"
        "KNN por similitud coseno sobre 9 audio features<br><br>"
        "<b style='color:#FFFFFF;'>Por preferencias</b><br>"
        "Define tu perfil ideal con sliders y encuentra canciones cercanas<br><br>"
        "<b style='color:#FFFFFF;'>Mi perfil</b><br>"
        "Historial ponderado (ultima × 1.5) y recomendaciones personalizadas"
        "</p>",
        unsafe_allow_html=True
    )


# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Cancion semilla",
    "Por preferencias",
    "Mi perfil",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Por cancion semilla
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c_ico, c_ttl = st.columns([1, 14])
    with c_ico:
        show_icon("music_note.png", width=40)
    c_ttl.markdown("### Canciones similares a la que elijas")

    st.markdown(
        "<p style='color:#B3B3B3;'>Busca una cancion, seleccionala y obtén recomendaciones "
        "basadas en sus caracteristicas de audio.</p>",
        unsafe_allow_html=True
    )
    st.divider()

    col_s, col_m, col_n = st.columns([4, 2, 1])
    with col_s:
        query1 = st.text_input("Artista o titulo", placeholder="Ej: Bad Bunny, Bohemian Rhapsody…", key="q1")
    with col_m:
        method = st.selectbox("Metodo", ["Hibrido (genero + similitud)", "Similitud global"])
    with col_n:
        n_recs1 = st.slider("Resultados", 5, 20, 10, key="n1")

    if query1:
        matches = rec.search_songs(query1, n_results=50)
        if matches.empty:
            st.info("No se encontraron resultados. Prueba con otro termino.")
        else:
            options = [
                f"{row['track_name']}  —  {row['artists']}   [{row['macro_genre']}]"
                for _, row in matches.iterrows()
            ]
            selected  = st.selectbox("Selecciona la cancion", options, key="sel1")
            sel_pos   = options.index(selected)
            sel_tid   = matches.iloc[sel_pos]["track_id"]

            df_match  = df[df["track_id"] == sel_tid]
            if df_match.empty:
                st.error("Cancion no encontrada en el dataset procesado.")
            else:
                seed_idx = df_match.index[0]
                seed     = df.iloc[seed_idx]

                st.divider()
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Cancion",     seed["track_name"][:28])
                col_b.metric("Artista",     seed["artists"][:22])
                col_c.metric("Genero",      seed.get("macro_genre", "—"))
                col_d.metric("Popularidad", int(seed["popularity"]))

                col_e, col_f, col_g = st.columns(3)
                col_e.metric(FEATURE_LABELS["danceability"], f"{seed['danceability']:.2f}")
                col_f.metric(FEATURE_LABELS["energy"],       f"{seed['energy']:.2f}")
                col_g.metric(FEATURE_LABELS["valence"],      f"{seed['valence']:.2f}")

                if st.button("Buscar canciones similares", key="btn1"):
                    with st.spinner("Calculando similitudes…"):
                        recs1 = (knn_hybrid(seed_idx, n_recs1)
                                 if "Hibrido" in method
                                 else knn_global(seed_idx, n_recs1))

                    st.markdown("#### Resultados")
                    recs1.index += 1
                    recs1.columns = ["Similitud", "Cancion", "Artista", "Genero", "Popularidad"]
                    st.dataframe(recs1, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Por preferencias
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c_ico2, c_ttl2 = st.columns([1, 14])
    with c_ico2:
        show_icon("vinyl.png", width=40)
    c_ttl2.markdown("### Define tu perfil musical")

    st.markdown(
        "<p style='color:#B3B3B3;'>Ajusta las caracteristicas de audio de tu cancion ideal "
        "y el sistema buscara las mas cercanas en el dataset.</p>",
        unsafe_allow_html=True
    )
    st.divider()

    col_sl, col_opts2 = st.columns([3, 1])

    with col_sl:
        s1, s2 = st.columns(2)
        with s1:
            dance    = st.slider(FEATURE_LABELS["danceability"],       0.0,   1.0,  0.5,  0.01)
            energy   = st.slider(FEATURE_LABELS["energy"],             0.0,   1.0,  0.5,  0.01)
            valence  = st.slider(FEATURE_LABELS["valence"],            0.0,   1.0,  0.5,  0.01)
            acoustic = st.slider(FEATURE_LABELS["acousticness"],       0.0,   1.0,  0.3,  0.01)
            speech   = st.slider(FEATURE_LABELS["speechiness"],        0.0,   1.0,  0.1,  0.01)
        with s2:
            tempo    = st.slider(FEATURE_LABELS["tempo"],             50.0, 220.0, 120.0, 1.0)
            loudness = st.slider(FEATURE_LABELS["loudness"],         -40.0,   0.0,  -8.0, 0.5)
            instrum  = st.slider(FEATURE_LABELS["instrumentalness"],   0.0,   1.0,  0.05, 0.01)
            liveness = st.slider(FEATURE_LABELS["liveness"],           0.0,   1.0,  0.15, 0.01)

    with col_opts2:
        st.markdown(
            "<p style='color:#B3B3B3; font-size:12px; text-transform:uppercase; "
            "letter-spacing:1px; font-weight:700;'>Filtros</p>",
            unsafe_allow_html=True
        )
        genre_filter = st.selectbox("Genero", ["Todos"] + macro_genres)
        n_recs2      = st.slider("Resultados", 5, 20, 10, key="n2")
        st.divider()
        st.markdown(
            "<p style='color:#B3B3B3; font-size:12px; text-transform:uppercase; "
            "letter-spacing:1px; font-weight:700;'>Tu perfil</p>",
            unsafe_allow_html=True
        )
        st.metric(FEATURE_LABELS["danceability"], f"{dance:.2f}")
        st.metric(FEATURE_LABELS["energy"],       f"{energy:.2f}")
        st.metric(FEATURE_LABELS["valence"],      f"{valence:.2f}")
        st.metric(FEATURE_LABELS["tempo"],        f"{tempo:.0f}")

    # Orden de features: danceability, energy, loudness, speechiness,
    #                    acousticness, instrumentalness, liveness, valence, tempo
    profile_vec = [dance, energy, loudness, speech, acoustic, instrum, liveness, valence, tempo]

    if st.button("Encontrar canciones", key="btn2"):
        with st.spinner("Buscando…"):
            recs2 = knn_by_profile(profile_vec, n_recs2, genre_filter)

        st.markdown("#### Resultados")
        recs2.index += 1
        recs2.columns = ["Similitud", "Cancion", "Artista", "Genero", "Popularidad"]
        st.dataframe(recs2, use_container_width=True)

        # Radar del perfil buscado
        features_radar = ["danceability", "energy", "valence",
                          "acousticness", "speechiness", "instrumentalness"]
        user_vals = [dance, energy, valence, acoustic, speech, instrum]
        radar_labels = [FEATURE_LABELS[f] for f in features_radar]

        if genre_filter != "Todos":
            gm = df[df["macro_genre"] == genre_filter][features_radar].mean()
        else:
            gm = df[features_radar].mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=user_vals, theta=radar_labels, fill="toself",
            name="Tu perfil",
            line=dict(color="#1DB954", width=2),
            fillcolor="rgba(29,185,84,0.15)"
        ))
        fig2.add_trace(go.Scatterpolar(
            r=gm.values, theta=radar_labels, fill="toself",
            name=f"Media {genre_filter}",
            line=dict(color="#B3B3B3", width=1.5, dash="dot"),
            fillcolor="rgba(179,179,179,0.08)"
        ))
        fig2.update_layout(
            **PLOTLY_DARK,
            polar=dict(
                bgcolor="#1a1a1a",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#333333", color="#B3B3B3"),
                angularaxis=dict(gridcolor="#333333", color="#FFFFFF")
            ),
            showlegend=True,
            legend=dict(font=dict(color="#B3B3B3")),
            height=380,
            margin=dict(t=40, b=20, l=20, r=20),
            title=dict(text="Tu perfil vs media del genero", font=dict(color="#FFFFFF", size=14))
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Mi perfil de usuario
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    c_ico3, c_ttl3 = st.columns([1, 14])
    with c_ico3:
        show_icon("user_icon.png", width=36)
    c_ttl3.markdown("### Tu perfil musical personalizado")

    st.markdown(
        "<p style='color:#B3B3B3;'>El sistema aprende de las canciones que escuchas "
        "y calcula un perfil sonoro ponderado para recomendarte musica afin.</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # Selector de usuario
    existing_users = rec.get_all_users()
    col_u1, col_u2 = st.columns([2, 3])
    with col_u1:
        username_text = st.text_input(
            "Nombre de usuario",
            placeholder="Ej: carlos_rdz",
            key="uname"
        )
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

        if stats["total_songs"] > 0:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Canciones",        stats["total_songs"])
            m2.metric("Genero favorito",  stats.get("genero_favorito", "—"))
            m3.metric(FEATURE_LABELS["danceability"], f"{stats.get('danceability_media', 0):.2f}")
            m4.metric(FEATURE_LABELS["energy"],       f"{stats.get('energy_media', 0):.2f}")
            m5.metric(FEATURE_LABELS["valence"],      f"{stats.get('valence_media', 0):.2f}")

        st.divider()

        result     = rec.recommend(username)
        last_songs = result["last_songs"]
        recs_main  = result["recommendations_main"]
        recs_sec   = result["recommendations_secondary"]

        col_left3, col_right3 = st.columns([4, 6])

        # ── Columna izquierda: historial + buscador ───────────────────────────
        with col_left3:
            st.markdown(
                "<p style='color:#B3B3B3; font-size:12px; text-transform:uppercase; "
                "letter-spacing:1px; font-weight:700;'>Ultimas escuchadas</p>",
                unsafe_allow_html=True
            )

            if last_songs.empty:
                st.info("Aun no has escuchado ninguna cancion. Busca y añade una abajo.")
            else:
                for i, (_, row) in enumerate(last_songs.iterrows()):
                    weight_label = "Ultima escuchada — peso 1.5×" if i == 0 else f"Peso 1.0×"
                    genre_color  = GENRE_COLORS.get(row.get("macro_genre", ""), "#7F8C8D")
                    border = "border-left: 3px solid #1DB954;" if i == 0 else "border-left: 3px solid #333;"
                    st.markdown(
                        f"<div style='background:#282828; border-radius:6px; padding:12px 16px; "
                        f"margin-bottom:8px; {border}'>"
                        f"<p style='margin:0; color:#FFFFFF; font-weight:700; font-size:14px;'>"
                        f"{row['track_name']}</p>"
                        f"<p style='margin:2px 0; color:#B3B3B3; font-size:12px;'>"
                        f"{row['artists']} &nbsp;·&nbsp; "
                        f"<span style='color:{genre_color}; font-weight:600;'>"
                        f"{row.get('macro_genre','—')}</span></p>"
                        f"<p style='margin:4px 0 0; color:#727272; font-size:11px;'>"
                        f"{weight_label}</p></div>",
                        unsafe_allow_html=True
                    )
                    c1, c2 = st.columns(2)
                    c1.progress(float(row.get("danceability", 0)),
                                text=f"{FEATURE_LABELS['danceability']}  {row.get('danceability',0):.2f}")
                    c2.progress(float(row.get("energy", 0)),
                                text=f"{FEATURE_LABELS['energy']}  {row.get('energy',0):.2f}")

            st.markdown(
                "<p style='color:#B3B3B3; font-size:12px; text-transform:uppercase; "
                "letter-spacing:1px; font-weight:700; margin-top:24px;'>Añadir cancion</p>",
                unsafe_allow_html=True
            )
            search_q = st.text_input(
                "Buscar por artista o titulo",
                placeholder="Ej: Feid, Shape of You…",
                key="search3"
            )
            if search_q:
                search_res = rec.search_songs(search_q, n_results=8)
                if search_res.empty:
                    st.info("Sin resultados. Prueba otro termino.")
                else:
                    for _, srow in search_res.iterrows():
                        gc = GENRE_COLORS.get(srow.get("macro_genre", ""), "#7F8C8D")
                        c_name, c_btn = st.columns([5, 2])
                        c_name.markdown(
                            f"<p style='margin:4px 0; font-size:13px; color:#FFFFFF;'>"
                            f"<b>{srow['track_name']}</b><br>"
                            f"<span style='color:#B3B3B3; font-size:12px;'>{srow['artists']}"
                            f" &nbsp;·&nbsp; "
                            f"<span style='color:{gc};'>{srow['macro_genre']}</span></span></p>",
                            unsafe_allow_html=True
                        )
                        if c_btn.button("Añadir", key=f"add3_{srow['track_id']}"):
                            added = rec.add_song_to_history(username, srow["track_id"])
                            if added:
                                st.success(f"Añadida: {srow['track_name']}")
                                st.rerun()
                            else:
                                st.info("Ya era tu ultima cancion.")

        # ── Columna derecha: recomendaciones ──────────────────────────────────
        with col_right3:
            st.markdown(
                "<p style='color:#B3B3B3; font-size:12px; text-transform:uppercase; "
                "letter-spacing:1px; font-weight:700;'>Recomendadas para ti</p>",
                unsafe_allow_html=True
            )

            if last_songs.empty:
                st.info("Añade canciones para recibir recomendaciones personalizadas.")
            else:
                genre_color = GENRE_COLORS.get(result["profile_genre"], "#1DB954")
                st.markdown(
                    f"<p style='color:#B3B3B3; font-size:13px;'>Perfil sonoro detectado: "
                    f"<span style='color:{genre_color}; font-weight:700;'>"
                    f"{result['profile_genre']}</span></p>",
                    unsafe_allow_html=True
                )

                with st.expander("Como funciona esta recomendacion"):
                    st.markdown(
                        f"<p style='color:#B3B3B3; font-size:13px;'>"
                        f"Se calcula tu <b style='color:#FFFFFF;'>perfil sonoro</b> como la "
                        f"media ponderada de tus ultimas {len(last_songs)} canciones. "
                        f"La mas reciente tiene peso <b style='color:#1DB954;'>1.5×</b>. "
                        f"Luego se buscan las canciones mas similares por "
                        f"<b style='color:#FFFFFF;'>similitud coseno</b> sobre 9 audio features, "
                        f"dentro del genero dominante "
                        f"(<b style='color:{genre_color};'>{result['profile_genre']}</b>).</p>",
                        unsafe_allow_html=True
                    )

                if not recs_main.empty:
                    st.markdown(
                        f"<p style='color:#FFFFFF; font-weight:700; margin-top:16px;'>"
                        f"Porque escuchas {result['profile_genre']}</p>",
                        unsafe_allow_html=True
                    )
                    d = recs_main[["track_name", "artists", "popularity", "similarity"]].copy()
                    d["similarity"] = (d["similarity"] * 100).round(1).astype(str) + " %"
                    d.columns = ["Cancion", "Artista", "Popularidad", "Similitud"]
                    st.dataframe(d, use_container_width=True, hide_index=True)

                if not recs_sec.empty:
                    sec_color = GENRE_COLORS.get(result["secondary_genre"], "#B3B3B3")
                    st.markdown(
                        f"<p style='color:#FFFFFF; font-weight:700; margin-top:16px;'>"
                        f"Puede que tambien te guste: "
                        f"<span style='color:{sec_color};'>{result['secondary_genre']}</span></p>",
                        unsafe_allow_html=True
                    )
                    d2 = recs_sec[["track_name", "artists", "popularity", "similarity"]].copy()
                    d2["similarity"] = (d2["similarity"] * 100).round(1).astype(str) + " %"
                    d2.columns = ["Cancion", "Artista", "Popularidad", "Similitud"]
                    st.dataframe(d2, use_container_width=True, hide_index=True)

        # ── Radar ─────────────────────────────────────────────────────────────
        if not last_songs.empty:
            st.divider()
            st.markdown(
                "<p style='color:#B3B3B3; font-size:12px; text-transform:uppercase; "
                "letter-spacing:1px; font-weight:700;'>Tu perfil sonoro</p>",
                unsafe_allow_html=True
            )

            profile_raw    = rec.compute_user_profile_raw(last_songs)
            features_radar = ["danceability", "energy", "valence",
                              "acousticness", "speechiness", "instrumentalness"]
            radar_labels   = [FEATURE_LABELS[f] for f in features_radar]
            genre_mean = (
                df[df["macro_genre"] == result["profile_genre"]][features_radar]
                .mean().to_dict()
            )
            gc = GENRE_COLORS.get(result["profile_genre"], "#1DB954")

            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(
                r=[profile_raw.get(f, 0) for f in features_radar],
                theta=radar_labels, fill="toself",
                name=username,
                line=dict(color="#1DB954", width=2),
                fillcolor="rgba(29,185,84,0.15)"
            ))
            fig3.add_trace(go.Scatterpolar(
                r=[genre_mean.get(f, 0) for f in features_radar],
                theta=radar_labels, fill="toself",
                name=f"Media {result['profile_genre']}",
                line=dict(color=gc, width=1.5, dash="dot"),
                fillcolor="rgba(255,107,53,0.08)"
            ))
            fig3.update_layout(
                **PLOTLY_DARK,
                polar=dict(
                    bgcolor="#1a1a1a",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="#333333", color="#B3B3B3"),
                    angularaxis=dict(gridcolor="#333333", color="#FFFFFF")
                ),
                showlegend=True,
                legend=dict(font=dict(color="#B3B3B3")),
                height=420,
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(
                "<p style='color:#727272; font-size:12px;'>"
                "Verde: tu perfil ponderado (la ultima cancion pesa 1.5×). "
                "Discontinuo: media del genero dominante.</p>",
                unsafe_allow_html=True
            )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"<p style='color:#333333; font-size:11px; text-align:center; margin-top:40px;'>"
    f"{len(df):,} canciones · {df['macro_genre'].nunique()} generos · "
    f"TFG — Grado en Estadistica · Universidad de Sevilla</p>",
    unsafe_allow_html=True
)
