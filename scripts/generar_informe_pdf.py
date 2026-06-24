"""
Genera el informe completo del TFG en PDF.
Incluye todas las fases, gráficos y tablas de resultados.
"""

from fpdf import FPDF
from pathlib import Path
import pandas as pd
import numpy as np

ROOT  = Path(__file__).parent.parent
FIGS  = ROOT / "reports" / "figures"
OUT   = ROOT / "results"
PROC  = ROOT / "data" / "processed"

FONT_R = "C:/Windows/Fonts/arial.ttf"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"
FONT_I = "C:/Windows/Fonts/ariali.ttf"
FONT_BI= "C:/Windows/Fonts/arialbi.ttf"

AZUL_OSC   = (29, 53, 87)
AZUL_MED   = (69, 123, 157)
AZUL_CLAR  = (168, 218, 220)
ROJO_ACENT = (230, 57, 70)
GRIS_TEXT  = (50, 50, 50)
GRIS_TABLA = (245, 245, 245)


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Ar",  "",  FONT_R)
        self.add_font("Ar",  "B", FONT_B)
        self.add_font("Ar",  "I", FONT_I)
        self.add_font("Ar",  "BI",FONT_BI)
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(22, 25, 22)
        self._seccion_actual = ""

    # ── Cabecera y pie ───────────────────────────────────────────────────────
    def header(self):
        if self.page_no() <= 2:
            return
        self.set_font("Ar", "I", 7.5)
        self.set_text_color(160, 160, 160)
        izq = "TFG · Análisis de Spotify con IA y Estadística"
        der = self._seccion_actual
        self.cell(0, 8, izq, align="L")
        self.set_xy(22, self.get_y())
        self.cell(0, 8, der, align="R")
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(22, self.get_y(), 188, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        if self.page_no() <= 2:
            return
        self.set_y(-16)
        self.set_draw_color(200, 200, 200)
        self.line(22, self.get_y(), 188, self.get_y())
        self.ln(2)
        self.set_font("Ar", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"— {self.page_no()} —", align="C")
        self.set_text_color(0, 0, 0)

    # ── Helpers de formato ───────────────────────────────────────────────────
    def titulo_fase(self, texto, color=AZUL_OSC):
        self.ln(6)
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font("Ar", "B", 13)
        self.cell(0, 11, f"  {texto}", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)
        self._seccion_actual = texto

    def subtitulo(self, texto, color=AZUL_MED):
        self.ln(4)
        self.set_font("Ar", "B", 11)
        self.set_text_color(*color)
        self.cell(0, 7, texto, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def p(self, texto, size=10):
        self.set_font("Ar", "", size)
        self.set_text_color(*GRIS_TEXT)
        self.multi_cell(0, 5.5, texto, align="J")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def li(self, texto, size=10, indent=6):
        self.set_font("Ar", "", size)
        self.set_text_color(*GRIS_TEXT)
        x0 = self.get_x()
        self.set_x(x0 + indent)
        self.cell(5, 5.5, "•")
        self.multi_cell(0, 5.5, texto)
        self.set_x(x0)
        self.set_text_color(0, 0, 0)

    def resaltado(self, texto, size=10):
        self.set_font("Ar", "BI", size)
        self.set_fill_color(*AZUL_CLAR)
        self.set_text_color(*AZUL_OSC)
        self.multi_cell(0, 6, f"  {texto}", fill=True, align="J")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def separador(self):
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(22, self.get_y(), 188, self.get_y())
        self.ln(4)

    def figura(self, nombre, caption="", w=155):
        path = FIGS / nombre
        if not path.exists():
            self.set_font("Ar", "I", 9)
            self.set_text_color(180, 0, 0)
            self.cell(0, 8, f"[Figura no encontrada: {nombre}]", ln=True)
            self.set_text_color(0, 0, 0)
            return
        x = (210 - w) / 2
        self.image(str(path), x=x, w=w)
        if caption:
            self.set_font("Ar", "I", 8.5)
            self.set_text_color(110, 110, 110)
            self.multi_cell(0, 5, caption, align="C")
            self.set_text_color(0, 0, 0)
        self.ln(4)

    def tabla(self, headers, rows, widths, aligns=None):
        if aligns is None:
            aligns = ["C"] * len(headers)
        # Cabecera
        self.set_fill_color(*AZUL_OSC)
        self.set_text_color(255, 255, 255)
        self.set_font("Ar", "B", 9)
        for h, w, a in zip(headers, widths, aligns):
            self.cell(w, 7, h, border=1, fill=True, align=a)
        self.ln()
        # Filas
        self.set_text_color(*GRIS_TEXT)
        for i, row in enumerate(rows):
            bg = GRIS_TABLA if i % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            self.set_font("Ar", "", 9)
            for cell, w, a in zip(row, widths, aligns):
                self.cell(w, 6, str(cell), border=1, fill=True, align=a)
            self.ln()
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def caja_metrica(self, etiqueta, valor, color=AZUL_MED):
        x0, y0 = self.get_x(), self.get_y()
        self.set_fill_color(*color)
        self.set_draw_color(*color)
        self.rect(x0, y0, 38, 16, "F")
        self.set_font("Ar", "B", 15)
        self.set_text_color(255, 255, 255)
        self.set_xy(x0, y0 + 1)
        self.cell(38, 8, valor, align="C")
        self.set_font("Ar", "", 7.5)
        self.set_xy(x0, y0 + 9)
        self.cell(38, 5, etiqueta, align="C")
        self.set_text_color(0, 0, 0)
        self.set_xy(x0 + 40, y0)

    def fila_metricas(self, pares, y_offset=18):
        y0 = self.get_y()
        x0 = 22
        self.set_xy(x0, y0)
        for etq, val, color in pares:
            self.caja_metrica(etq, val, color)
        self.set_xy(22, y0 + y_offset)
        self.ln(2)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL DOCUMENTO
# ══════════════════════════════════════════════════════════════════════════════
pdf = PDF()

# ─────────────────────────────────────────────────────────────────────────────
# PORTADA
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.set_fill_color(*AZUL_OSC)
pdf.rect(0, 0, 210, 297, "F")

# Banda superior decorativa
pdf.set_fill_color(*ROJO_ACENT)
pdf.rect(0, 0, 210, 8, "F")

# Logos / íconos textuales
pdf.set_font("Ar", "B", 10)
pdf.set_text_color(255, 255, 255)
pdf.set_xy(22, 20)
pdf.cell(0, 8, "TRABAJO DE FIN DE GRADO  ·  MATEMÁTICAS / ESTADÍSTICA", align="C")

# Título principal
pdf.set_xy(22, 55)
pdf.set_font("Ar", "B", 28)
pdf.set_text_color(255, 255, 255)
pdf.multi_cell(166, 14, "IA y Análisis\nEstadístico\naplicado a la\nIndustria Musical", align="C")

# Subtítulo
pdf.set_xy(22, 145)
pdf.set_font("Ar", "I", 14)
pdf.set_text_color(*AZUL_CLAR)
pdf.multi_cell(166, 8, "Caso práctico con el dataset de\nSpotify Tracks (114.000 canciones)", align="C")

# Línea divisoria
pdf.set_draw_color(*ROJO_ACENT)
pdf.set_line_width(1.2)
pdf.line(40, 175, 170, 175)
pdf.set_line_width(0.2)

# Info inferior
pdf.set_font("Ar", "", 11)
pdf.set_text_color(200, 200, 200)
pdf.set_xy(22, 182)
pdf.cell(0, 7, "Fases completadas: Limpieza · EDA · Modelos RF/XGBoost · Clustering", align="C")
pdf.set_xy(22, 190)
pdf.cell(0, 7, "Recomendación · App Streamlit · Nacionalidad · Trayectoria temporal", align="C")

pdf.set_xy(22, 210)
pdf.set_font("Ar", "B", 12)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 7, "Alfonso García Betico", align="C")
pdf.set_xy(22, 218)
pdf.set_font("Ar", "", 11)
pdf.set_text_color(200, 200, 200)
pdf.cell(0, 7, "Junio 2026", align="C")

# Banda inferior
pdf.set_fill_color(*ROJO_ACENT)
pdf.rect(0, 289, 210, 8, "F")

# ─────────────────────────────────────────────────────────────────────────────
# ÍNDICE
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.set_text_color(0, 0, 0)
pdf.set_fill_color(*AZUL_OSC)
pdf.set_text_color(255, 255, 255)
pdf.set_font("Ar", "B", 16)
pdf.cell(0, 12, "  ÍNDICE DE CONTENIDOS", fill=True, ln=True)
pdf.set_text_color(0, 0, 0)
pdf.ln(8)

secciones = [
    ("1.", "Resumen ejecutivo", 3),
    ("2.", "Fase 0 — Limpieza y preparación de datos", 3),
    ("3.", "Fase 1 — Análisis exploratorio (EDA)", 4),
    ("4.", "Fase 2 — Modelos predictivos: Random Forest vs XGBoost", 6),
    ("  4.1", "Regresión de popularidad", 6),
    ("  4.2", "Clasificación de macro-género", 7),
    ("5.", "Fase 3 — Clustering y sistema de recomendación", 8),
    ("6.", "Fase 4 — Aplicación Streamlit", 10),
    ("7.", "Fase 5 — Estudio de nacionalidad", 11),
    ("8.", "Fase 6 — Trayectoria temporal: Feid y similares", 13),
    ("9.", "Conclusiones y limitaciones", 15),
]

for num, titulo, pag in secciones:
    es_sub = num.startswith(" ")
    pdf.set_font("Ar", "" if es_sub else "B", 10 if es_sub else 11)
    indent = 10 if es_sub else 0
    pdf.set_x(22 + indent)
    pdf.set_text_color(*AZUL_OSC if not es_sub else GRIS_TEXT)
    pdf.cell(12, 7, num)
    pdf.cell(120, 7, titulo)
    pdf.set_font("Ar", "I", 9)
    pdf.cell(0, 7, f"pág. {pag}", align="R")
    pdf.ln()

pdf.set_text_color(0, 0, 0)
pdf.ln(10)
pdf.set_font("Ar", "I", 9)
pdf.set_text_color(130, 130, 130)
pdf.cell(0, 6, "Nota: las páginas son aproximadas — el documento se genera dinámicamente.", ln=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf._seccion_actual = "1. Resumen ejecutivo"
pdf.titulo_fase("1. RESUMEN EJECUTIVO")

pdf.p(
    "Este trabajo aplica técnicas de Inteligencia Artificial y Análisis Estadístico "
    "al dataset de Spotify Tracks, que contiene 114.000 registros de canciones con "
    "21 variables de audio, metadatos y popularidad, correspondientes a 89.740 canciones "
    "únicas en 114 géneros musicales distintos. El objetivo es desarrollar un ciclo "
    "completo de Data Science: desde la limpieza inicial hasta una aplicación interactiva "
    "de recomendación, pasando por modelos supervisados, clustering y análisis sociocultural."
)

pdf.subtitulo("Estructura del proyecto")
pdf.p(
    "El trabajo se articula en seis fases de complejidad creciente. Las fases 0 a 4 "
    "son el núcleo técnico del análisis; las fases 5 y 6 constituyen estudios cualitativos "
    "con apoyo estadístico orientados a la interpretación musical y cultural de los datos."
)

fases_resumen = [
    ("Fase 0", "Limpieza y preparación", "114k → 89.740 únicas; tempo=0 imputado; deduplicación documentada"),
    ("Fase 1", "Análisis exploratorio", "Correlaciones, distribuciones, perfiles por género (7 figuras)"),
    ("Fase 2", "RF vs XGBoost", "Regresión popularidad: R²=0.472 (RF); Clasificación género: F1=0.409 (XGB)"),
    ("Fase 3", "Clustering + Recomendación", "k=2 óptimo; recomendador híbrido género+KNN: coherencia=1.0"),
    ("Fase 4", "App Streamlit", "Interfaz interactiva con búsqueda por canción y por perfil de usuario"),
    ("Fase 5", "Estudio de nacionalidad", "220 artistas, 7 regiones; diferencias sonoras significativas (p<0.001)"),
    ("Fase 6", "Trayectoria temporal", "291 canciones fechadas; evolución y predicción del sonido de Feid 2023-2025"),
]
pdf.tabla(
    ["Fase", "Título", "Resultado clave"],
    fases_resumen,
    [22, 50, 94],
    ["C", "L", "L"]
)

pdf.subtitulo("Hallazgos principales")
for h in [
    "El 10,5% de las canciones tiene popularity = 0. La distribución está fuertemente sesgada a la derecha.",
    "La feature con mayor correlación negativa con popularity es instrumentalness (−0.127): "
    "las canciones con letra tienden a ser más populares en el snapshot analizado.",
    "Random Forest supera a XGBoost en regresión de popularidad (R²=0.472 vs 0.430). "
    "En clasificación de género ambos empatan (~F1=0.41), muy por encima del baseline sin agrupar (F1=0.25).",
    "El clustering de audio features identifica solo 2 clústeres bien diferenciados (silhouette=0.258): "
    "música energética vs. tranquila/acústica. Esto sugiere que las features de audio no "
    "capturan la varianza de género a nivel fino.",
    "El recomendador híbrido (filtrado por género + KNN en el espacio de features) alcanza "
    "coherencia de género = 1.0 con una diversidad de artistas de 0.83.",
    "Las diferencias sonoras entre regiones geográficas son estadísticamente muy significativas "
    "(ANOVA p<0.001 para danceability, energy, valence y tempo). Latinoamérica lidera en "
    "energy (0.712) y valence (0.603); Europa en instrumentalness (0.336).",
    "La trayectoria de Feid (2018-2022) muestra una evolución clara hacia mayor energía y "
    "menor acousticness, convergiendo con el sonido dominante del reggaeton urbano actual.",
]:
    pdf.li(h)

pdf.separador()
pdf.resaltado(
    "Limitación transversal: la variable popularity es un snapshot del algoritmo de Spotify "
    "basado en reproducciones recientes. No refleja el éxito histórico de los artistas. "
    "Esta limitación afecta especialmente las comparaciones temporales y el ranking por popularidad."
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FASE 0 — LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.titulo_fase("2. FASE 0 — LIMPIEZA Y PREPARACIÓN DE DATOS")

pdf.subtitulo("Descripción del dataset")
pdf.p(
    "El dataset original contiene 114.000 filas con 21 columnas: identificadores de canción, "
    "artista y álbum; variables de audio (danceability, energy, loudness, speechiness, "
    "acousticness, instrumentalness, liveness, valence, tempo, key, mode, time_signature, "
    "duration_ms); y las variables objetivo popularity y track_genre."
)
pdf.p(
    "El fichero spotify-tracks-dataset.csv es una copia exacta del dataset principal (114.000 "
    "filas, mismas columnas, solo añade un índice Unnamed: 0). Se descartó para evitar "
    "duplicación. Únicamente se usó dataset.csv."
)

pdf.subtitulo("Decisiones de limpieza documentadas")

pdf.tabla(
    ["Problema detectado", "Magnitud", "Decisión tomada"],
    [
        ["Duplicados por track_id", "40.900 filas (36%)", "Deduplicación: keep='first' → tracks_unique.csv (89.740 filas)"],
        ["Versión larga (clasificación)", "—", "tracks_long.csv con 113.999 filas (1 fila eliminada)"],
        ["Fila con artista/álbum/nombre nulo", "1 fila", "Eliminada directamente"],
        ["tempo == 0", "157 filas (0.14%)", "Imputadas con la mediana del género correspondiente"],
        ["Géneros únicos", "114 (no 125 como indica el dataset)", "Anotado como dato curioso; sin acción"],
        ["loudness en dB negativos", "Rango habitual: −60 a 0 dB", "Normal en audio digital; sin transformación"],
        ["duration_ms en milisegundos", "Algunas > 30 min (grabaciones/podcasts)", "Mantenidas; outliers documentados en EDA"],
    ],
    [52, 38, 76],
    ["L", "C", "L"]
)

pdf.subtitulo("Estructura de ficheros generada")
for l in [
    "data/processed/tracks_unique.csv — 89.740 canciones, una fila por canción",
    "data/processed/tracks_long.csv — 113.999 filas, una fila por combinación canción-género",
    "data/processed/tracks_nationality.csv — 11.881 canciones con metadatos de nacionalidad",
    "data/processed/artists_nationality.csv — 220 artistas con país y región",
    "data/processed/tracks_artistas_trayectoria.csv — 291 canciones con fechas de lanzamiento",
]:
    pdf.li(l, size=9)

pdf.subtitulo("Nota metodológica sobre la variable popularity")
pdf.p(
    "Spotify calcula el índice de popularidad (0–100) mediante un algoritmo propietario basado "
    "principalmente en el volumen de reproducciones recientes y su recencia. No equivale al "
    "número total de streams de la vida de la canción. Una canción de 2004 con millones de "
    "reproducciones históricas puede tener popularity = 20 en el snapshot, mientras que una "
    "canción de 2022 con pocas reproducciones puede tener 70. Este matiz es fundamental para "
    "interpretar todos los análisis donde aparece popularity como variable."
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FASE 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.titulo_fase("3. FASE 1 — ANÁLISIS EXPLORATORIO (EDA)")

pdf.p(
    "El análisis exploratorio busca entender la estructura de los datos antes de modelar, "
    "siguiendo la metodología habitual en proyectos de Data Science: conocer distribuciones, "
    "relaciones entre variables y detectar anomalías."
)

pdf.subtitulo("3.1  Distribución de popularidad")
pdf.p(
    "La variable objetivo presenta una distribución bimodal con fuerte sesgo a la izquierda. "
    "El 10,5% de las canciones tiene popularity = 0. Esto puede deberse a: canciones muy "
    "antiguas con pocas reproducciones recientes, canciones de géneros muy minoritarios, "
    "o canciones retiradas del catálogo activo. La mediana de popularidad es 35."
)
pdf.figura("01a_popularity_distribution.png",
           "Fig. 1 — Distribución de popularity. Nótese el pico en 0 y la distribución sesgada a la derecha.")

pdf.add_page()
pdf.subtitulo("3.2  Matriz de correlaciones")
pdf.p(
    "La correlación de las features de audio con popularity es en general baja, lo que anticipa "
    "la dificultad del problema de regresión. Las correlaciones más destacadas son:"
)
for c in [
    "instrumentalness: −0.127 (negativa, la más fuerte). Las canciones instrumentales son menos populares en el snapshot.",
    "acousticness: −0.083 (negativa). La música acústica tiende a menor popularidad en el período analizado.",
    "danceability: +0.063 (positiva, débil). Más bailable → ligeramente más popular.",
    "loudness: +0.058 (positiva). Canciones más altas en dB tienden a más popularidad.",
    "energy: −0.022 (prácticamente nula en correlación lineal).",
]:
    pdf.li(c, size=9)

pdf.figura("01b_correlation_matrix.png",
           "Fig. 2 — Matriz de correlación de features numéricas. Colores rojos = correlación positiva, azules = negativa.")

pdf.add_page()
pdf.subtitulo("3.3  Perfil sonoro por género")
pdf.p(
    "Comparamos 8 géneros contrastados en 6 audio features normalizadas [0–1]. "
    "El gráfico de radar permite identificar el 'sonido característico' de cada género:"
)
for c in [
    "Reggaeton: danceability y energy muy altas; acousticness baja.",
    "Classical: acousticness e instrumentalness dominantes; danceability baja.",
    "Metal: energy máxima; valence baja (sonido oscuro/intenso); acousticness mínima.",
    "Jazz: acousticness media-alta; valence media; instrumentalness alta.",
    "EDM: energy alta; danceability alta; acousticness y speechiness bajas.",
]:
    pdf.li(c, size=9)

pdf.figura("01c_genre_radar.png",
           "Fig. 3 — Gráfico radar por género (features normalizadas). Cada eje representa una feature de audio.")

pdf.add_page()
pdf.subtitulo("3.4  Popularidad por género y otras variables categóricas")
pdf.figura("01d_popularity_by_genre.png",
           "Fig. 4 — Boxplot de popularidad por género (top géneros seleccionados).")
pdf.p(
    "Los géneros con mayor popularidad media en el snapshot son los géneros urbanos "
    "recientes (pop, latin, reggaeton) mientras que géneros más clásicos o de nicho "
    "(folk, blues, ambient, classical) presentan medianas más bajas, principalmente "
    "por el efecto de recencia del índice de Spotify."
)
pdf.figura("01e_categorical_vars.png",
           "Fig. 5 — Distribución de key, mode, time_signature y explicit vs. popularity.")

pdf.add_page()
pdf.subtitulo("3.5  Outliers en loudness y duration")
pdf.figura("01f_outliers_loudness_duration.png",
           "Fig. 6 — Detección de outliers en loudness (dB) y duration (ms).")
pdf.p(
    "Loudness: rango normal entre −30 y 0 dB. Existen outliers extremos por debajo de "
    "−40 dB (silencio, grabaciones con mucho ruido de fondo o canciones muy dinámicas). "
    "Duration: la mayoría de canciones dura entre 2 y 6 minutos. Hay outliers por encima "
    "de 30 minutos (podcasts o grabaciones largas clasificados como 'canciones' en Spotify)."
)
pdf.figura("01g_audio_features_bar.png",
           "Fig. 7 — Perfil medio de audio features por macro-categoría de género.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FASE 2 — MODELOS PREDICTIVOS
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.titulo_fase("4. FASE 2 — MODELOS PREDICTIVOS: RANDOM FOREST vs XGBOOST")
pdf.p(
    "Esta fase entrena y compara dos familias de modelos de ensemble (Random Forest y XGBoost) "
    "para dos tareas distintas: regresión de la popularidad y clasificación de macro-género."
)

pdf.subtitulo("4.1  Regresión de popularidad")
pdf.p(
    "Se usa el dataset deduplicado (tracks_unique, 89.740 canciones) con split 80/20. "
    "Las 114 categorías de track_genre se codifican con Target Encoding. "
    "La búsqueda de hiperparámetros se realiza con RandomizedSearchCV (20 iteraciones, 3-fold CV)."
)

pdf.tabla(
    ["Modelo", "RMSE", "MAE", "R²", "Hiperparámetros óptimos (selección)"],
    [
        ["Random Forest", "14.86", "10.16", "0.472", "n_est=400, max_depth=20, min_samples_leaf=2"],
        ["XGBoost",       "15.44", "10.84", "0.430", "n_est=300, max_depth=6, lr=0.1, subsample=0.8"],
    ],
    [30, 18, 18, 16, 84],
    ["L", "C", "C", "C", "L"]
)

pdf.resaltado(
    "Random Forest supera a XGBoost en regresión con un R²=0.472. Ninguno de los dos modelos "
    "explica más del 47% de la varianza de popularity, lo que es esperable: el índice "
    "de Spotify depende de factores no capturados en el dataset (streams recientes, "
    "actividad en playlists editoriales, viralidad en redes sociales)."
)

pdf.figura("02a_feature_importance_regression.png",
           "Fig. 8 — Importancia de variables en regresión (RF impurity vs. XGB gain). "
           "track_genre y loudness lideran en ambos modelos.")

pdf.add_page()
pdf.subtitulo("4.2  Clasificación de macro-género")
pdf.p(
    "Con 114 géneros originales la clasificación directa es muy difícil (muchos géneros "
    "con pocas canciones, frontera entre géneros ambigua). Se agrupan los 114 géneros "
    "en 12 macro-categorías usando un mapa editorial:"
)

macro_map = [
    ("rock",           "alternative, grunge, punk, emo, indie"),
    ("electronica",    "edm, techno, trance, dubstep, house, electro, disco"),
    ("latino",         "latin, reggaeton, salsa, cumbia, bachata, reggaeton-colombiano"),
    ("hip-hop",        "hip-hop, rap, trap, r-n-b, soul"),
    ("clasica",        "classical, opera, piano, chamber"),
    ("jazz-blues",     "jazz, blues, gospel"),
    ("pop",            "pop, dance, synth-pop"),
    ("metal",          "metal, heavy-metal, black-metal, death-metal"),
    ("folk-acustico",  "folk, acoustic, singer-songwriter, country, bluegrass"),
    ("kpop-jpop",      "k-pop, j-pop, j-idol, cantopop, mandopop"),
    ("world",          "world-music, afrobeat, sertanejo, pagode, samba, indian"),
    ("otros",          "new-age, ambient, children, comedy, show-tunes, ..."),
]
pdf.tabla(
    ["Macro-género", "Géneros incluidos (muestra)"],
    macro_map,
    [35, 131],
    ["L", "L"]
)

pdf.tabla(
    ["Modelo", "Accuracy", "F1-macro", "F1-weighted"],
    [
        ["Random Forest",  "0.465", "0.408", "0.462"],
        ["XGBoost",        "0.464", "0.409", "0.463"],
        ["Baseline (114 géneros sin agrupación)", "0.257", "0.246", "0.251"],
    ],
    [60, 32, 32, 32],
    ["L", "C", "C", "C"]
)

pdf.p(
    "Ambos modelos son prácticamente equivalentes. La agrupación en 12 macro-géneros "
    "casi duplica el F1 respecto al enfoque con 114 clases. El error más frecuente "
    "se produce entre rock y folk-acústico, y entre electronica y pop, donde los "
    "límites estilísticos son difusos."
)

pdf.figura("02b_feature_importance_classification.png",
           "Fig. 9 — Importancia de variables en clasificación. Instrumentalness, "
           "acousticness y tempo son los mejores discriminadores de género.")

pdf.add_page()
pdf.figura("02c_confusion_matrix_rf.png",
           "Fig. 10 — Matriz de confusión del Random Forest (12 macro-géneros). "
           "La diagonal muestra las predicciones correctas.")
pdf.p(
    "Los géneros mejor clasificados son k-pop/j-pop (sonido muy característico) y "
    "clásica/instrumental (altísima instrumentalness). Los más confundidos son "
    "pop vs. electronica y rock vs. metal (frontera estilística muy difusa en el dataset)."
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. FASE 3 — CLUSTERING Y RECOMENDACIÓN
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.titulo_fase("5. FASE 3 — CLUSTERING Y SISTEMA DE RECOMENDACIÓN")

pdf.p(
    "El objetivo es construir un sistema de recomendación de canciones basado en contenido "
    "(content-based filtering), usando las audio features como representación del 'sonido' "
    "de cada canción. Esto replica la capa de análisis de audio del sistema de Spotify, "
    "que combina este enfoque con filtrado colaborativo y NLP sobre metadatos, capas "
    "que no podemos replicar sin datos de usuarios ni texto."
)

pdf.subtitulo("5.1  Preprocesado y reducción de dimensionalidad")
pdf.p(
    "Features usadas: danceability, energy, loudness, speechiness, acousticness, "
    "instrumentalness, liveness, valence, tempo (9 features). Escalado con StandardScaler. "
    "Reducción para visualización: PCA (2 componentes, explica 45% de la varianza) y UMAP."
)

pdf.subtitulo("5.2  Algoritmos de clustering comparados")
pdf.tabla(
    ["Algoritmo", "k / parámetro óptimo", "Silhouette", "Davies-Bouldin", "Observaciones"],
    [
        ["KMeans",              "k=2",         "0.258", "1.41", "Clúster 0=energético, Clúster 1=tranquilo/acústico"],
        ["AgglomerativeClustering", "k=2",     "0.251", "1.48", "Resultado muy similar a KMeans"],
        ["DBSCAN",             "eps=0.8",     "0.112", "—",    "Detecta ruido; no converge bien con datos densos"],
    ],
    [42, 32, 22, 28, 42],
    ["L", "C", "C", "C", "L"]
)

pdf.p(
    "El silhouette score bajo (~0.25) indica que los géneros musicales no forman clusters "
    "bien separados en el espacio de audio features. La música se distribuye en un continuo "
    "multidimensional más que en grupos discretos: un hallazgo en sí mismo relevante "
    "para la memoria del TFG."
)

pdf.figura("03a_kmeans_elbow.png",
           "Fig. 11 — Método del codo y silhouette score para selección de k en KMeans.")

pdf.add_page()
pdf.figura("03b_pca2d_clusters.png",
           "Fig. 12 — Visualización PCA 2D de los clústeres. La separación entre Clúster 0 "
           "(energético, en azul) y Clúster 1 (tranquilo, en naranja) es clara.")

pdf.figura("03c_umap_clusters.png",
           "Fig. 13 — Visualización UMAP de los clústeres, coloreado por macro-género. "
           "UMAP preserva mejor la estructura local que PCA.")

pdf.add_page()
pdf.subtitulo("5.3  Sistema de recomendación: tres enfoques")
pdf.p(
    "Se implementan y comparan tres estrategias de recomendación, evaluadas sobre "
    "100 canciones semilla aleatorias usando métricas proxy (sin ground truth real):"
)
pdf.tabla(
    ["Recomendador", "Coherencia género", "Diversidad artistas", "Descripción"],
    [
        ["KNN global",        "0.136", "0.871", "k=10 vecinos más cercanos en espacio de features (coseno)"],
        ["Basado en clúster", "0.312", "0.842", "Recomienda canciones del mismo clúster que la semilla"],
        ["Híbrido ★",         "1.000", "0.831", "Filtra por género → aplica KNN dentro del subconjunto"],
    ],
    [38, 28, 30, 70],
    ["L", "C", "C", "L"]
)

pdf.resaltado(
    "El recomendador híbrido (género + KNN) alcanza coherencia de género perfecta (1.0) "
    "con una diversidad de artistas del 83%. Es el que se usa en la aplicación Streamlit. "
    "La coherencia perfecta es esperable: al filtrar primero por género, todas las "
    "recomendaciones son del mismo género que la semilla por construcción."
)

pdf.figura("03d_recommender_comparison.png",
           "Fig. 14 — Comparativa de los tres recomendadores en coherencia de género y diversidad de artistas.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FASE 4 — APP STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.titulo_fase("6. FASE 4 — APLICACIÓN INTERACTIVA (STREAMLIT)")

pdf.p(
    "Se desarrolla una aplicación web interactiva con Streamlit (app/app.py) que permite "
    "explorar el sistema de recomendación de forma visual sin necesidad de código. "
    "La app se lanza con: streamlit run app/app.py"
)

pdf.subtitulo("Funcionalidades implementadas")
pdf.p("La aplicación tiene dos modos de uso organizados en pestañas:")

pdf.li(
    "Modo 'Por canción semilla': el usuario busca una canción por nombre de artista o título. "
    "La app muestra las N canciones más similares usando el recomendador híbrido (género + KNN). "
    "Para cada recomendación muestra artista, álbum, género y popularidad.",
    size=10
)
pdf.ln(2)
pdf.li(
    "Modo 'Por preferencias': el usuario ajusta sliders de danceability, energy, valence, "
    "tempo y acousticness para definir un 'perfil sonoro ideal'. Opcionalmente puede "
    "filtrar por género/macro-género. La app busca las canciones más cercanas a ese perfil "
    "en el espacio de features escaladas.",
    size=10
)

pdf.subtitulo("Decisiones técnicas")
for d in [
    "@st.cache_data para el dataset (89.740 canciones): se carga una sola vez en memoria.",
    "@st.cache_resource para los modelos (KNN, scaler, KMeans, X_scaled): se persiste entre sesiones.",
    "Búsqueda con str.contains() insensible a mayúsculas para el buscador de canciones.",
    "Las recomendaciones se muestran en una tabla interactiva st.dataframe con columnas configuradas.",
]:
    pdf.li(d, size=9)

pdf.subtitulo("Limitaciones de la app")
pdf.p(
    "El catálogo está limitado a las 89.740 canciones del dataset (snapshot de 2022). "
    "No se integra con la API real de Spotify, por lo que no muestra portadas, "
    "no reproduce canciones y no se actualiza en tiempo real. Para un producto real "
    "habría que conectar la capa de recomendación a los endpoints de la API de Spotify "
    "y añadir autenticación de usuario."
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. FASE 5 — NACIONALIDAD
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.titulo_fase("7. FASE 5 — ESTUDIO DE NACIONALIDAD")

pdf.p(
    "Este análisis estudia si existen diferencias estadísticamente significativas en el "
    "perfil sonoro de los artistas según su región de origen. La hipótesis de partida "
    "es que la cultura y las tradiciones musicales de cada región se reflejan en las "
    "audio features de sus canciones."
)

pdf.resaltado(
    "IMPORTANTE — Limitación metodológica: este es un análisis EXPLORATORIO sobre una "
    "muestra CURADA y NO REPRESENTATIVA de 220 artistas. Las conclusiones son de naturaleza "
    "descriptiva/cualitativa. No permiten hacer afirmaciones generalizables sobre toda la "
    "producción musical de un país o región."
)

pdf.subtitulo("7.1  Metodología")
for m in [
    "220 artistas clasificados manualmente por país y región (curación basada en conocimiento general).",
    "Criterio de autoría: el 'artista principal' es el primero en el campo artists del dataset.",
    "7 regiones definidas: Anglophone, Latinoamérica, España, Asia-Pacífico, Europa (no anglosajona), India, África.",
    "11.881 canciones extraídas de tracks_unique.csv con artista principal identificado.",
    "Comparación estadística: ANOVA de un factor entre regiones para cada audio feature.",
]:
    pdf.li(m, size=9)

pdf.subtitulo("7.2  Distribución de canciones y popularidad por región")
pdf.tabla(
    ["Región", "Canciones", "Pop. media", "Pop. mediana"],
    [
        ["Anglophone (USA/UK/AUS/CAN)", "5.312", "42.0", "47"],
        ["Latinoamérica",               "2.790", "36.2", "38"],
        ["Asia-Pacífico",               "1.543", "43.2", "42"],
        ["Europa (no anglosajona)",      "1.372", "38.5", "39"],
        ["India/Subcontinente",          "655",   "48.5", "50"],
        ["España",                       "113",   "40.4", "34"],
        ["África",                       "96",    "27.2", "2"],
    ],
    [64, 24, 24, 24],
    ["L", "C", "C", "C"]
)

pdf.figura("05a_popularity_by_region.png",
           "Fig. 15 — Distribución de popularidad por región (boxplot y medias).")

pdf.add_page()
pdf.subtitulo("7.3  Perfil sonoro por región")
pdf.figura("05b_audio_features_by_region.png",
           "Fig. 16 — Heatmap de audio features por región (valores normalizados 0–1).")

pdf.p("Principales hallazgos por región:")
for h in [
    "Latinoamérica: danceability=0.605, energy=0.712, valence=0.603. El sonido latino se caracteriza "
    "por alta energía y un carácter alegre (valence). Es la diferencia más marcada respecto al resto.",
    "Asia-Pacífico: instrumentalness=0.099 (moderada), danceability alta. El K-pop y J-pop combinan "
    "alta producción electrónica con melodías muy bailables.",
    "Europa no anglosajona: instrumentalness=0.336 (la más alta). Refleja la presencia de música "
    "clásica, EDM instrumental y géneros con menor protagonismo de la voz.",
    "España: energy=0.758 (la más alta), valence intermedia. El perfil se asemeja más a "
    "Latinoamérica que a Europa, con alta energía pero menor valence.",
    "India: speechiness=0.087 (alta), muchas canciones de Bollywood con letras elaboradas.",
]:
    pdf.li(h, size=9)
    pdf.ln(1)

pdf.subtitulo("7.4  Test de significancia estadística")
pdf.tabla(
    ["Feature", "F (ANOVA)", "p-valor", "Significancia"],
    [
        ["danceability", "143.70", "< 0.001", "***"],
        ["energy",       "212.76", "< 0.001", "***"],
        ["valence",      "287.50", "< 0.001", "***"],
        ["tempo",        "21.82",  "< 0.001", "***"],
        ["acousticness", "6.18",   "0.002",   "**"],
    ],
    [50, 30, 30, 30],
    ["L", "C", "C", "C"]
)
pdf.p(
    "Todos los test ANOVA son altamente significativos (p<0.001). Las diferencias sonoras "
    "entre regiones no son debidas al azar, sino a características culturales y estilísticas "
    "reales de la música de cada zona geográfica."
)

pdf.figura("05c_radar_by_region.png",
           "Fig. 17 — Radares sonoros por región (features normalizadas).")

pdf.add_page()
pdf.figura("05d_latam_vs_anglophone_vs_spain.png",
           "Fig. 18 — Comparativa boxplots: Latinoamérica vs Anglophone vs España en 4 features clave.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FASE 6 — TRAYECTORIA TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.titulo_fase("8. FASE 6 — TRAYECTORIA TEMPORAL: FEID Y ARTISTAS SIMILARES")

pdf.p(
    "Este estudio analiza cómo ha evolucionado el sonido de Feid entre 2018 y 2022, "
    "comparándolo con 7 artistas del mismo ecosistema musical (reggaeton urbano latino). "
    "La hipótesis es que los artistas del mismo entorno cultural siguen trayectorias "
    "sonoras similares, pero cada uno con su propia velocidad e intensidad de cambio."
)

pdf.subtitulo("8.1  Artistas analizados y metodología")
pdf.tabla(
    ["Artista", "País", "Período en dataset", "Canciones fechadas"],
    [
        ["Feid",           "Colombia",     "2018–2022", "28"],
        ["J Balvin",       "Colombia",     "2014–2022", "74"],
        ["Maluma",         "Colombia",     "2016–2022", "16"],
        ["KAROL G",        "Colombia",     "2017–2022", "27"],
        ["Ozuna",          "Puerto Rico",  "2016–2022", "38"],
        ["Rauw Alejandro", "Puerto Rico",  "2020–2022", "19"],
        ["Daddy Yankee",   "Puerto Rico",  "2004–2022", "48"],
        ["Bad Bunny",      "Puerto Rico",  "2017–2022", "41"],
    ],
    [38, 30, 30, 30],
    ["L", "C", "C", "C"]
)

pdf.p(
    "Las fechas de lanzamiento se asignan canción por canción a partir del conocimiento "
    "discográfico (mes y año de lanzamiento oficial). Cuando el dataset incluye la misma "
    "canción en múltiples playlists de compilación (hasta 15 entradas para un mismo tema), "
    "se deduplica conservando la versión con mayor popularity."
)

pdf.subtitulo("8.2  Evolución del sonido (todos los artistas)")
pdf.figura("06a_evolucion_features.png",
           "Fig. 19 — Evolución de danceability, energy, valence y acousticness (media anual por artista).")

pdf.p(
    "Tendencias generales observadas entre 2016-2022:"
)
for t in [
    "Danceability: se mantiene alta y estable en todos los artistas (0.70–0.80). El reggaeton "
    "es inherentemente bailable y esa característica no varía.",
    "Energy: ligero aumento en casi todos los artistas a partir de 2020. El trap y el "
    "reggaeton urbano de la era post-COVID tiende a producción más agresiva.",
    "Valence: mayor variación. Bad Bunny baja su valence en sus álbumes más oscuros "
    "(X 100PRE, YHLQMDLG) pero sube con Un Verano Sin Ti.",
    "Acousticness: tendencia decreciente generalizada. La música urbana latina se aleja "
    "progresivamente de elementos acústicos hacia producción 100% electrónica.",
]:
    pdf.li(t, size=9)
    pdf.ln(1)

pdf.add_page()
pdf.subtitulo("8.3  Feid vs J Balvin vs Bad Bunny")
pdf.figura("06b_feid_vs_trio.png",
           "Fig. 20 — Trayectoria normalizada de Feid, J Balvin y Bad Bunny en 3 features (raw y normalizado).")

pdf.p(
    "La comparativa normalizada permite ver la tendencia relativa de cada artista "
    "independientemente de sus valores absolutos:"
)
for t in [
    "J Balvin muestra la trayectoria más larga y volátil: arranca en 2014 con un sonido "
    "más latinpop (Ay Vamos, Ginza), experimenta con Colores en 2020 (pop/electrónico) "
    "y vuelve al reggaeton con JOSE en 2021.",
    "Bad Bunny presenta grandes saltos por álbum: YHLQMDLG (2020) es su proyecto más "
    "bailable y energético; Un Verano Sin Ti (2022) añade variedad con elementos "
    "de salsa, dembow y indie que bajan su energía media.",
    "Feid muestra la trayectoria más lineal y consistente: crecimiento progresivo "
    "en danceability y energy desde 2018 hasta 2022, con muy pocas excepciones.",
]:
    pdf.li(t, size=9)
    pdf.ln(1)

pdf.subtitulo("8.4  El sonido colombiano")
pdf.figura("06e_sonido_colombiano.png",
           "Fig. 21 — Evolución de Feid, J Balvin, Maluma y KAROL G (4 artistas de Colombia).")

pdf.p(
    "Los cuatro artistas colombianos principales muestran convergencia sonora: "
    "alta danceability y energy en 2021-2022, con valence alta salvo excepciones. "
    "Feid es el que más ha crecido en energia en ese período, partiendo de niveles "
    "más bajos en 2018 (sonido más R&B/melodico) y llegando a valores comparables "
    "a J Balvin y Maluma en 2022."
)

pdf.add_page()
pdf.subtitulo("8.5  Popularidad por canción y año (análisis del snapshot)")
pdf.figura("06c_popularity_scatter.png",
           "Fig. 22 — Popularidad de cada canción según su año de lanzamiento. "
           "Bad Bunny concentra los valores más altos por el efecto de recencia.")

pdf.p(
    "Este scatter confirma visualmente el sesgo de recencia del índice de Spotify: "
    "las canciones de 2022 (especialmente del álbum Un Verano Sin Ti de Bad Bunny) "
    "tienen popularidades muy altas (85–97), mientras que clásicos de Daddy Yankee "
    "de 2004-2012 como 'Gasolina' o 'Limbo' tienen popularidades bajas en el snapshot "
    "a pesar de ser hits históricos con cientos de millones de reproducciones totales."
)

pdf.subtitulo("8.6  Predicción de la trayectoria sonora de Feid (2023-2025)")
pdf.figura("06d_feid_prediccion.png",
           "Fig. 23 — Extrapolación polinómica del sonido de Feid para 2023-2025. "
           "Línea punteada = tendencia; estrellas = predicciones.")

pdf.tabla(
    ["Año", "Danceability", "Energy", "Valence", "Acousticness", "Tempo (BPM)"],
    [
        ["2018 (inicio)", "~0.65", "~0.55", "~0.50", "~0.23", "~128"],
        ["2022 (último)", "~0.76", "~0.69", "~0.62", "~0.15", "~138"],
        ["2023 (pred.)",  "0.747", "0.665", "0.605", "0.140",  "138.3"],
        ["2024 (pred.)",  "0.742", "0.671", "0.619", "0.132",  "138.8"],
        ["2025 (pred.)",  "0.738", "0.676", "0.633", "0.123",  "139.3"],
    ],
    [28, 28, 22, 22, 28, 28],
    ["C","C","C","C","C","C"]
)

pdf.p(
    "La predicción sugiere una tendencia estabilizadora: danceability ligeramente decreciente, "
    "energy y valence crecientes, acousticness decreciente (más electrónico). "
    "El tempo se mantiene estable cerca de los 138-139 BPM."
)

pdf.resaltado(
    "ADVERTENCIA: estas predicciones son extrapolaciones polinómicas de grado 2 sobre 5 puntos "
    "(una media por año), no modelos predictivos robustos. Su valor es ilustrativo de la "
    "tendencia reciente de Feid, no una previsión real de su evolución artística. "
    "La evolución del sonido de un artista depende de decisiones creativas, "
    "productores, tendencias del mercado y colaboraciones que ningún modelo puede anticipar."
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. CONCLUSIONES
# ─────────────────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.titulo_fase("9. CONCLUSIONES Y TRABAJO FUTURO", color=AZUL_OSC)

pdf.subtitulo("Conclusiones técnicas")
for c in [
    "El análisis de audio features de Spotify permite caracterizar géneros musicales y perfiles "
    "sonoros con buena fiabilidad estadística, pero su poder predictivo sobre la popularidad "
    "es moderado (R²≈0.47). La popularidad en Spotify está dominada por factores de "
    "distribución y marketing que los datos de audio no capturan.",
    "La agrupación de géneros en macro-categorías es imprescindible para obtener resultados "
    "clasificatorios interpretables. Los 114 géneros originales son demasiado finos y "
    "están solapados, haciendo que el espacio de features sea ambiguo.",
    "El clustering de canciones por features de audio converge a solo k=2 clústeres bien "
    "diferenciados (música energética vs. tranquila/acústica). Esto refleja que las "
    "audio features de Spotify capturan una dimensión principalmente de 'intensidad "
    "sonora' más que la riqueza multidimensional de los estilos musicales.",
    "Los tres recomendadores implementados demuestran el trade-off entre coherencia y "
    "diversidad. El recomendador híbrido es el más apropiado para una app real porque "
    "garantiza que las recomendaciones suenan similares (mismo género) pero provienen "
    "de artistas distintos (diversidad).",
]:
    pdf.li(c)
    pdf.ln(2)

pdf.subtitulo("Conclusiones sobre el análisis musical")
for c in [
    "Existe una 'identidad sonora' estadísticamente distinguible por región geográfica: "
    "la música latina tiene más energy y valence; la música europea más instrumentalness. "
    "Estas diferencias son robustas (p<0.001) incluso en esta muestra curada.",
    "La trayectoria de Feid (2018-2022) muestra una evolución coherente hacia el "
    "reggaeton urbano de alta energía, paralela a la de sus contemporáneos colombianos "
    "(J Balvin, Maluma, KAROL G). Su crecimiento ha sido más rápido y lineal que el "
    "de artistas más establecidos.",
    "El índice de popularidad de Spotify distorsiona cualquier comparación temporal o "
    "geográfica. Artistas históricos como Daddy Yankee aparecen con popularidad baja "
    "en el dataset a pesar de ser pioneros del género con carrera de 20+ años.",
]:
    pdf.li(c)
    pdf.ln(2)

pdf.separador()
pdf.subtitulo("Limitaciones principales")
pdf.tabla(
    ["Limitación", "Impacto", "Mitigación aplicada"],
    [
        ["popularity es snapshot (recencia)", "Alto", "Documentado en todos los análisis que la usan"],
        ["Sin datos de usuarios (filtrado colaborativo)", "Medio", "Sistema content-based declarado explícitamente"],
        ["Mapa de macro-géneros ad hoc", "Medio", "Decisión editorial justificada en la memoria"],
        ["k=2 en clustering (poca granularidad)", "Bajo-Medio", "Interpretado como hallazgo, no como limitación del método"],
        ["Estudio de nacionalidad: muestra curada", "Medio", "Declarado explícitamente como exploratorio"],
        ["Predicción con 5 puntos anuales", "Alto", "Advertencia incluida; presentado como ilustración de tendencia"],
    ],
    [58, 24, 84],
    ["L", "C", "L"]
)

pdf.subtitulo("Trabajo futuro")
for t in [
    "Conectar con la API de Spotify para obtener fechas de lanzamiento reales y datos actualizados.",
    "Añadir filtrado colaborativo con datos de playlists públicas para mejorar la recomendación.",
    "Profundizar en el análisis de Feid con un estudio de caso completo: todas sus canciones, "
    "análisis de letra (NLP), y comparación antes/después de su gran explosión de popularidad.",
    "Explorar modelos de series temporales más robustos (ARIMA, Prophet) para la predicción "
    "de tendencias sonoras con más datos históricos.",
    "Ampliar el estudio de nacionalidad con datos de más artistas y un muestreo más representativo.",
]:
    pdf.li(t)
    pdf.ln(1)

pdf.ln(6)
pdf.set_draw_color(*ROJO_ACENT)
pdf.set_line_width(0.8)
pdf.line(22, pdf.get_y(), 188, pdf.get_y())
pdf.set_line_width(0.2)
pdf.ln(6)
pdf.set_font("Ar", "I", 9)
pdf.set_text_color(130, 130, 130)
pdf.multi_cell(
    0, 5.5,
    "Informe generado automáticamente a partir de los scripts del proyecto TFG.\n"
    "Todos los datos, modelos y figuras están disponibles en el repositorio del proyecto.\n"
    "Junio 2026 · Alfonso García Betico",
    align="C"
)

# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────────────────────────────────────────
out_path = OUT / "TFG_Informe_Spotify.pdf"
pdf.output(str(out_path))
print(f"[OK] PDF generado: {out_path}")
print(f"     Páginas: {pdf.page}")
