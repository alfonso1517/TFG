# =============================================================================
# FASE A — LIMPIEZA DE DATOS EXHAUSTIVA
# TFG: IA y Análisis Estadístico aplicado a la Industria Musical (Spotify)
# =============================================================================
# Entrada:  data/processed/tracks_unique.csv  (89.740 canciones, 1 por track_id)
# Salida:   data/processed/tracks_model.csv   (dataset limpio + feature engineering)
#           results/A_limpieza_exhaustiva.md
# =============================================================================

import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

ROOT  = Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
RES   = ROOT / "results"

print("=" * 70)
print("FASE A — LIMPIEZA DE DATOS EXHAUSTIVA")
print("=" * 70)

# ─── Carga ────────────────────────────────────────────────────────────────────
df = pd.read_csv(PROC / "tracks_unique.csv")
n_inicial = len(df)
print(f"\n[A.0] Dataset cargado: {n_inicial:,} filas, {df.shape[1]} columnas")

# =============================================================================
# A.1 AUDITORÍA COMPLETA DE CALIDAD
# =============================================================================
print("\n" + "─" * 60)
print("A.1  AUDITORÍA DE CALIDAD")
print("─" * 60)

print("\n--- Tipos de datos ---")
print(df.dtypes.to_string())

print("\n--- Nulos por columna ---")
nulos = df.isnull().sum()
print(nulos[nulos > 0] if nulos.sum() > 0 else "Sin nulos")

print("\n--- track_id únicos ---")
assert df["track_id"].nunique() == len(df), "¡Hay duplicados de track_id!"
print(f"OK — {df['track_id'].nunique():,} track_ids únicos (= total filas)")

print("\n--- Géneros por frecuencia (top 20) ---")
genre_counts = df["track_genre"].value_counts()
print(genre_counts.head(20).to_string())
print(f"\nGéneros con < 100 canciones: {(genre_counts < 100).sum()}")

print("\n--- tempo == 0 ---")
n_tempo_0 = (df["tempo"] == 0).sum()
print(f"Filas con tempo=0: {n_tempo_0}")

print("\n--- popularity == 0 ---")
n_pop_0 = (df["popularity"] == 0).sum()
print(f"Filas con popularity=0: {n_pop_0} ({n_pop_0/len(df)*100:.1f}%)")

print("\n--- Rango de variables numéricas ---")
num_cols = ["popularity", "duration_ms", "danceability", "energy", "loudness",
            "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo"]

stats_rows = []
for col in num_cols:
    stats_rows.append({
        "feature":  col,
        "min":      round(df[col].min(), 3),
        "max":      round(df[col].max(), 3),
        "mean":     round(df[col].mean(), 3),
        "median":   round(df[col].median(), 3),
        "skew":     round(df[col].skew(), 3),
    })
stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False))

# =============================================================================
# A.2 TRATAMIENTO DE OUTLIERS
# =============================================================================
print("\n" + "─" * 60)
print("A.2  TRATAMIENTO DE OUTLIERS")
print("─" * 60)

# --- Tempo == 0: imputar con mediana del género ---
if n_tempo_0 > 0:
    mediana_por_genero = df.groupby("track_genre")["tempo"].transform(
        lambda x: x[x > 0].median() if (x > 0).any() else x.median()
    )
    df.loc[df["tempo"] == 0, "tempo"] = mediana_por_genero[df["tempo"] == 0]
    print(f"  [OK] {n_tempo_0} filas con tempo=0 imputadas con mediana del género")
else:
    print("  [OK] No hay filas con tempo=0 (ya imputadas en Fase 0)")

# --- Duration: convertir a minutos y filtrar outliers ---
df["duration_min"] = df["duration_ms"] / 60_000

Q1 = df["duration_min"].quantile(0.25)
Q3 = df["duration_min"].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR
print(f"\n  duration_min → IQR: [{Q1:.2f}, {Q3:.2f}], límites: [{lim_inf:.2f}, {lim_sup:.2f}]")

muy_largas  = df[df["duration_min"] > 15]
muy_cortas  = df[df["duration_min"] < 0.5]
print(f"  Canciones > 15 min: {len(muy_largas)}")
print(f"  Canciones < 0.5 min: {len(muy_cortas)}")
if len(muy_largas) > 0:
    print("  Muestra > 15 min:")
    print(muy_largas[["track_name", "artists", "duration_min", "track_genre"]].head(5).to_string(index=False))

# Excluir para el conjunto de modelado
mask_dur = (df["duration_min"] >= 0.5) & (df["duration_min"] <= 15)
n_antes_dur = len(df)
df_model = df[mask_dur].copy()
n_despues_dur = len(df_model)
print(f"\n  [DECISIÓN] Excluir duración < 0.5 min o > 15 min: "
      f"{n_antes_dur - n_despues_dur} filas eliminadas")

# --- Loudness: excluir outliers extremos < -40 dB ---
n_loud_outlier = (df_model["loudness"] < -40).sum()
print(f"\n  loudness < -40 dB: {n_loud_outlier} filas")
df_model = df_model[df_model["loudness"] >= -40].copy()
print(f"  [DECISIÓN] Excluidas {n_loud_outlier} filas con loudness extremo")

n_tras_outliers = len(df_model)
print(f"\n  Pipeline de filas: {n_inicial:,} → {n_tras_outliers:,} "
      f"(perdidas: {n_inicial - n_tras_outliers})")

# --- Canciones muy instrumentales pero muy populares (hallazgo interesante) ---
print("\n  Canciones instrumentalness > 0.9 Y popularity > 60:")
mask_instr = (df_model["instrumentalness"] > 0.9) & (df_model["popularity"] > 60)
casos_instr = df_model[mask_instr][["track_name", "artists", "track_genre",
                                     "instrumentalness", "popularity"]]
print(f"  Total: {len(casos_instr)} canciones")
if len(casos_instr) > 0:
    print(casos_instr.sort_values("popularity", ascending=False).head(10).to_string(index=False))

# --- Correlaciones entre features (multicolinealidad) ---
print("\n  Correlaciones de Spearman altas (|r| > 0.7) entre features:")
corr_matrix = df_model[num_cols + ["duration_min"]].corr(method="spearman")
high_corr = []
cols_c = corr_matrix.columns.tolist()
for i in range(len(cols_c)):
    for j in range(i + 1, len(cols_c)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            high_corr.append((cols_c[i], cols_c[j], round(r, 4)))

if high_corr:
    for a, b, r in high_corr:
        print(f"    {a} ↔ {b}: r={r}")
else:
    print("  No se detectaron pares con |r| > 0.7")

# =============================================================================
# A.3 FEATURE ENGINEERING
# =============================================================================
print("\n" + "─" * 60)
print("A.3  FEATURE ENGINEERING")
print("─" * 60)

# Log-transforms para variables con distribución sesgada
df_model["log_instrumentalness"] = np.log1p(df_model["instrumentalness"])
df_model["log_speechiness"]      = np.log1p(df_model["speechiness"])
df_model["log_acousticness"]     = np.log1p(df_model["acousticness"])
print("  [OK] Log-transforms: log_instrumentalness, log_speechiness, log_acousticness")

# Variable binaria: canción popular (umbral = 50)
df_model["is_popular"] = (df_model["popularity"] >= 50).astype(int)
n_popular   = df_model["is_popular"].sum()
n_no_popular = len(df_model) - n_popular
print(f"  [OK] is_popular (umbral=50): {n_popular:,} populares ({n_popular/len(df_model)*100:.1f}%), "
      f"{n_no_popular:,} no populares ({n_no_popular/len(df_model)*100:.1f}%)")

# Ratio energy / acousticness (proxy de "electronización")
df_model["electronic_ratio"] = df_model["energy"] / (df_model["acousticness"] + 0.01)
print("  [OK] electronic_ratio = energy / (acousticness + 0.01)")

# Macro-género (mapa editorial de 12 categorías — versión expandida cubriendo los 114 géneros)
genre_map = {
    # folk / acústico
    "acoustic": "folk-acustico", "singer-songwriter": "folk-acustico",
    "songwriter": "folk-acustico", "folk": "folk-acustico",
    "country": "folk-acustico", "bluegrass": "folk-acustico",
    "guitar": "folk-acustico", "road-trip": "folk-acustico",
    # pop
    "pop": "pop", "dance": "pop", "synth-pop": "pop",
    "indie-pop": "pop", "pop-film": "pop", "power-pop": "pop",
    "disney": "pop", "happy": "pop", "romance": "pop",
    "sad": "pop", "party": "pop", "children": "pop",
    "kids": "pop", "chill": "pop",
    # electrónica
    "edm": "electronica", "techno": "electronica", "trance": "electronica",
    "dubstep": "electronica", "house": "electronica", "electro": "electronica",
    "disco": "electronica", "minimal-techno": "electronica", "detroit-techno": "electronica",
    "chicago-house": "electronica", "deep-house": "electronica",
    "club": "electronica", "breakbeat": "electronica", "hardstyle": "electronica",
    "progressive-house": "electronica", "electronic": "electronica",
    "j-dance": "electronica", "dub": "electronica",
    # hip-hop / urban
    "hip-hop": "hip-hop", "rap": "hip-hop", "trap": "hip-hop",
    "r-n-b": "hip-hop", "soul": "hip-hop", "funk": "hip-hop",
    "trip-hop": "hip-hop",
    # latino
    "latin": "latino", "reggaeton": "latino", "salsa": "latino",
    "cumbia": "latino", "bachata": "latino", "reggaeton-colombiano": "latino",
    "latin-alternative": "latino", "spanish": "latino",
    "tango": "latino", "bossanova": "latino",
    # rock
    "rock": "rock", "alternative": "rock", "grunge": "rock",
    "punk": "rock", "emo": "rock", "indie": "rock", "psych-rock": "rock",
    "alt-rock": "rock", "punk-rock": "rock", "rock-n-roll": "rock",
    "garage": "rock", "british": "rock", "ska": "rock",
    "j-rock": "rock",
    # metal
    "metal": "metal", "heavy-metal": "metal", "black-metal": "metal",
    "death-metal": "metal", "hard-rock": "metal", "metalcore": "metal",
    # clásica / instrumental
    "classical": "clasica", "opera": "clasica", "piano": "clasica",
    "chamber": "clasica", "orchestra": "clasica",
    # jazz / blues
    "jazz": "jazz-blues", "blues": "jazz-blues", "gospel": "jazz-blues",
    "mpb": "jazz-blues",
    # k-pop / j-pop / asiática
    "k-pop": "kpop-jpop", "j-pop": "kpop-jpop", "j-idol": "kpop-jpop",
    "cantopop": "kpop-jpop", "mandopop": "kpop-jpop",
    "anime": "kpop-jpop", "j-rock": "kpop-jpop",
    # world / regional
    "world-music": "world", "afrobeat": "world", "sertanejo": "world",
    "pagode": "world", "samba": "world", "indian": "world",
    "forro": "world", "reggae": "world", "malay": "world",
    "french": "world", "german": "world", "swedish": "world",
    "turkish": "world", "iranian": "world", "philippines-opm": "world",
    "dub": "world",
    # world / regional (adicionales)
    "brazil": "world", "dancehall": "world",
    # electrónica (adicionales)
    "drum-and-bass": "electronica", "idm": "electronica", "industrial": "electronica",
    # rock / metal (adicionales)
    "goth": "metal", "grindcore": "metal", "hardcore": "metal",
    "rockabilly": "rock", "groove": "hip-hop",
    # folk-acustico (adicionales)
    "honky-tonk": "folk-acustico",
    # latino (adicionales — el género se llama "latino" además de "latin")
    "latino": "latino",
    # otros (ambient, sleep, comedy, etc.)
    "ambient": "otros", "new-age": "otros", "comedy": "otros",
    "show-tunes": "otros", "sleep": "otros", "study": "otros",
    "new-release": "otros",
}

df_model["macro_genre"] = df_model["track_genre"].map(genre_map).fillna("otros")
print("\n  Distribución de macro-géneros:")
mg_counts = df_model["macro_genre"].value_counts()
for mg, cnt in mg_counts.items():
    print(f"    {mg:<20} {cnt:>6,} ({cnt/len(df_model)*100:.1f}%)")

generos_sin_mapear = df_model[df_model["macro_genre"] == "otros"]["track_genre"].unique()
print(f"\n  Géneros mapeados a 'otros': {len(generos_sin_mapear)}")
if len(generos_sin_mapear) <= 20:
    print("  ", generos_sin_mapear.tolist())

# Tempo normalizado a [0,1] (para radar en Fase B)
df_model["tempo_norm"] = (df_model["tempo"] - df_model["tempo"].min()) / \
                         (df_model["tempo"].max() - df_model["tempo"].min())
print("\n  [OK] tempo_norm: tempo normalizado a [0,1]")

# =============================================================================
# GUARDAR tracks_model.csv
# =============================================================================
out_path = PROC / "tracks_model.csv"
df_model.to_csv(out_path, index=False)
print(f"\n[OK] tracks_model.csv guardado: {len(df_model):,} filas, {df_model.shape[1]} columnas")

# =============================================================================
# RESUMEN MARKDOWN
# =============================================================================
md = f"""# Fase A — Limpieza de Datos Exhaustiva

## Pipeline de filas

| Etapa | Filas | Cambio |
|-------|-------|--------|
| Dataset original (dataset.csv) | 114.000 | — |
| Tras eliminar fila nula | 113.999 | −1 |
| Tras deduplicación por track_id (tracks_unique.csv) | {n_inicial:,} | −{114000 - n_inicial:,} |
| Tras excluir duration < 0.5 min o > 15 min | {n_antes_dur - (n_antes_dur - n_despues_dur):,} | −{n_antes_dur - n_despues_dur} |
| Tras excluir loudness < −40 dB | {n_tras_outliers:,} | −{n_despues_dur - n_tras_outliers} |
| **tracks_model.csv (final)** | **{n_tras_outliers:,}** | — |

## Decisiones de limpieza

| Variable | Problema | Decisión | Justificación |
|----------|----------|----------|---------------|
| `tempo` | {n_tempo_0} filas con valor 0 | Imputar con mediana del género | No son canciones sin ritmo; error de detección de Spotify |
| `duration_ms` | Outliers extremos (< 0.5 min o > 15 min) | Excluir del conjunto de modelado | Probables podcasts, grabaciones especiales o intros; no son canciones típicas |
| `loudness` | {n_loud_outlier} filas < −40 dB | Excluir | Grabaciones con problemas técnicos o silencio; valores anómalos |
| `popularity == 0` | {n_pop_0} filas (10.5%) | Mantener | No son errores; son canciones con pocas reproducciones recientes |

## Hallazgo: canciones muy instrumentales y muy populares

Se identificaron **{len(casos_instr)} canciones** con `instrumentalness > 0.9` y `popularity > 60`.
Estas son excepciones a la tendencia general (correlación negativa entre instrumentalness y popularidad).
Son candidatas interesantes para análisis cualitativo en la memoria del TFG.

## Correlaciones altas detectadas (|r_Spearman| > 0.7)

"""
if high_corr:
    for a, b, r in high_corr:
        md += f"- **{a} ↔ {b}**: r = {r}  \n"
    md += "\nEstas correlaciones implican posible multicolinealidad. En modelos lineales habría que actuar, pero Random Forest y XGBoost son robustos frente a este problema.\n"
else:
    md += "No se detectaron pares con correlación > 0.7. Las features de audio de Spotify tienen baja multicolinealidad.\n"

md += f"""
## Variables creadas (feature engineering)

| Variable | Fórmula | Justificación |
|----------|---------|---------------|
| `duration_min` | `duration_ms / 60000` | Más interpretable que milisegundos |
| `log_instrumentalness` | `log(1 + instrumentalness)` | Distribución muy sesgada a la derecha; log reduce el sesgo |
| `log_speechiness` | `log(1 + speechiness)` | Ídem |
| `log_acousticness` | `log(1 + acousticness)` | Ídem |
| `is_popular` | `popularity >= 50 → 1, else 0` | {n_popular/len(df_model)*100:.1f}% populares. Umbral 50 separa el cuartil superior |
| `electronic_ratio` | `energy / (acousticness + 0.01)` | Proxy de cuánto suena "electrónica" vs "acústica" la canción |
| `macro_genre` | Mapa editorial de 114 → 12 categorías | Agrupa géneros similares; mejora clasificación y visualización |
| `tempo_norm` | `(tempo − min) / (max − min)` | Normalización para comparación en gráficos radar |

## Distribución de macro-géneros

| Macro-género | Canciones | % |
|-------------|-----------|---|
"""
for mg, cnt in mg_counts.items():
    md += f"| {mg} | {cnt:,} | {cnt/len(df_model)*100:.1f}% |\n"

RES.mkdir(exist_ok=True)
(RES / "A_limpieza_exhaustiva.md").write_text(md, encoding="utf-8")
print("\n[OK] results/A_limpieza_exhaustiva.md guardado")
print("\n[FASE A COMPLETADA]")
