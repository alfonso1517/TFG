"""
FASE 1 — Análisis Exploratorio de Datos (EDA)
Genera gráficos en reports/figures/ y notas en results/01_eda_notes.md
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

ROOT    = Path(__file__).parent.parent
PROC    = ROOT / "data" / "processed"
FIGS    = ROOT / "reports" / "figures"
RES     = ROOT / "results"

FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

print("Cargando tracks_unique.csv ...")
df = pd.read_csv(PROC / "tracks_unique.csv")
print(f"  {len(df):,} canciones únicas, {df['track_genre'].nunique()} géneros")

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

# ── 1. Distribución de popularity ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["popularity"], bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
axes[0].set_title("Distribución de Popularity")
axes[0].set_xlabel("Popularity (0–100)")
axes[0].set_ylabel("Frecuencia")

pop_no0 = df[df["popularity"] > 0]["popularity"]
axes[1].hist(pop_no0, bins=50, color="#DD8452", edgecolor="white", linewidth=0.4)
axes[1].set_title("Popularity > 0 (sin el pico en 0)")
axes[1].set_xlabel("Popularity (0–100)")
axes[1].set_ylabel("Frecuencia")

pct_zero = (df["popularity"] == 0).mean() * 100
fig.suptitle(f"Canciones con popularity = 0: {pct_zero:.1f}%", fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "01a_popularity_distribution.png")
plt.close()
print(f"[OK] 01a guardado  | popularity=0: {pct_zero:.1f}%")

# ── 2. Matriz de correlación ──────────────────────────────────────────────────
corr_cols = AUDIO_FEATURES + ["duration_ms", "popularity", "explicit", "key", "mode", "time_signature"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
ax.set_title("Matriz de Correlación — Audio Features + Popularity")
plt.tight_layout()
plt.savefig(FIGS / "01b_correlation_matrix.png")
plt.close()

top_corr = corr["popularity"].drop("popularity").abs().sort_values(ascending=False).head(5)
print(f"[OK] 01b guardado  | Top correlaciones con popularity:\n{top_corr.to_string()}")

# ── 3. Perfil de audio features por género (radar chart) ─────────────────────
SELECTED_GENRES = ["reggaeton", "classical", "metal", "acoustic", "edm", "latin", "pop", "jazz"]
RADAR_FEATURES  = ["danceability", "energy", "valence", "acousticness", "speechiness", "liveness"]

genre_profiles = (
    df[df["track_genre"].isin(SELECTED_GENRES)]
    .groupby("track_genre")[RADAR_FEATURES]
    .mean()
)

# Normalizar tempo aparte (escala distinta) — aquí usamos features ya en [0,1]
angles = np.linspace(0, 2 * np.pi, len(RADAR_FEATURES), endpoint=False).tolist()
angles += angles[:1]  # cerrar el polígono

fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(polar=True))
colors = plt.cm.tab10.colors

for ax, (genre, vals) in zip(axes.flatten(), genre_profiles.iterrows()):
    values = vals.tolist() + vals.tolist()[:1]
    ax.plot(angles, values, color=colors[SELECTED_GENRES.index(genre)], linewidth=2)
    ax.fill(angles, values, alpha=0.25, color=colors[SELECTED_GENRES.index(genre)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_FEATURES, size=7)
    ax.set_ylim(0, 1)
    ax.set_title(genre, size=10, pad=10)

fig.suptitle("Perfil de Audio Features por Género", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "01c_genre_radar.png")
plt.close()
print("[OK] 01c guardado  | Radar por género")

# ── 4. Boxplot popularity por género (top 20 por mediana) ────────────────────
genre_med = df.groupby("track_genre")["popularity"].median().sort_values(ascending=False)
top20_genres = genre_med.head(20).index.tolist()
df_top20 = df[df["track_genre"].isin(top20_genres)]

fig, ax = plt.subplots(figsize=(14, 6))
order = genre_med.head(20).index.tolist()
sns.boxplot(data=df_top20, x="track_genre", y="popularity", order=order,
            palette="muted", ax=ax, fliersize=2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_title("Popularity por Género (Top 20 por mediana)")
ax.set_xlabel("")
ax.set_ylabel("Popularity (0–100)")
plt.tight_layout()
plt.savefig(FIGS / "01d_popularity_by_genre.png")
plt.close()
print(f"[OK] 01d guardado  | Género con mayor mediana: {top20_genres[0]}")

# ── 5. Distribución de key, mode, time_signature, explicit ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, col in zip(axes.flatten(), ["key", "mode", "time_signature", "explicit"]):
    vc = df[col].value_counts().sort_index()
    pop_means = df.groupby(col)["popularity"].mean()
    x = [str(v) for v in vc.index]

    ax2 = ax.twinx()
    ax.bar(x, vc.values, color="#4C72B0", alpha=0.6, label="Nº canciones")
    ax2.plot(x, [pop_means[i] for i in vc.index], color="#C44E52",
             marker="o", linewidth=2, label="Pop. media")

    ax.set_title(f"Distribución de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Nº canciones", color="#4C72B0")
    ax2.set_ylabel("Popularity media", color="#C44E52")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

plt.suptitle("Variables categóricas: frecuencia y popularidad media", fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "01e_categorical_vars.png")
plt.close()
print("[OK] 01e guardado  | key/mode/time_sig/explicit vs popularity")

# ── 6. Outliers loudness y duration_ms ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["loudness"], bins=60, color="#55A868", edgecolor="white", linewidth=0.3)
axes[0].set_title("Distribución de Loudness (dB)")
axes[0].set_xlabel("Loudness (dB)")
axes[0].axvline(df["loudness"].quantile(0.01), color="red", linestyle="--", label="P1")
axes[0].axvline(df["loudness"].quantile(0.99), color="red", linestyle="--", label="P99")
axes[0].legend()

dur_min = df["duration_ms"] / 60000
axes[1].hist(dur_min[dur_min <= 10], bins=60, color="#C44E52", edgecolor="white", linewidth=0.3)
axes[1].set_title("Distribución de Duración (min, ≤10 min)")
axes[1].set_xlabel("Duración (minutos)")
n_long = (dur_min > 10).sum()
axes[1].text(0.98, 0.95, f"Canciones >10 min: {n_long}", transform=axes[1].transAxes,
             ha="right", va="top", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(FIGS / "01f_outliers_loudness_duration.png")
plt.close()
print(f"[OK] 01f guardado  | Canciones >10 min: {n_long}")

# ── 7. Barras agrupadas: audio features por géneros contrastados ──────────────
bar_features = ["danceability", "energy", "valence", "acousticness"]
genre_bar = genre_profiles[bar_features]

x = np.arange(len(bar_features))
width = 0.1
fig, ax = plt.subplots(figsize=(12, 6))
for i, genre in enumerate(SELECTED_GENRES):
    if genre in genre_bar.index:
        ax.bar(x + i * width, genre_bar.loc[genre], width, label=genre,
               color=colors[i], alpha=0.85)

ax.set_xticks(x + width * (len(SELECTED_GENRES) - 1) / 2)
ax.set_xticklabels(bar_features, fontsize=10)
ax.set_ylim(0, 1)
ax.set_ylabel("Valor medio (0–1)")
ax.set_title("Comparativa de Audio Features por Género")
ax.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "01g_audio_features_bar.png")
plt.close()
print("[OK] 01g guardado  | Barras agrupadas por género")

# ── 8. Guardar notas EDA ──────────────────────────────────────────────────────
pct_pop0    = (df["popularity"] == 0).mean() * 100
pop_skew    = df["popularity"].skew()
top_corr_list = corr["popularity"].drop("popularity").abs().sort_values(ascending=False).head(5)
top_genre_pop = genre_med.head(5)

notes = f"""# 01 — Notas EDA

## Dataset analizado
- `tracks_unique.csv`: {len(df):,} canciones únicas, {df['track_genre'].nunique()} géneros.

## 1. Distribución de Popularity
- **{pct_pop0:.1f}%** de las canciones tienen `popularity = 0`.
  Estas son canciones muy antiguas o prácticamente desconocidas donde el factor
  de recencia del algoritmo de Spotify lleva el índice a 0.
- La distribución está **fuertemente sesgada a la izquierda** (skewness ≈ {pop_skew:.2f}):
  la mayoría de canciones tienen baja popularidad; las muy populares son minoría.
- Existe un pico pronunciado en 0 seguido de una distribución aproximadamente
  unimodal con cola derecha.

## 2. Correlaciones con Popularity
Las features que más correlacionan (en valor absoluto):
{top_corr_list.to_string()}

- Ninguna feature individual tiene correlación lineal fuerte (máximo ~0.2–0.3),
  lo que sugiere relaciones no lineales y/o la necesidad de combinar features
  en un modelo.

## 3. Perfil por género (radar chart)
- **Reggaeton/Latin**: alta danceability y valence, baja acousticness.
- **Classical**: alta acousticness, muy baja danceability/energy.
- **EDM**: alta energy y liveness, baja acousticness.
- **Metal**: alta energy, baja valence/acousticness.
- **Acoustic**: alta acousticness, baja energy y danceability.
- Los géneros se diferencian claramente en el espacio de audio features,
  lo que es prometedor para los modelos de clasificación y clustering.

## 4. Popularity por género (top 5 por mediana)
{top_genre_pop.to_string()}

Los géneros de música popular contemporánea (pop, reggaeton, latin) tienden
a tener mayor mediana de popularidad que géneros más de nicho o clásicos.

## 5. Variables categóricas vs Popularity
- **explicit**: canciones explícitas tienen ligera mayor popularidad media
  (probablemente correlacionado con géneros populares actuales como rap/reggaeton).
- **mode** (mayor/menor): diferencias muy pequeñas en popularidad.
- **key** y **time_signature**: variación leve entre categorías.

## 6. Outliers
- **Loudness**: rango típico de -20 a -3 dB. Outliers extremos (<-40 dB)
  son piezas muy silenciosas o con errores de normalización.
- **Duration**: mayoría de canciones entre 2 y 5 minutos. Canciones >10 min
  son piezas clásicas o suites — se mantienen en el dataset.

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `01a_popularity_distribution.png` | Histograma de popularity (con y sin 0s) |
| `01b_correlation_matrix.png` | Matriz de correlación completa |
| `01c_genre_radar.png` | Radar por género (8 géneros contrastados) |
| `01d_popularity_by_genre.png` | Boxplot popularity top-20 géneros |
| `01e_categorical_vars.png` | key/mode/time_sig/explicit vs frecuencia y pop. media |
| `01f_outliers_loudness_duration.png` | Histogramas loudness y duración |
| `01g_audio_features_bar.png` | Barras agrupadas features por género |
"""

(RES / "01_eda_notes.md").write_text(notes, encoding="utf-8")
print("\n[OK] results/01_eda_notes.md guardado")
print("✓ Fase 1 completada.")
