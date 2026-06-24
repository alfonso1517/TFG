"""
FASE 0 — Limpieza y preparación de datos
Genera:
  - data/processed/tracks_unique.csv   (una fila por track_id)
  - data/processed/tracks_long.csv     (todas las combinaciones cancion-genero)
  - results/00_data_cleaning_notes.md
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES  = ROOT / "results"

PROC.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

# ── 1. Carga ──────────────────────────────────────────────────────────────────
print("Cargando dataset.csv ...")
df = pd.read_csv(RAW / "dataset.csv")
print(f"  Filas iniciales : {len(df):,}")
print(f"  Columnas        : {list(df.columns)}")

# ── 2. Eliminar fila con artista/album/track nulos ────────────────────────────
null_mask = df[["artists", "album_name", "track_name"]].isnull().any(axis=1)
n_null = null_mask.sum()
df = df[~null_mask].reset_index(drop=True)
print(f"\n[OK] Filas nulas eliminadas: {n_null}")

# ── 3. Estadísticas previas a deduplicación ───────────────────────────────────
n_total       = len(df)
n_unique_ids  = df["track_id"].nunique()
n_genres      = df["track_genre"].nunique()
n_dup_rows    = n_total - n_unique_ids
tempo_zeros   = (df["tempo"] == 0).sum()

print(f"\n── Estadísticas pre-deduplicación ──")
print(f"  Filas totales            : {n_total:,}")
print(f"  track_id únicos          : {n_unique_ids:,}")
print(f"  Filas duplicadas (dif. género): {n_dup_rows:,}")
print(f"  Géneros únicos           : {n_genres}")
print(f"  Filas con tempo == 0     : {tempo_zeros}")

# ── 4. Tratar tempo == 0 (imputar mediana por género) ────────────────────────
#    Se imputa ANTES de deduplicar para que la mediana use más contexto.
genre_tempo_median = df[df["tempo"] > 0].groupby("track_genre")["tempo"].median()
def impute_tempo(row):
    if row["tempo"] == 0:
        return genre_tempo_median.get(row["track_genre"], df[df["tempo"] > 0]["tempo"].median())
    return row["tempo"]

df["tempo"] = df.apply(impute_tempo, axis=1)
print(f"\n[OK] tempo == 0 imputado con mediana del género ({tempo_zeros} filas)")

# ── 5. Versión LONG (todas las combinaciones cancion-genero, post-limpieza) ───
tracks_long = df.copy()
tracks_long.to_csv(PROC / "tracks_long.csv", index=False)
print(f"\n[OK] tracks_long.csv guardado: {len(tracks_long):,} filas")

# ── 6. Deduplicar por track_id → tracks_unique ───────────────────────────────
#    Estrategia: primera aparición (orden original del CSV).
#    La columna track_genre de tracks_unique representa el "primer género" asignado.
tracks_unique = df.drop_duplicates(subset="track_id", keep="first").reset_index(drop=True)
tracks_unique.to_csv(PROC / "tracks_unique.csv", index=False)
print(f"[OK] tracks_unique.csv guardado: {len(tracks_unique):,} filas")

# ── 7. Verificación de rangos numéricos ──────────────────────────────────────
numeric_cols = [
    "popularity", "duration_ms", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature"
]
print("\n── Rangos numéricos (tracks_unique) ──")
desc = tracks_unique[numeric_cols].describe().T[["min", "max", "mean", "50%"]]
desc.columns = ["min", "max", "mean", "median"]
print(desc.to_string())

# duración en minutos
tracks_unique["duration_min"] = tracks_unique["duration_ms"] / 60000
print(f"\n  duration_min: min={tracks_unique['duration_min'].min():.2f}  "
      f"max={tracks_unique['duration_min'].max():.2f}  "
      f"median={tracks_unique['duration_min'].median():.2f}")

# ── 8. Outliers en loudness y duration_ms ────────────────────────────────────
loud_q01 = tracks_unique["loudness"].quantile(0.01)
loud_q99 = tracks_unique["loudness"].quantile(0.99)
dur_q99  = tracks_unique["duration_min"].quantile(0.99)
print(f"\n  loudness  P1={loud_q01:.1f} dB  P99={loud_q99:.1f} dB  "
      f"(valores negativos son normales — 0 dB es el máximo teórico)")
print(f"  duration  P99={dur_q99:.1f} min  "
      f"(canciones >10 min: {(tracks_unique['duration_min']>10).sum()})")

# ── 9. Guardar notas de limpieza ──────────────────────────────────────────────
notes = f"""# 00 — Notas de limpieza de datos

## Dataset original
- Fichero: `data/raw/dataset.csv`
- Filas totales (raw): {n_total + n_null:,}
- Columnas: 20

## Decisiones tomadas

### Fila nula eliminada
- {n_null} fila(s) con `artists`, `album_name` y `track_name` nulos → eliminada(s).

### Deduplicación por `track_id`
- **Problema**: {n_total:,} filas pero solo {n_unique_ids:,} `track_id` únicos
  ({n_dup_rows:,} filas son la misma canción bajo distintos géneros).
- **Para regresión/clustering/recomendación**: se usa `tracks_unique.csv`
  (primera aparición por `track_id`). La columna `track_genre` contiene el
  primer género asignado.
- **Para clasificación de género**: se usa `tracks_long.csv` (todas las
  combinaciones canción-género son observaciones válidas).

### Tratamiento de `tempo == 0`
- **Problema**: {tempo_zeros} filas con `tempo == 0` — errores de detección
  de Spotify, no canciones sin ritmo.
- **Decisión**: imputación con la **mediana del género** correspondiente.
  Se imputa antes de deduplicar para disponer de mayor contexto estadístico.

### Géneros únicos
- Se detectaron **{n_genres} géneros** (la descripción del dataset dice 125 —
  diferencia menor, sin impacto en el análisis).

## Rangos numéricos relevantes
- `loudness`: valores negativos (dB) son normales; 0 dB es el máximo teórico.
- `duration_ms`: rango amplio; canciones >10 min son outliers pero reales
  (piezas clásicas o mixes). Se mantienen en el dataset.
- `popularity`: índice 0-100 de Spotify basado en reproducciones recientes —
  **no** es conteo total de streams. Alta proporción de canciones con
  popularity = 0 (canciones antiguas o poco conocidas con factor de recencia
  bajo).

## Ficheros generados
| Fichero | Filas | Descripción |
|---------|-------|-------------|
| `data/processed/tracks_long.csv` | {len(tracks_long):,} | Todas las combinaciones canción-género |
| `data/processed/tracks_unique.csv` | {len(tracks_unique):,} | Una fila por `track_id` (primer género) |
"""

(RES / "00_data_cleaning_notes.md").write_text(notes, encoding="utf-8")
print("\n[OK] results/00_data_cleaning_notes.md guardado")
print("\n✓ Fase 0 completada.")
