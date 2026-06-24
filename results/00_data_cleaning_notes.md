# 00 — Notas de limpieza de datos

## Dataset original
- Fichero: `data/raw/dataset.csv`
- Filas totales (raw): 114,000
- Columnas: 20

## Decisiones tomadas

### Fila nula eliminada
- 1 fila(s) con `artists`, `album_name` y `track_name` nulos → eliminada(s).

### Deduplicación por `track_id`
- **Problema**: 113,999 filas pero solo 89,740 `track_id` únicos
  (24,259 filas son la misma canción bajo distintos géneros).
- **Para regresión/clustering/recomendación**: se usa `tracks_unique.csv`
  (primera aparición por `track_id`). La columna `track_genre` contiene el
  primer género asignado.
- **Para clasificación de género**: se usa `tracks_long.csv` (todas las
  combinaciones canción-género son observaciones válidas).

### Tratamiento de `tempo == 0`
- **Problema**: 157 filas con `tempo == 0` — errores de detección
  de Spotify, no canciones sin ritmo.
- **Decisión**: imputación con la **mediana del género** correspondiente.
  Se imputa antes de deduplicar para disponer de mayor contexto estadístico.

### Géneros únicos
- Se detectaron **114 géneros** (la descripción del dataset dice 125 —
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
| `data/processed/tracks_long.csv` | 113,999 | Todas las combinaciones canción-género |
| `data/processed/tracks_unique.csv` | 89,740 | Una fila por `track_id` (primer género) |
