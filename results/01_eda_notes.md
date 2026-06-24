# 01 — Notas EDA

## Dataset analizado
- `tracks_unique.csv`: 89,740 canciones únicas, 113 géneros.

## 1. Distribución de Popularity
- **10.5%** de las canciones tienen `popularity = 0`.
  Estas son canciones muy antiguas o prácticamente desconocidas donde el factor
  de recencia del algoritmo de Spotify lleva el índice a 0.
- La distribución está **fuertemente sesgada a la izquierda** (skewness ≈ 0.07):
  la mayoría de canciones tienen baja popularidad; las muy populares son minoría.
- Existe un pico pronunciado en 0 seguido de una distribución aproximadamente
  unimodal con cola derecha.

## 2. Correlaciones con Popularity
Las features que más correlacionan (en valor absoluto):
instrumentalness    0.127477
loudness            0.071674
danceability        0.064275
explicit            0.054898
speechiness         0.047086

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
track_genre
pop         65.0
metal       63.0
k-pop       61.0
pop-film    60.0
hip-hop     59.0

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
