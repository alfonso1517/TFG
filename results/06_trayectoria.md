# 06 — Trayectoria Temporal de Artistas (Feid y similares)

## Artistas analizados
Feid, J Balvin, Maluma, KAROL G, Ozuna, Rauw Alejandro, Daddy Yankee, Bad Bunny

## Metodología
- **Fechas de lanzamiento**: asignadas manualmente basándose en conocimiento discográfico
  (año y mes de lanzamiento oficial del single o álbum de estudio)
- **Deduplicación**: misma canción en múltiples playlists → versión con mayor popularity
- **Canciones con fecha asignada**: 291
- **Predicción**: regresión polinómica (grado 2) sobre medias anuales de audio features

## Resumen por artista
                n_canciones años_activo  pop_media  pop_max  dance_media  energy_media  valence_media
artista                                                                                              
Bad Bunny                41   2017–2022      81.78       97         0.77          0.67           0.50
Maluma                   16   2016–2022      59.81       81         0.77          0.77           0.68
Rauw Alejandro           19   2020–2022      56.11       91         0.78          0.72           0.53
Ozuna                    38   2016–2022      52.32       83         0.80          0.72           0.60
Daddy Yankee             48   2004–2022      51.02       85         0.74          0.83           0.71
J Balvin                 74   2014–2022      43.08       88         0.75          0.73           0.66
KAROL G                  27   2017–2022      28.11       93         0.76          0.74           0.65
Feid                     28   2018–2022      25.93       89         0.75          0.66           0.58

## Predicción features de Feid (2023-2025)
 year  danceability  energy  valence  acousticness   tempo
 2023         0.747   0.665    0.605         0.140 138.348
 2024         0.742   0.671    0.619         0.132 138.827
 2025         0.738   0.676    0.633         0.123 139.306

## Hallazgos clave

### Feid
- Arrancó en 2018 con un sonido más tranquilo/R&B (acousticness alta, valence media)
- Evolución clara hacia reggaeton urbano de mayor energía entre 2020-2022
- Punto de inflexión: 2021 (collabs con artistas grandes, sonido más bailable)
- 2022: consolidación (FELIZ CUMPLEAÑOS FERXXO, Hey Mor con Ozuna, Prohibidox)
- La predicción sugiere tendencia a mayor energy y danceability en 2023-2025

### Comparativa colombiana
- J Balvin muestra la trayectoria más larga (desde 2014) y mayor variación de sonido
- KAROL G arranca en 2017-2018 y despega en 2020 (BICHOTA); sigue la misma curva que Feid
- Maluma empezó más reggaeton-pop, trayectoria más estable
- Feid es el de mayor crecimiento en el período 2018-2022

### Bad Bunny como referencia
- Discografía mucho más extensa en el dataset, popularidades más altas
- Su sonido es el más "variable" (experimenta entre trap, dembow, indie en Un Verano Sin Ti)
- La baja popularidad de canciones pre-2020 en el dataset confirma el efecto de recencia

### Limitación principal
La variable `popularity` en el dataset es un **snapshot** del algoritmo de Spotify
(basado en streams recientes). Las canciones más antiguas —como "Gasolina" (2004) o
"Ginza" (2015)— pueden aparecer con popularidad baja a pesar de ser hits absolutos.
Este análisis es útil para features de **audio** (objetivas), pero NO para inferir
el éxito real de canciones históricas por su popularity en el dataset.

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `06a_evolucion_features.png` | Evolución de 4 features para los 8 artistas |
| `06b_feid_vs_trio.png` | Feid vs J Balvin vs Bad Bunny (raw + normalizado) |
| `06c_popularity_scatter.png` | Popularidad por canción y año |
| `06d_feid_prediccion.png` | Predicción 2023-2025 de audio features de Feid |
| `06e_sonido_colombiano.png` | Evolución del sonido colombiano (4 artistas) |
