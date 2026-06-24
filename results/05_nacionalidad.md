# 05 — Estudio de Nacionalidad

## Metodología
- **Artistas incluidos**: 220 artistas curados manualmente
  (top-200 por score de presencia×popularidad + artistas latinos/españoles añadidos)
- **Criterio de autoría**: el "artista principal" es el **primer nombre** en el campo
  `artists` del dataset (p.ej. en "Ozuna;Feid", Ozuna es el principal)
- **Canciones analizadas**: 11,881 canciones únicas con artista principal identificado
- **Limitación**: análisis **exploratorio sobre muestra curada y no representativa**.
  No es generalizable a toda la producción musical mundial.

## Distribución de canciones por región
region_label
Anglophone (USA/UK/AUS/CAN)    5312
Latinoamérica                  2790
Asia-Pacífico                  1543
Europa (no anglosajona)        1372
India/Subcontinente             655
España                          113
África                           96

## Popularidad media por región (ranking)
region_label
India/Subcontinente            48.5
Asia-Pacífico                  43.2
Anglophone (USA/UK/AUS/CAN)    42.0
España                         40.4
Europa (no anglosajona)        38.5
Latinoamérica                  36.2
África                         27.2

## Hallazgos de audio features (normalizado 0-1)
                             danceability  energy  loudness  speechiness  acousticness  instrumentalness  liveness  valence  tempo
region_label                                                                                                                      
Anglophone (USA/UK/AUS/CAN)         0.539   0.601     0.770        0.080         0.332             0.121     0.212    0.462  0.489
Asia-Pacífico                       0.574   0.641     0.791        0.068         0.321             0.099     0.198    0.488  0.518
España                              0.558   0.758     0.843        0.078         0.237             0.006     0.227    0.499  0.495
Europa (no anglosajona)             0.491   0.620     0.724        0.064         0.306             0.336     0.232    0.357  0.488
India/Subcontinente                 0.637   0.616     0.789        0.087         0.373             0.024     0.175    0.542  0.471
Latinoamérica                       0.605   0.712     0.814        0.087         0.317             0.027     0.334    0.603  0.515
África                              0.780   0.597     0.812        0.140         0.183             0.011     0.157    0.622  0.402

## Interpretación
- **Latinoamérica**: perfil caracterizado por alta danceability y valence
  (música alegre y bailable), baja acousticness en géneros urbanos.
- **Anglophone**: energía y diversidad alta; abarca desde rock a pop a hip-hop.
- **Asia-Pacífico**: destacan acousticness e instrumentalness (K-pop y J-pop
  tienen mucha producción instrumental); alta energía en K-pop.
- **Europa (no anglosajona)**: mayor instrumentalness (clásica, EDM sin vocals),
  diversidad estilística amplia.
- **España**: perfil intermedio entre Latino y Anglophone; mayor acousticness
  que Latinoamérica.
- **India**: alta speechiness (géneros con mucha letra), energía media.

## Géneros dominantes por región
- **Anglophone (USA/UK/AUS/CAN)**: honky-tonk, world-music, grunge
- **Asia-Pacífico**: j-idol, cantopop, k-pop
- **España**: spanish, latin
- **Europa (no anglosajona)**: goth, german, happy
- **India/Subcontinente**: k-pop, hip-hop, malay
- **Latinoamérica**: heavy-metal, pagode, sertanejo
- **África**: dancehall, dance

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `05a_popularity_by_region.png` | Boxplot y barras de popularidad por región |
| `05b_audio_features_by_region.png` | Heatmap de audio features por región |
| `05c_radar_by_region.png` | Radares sonoros por región |
| `05d_latam_vs_anglophone_vs_spain.png` | Comparativa boxplots Latam/Anglophone/España |
