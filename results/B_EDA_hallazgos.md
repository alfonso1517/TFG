# Fase B — EDA Profundo: 10 Hallazgos Principales

Dataset analizado: **89,550 canciones** en `tracks_model.csv` (29 variables).

---

## Hallazgo 1: B1 — Distribución de Popularity

La distribución de popularity presenta sesgo de 0.07 y un pico pronunciado en 0 (10.5% de canciones). Los tests Shapiro-Wilk (p=7.56e-31) y Kolmogorov-Smirnov (p=1.55e-15) rechazan la normalidad. La mediana (33) es prácticamente igual a la media (33.2), pero la distribución es bimodal: un gran grupo de canciones con 0 reproducciones recientes y una distribución más uniforme para el resto. Esto anticipa la dificultad de regresión: el modelo debe aprender a predecir tanto el pico en 0 como la distribución del cuerpo principal.

![B1 — Distribución de Popularity](../reports/figures/B1_B1.png)

---

## Hallazgo 2: B2 — Correlaciones de Spearman

Las tres features con mayor correlación POSITIVA con popularity son: ['explicit', 'danceability', 'loudness']. Las tres con mayor correlación NEGATIVA son: ['instrumentalness', 'log_instrumentalness', 'speechiness']. La correlación más fuerte detectada es energy-loudness (r=0.75), que indica multicolinealidad entre estas dos variables. Todas las correlaciones con popularity son bajas (|r| < 0.30), confirmando que la popularidad no es bien predecible desde las features de audio puras: el algoritmo de Spotify incorpora factores de marketing y distribución no capturados aquí.

![B2 — Correlaciones de Spearman](../reports/figures/B2_B2.png)

---

## Hallazgo 3: B3 — Violin plots por macro-género

Los violin plots revelan diferencias claras entre macro-géneros. En popularity, el género con mediana más alta es 'hip-hop' (43) y el más bajo es 'clasica' (23). En danceability, latinos y kpop tienen distribuciones muy elevadas (0.7-0.85). En energy, metal y rock tienen distribuciones concentradas en valores altos (0.8-0.95). La clasica presenta la distribución de acousticness más alta y más concentrada. Los violin plots también muestran bimodalidades: en varios géneros hay subpoblaciones con características muy distintas dentro de la misma macro-categoría.

![B3 — Violin plots por macro-género](../reports/figures/B3_B3.png)

---

## Hallazgo 4: B4 — Scatter matrix

La scatter matrix confirma la estructura del espacio de audio features. Los pares más informativos son energy-acousticness (correlación negativa clara, con dos nubes: música electrónica/metal en zona alta-energy/baja-acousticness y música acústica/clásica en zona contraria) y danceability-energy (distribución diagonal positiva, con reggaeton/latin en zona alta-alta). instrumentalness es la feature más bimodal: la mayoría de canciones tiene valores cercanos a 0 (con letra) y un segundo grupo con valores > 0.8 (música clásica, ambient, EDM instrumental). La separación por colores muestra que la mayor parte de la varianza está capturada por el eje energy/acousticness, coherente con que KMeans converge a k=2.

![B4 — Scatter matrix](../reports/figures/B4_B4.png)

---

## Hallazgo 5: B5 — Features vs Popularity

Los scatter plots de cada feature contra popularity muestran relaciones muy débiles y ruidosas. La relación más visible es loudness-popularity (r positivo: canciones más 'altas' tienden a más popularidad) y instrumentalness-popularity (r negativo: canciones sin letra son menos populares en el snapshot). En la versión log-transformada (log_instrumentalness, log_acousticness) las relaciones lineales son ligeramente más claras, justificando el uso de estas transformaciones en los modelos. El ruido extremo en todos los gráficos confirma que ninguna feature de audio individual es suficiente para predecir popularity: el problema requiere modelos que capturen interacciones entre features.

![B5 — Features vs Popularity](../reports/figures/B5_B5.png)

---

## Hallazgo 6: B6 — Radar por macro-género

El gráfico radar muestra con claridad el 'sonido característico' de cada macro-género. Metal destaca en energy y tiene el valence más bajo (sonido intenso y oscuro). Clásica domina en instrumentalness y acousticness. Latino lidera en danceability y valence (alegre y bailable). Hip-hop tiene el mayor speechiness (muchas letras, rap). Electronica presenta alta energy pero baja acousticness (producción electrónica). Folk-acústico combina alta acousticness con danceability moderada. K-pop/J-pop tiene un perfil equilibrado pero con danceability y energy altos. Este radar puede usarse directamente en la memoria como ilustración de que las audio features de Spotify capturan características culturales y estilísticas reales.

![B6 — Radar por macro-género](../reports/figures/B6_B6.png)

---

## Hallazgo 7: B7 — Popularidad por macro-género con IC 95%

El macro-género más popular en el snapshot es 'kpop-jpop' (media=41.2) y el menos popular es 'clasica' (media=27.6). Los intervalos de confianza son muy estrechos porque los grupos son grandes (n > 1.000 en casi todos). Las diferencias son estadísticamente significativas. El efecto de recencia del índice de Spotify favorece géneros urbanos y contemporáneos (pop, hip-hop, latino) frente a géneros clásicos (clásica, jazz-blues) o instrumentales (ambient en 'otros'). Esta diferencia no refleja el valor artístico ni el éxito histórico, sino principalmente cuándo se lanzaron las canciones más populares de cada género.

![B7 — Popularidad por macro-género con IC 95%](../reports/figures/B7_B7.png)

---

## Hallazgo 8: B8 — Variables categóricas vs Popularity

Las canciones explícitas tienen una popularidad media significativamente mayor (Mann-Whitney p=0.0000): el contenido adulto tiende a ir asociado con géneros urbanos (hip-hop, trap, reggaeton) que son muy populares en el período del snapshot. El modo musical (Mayor/Menor) tiene un efecto muy pequeño pero estadísticamente significativo: las canciones en modo Mayor (sonido 'alegre') tienen mediana de popularidad ligeramente superior. La tonalidad (key) no muestra diferencias prácticas relevantes (ANOVA F=4.74, p=0.0000): las diferencias entre tonalidades son mínimas y probablemente se deben al tamaño de muestra grande, no a un efecto real de la tonalidad sobre la popularidad.

![B8 — Variables categóricas vs Popularity](../reports/figures/B8_B8.png)

---

## Hallazgo 9: B9 — Top artistas: presencia vs popularidad

El gráfico revela una paradoja importante: los artistas con más canciones en el dataset (The Beatles, George Jones, Stevie Wonder, Linkin Park, Ella Fitzgerald) no son necesariamente los que tienen mayor popularidad en el snapshot de Spotify. Esto confirma el efecto de recencia del índice: artistas clásicos con enormes catálogos tienen popularidad media baja porque sus canciones son antiguas. Los artistas con mayor popularidad media entre el top-20 son: ['BTS', 'The Beatles', 'Pritam']. Este hallazgo tiene implicaciones directas para el análisis de trayectorias (Fase 6): la popularidad del dataset no es un indicador de éxito histórico sino de actividad reciente.

![B9 — Top artistas: presencia vs popularidad](../reports/figures/B9_B9.png)

---

## Hallazgo 10: B10 — Análisis de canciones con popularity=0

El 10.5% de las canciones tiene popularity=0 (9,410 canciones). El macro-género con mayor tasa de popularity=0 es 'jazz-blues' (25.7%). Comparando las features de audio entre canciones con pop=0 y pop>0, se observa que las canciones con pop=0 tienen mayor instrumentalness (media=0.124 vs 0.179) y mayor acousticness (media=0.386 vs 0.321). Esto sugiere que las canciones instrumentales y acústicas (clásica, ambient, folk antiguo) tienen mayor probabilidad de tener popularity=0 en el snapshot. Estas canciones pueden ser perfectamente válidas y exitosas históricamente, pero tienen pocas reproducciones recientes.

![B10 — Análisis de canciones con popularity=0](../reports/figures/B10_B10.png)

---

## Resumen ejecutivo del EDA

| Variable | Hallazgo clave |
|----------|----------------|
| `popularity` | Distribución no normal, bimodal con pico en 0 (10.5% de canciones) |
| `instrumentalness` | Mayor correlación negativa con popularity (r_Spearman=-0.124) |
| `energy-loudness` | Correlación alta entre sí (r=0.75): posible redundancia en modelos lineales |
| `explicit` | Canciones explícitas significativamente más populares (Mann-Whitney p<0.001) |
| `key` | Sin efecto relevante sobre popularity (ANOVA p=0.0000) |
| `macro_genre` | Diferencias significativas en popularity: pop/hip-hop > clasica/jazz |
| `popularity=0` | 9,410 canciones (10.5%) — más instrumentales y acústicas que el resto |

