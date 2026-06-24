# CLAUDE.md — TFG: IA y análisis estadístico aplicado a la industria musical (Spotify)

## Contexto general

Este proyecto es la base del caso práctico de un Trabajo de Fin de Grado en
Matemáticas/Estadística. El objetivo de la sesión de hoy es hacer un **primer
acercamiento amplio** a varias líneas de trabajo para luego decidir en cuál
profundizar. Prioriza tener algo funcionando en cada fase antes de pulir
detalles. Documenta hallazgos, decisiones de diseño y resultados a medida que
avances (en README o en un notebook/markdown de resultados) porque luego hay
que redactar la memoria.

## Datos disponibles

En la carpeta del proyecto hay dos ficheros: `dataset.csv` y
`spotify-tracks-dataset.csv`.

**IMPORTANTE — ya verificado:** ambos ficheros son **el mismo dataset
duplicado** (114.000 filas, mismas columnas, mismos datos; el segundo solo
tiene una columna de índice extra `Unnamed: 0`). No hay un dataset adicional
"más reciente" — usa solo `dataset.csv` y descarta el otro (o muévelo a una
carpeta `raw/` por si acaso).

### Columnas
`track_id, artists, album_name, track_name, popularity, duration_ms, explicit,
danceability, energy, key, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, time_signature, track_genre`

### Calidad de datos — hallazgos a tener en cuenta

- **114.000 filas, pero solo 89.741 `track_id` únicos.** 40.900 filas son la
  misma canción repetida bajo distintos `track_genre` (Spotify etiqueta una
  canción con varios géneros y el dataset crea una fila por combinación
  canción-género). **Decisión a tomar y documentar:**
  - Para clasificación de género: mantener todas las filas (cada
    combinación canción-género es una observación válida), o quedarte con
    un género "principal" por canción — decide y justifica.
  - Para clustering/recomendación/regresión de popularidad: probablemente
    interese **deduplicar por `track_id`** (quedarte con una fila por
    canción, p.ej. la primera) para no sobre-representar canciones con
    muchos géneros asignados.
- **114 géneros únicos** (no 125 como dice la descripción del dataset —
  anótalo como dato curioso/menor).
- **1 fila con `artists`, `album_name`, `track_name` nulos** — elimínala o
  trátala aparte.
- **157 filas con `tempo == 0`** — son errores de detección de tempo de
  Spotify, no canciones sin ritmo. Decide si las imputas (mediana del
  género) o las excluyes de los modelos que usan `tempo`, y documéntalo.
- Artistas con más canciones en el dataset: The Beatles (279), George Jones
  (271), Stevie Wonder (236), Linkin Park (224), Ella Fitzgerald (222),
  Prateek Kuhad (217), **Feid (202)**, Chuck Berry (190), Håkan Hellström
  (183), OneRepublic (181). Útiles como candidatos para el estudio de
  trayectoria temporal (Fase 6).

### Sobre la variable `popularity`

Es un índice 0-100 propio de Spotify, calculado por un algoritmo no público
basado principalmente en el volumen de reproducciones recientes y su
recencia (no es un conteo de streams totales). Trátalo como una "instantánea"
del momento en que se extrajo el dataset. Esto es relevante sobre todo para
la Fase 6 (series temporales por artista): canciones antiguas pueden tener
`popularity` baja simplemente por el componente de recencia del algoritmo,
no porque fracasaran en su momento. Documenta esta limitación en cualquier
análisis que compare popularidad entre canciones de distintas épocas.

---

## Estructura de carpetas sugerida

```
proyecto/
├── data/
│   ├── raw/                  # dataset.csv original + el duplicado descartado
│   └── processed/            # datasets limpios/deduplicados
├── notebooks/ o scripts/      # un script o notebook por fase
├── models/                    # modelos entrenados (.pkl/.json)
├── app/                        # interfaz Streamlit
├── reports/figures/            # gráficos generados
└── results/                    # tablas de métricas, hallazgos en markdown
```

---

## FASE 0 — Setup y limpieza (prioridad alta, ~30 min)

1. Cargar `dataset.csv`, eliminar la fila nula, decidir y aplicar la
   estrategia de deduplicación por `track_id` (crear `data/processed/
   tracks_unique.csv` con una fila por canción, y mantener también una
   versión "long" con todas las combinaciones género-canción si se necesita
   para clasificación de género).
2. Tratar los `tempo == 0`.
3. Comprobar rangos/tipos de todas las variables numéricas (duration_ms en
   minutos puede ser más legible, loudness en dB negativo es normal, etc.).
4. Guardar un pequeño resumen de "decisiones de limpieza" en
   `results/00_data_cleaning_notes.md` — esto va directo a la memoria.

---

## FASE 1 — Análisis descriptivo (prioridad alta, ~1h)

Objetivo: entender la estructura de los datos antes de modelar (igual que en
el TFG de referencia con los salarios).

- Distribución de `popularity` (histograma, ¿está sesgada? ¿hay muchas
  canciones con popularity = 0?).
- Matriz de correlación entre las features de audio numéricas y `popularity`.
  ¿Cuáles correlacionan más?
- Perfil medio de audio features por género: elegir 6-8 géneros
  contrastados (p.ej. reggaeton, classical, metal, acoustic, edm, latin,
  pop, jazz) y compararlos con un gráfico de radar o barras agrupadas
  (danceability, energy, valence, tempo, acousticness).
- Boxplots de `popularity` por género — ¿qué géneros tienden a tener
  popularidad media más alta?
- Distribución de `key`, `mode`, `time_signature`, `explicit` — ¿influyen en
  popularidad?
- Detectar outliers en `loudness` y `duration_ms`.

Guardar gráficos en `reports/figures/` y un resumen de conclusiones en
`results/01_eda_notes.md`.

---

## FASE 2 — Modelos predictivos: Random Forest vs XGBoost (prioridad alta, ~2-3h)

### 2.1 Regresión de `popularity`

- Features: todas las audio features numéricas + `duration_ms`, `explicit`,
  `key`, `mode`, `time_signature`, y `track_genre` (codificada — con 114
  categorías, probar one-hot puede ser pesado; considera target encoding o
  frequency encoding como alternativa, y compara).
- Split train/test (80/20), preferiblemente con `tracks_unique` (una fila
  por canción).
- Modelos: `RandomForestRegressor` y `XGBRegressor`.
- Búsqueda de hiperparámetros: usar `RandomizedSearchCV` (con 114k filas un
  `GridSearchCV` exhaustivo puede ser muy lento; si hay tiempo, opcionalmente
  probar `optuna` para una búsqueda más eficiente) con validación cruzada
  (3-5 folds).
  - RF: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
  - XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
    `colsample_bytree`, `gamma`.
- Métricas: RMSE, MAE, R². Comparar ambos modelos en una tabla.
- Importancia de variables (impurity + permutation importance) para ambos
  modelos — comparar qué variables destacan cada uno.

### 2.2 Clasificación de género

- 114 clases es mucho para un primer modelo. Estrategia recomendada:
  1. Primero, probar con todas las clases como baseline rápido (para tener
     un número de referencia).
  2. Después, agrupar los 114 géneros en ~10-15 macro-categorías razonables
     (p.ej. "rock", "electrónica/edm", "latino/reggaeton", "hip-hop/rap",
     "clásica/instrumental", "jazz/blues", "pop", "metal", "folk/acoustic",
     "world/regional", etc. — usa tu criterio musical para el mapeo, es una
     decisión defendible en la memoria) y repetir la clasificación con estas
     macro-categorías, que debería dar resultados mucho más interpretables.
- Modelos: `RandomForestClassifier` y `XGBClassifier`, con
  `RandomizedSearchCV` igual que en 2.1.
- Métricas: accuracy, F1-macro, matriz de confusión (sobre todo para las
  macro-categorías).
- Importancia de variables: ¿qué features de audio discriminan mejor el
  género? (esperable: tempo/energy para electrónica vs acousticness para
  folk, etc.)

Guardar tablas comparativas RF vs XGBoost en `results/02_modelos_predictivos.md`.

---

## FASE 3 — Clustering y sistema de recomendación (prioridad alta, ~2h)

### Contexto (para la memoria)

El sistema real de recomendación de Spotify combina tres pilares: filtrado
colaborativo (qué escuchan usuarios similares), análisis de audio mediante
redes neuronales sobre espectrogramas (similitud de sonido) y NLP sobre
metadatos/blogs/redes sociales (contexto cultural). Nuestro dataset no tiene
datos de usuarios ni texto, así que el sistema que construyamos es un
**recomendador basado en contenido (content-based)** usando las audio
features como proxy del "embedding de audio" — es la pieza de Spotify que
sí podemos replicar con estos datos. Esto hay que dejarlo explícito en la
memoria como justificación del enfoque y como limitación.

### Tareas

1. Preprocesado: seleccionar audio features numéricas relevantes, escalar
   (`StandardScaler`), usar `tracks_unique`.
2. Reducción de dimensionalidad para visualización: PCA (2-3 componentes) y,
   si hay tiempo, UMAP o t-SNE para ver si los clusters se separan mejor.
3. Probar y comparar varios algoritmos de clustering:
   - `KMeans` (elegir k con método del codo + silhouette score).
   - `AgglomerativeClustering` (jerárquico).
   - `DBSCAN` (para ver si detecta géneros "atípicos" como ruido).
   - Comparar resultados: silhouette score, Davies-Bouldin, y sobre todo
     **¿los clusters se corresponden con géneros reales?** (tabla cruzada
     cluster vs género).
4. Sistema de recomendación — comparar al menos 3 enfoques:
   - **KNN sobre audio features escaladas** (similitud coseno o euclídea):
     dada una canción semilla, devolver las N más cercanas.
   - **Recomendación basada en cluster**: recomendar canciones del mismo
     cluster que la semilla.
   - **Híbrido**: filtrar primero por género (o macro-género) y luego aplicar
     KNN dentro de ese subconjunto.
   - Evaluación (no hay ground truth real, así que usar métricas proxy):
     coherencia de género de las recomendaciones respecto a la semilla,
     diversidad de artistas recomendados, distribución de popularidad de las
     recomendaciones.

Guardar comparativa en `results/03_clustering_recomendacion.md`.

---

## FASE 4 — Interfaz de recomendación (prioridad media, ~1h)

Aplicación sencilla en Streamlit (un solo fichero `app/app.py`), con dos modos:

1. **Por canción semilla**: el usuario busca/selecciona una canción del
   dataset (buscador por nombre de artista/canción) y la app devuelve el
   top-N de canciones similares (usando el mejor recomendador de la Fase 3).
2. **Por preferencias**: sliders para danceability, energy, valence, tempo,
   acousticness, etc., más un selector opcional de género/macro-género, y la
   app busca las canciones más cercanas en el espacio de features a ese
   "perfil ideal".

Mantenlo simple y funcional — cachear la carga de datos/modelo con
`@st.cache_data` / `@st.cache_resource`.

---

## FASE 5 — Estudio de nacionalidad (prioridad media-baja, curado a mano)

No es viable ni fiable automatizar la nacionalidad para ~9.000 artistas
distintos. Enfoque curado:

1. Seleccionar un conjunto manejable y con sentido: por ejemplo, los 30-50
   artistas más presentes/populares del dataset, o un conjunto temático
   (p.ej. comparar artistas anglosajones vs latinos vs asiáticos en géneros
   específicos como pop, reggaeton, k-pop).
2. Para cada artista de ese conjunto, buscar/investigar su nacionalidad
   (puedes usar tu conocimiento o una búsqueda puntual) y construir una
   tabla auxiliar `artists_nationality.csv` (artista, país, región/continente).
3. Cruzar con el dataset para ese subconjunto y analizar: distribución de
   `popularity` por país/región, perfiles de audio features por región
   (¿el "sonido latino" es estadísticamente distinto del "sonido
   anglosajón" en danceability/tempo/valence?).
4. **Importante para la memoria**: dejar muy claro que esto es un análisis
   **exploratorio sobre una muestra curada y no representativa**, no una
   afirmación generalizable sobre "probabilidad de éxito según
   nacionalidad". Es más un estudio descriptivo/cualitativo con apoyo
   estadístico que una inferencia poblacional.

---

## FASE 6 — Caso práctico: trayectoria temporal de un artista (prioridad media-baja)

Para 1-2 artistas concretos con buena presencia en el dataset (Feid es buen
candidato por relevancia para el TFG y por tener 202 canciones; también
sirven Linkin Park, OneRepublic, etc. si se prefiere algo con carrera más
larga):

1. Extraer todas sus canciones del dataset.
2. Para cada canción, investigar (búsqueda puntual, no hace falta precisión
   exacta — basta mes/año aproximado) la fecha de lanzamiento.
3. Ordenar cronológicamente y construir una serie temporal de:
   - Evolución de audio features (¿ha cambiado su "sonido" con el tiempo?
     tempo, energy, valence...).
   - Evolución de `popularity` (con el matiz mencionado arriba sobre qué
     significa esta variable).
4. Visualizar como líneas temporales / gráfico de evolución, y comentar si
   se observan tendencias (p.ej. cambio de género/sonido en algún momento de
   la carrera, canciones "outlier" que rompen la tendencia, etc.).
5. Si da tiempo, un modelo simple de tendencia (regresión lineal sobre el
   tiempo para alguna feature) — sin sobre-ingeniería, es más para narrativa
   que para predicción robusta.

---

## Entregables al final del día

- `data/processed/tracks_unique.csv` + notas de limpieza.
- EDA con gráficos guardados.
- Modelos RF y XGBoost (regresión popularidad + clasificación género) con
  tablas comparativas de métricas e importancia de variables.
- Comparativa de clustering + sistema de recomendación con al menos 3
  enfoques evaluados.
- App de Streamlit funcional.
- Mini-estudio de nacionalidad sobre conjunto curado.
- 1-2 casos de trayectoria temporal de artista.
- Un fichero `results/RESUMEN_HALLAZGOS.md` con los puntos más interesantes
  de cada fase, pensado para poder decidir después en qué profundizar para
  la memoria final.

## Prioridad si falta tiempo

Si no da tiempo a todo, el orden de prioridad es: **Fase 0 → Fase 1 → Fase 2
→ Fase 3 → Fase 4**. Las fases 5 y 6 son curadas/manuales y pueden hacerse
en otra sesión con menos urgencia, ya que dependen de elegir bien los
artistas/casos antes de invertir tiempo en investigación.
