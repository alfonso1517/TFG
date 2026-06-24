# Registro de Trabajo — TFG Spotify

> Documento generado automáticamente durante la sesión de trabajo.
> Contiene decisiones técnicas, librerías instaladas, scripts creados y resultados obtenidos.

---

## Entorno y setup

### Versiones
- Python: 3.12.6
- Sistema: Windows 10 Pro Education
- Directorio: `c:\Users\Fonso\desktop\tfg_robe`

### Librerías instaladas

| Librería | Versión instalada | Uso |
|----------|------------------|-----|
| pandas | 3.0.3 | Manipulación de datos |
| numpy | 2.4.6 | Cómputo numérico |
| matplotlib | — | Gráficos base |
| seaborn | 0.13.2 | Gráficos estadísticos |
| plotly | 6.8.0 | Gráficos interactivos |
| scikit-learn | 1.9.0 | ML clásico (RF, KMeans, PCA, etc.) |
| xgboost | 3.2.0 | Gradient Boosting |
| streamlit | 1.58.0 | App web de recomendación |
| joblib | 1.5.3 | Serialización de modelos |
| scipy | 1.17.1 | Estadística |
| category_encoders | 2.9.0 | Target/frequency encoding |
| umap-learn | 0.5.12 | Reducción dimensionalidad no lineal |
| optuna | 4.9.0 | Búsqueda bayesiana de hiperparámetros |
| tqdm | 4.68.2 | Barras de progreso |

**Nota**: se detectó un conflicto de versiones numpy/pandas al instalar
(numpy 2.1.1 incompatible con pandas 2.0.3). Se resolvió reinstalando ambos
con `pip install --upgrade numpy pandas --force-reinstall`, obteniendo
numpy 2.4.6 + pandas 3.0.3 compatibles.

**Nota**: los scripts deben ejecutarse con `python -X utf8` o con
`$env:PYTHONIOENCODING = "utf-8"` para evitar errores de codificación
en la consola de Windows (caracteres españoles en los prints).

---

## Estructura de carpetas creada

```
tfg_robe/
├── data/
│   ├── raw/                  ← dataset.csv original (copia)
│   └── processed/            ← tracks_unique.csv, tracks_long.csv, tracks_unique_clustered.csv
├── scripts/                  ← un script por fase (00 al 03)
├── models/                   ← modelos .pkl y arrays .npy
├── app/                      ← app.py (Streamlit)
├── reports/
│   └── figures/              ← gráficos .png generados
├── results/                  ← notas .md por fase + este registro
└── CLAUDE.md                 ← instrucciones del proyecto
```

---

## FASE 0 — Limpieza de datos

**Script**: `scripts/00_data_cleaning.py`
**Ejecutado**: sí ✓
**Tiempo**: ~30 segundos

### Hallazgos
- Dataset original: 114.000 filas, 21 columnas (incluye `Unnamed: 0` como índice)
- 1 fila eliminada (artista/album/track nulos)
- 113.999 filas limpias, 89.740 `track_id` únicos → 24.259 filas duplicadas por multi-género
- 114 géneros únicos (confirmado; la descripción del dataset dice 125 — discrepancia menor)
- 157 filas con `tempo == 0` → imputadas con mediana del género

### Decisiones de limpieza
| Decisión | Justificación |
|----------|---------------|
| Deduplicar por `track_id` para regresión/clustering | Evitar sobre-representar canciones con muchos géneros |
| Mantener long format para clasificación de género | Cada combinación canción-género es una observación válida |
| Imputar tempo=0 con mediana del género | Son errores de detección de Spotify, no canciones sin ritmo |

### Ficheros generados
- `data/processed/tracks_unique.csv` — 89.740 filas
- `data/processed/tracks_long.csv` — 113.999 filas
- `results/00_data_cleaning_notes.md`

---

## FASE 1 — Análisis Exploratorio (EDA)

**Script**: `scripts/01_eda.py`
**Ejecutado**: sí ✓
**Tiempo**: ~45 segundos

### Hallazgos clave
- **10.5% de canciones** tienen `popularity = 0` (sesgo de recencia del algoritmo Spotify)
- Distribución de `popularity` fuertemente sesgada a la izquierda (skewness > 1)
- Correlaciones más altas con popularity (en valor absoluto):
  - `instrumentalness` (negativa, ~0.13): canciones instrumentales tienden a ser menos populares
  - `loudness` (positiva, ~0.07): canciones más "fuertes" ligeramente más populares
  - `danceability` (positiva, ~0.06)
  - Ninguna correlación lineal fuerte → relaciones no lineales, modelos de árbol pertinentes
- **Pop** es el género con mayor mediana de popularidad entre los top-20
- Géneros se distinguen claramente en el espacio de audio features (radar charts)
- 550 canciones tienen duración > 10 min (piezas clásicas, suites)

### Figuras generadas
- `reports/figures/01a_popularity_distribution.png`
- `reports/figures/01b_correlation_matrix.png`
- `reports/figures/01c_genre_radar.png`
- `reports/figures/01d_popularity_by_genre.png`
- `reports/figures/01e_categorical_vars.png`
- `reports/figures/01f_outliers_loudness_duration.png`
- `reports/figures/01g_audio_features_bar.png`

---

## FASE 2 — Modelos Predictivos (RF vs XGBoost)

**Script**: `scripts/02_modelos_predictivos.py`
**Ejecutado**: en proceso / pendiente de resultados
**Tiempo estimado**: 20-40 minutos

### Estrategia implementada
- **Regresión de popularity**: target encoding de `track_genre` (114 categorías → valor numérico),
  split 80/20, RandomizedSearchCV con 20 iteraciones y 3-fold CV
- **Clasificación de género**: 114 géneros → 12 macro-categorías (mapeo musical ad hoc);
  baseline con 114 clases para referencia; RandomizedSearchCV con 15 iteraciones y 3-fold CV

### Macro-géneros (mapeo)
rock | metal | electronic | hip-hop | latino | pop | jazz-blues |
classical | folk-country | reggae-ska | world | ambient | other

### Resultados finales ✓

**Regresión de Popularity** (test 20%, tracks_unique):
| Modelo | RMSE | MAE | R² |
|--------|------|-----|----|
| Random Forest | 14.864 | 10.160 | **0.472** |
| XGBoost | 15.444 | 10.841 | 0.430 |

RF gana ligeramente. R²≈0.47 es razonable dado que popularity depende de factores externos al audio.

**Clasificación de Género** (tracks_long, 80/20):
| Modelo | Accuracy | F1-macro |
|--------|----------|----------|
| Baseline RF (114 clases) | 0.257 | 0.246 |
| RF Macro-género (~12 cat.) | **0.465** | **0.408** |
| XGB Macro-género (~12 cat.) | 0.464 | 0.409 |

RF y XGB prácticamente empatan. La agrupación en macro-géneros duplica las métricas.

**Incidencia durante ejecución**: el segundo proceso lanzado falló con OOM
(`TerminatedWorkerError`) al usar `n_jobs=-1` en ambas capas (RF interno + CV externo)
con 113k filas. El primer proceso (lanzado antes, sin la bandera `-u`) terminó correctamente.
Script corregido: la sección de regressores ahora salta si los modelos ya existen,
y los clasificadores usan `n_jobs=2` en vez de `-1` para evitar explosión de memoria.

### Modelos guardados ✓
- `models/rf_regressor.pkl`
- `models/xgb_regressor.pkl`
- `models/rf_classifier.pkl`
- `models/xgb_classifier.pkl`
- `models/target_encoder_reg.pkl`
- `models/label_encoder_macro.pkl`
- `models/genre_map.pkl`

---

## FASE 3 — Clustering y Recomendación

**Script**: `scripts/03_clustering_recomendacion.py`
**Ejecutado**: sí ✓
**Tiempo**: ~8 minutos (incluye UMAP sobre 8k canciones)

### Resultados de clustering

| Algoritmo | k | Silhouette | Davies-Bouldin |
|-----------|---|------------|----------------|
| KMeans | **2** (óptimo por silhouette) | 0.2581 | 1.5729 |
| Agglomerative (Ward) | 2 | 0.1853 | 1.8648 |
| DBSCAN (eps=1.303) | 2+ruido | -0.037 | 1.044 |

- PCA 2D: varianza explicada = 48% (PC1=32%, PC2=16%)
- k=2 refleja la división fundamental energético vs tranquilo/acústico
- DBSCAN: 7.4% de canciones marcadas como "ruido" (sonido atípico)

### Géneros por cluster (KMeans k=2)
- **Cluster 0** (67.369 canciones, energético): breakbeat, chicago-house, heavy-metal, happy, forro
- **Cluster 1** (22.371 canciones, tranquilo): sleep, ambient, new-age, classical, romance

### Resultados del recomendador (100 semillas)

| Enfoque | Coherencia género | Diversidad artistas | Pop. media |
|---------|------------------|---------------------|------------|
| KNN global | 0.136 | 0.871 | 31.8 |
| Basado en cluster | 0.125 | 0.864 | 32.2 |
| **Híbrido** | **1.000** | 0.831 | 33.5 |

### Ficheros generados
- `data/processed/tracks_unique_clustered.csv`
- `models/scaler_cluster.pkl`, `models/kmeans.pkl`
- `models/knn_recommender.pkl`, `models/X_scaled.npy`
- `reports/figures/03a_kmeans_elbow.png` a `03d_recommender_comparison.png`
- `results/03_clustering_recomendacion.md`

---

## FASE 4 — App Streamlit

**Fichero**: `app/app.py`
**Estado**: ✓ Ejecutándose en http://localhost:8501
**Para lanzar**:
```bash
cd c:\Users\Fonso\desktop\tfg_robe
streamlit run app/app.py
```

### Funcionalidades
1. **Búsqueda por canción semilla**: buscar por nombre/artista → seleccionar → recomendar top-N
2. **Búsqueda por perfil**: sliders de danceability/energy/valence/tempo/etc. + filtro género opcional

---

## Notas para la memoria del TFG

### Limitaciones documentadas
1. `popularity` es una "instantánea" de Spotify basada en reproducciones recientes (recencia).
   Canciones antiguas tienden a tener `popularity` baja aunque fueran exitosas en su momento.
   Esto afecta especialmente la interpretación de la regresión y cualquier análisis temporal.

2. El sistema de recomendación es **content-based puro** (basado en audio features).
   El sistema real de Spotify combina además filtrado colaborativo y NLP sobre metadatos.
   Nuestro sistema replica solo la componente de análisis de audio.

3. La clasificación de macro-géneros usa un mapeo musical **ad hoc** (basado en criterio musical).
   Este mapeo es una decisión editorial defendible pero no única — otra persona podría
   hacer agrupaciones diferentes. Debe explicitarse en la memoria.

4. Datos de nacionalidad: no automatizados (inviable con ~9k artistas), requieren curación manual.

5. El dataset tiene 113 géneros efectivos (no 114 tras la deduplicación — posible duplicado
   o género que solo aparecía en la fila nula eliminada).

### Hallazgos más relevantes para la memoria
*(se actualizará al completar todas las fases)*
- Distribución de popularity muy sesgada, dominada por canciones con popularity = 0
- Las audio features tienen correlación lineal débil con popularity individualmente
  → justifica el uso de modelos no lineales como RF/XGBoost
- Los géneros se separan razonablemente en el espacio de audio features
  → clustering y clasificación son viables aunque imperfectos
