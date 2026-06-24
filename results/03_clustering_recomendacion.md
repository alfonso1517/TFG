# 03 — Clustering y Sistema de Recomendación

## Datos y preprocesado
- Dataset: `tracks_unique.csv` (89,740 canciones)
- Features de clustering: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- Escalado: `StandardScaler` (media 0, desv. típica 1)

## Reducción de dimensionalidad
- PCA 2D: varianza explicada = 48.0% (PC1=32.0%, PC2=16.0%)
- UMAP 2D: calculado sobre 8k canciones

## Comparativa de algoritmos de clustering

| Algoritmo | k/parámetros | Silhouette | Davies-Bouldin |
|-----------|-------------|------------|----------------|
| KMeans | k=2 | 0.2581 | 1.5729 |
| Agglomerative (Ward) | k=2 (10k muestra) | 0.1853 | 1.8648 |
| DBSCAN | eps=1.303, minPts=10 (5k muestra) | -0.0367 | 1.0441 |

**Análisis**:
- KMeans proporciona clusters de tamaño más equilibrado y permite asignar
  etiquetas a nuevas canciones (predicción rápida).
- Agglomerative (Ward) tiende a resultados similares pero sin método predict.
- DBSCAN detecta 2 clusters y 372 puntos de ruido
  (7.4% de la muestra de 5k). Útil para identificar canciones
  "atípicas" que no encajan en ningún cluster principal.
- **KMeans k=2** se elige para el sistema de recomendación por su
  equilibrio rendimiento/interpretabilidad.

## Top géneros por cluster KMeans
- **Cluster 0**: happy, forro, breakbeat, chicago-house, heavy-metal
- **Cluster 1**: sleep, ambient, new-age, classical, romance


## Sistema de Recomendación — Comparativa

| Enfoque | Coherencia Género | Diversidad Artistas | Popularidad Media |
|---------|------------------|---------------------|-------------------|
| KNN (coseno) | 0.136 | 0.871 | 31.8 |
| Basado en Cluster | 0.125 | 0.864 | 32.2 |
| Híbrido (género+KNN) | 1.000 | 0.831 | 33.5 |

**Interpretación**:
- **Híbrido** ofrece mayor coherencia de género (las recomendaciones son del
  mismo género que la semilla), a costa de menor diversidad de artistas.
- **KNN global** es más diverso y equilibrado en popularidad.
- **Cluster** funciona bien como compromiso, aunque la granularidad del cluster
  puede ser insuficiente para géneros muy específicos.
- Para la app Streamlit se usará el **enfoque híbrido** como principal y KNN
  como alternativa configurable.

## Nota sobre el sistema de Spotify real
Nuestro sistema es un **recomendador basado en contenido (content-based)**
usando audio features como proxy del embedding de audio. El sistema real de
Spotify combina adicionalmente filtrado colaborativo (comportamiento de usuarios)
y NLP sobre metadatos. Nuestra aproximación replica solo la componente de
análisis de audio, que es la única replicable con este dataset.

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `03a_kmeans_elbow.png` | Codo + silhouette para selección de k |
| `03b_pca2d_clusters.png` | PCA 2D coloreado por cluster y por género |
| `03c_umap_clusters.png` | UMAP 2D (si disponible) |
| `03d_recommender_comparison.png` | Métricas comparativas de los 3 enfoques |
