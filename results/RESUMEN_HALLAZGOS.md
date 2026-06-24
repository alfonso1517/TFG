# Resumen de Hallazgos — TFG Spotify

> Este documento recoge los puntos más relevantes de cada fase para
> facilitar la redacción de la memoria final y decidir en qué profundizar.
> Se irá completando a medida que se ejecutan las fases.

---

## FASE 0 — Limpieza de datos ✓

| Hallazgo | Relevancia para la memoria |
|----------|---------------------------|
| 114.000 filas → 89.740 canciones únicas | Dataset con multi-etiquetado de género (24.259 duplicados por género) |
| 114 géneros (descripción dice 125) | Discrepancia menor, anotar como curiosidad |
| 157 canciones con tempo=0 | Errores de detección Spotify; imputar con mediana del género |
| 1 fila con metadatos nulos | Ruido mínimo, eliminada sin impacto estadístico |
| Popularity: índice 0-100 basado en recencia | **Limitación crítica**: no es conteo de streams; canciones antiguas infra-representadas |

**Decisiones clave documentadas**: deduplicación por `track_id` para modelos predictivos/clustering; formato long para clasificación de género.

---

## FASE 1 — Análisis Exploratorio ✓

### Hallazgos sobre `popularity`
- **10.5% de canciones con popularity = 0** — efecto del algoritmo de recencia de Spotify
- Distribución fuertemente sesgada a la izquierda (skewness > 1): la popularidad "alta" es rara
- Las canciones con popularity > 0 siguen una distribución aproximadamente unimodal centrada en ~35-45

### Correlaciones con popularity (top 5, valor absoluto)
| Feature | Correlación | Interpretación |
|---------|-------------|----------------|
| instrumentalness | ~-0.13 | Canciones instrumentales menos populares (sin voz) |
| loudness | ~+0.07 | Canciones más "fuertes" (masterizadas) ligeramente más populares |
| danceability | ~+0.06 | Canciones bailables ligeramente más populares |
| explicit | ~+0.05 | Contenido explícito correlaciona con géneros populares actuales |
| speechiness | ~+0.05 | Más discurso/rap ligera correlación positiva |

**Conclusión clave**: las correlaciones lineales son débiles → justifica el uso de modelos no lineales (RF, XGBoost). El modelo tendrá R² moderado porque la popularidad depende de factores externos (playlists, tendencias virales) no capturados en el audio.

### Géneros y popularidad
- **Pop** tiene la mayor mediana de popularidad entre los top-20 géneros
- Géneros contemporáneos (pop, reggaeton, hip-hop) > géneros clásicos en popularidad media
- Los géneros se separan claramente en el espacio de audio features → clustering viable

### Variables categóricas
- Canciones con `explicit=True` tienen popularidad media ligeramente mayor
- `mode` (mayor/menor) y `key` tienen efectos mínimos sobre popularidad
- `time_signature=4` domina (>80% de canciones)

---

## FASE 2 — Modelos Predictivos *(en proceso)*

### 2.1 Regresión de Popularity ✓

| Modelo | RMSE | MAE | R² |
|--------|------|-----|----|
| Random Forest | **14.86** | **10.16** | **0.472** |
| XGBoost | 15.44 | 10.84 | 0.430 |

**Interpretación**: R²≈0.47 significa que el modelo explica ~47% de la varianza en popularity. El 53% restante corresponde a factores externos (viralidad, playlists, momento de lanzamiento) no capturados en las audio features. RF supera ligeramente a XGB.

### 2.2 Clasificación de género ✓

| Modelo | Accuracy | F1-macro |
|--------|----------|----------|
| Baseline RF (114 clases) | 0.257 | 0.246 |
| RF Macro-género | **0.465** | **0.408** |
| XGB Macro-género | 0.464 | 0.409 |

**Interpretación**: Agrupar 114 géneros en ~12 macro-categorías duplica las métricas. RF y XGB prácticamente empatan — ambos son viables. Un F1-macro de 0.41 con 12 clases es resultado moderado-bueno, teniendo en cuenta que muchos géneros comparten perfiles de audio similares (ej: pop y latin, o metal y rock).

---

## FASE 3 — Clustering y Recomendación ✓

### Clustering — hallazgos clave
El espacio de audio features **no crea muchos clusters bien separados**. El k óptimo por silhouette es **k=2**, lo que refleja que la división más natural en el espacio de audio es binaria:
- **Cluster 0 (energético)**: breakbeat, chicago-house, heavy-metal, happy, forro → alta energy, alta danceability
- **Cluster 1 (tranquilo)**: sleep, ambient, new-age, classical, romance → alta acousticness, baja energy

Esto es un hallazgo relevante para la memoria: las audio features capturan bien el "eje energía/acústica" pero tienen dificultad para discriminar géneros más matizados.

| Algoritmo | k | Silhouette | Davies-Bouldin |
|-----------|---|------------|----------------|
| KMeans | 2 | 0.2581 | 1.5729 |
| Agglomerative (Ward) | 2 | 0.1853 | 1.8648 |
| DBSCAN (eps=1.303) | 2+ruido | -0.037 | 1.044 |

PCA 2D explica el 48% de la varianza (PC1=32%, PC2=16%) — moderado, justifica usar UMAP para mejor visualización.

### Sistema de recomendación

| Enfoque | Coherencia género | Diversidad artistas | Pop. media |
|---------|------------------|---------------------|------------|
| KNN global (coseno) | 0.136 | **0.871** | 31.8 |
| Basado en cluster | 0.125 | 0.864 | 32.2 |
| **Híbrido (género+KNN)** | **1.000** | 0.831 | 33.5 |

El híbrido tiene coherencia perfecta porque filtra primero por género exacto — garantiza que las recomendaciones sean del mismo género. Contrapartida: menor diversidad de artistas.

**Para la app**: se usan los 3 enfoques, con híbrido como predeterminado.

---

## FASE 4 — App Streamlit ✓

App funcional en `app/app.py`, lanzada en http://localhost:8501.

**Modo 1 — Por canción semilla**: busca por nombre/artista → selecciona → recomienda top-N (híbrido o KNN).
**Modo 2 — Por preferencias**: sliders de danceability/energy/valence/tempo/acousticness + filtro de género → canciones más cercanas al perfil.

---

## Preguntas abiertas para profundizar en la memoria

1. **¿Qué factores hacen que una canción sea popular en Spotify?**
   Los modelos de regresión dan pistas, pero la explicación completa requiere datos externos (playlists, redes sociales). Interesante como sección de discusión/limitaciones.

2. **¿Tienen las audio features poder discriminativo suficiente para clasificar género?**
   Los resultados de Fase 2 responden esto. Si el F1-macro con macro-géneros es alto (>0.7), sí; si es moderado (0.5-0.7), el espacio de features tiene solapamiento significativo.

3. **¿Los clusters de audio corresponden a géneros musicales reales?**
   La tabla cruzada cluster-género de Fase 3 responde esto directamente.

4. **¿Cómo de buenas son las recomendaciones del sistema content-based?**
   Las métricas proxy (coherencia de género, diversidad de artistas) de Fase 3 dan una respuesta cuantitativa.

5. **¿Ha evolucionado el sonido de artistas concretos a lo largo de su carrera?**
   Fase 6 (pendiente) responde esto para 1-2 artistas.

---

## Aspectos a destacar en la introducción de la memoria

- El dataset refleja la diversidad musical global (114 géneros, artistas de todo el mundo)
- La variable `popularity` es un proxy imperfecto del éxito comercial (limitación bien documentada)
- El proyecto replica una parte del pipeline de ML de Spotify (análisis de contenido de audio)
  con datos públicos y herramientas open-source
- La combinación EDA → modelos supervisados → clustering → recomendación → interfaz
  cubre el ciclo completo de un proyecto de Data Science aplicado
