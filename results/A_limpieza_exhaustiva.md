# Fase A — Limpieza de Datos Exhaustiva

## Pipeline de filas

| Etapa | Filas | Cambio |
|-------|-------|--------|
| Dataset original (dataset.csv) | 114.000 | — |
| Tras eliminar fila nula | 113.999 | −1 |
| Tras deduplicación por track_id (tracks_unique.csv) | 89,740 | −24,260 |
| Tras excluir duration < 0.5 min o > 15 min | 89,585 | −155 |
| Tras excluir loudness < −40 dB | 89,550 | −35 |
| **tracks_model.csv (final)** | **89,550** | — |

## Decisiones de limpieza

| Variable | Problema | Decisión | Justificación |
|----------|----------|----------|---------------|
| `tempo` | 0 filas con valor 0 | Imputar con mediana del género | No son canciones sin ritmo; error de detección de Spotify |
| `duration_ms` | Outliers extremos (< 0.5 min o > 15 min) | Excluir del conjunto de modelado | Probables podcasts, grabaciones especiales o intros; no son canciones típicas |
| `loudness` | 35 filas < −40 dB | Excluir | Grabaciones con problemas técnicos o silencio; valores anómalos |
| `popularity == 0` | 9447 filas (10.5%) | Mantener | No son errores; son canciones con pocas reproducciones recientes |

## Hallazgo: canciones muy instrumentales y muy populares

Se identificaron **194 canciones** con `instrumentalness > 0.9` y `popularity > 60`.
Estas son excepciones a la tendencia general (correlación negativa entre instrumentalness y popularidad).
Son candidatas interesantes para análisis cualitativo en la memoria del TFG.

## Correlaciones altas detectadas (|r_Spearman| > 0.7)

- **duration_ms ↔ duration_min**: r = 1.0  
- **energy ↔ loudness**: r = 0.7526  
- **energy ↔ acousticness**: r = -0.712  

Estas correlaciones implican posible multicolinealidad. En modelos lineales habría que actuar, pero Random Forest y XGBoost son robustos frente a este problema.

## Variables creadas (feature engineering)

| Variable | Fórmula | Justificación |
|----------|---------|---------------|
| `duration_min` | `duration_ms / 60000` | Más interpretable que milisegundos |
| `log_instrumentalness` | `log(1 + instrumentalness)` | Distribución muy sesgada a la derecha; log reduce el sesgo |
| `log_speechiness` | `log(1 + speechiness)` | Ídem |
| `log_acousticness` | `log(1 + acousticness)` | Ídem |
| `is_popular` | `popularity >= 50 → 1, else 0` | 23.7% populares. Umbral 50 separa el cuartil superior |
| `electronic_ratio` | `energy / (acousticness + 0.01)` | Proxy de cuánto suena "electrónica" vs "acústica" la canción |
| `macro_genre` | Mapa editorial de 114 → 12 categorías | Agrupa géneros similares; mejora clasificación y visualización |
| `tempo_norm` | `(tempo − min) / (max − min)` | Normalización para comparación en gráficos radar |

## Distribución de macro-géneros

| Macro-género | Canciones | % |
|-------------|-----------|---|
| electronica | 15,635 | 17.5% |
| world | 14,214 | 15.9% |
| pop | 11,720 | 13.1% |
| rock | 9,219 | 10.3% |
| metal | 7,192 | 8.0% |
| folk-acustico | 6,160 | 6.9% |
| kpop-jpop | 5,961 | 6.7% |
| otros | 5,813 | 6.5% |
| latino | 4,156 | 4.6% |
| hip-hop | 4,079 | 4.6% |
| jazz-blues | 2,913 | 3.3% |
| clasica | 2,488 | 2.8% |
