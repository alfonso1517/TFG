# 02 — Resultados Modelos Predictivos

## 2.1 Regresión de Popularity

### Métricas comparativas (test 20%)
| Modelo | RMSE | MAE | R² |
|--------|------|-----|----|
| Random Forest | 14.864 | 10.160 | 0.472 |
| XGBoost | 15.444 | 10.841 | 0.430 |

### Mejores hiperparámetros
- **RF**: {'n_estimators': 300, 'min_samples_leaf': 2, 'max_features': 0.5, 'max_depth': 30}
- **XGB**: {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'gamma': 0, 'colsample_bytree': 0.8}

### Top-3 features más importantes
- **RF**:  track_genre, acousticness, duration_ms
- **XGB**: track_genre, explicit, acousticness

### Interpretación
El R² relativamente bajo es esperable: `popularity` es un índice calculado
por Spotify con factores externos al audio (tendencias virales, lanzamientos
recientes, playlists editoriales) que no están en el dataset. Las features de
audio capturan la "calidad musical" pero no el "momento de lanzamiento".
La variable `track_genre` (codificada) suele aparecer entre las más importantes,
confirmando que el género tiene un impacto significativo en la popularidad.

---

## 2.2 Clasificación de Género

### Baseline con 114 géneros (RF, sin tuning)
| Métrica | Valor |
|---------|-------|
| Accuracy | 0.257 |
| F1-macro | 0.246 |

*Con 114 clases muy solapadas el F1-macro bajo es esperable.*

### Macro-géneros (~12 categorías)
| Modelo | Accuracy | F1-macro |
|--------|----------|----------|
| Random Forest | 0.465 | 0.408 |
| XGBoost | 0.464 | 0.409 |

### Mejores hiperparámetros
- **RF**: {'n_estimators': 300, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 30}
- **XGB**: {'subsample': 0.7, 'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05, 'colsample_bytree': 0.7}

### Top-3 features más importantes (clasificación)
- **RF**:  acousticness, danceability, duration_ms
- **XGB**: acousticness, explicit, instrumentalness

### Interpretación
Agrupar 114 géneros en ~12 macro-categorías mejora drásticamente las métricas.
Las features más discriminativas suelen ser `acousticness`, `energy` y
`instrumentalness`, lo que tiene sentido: clásica (alta acousticness +
instrumentalness), metal (alta energy, baja acousticness), electrónica
(alta energy + tempo), etc.

### Mapeo de macro-géneros utilizado
- **rock**: rock, alt-rock, alternative, grunge, indie, punk, emo, garage…
- **metal**: metal, heavy-metal, death-metal, black-metal, metalcore…
- **electronic**: edm, techno, trance, house, dubstep, drum-and-bass…
- **hip-hop**: hip-hop, rap, trap, gangster-rap
- **latino**: reggaeton, latin, salsa, samba, bossanova…
- **pop**: pop, indie-pop, k-pop, j-pop, dream-pop…
- **jazz-blues**: jazz, blues, soul, funk, r-n-b, gospel…
- **classical**: classical, opera, piano, orchestra
- **folk-country**: folk, acoustic, country, bluegrass, singer-songwriter
- **reggae-ska**: reggae, ska, dub, dancehall
- **world**: afrobeat, turkish, swedish, french, tango, fado, mandopop…
- **ambient**: ambient, chill, sleep, anime, children, comedy, disney…
- **other**: géneros no mapeados

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `02a_feature_importance_regression.png` | Importancia variables (RF vs XGB, regresión) |
| `02b_feature_importance_classification.png` | Importancia variables (RF vs XGB, clasificación) |
| `02c_confusion_matrix_rf.png` | Matriz confusión RF macro-género |
