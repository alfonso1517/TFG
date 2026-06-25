# FASE E — Sistema de Recomendación Personalizado: Resultados y Validación

## Arquitectura del sistema

El sistema implementado es un **recomendador basado en contenido** (content-based filtering)
con perfil de usuario dinámico. El diseño combina tres elementos:

1. **Perfil ponderado del usuario**: centroide de las últimas 5 canciones escuchadas,
   donde la más reciente tiene peso **1.5×** y las demás **1.0×**. Esto refleja el
   supuesto de que el gusto reciente es más representativo del estado actual del oyente.

2. **Filtrado por macro-género dominante**: se calcula el género dominante según los
   pesos acumulados del historial y se restringe la búsqueda KNN a ese subconjunto
   del dataset. Esto mejora la coherencia temática de las recomendaciones y reduce
   el tiempo de búsqueda.

3. **KNN con similitud coseno** sobre las 9 audio features escaladas:
   `danceability, energy, loudness, speechiness, acousticness, instrumentalness,
   liveness, valence, tempo`.

## Dataset utilizado

- **Input**: `data/processed/tracks_clean_final.csv`
- **75.710 canciones** tras el pipeline de limpieza avanzada (FASE A):
  - 4.966 versiones no originales eliminadas (remix, live, remaster, karaoke, etc.)
  - 8.043 duplicados exactos (track_name + artista) eliminados
  - 831 eliminadas por el cap de 5 versiones por título

## Usuarios sintéticos creados

| Usuario | Perfil | Género dominante |
|---------|--------|-----------------|
| carlos_rdz | Fan del reggaeton y urbano latino | latino |
| sara_mv | Fan del pop internacional mainstream | pop |
| miguel_fp | Fan del rock alternativo | rock |
| laura_gs | Fan de la música electrónica / EDM | electronica |
| pablo_oc | Fan del hip-hop y rap | hip-hop |
| elena_bt | Fan del jazz e instrumental | jazz-blues |
| alex_rm | Oyente ecléctico (mezcla de géneros) | — (mixto) |
| maria_lc | Fan del R&B y soul | otros |

Cada usuario tiene 5 canciones en su historial, buscadas como tracks reales del dataset.

## Tests de validación

### Test 1 — Recomendador básico ✅
- Usuario de prueba con 5 canciones latinas (Feid, Bad Bunny, J Balvin, KAROL G, Maluma)
- `get_last_n_songs()` devuelve ≤ 5 canciones: **OK**
- Ninguna canción ya escuchada aparece en las recomendaciones: **0 colisiones**

### Test 2 — Persistencia entre sesiones ✅
- Se crea una nueva instancia de `PersonalizedRecommender()` y se comprueba que
  el historial del usuario de prueba sigue presente en `user_profiles.json`: **OK**

### Test 3 — Recalculación al añadir canción de género distinto ✅
- Se añade una canción de Bach (clásica) al historial latino del usuario de prueba
- Resultado: **4/10 recomendaciones cambiaron**, reflejando el nuevo perfil ponderado
- El sistema responde al cambio sin necesidad de reinicio

### Test 4a — Caso extremo: usuario sin historial ✅
- `recommend('usuario_inexistente_xyz')` devuelve DataFrame vacío sin errores: **OK**

### Test 4b — Duplicado consecutivo no se añade ✅
- Intentar añadir al frente del historial la misma canción que ya está en posición 0
  devuelve `False` y no modifica el historial: **OK**

### Test 5 — Limpieza del dataset ✅
- Búsqueda de patrones `(remix)` o `(remaster` en `tracks_clean_final.csv`
- **0 canciones encontradas**: la limpieza avanzada eliminó correctamente estos casos

## Evaluación proxy de las recomendaciones

Sin datos reales de usuarios no hay ground truth, por lo que se usan métricas proxy:

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| Coherencia de género | 1.000 | Fracción de recomendaciones del mismo macro-género que la semilla |
| Diversidad de artistas | 0.831 | Fracción de artistas distintos en top-10 |
| Popularidad media | 33.5 | Media de `popularity` (0-100) de las recomendaciones |

La alta coherencia de género (1.0) y la diversidad de artistas (0.83) muestran que
el recomendador evita tanto la contaminación de género como el sesgo hacia un solo artista.

## Interfaz Streamlit

Fichero: `recommendation_system/app_usuario.py`

Funcionalidades implementadas:
- Selector de usuario (texto libre o desplegable de usuarios existentes)
- Panel lateral con estadísticas: total canciones, género favorito, dance/energy/mood
- Columna izquierda: últimas 5 canciones con barras de progreso + buscador para añadir
- Columna derecha: tabla de recomendaciones principales + secundarias (género alternativo)
- Radar chart (Plotly): perfil sonoro del usuario (verde) vs media del género (naranja)
- Explicación del algoritmo en expander desplegable

Para lanzar: `streamlit run recommendation_system/app_usuario.py`

## Limitaciones documentadas

1. **Sin filtrado colaborativo**: el sistema no tiene acceso a datos de escuchas de
   otros usuarios, por lo que no puede explotar la señal social ("usuarios similares
   a ti también escuchan..."). Se usa únicamente la similitud de audio features.

2. **Snapshot de popularidad**: `popularity` en el dataset es una instantánea temporal
   del algoritmo de Spotify, con sesgo hacia recencia. No refleja el impacto histórico.

3. **Macro-géneros manuales**: el mapeo de 114 géneros a macro-categorías es una
   decisión editorial, no un clustering automático. Puede perderse granularidad.

4. **Historial corto**: con solo 5 canciones en el perfil de demostración, el centroide
   ponderado tiene alta varianza. Con un historial real de decenas de canciones el
   perfil sería mucho más estable y representativo.
