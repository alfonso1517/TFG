# Limpieza Avanzada del Dataset

## Pipeline completo

| Etapa | Filas | Decision |
|-------|-------|----------|
| Dataset original | 114000 | Punto de partida |
| Eliminar fila nula | 113999 | Drop fila con artista/album/nombre nulo |
| Deduplicar por track_id | 89741 | keep=first; misma cancion en varios generos |
| Imputar tempo=0 | 89741 | Mediana del genero; 157 filas corregidas |
| Eliminar outliers duracion | ~89k | >15 min o <0.5 min |
| Eliminar no originales | 76541 | Regex en track_name; feat. conservado |
| Deduplicar (track_name+artists) | 76541 | keep=first por popularidad |
| Cap versiones (max 5 por titulo) | 75710 | Top-5 por popularidad; afecta villancicos |

## Decisiones clave

- **feat.** se conserva: las colaboraciones son canciones originales
- **Cap en 5** versiones por titulo (no 1): evita sesgo hacia artistas mainstream en villancicos y clasicos
- **live/unplugged** eliminados: grabaciones en directo tienen features de audio distintas que distorsionarian el clustering
- **Falsos positivos potenciales**: revisar muestra de removed_non_originals.csv
