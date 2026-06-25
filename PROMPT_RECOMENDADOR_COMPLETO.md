# Prompt para Claude Code — Limpieza Avanzada + Sistema de Recomendación Personalizado

## CONTEXTO GENERAL

Estás trabajando en el TFG de Análisis Estadístico aplicado a la Industria
Musical. Partes del dataset ya limpio en `data/processed/tracks_model.csv`
(~89.740 canciones únicas por `track_id`, con audio features y `macro_genre`)
y del scaler de clustering en `models/scaler_cluster.pkl`.

Este prompt cubre dos fases que deben ejecutarse en orden:

**FASE A** — Limpieza avanzada del dataset: eliminar remixes/live/remasters
y resolver el problema de canciones con el mismo nombre.

**FASE B** — Sistema de recomendación personalizado por usuario: construcción
de la clase `PersonalizedRecommender`, base de datos de 8 usuarios sintéticos,
aplicación Streamlit y tests de validación.

El dataset de entrada de la Fase A es `tracks_model.csv`.
El dataset de entrada de la Fase B es `tracks_clean_final.csv` (salida de A).
**Ejecuta siempre A antes que B.**

---

## LO QUE YA SABEMOS DEL DATASET (análisis previo)

- 89.741 canciones únicas por `track_id` en el dataset actual
- **4.364** son remixes/live/remastered detectables por regex en `track_name`
- **1.966** tienen "(feat. X)" en el título → son originales, NO eliminar
- **7.991 nombres** de canciones se repiten con artistas distintos; la causa
  dominante es música navideña ("Rockin' Around The Christmas Tree" tiene
  48 versiones de artistas distintos, "Frosty The Snowman" tiene 45, etc.)

---

## FASE A — LIMPIEZA AVANZADA DEL DATASET

### A.1 Eliminar versiones no originales (remix, live, remaster…)

El patrón busca palabras clave SOLO cuando aparecen dentro de paréntesis,
corchetes o tras guion/raya al final del título. Esto evita falsos positivos
como "Heartbreak Anniversary" o "Live Forever" de Oasis.

```python
import pandas as pd
import numpy as np
import re
import os

df = pd.read_csv('data/processed/tracks_model.csv')
print(f"Canciones antes de limpieza avanzada: {len(df)}")

# Patrón: versiones no originales identificables por sufijo entre delimitadores
# feat. EXCLUIDO a propósito — las colaboraciones son canciones originales
NON_ORIGINAL_PATTERN = re.compile(
    r'(?i)'
    r'[\(\[\-–—]'
    r'\s*'
    r'('
    r'remix'
    r'|remaster(?:ed)?(?:\s+\d{4})?'
    r'|live(?:\s+at\s+[^\)\]]+)?'
    r'|acoustic(?:\s+version)?'
    r'|(?:radio\s?)?edit'
    r'|extended(?:\s+(?:mix|version))?'
    r'|instrumental(?:\s+version)?'
    r'|karaoke(?:\s+version)?'
    r'|cover(?:\s+version)?'
    r'|(?:original\s+)?mix'
    r'|vip(?:\s+mix)?'
    r'|club(?:\s+mix)?'
    r'|demo(?:\s+version)?'
    r'|re\-?recorded?'
    r'|stripped(?:\s+version)?'
    r'|unplugged'
    r'|reprise'
    r'|medley'
    r'|mashup'
    r'|interlude'
    r'|bonus\s+track'
    r'|anniversary\s+edition'
    r'|deluxe\s+(?:edition|version)'
    r'|reissue'
    r'|piano\s+version'
    r'|(?:\d{4}\s+)?remaster'
    r')',
    re.IGNORECASE
)

df['is_non_original'] = df['track_name'].str.contains(
    NON_ORIGINAL_PATTERN, na=False
)

# Casos especiales que no usan delimitadores pero siempre indican no original
ALWAYS_NON_ORIGINAL = [
    r'(?i)\bkaraoke\b',
    r'(?i)\btribute\b',
    r'(?i)\bsound.?alike\b',
    r'(?i)\bcover\s+version\b',
    r'(?i)\bmade\s+famous\s+by\b',
    r'(?i)\boriginally\s+(?:performed|recorded)\s+by\b',
]
for pattern in ALWAYS_NON_ORIGINAL:
    df['is_non_original'] |= df['track_name'].str.contains(pattern, na=False)

# Verificación: las canciones con 'feat' NO deben marcarse como no originales
feat_check = df[df['track_name'].str.contains(r'feat', case=False, na=False)]
feat_non_orig = feat_check[feat_check['is_non_original']]
print(f"Canciones con 'feat' marcadas como no originales (debe ser 0): {len(feat_non_orig)}")
# Si > 0, hay un problema en el patrón — revisar antes de continuar

print(f"Detectadas como no originales: {df['is_non_original'].sum()}")

# Guardar log de lo eliminado (útil para la memoria)
removed_df = df[df['is_non_original']]
removed_df[['track_name', 'artists', 'macro_genre', 'popularity']].to_csv(
    'data/processed/removed_non_originals.csv', index=False
)
print("Log guardado en data/processed/removed_non_originals.csv")
print("Distribución por macro-género de lo eliminado:")
print(removed_df['macro_genre'].value_counts().head(10))

df_originals = df[~df['is_non_original']].copy().reset_index(drop=True)
print(f"Canciones restantes tras eliminar no originales: {len(df_originals)}")
```

### A.2 Limpieza de nombres duplicados (quirúrgica, tres capas)

**Principio:** el mismo título con artistas distintos son canciones distintas
y se conservan. Solo se elimina redundancia real.

```python
# Capa 1: Deduplicar por (track_name + artists) exacto
# Queda con la copia de mayor popularidad
before = len(df_originals)
df_originals = df_originals.sort_values('popularity', ascending=False)
df_originals = df_originals.drop_duplicates(
    subset=['track_name', 'artists'], keep='first'
).reset_index(drop=True)
print(f"Eliminadas por (nombre+artista) duplicado exacto: {before - len(df_originals)}")

# Capa 2: Cap de 5 versiones máximo por título
# Afecta principalmente a villancicos con 30-48 versiones de artistas distintos
# Se justifica el cap en 5 (no en 1) porque en Spotify coexisten múltiples
# versiones legítimas del mismo clásico — un cap más agresivo sesgaría
# el dataset hacia artistas mainstream
name_counts = df_originals['track_name'].value_counts()
crowded_names = name_counts[name_counts > 5].index
print(f"Títulos con más de 5 versiones distintas: {len(crowded_names)}")
print("Top 10:")
print(name_counts[name_counts > 5].head(10))

def cap_versions(group, max_versions=5):
    if len(group) <= max_versions:
        return group
    return group.nlargest(max_versions, 'popularity')

df_crowded = df_originals[df_originals['track_name'].isin(crowded_names)]
df_not_crowded = df_originals[~df_originals['track_name'].isin(crowded_names)]
df_crowded_capped = df_crowded.groupby('track_name', group_keys=False).apply(
    lambda g: cap_versions(g, max_versions=5)
)
df_clean = pd.concat([df_not_crowded, df_crowded_capped]).reset_index(drop=True)

print(f"Eliminadas por cap de versiones: {len(df_originals) - len(df_clean)}")
print(f"Dataset final limpio: {len(df_clean)} canciones")
```

### A.3 Guardar dataset final y tabla resumen del pipeline

```python
df_clean.to_csv('data/processed/tracks_clean_final.csv', index=False)
print(f"✅ Dataset guardado: data/processed/tracks_clean_final.csv")

# Tabla resumen del pipeline completo (va directamente a la memoria)
pipeline = [
    ('Dataset original',                  114000, 'Punto de partida'),
    ('Eliminar fila nula',                113999, 'Drop fila con artista/álbum/nombre nulo'),
    ('Deduplicar por track_id',            89741, 'keep="first"; misma canción en varios géneros'),
    ('Imputar tempo=0',                    89741, 'Mediana del género; 157 filas corregidas'),
    ('Eliminar outliers duración',        '~89k', '>15 min o <0.5 min'),
    ('Eliminar no originales',            '~85k', 'Regex en track_name; feat. conservado'),
    ('Deduplicar (track_name+artists)',   '~84k', 'keep="first" por popularidad'),
    ('Cap versiones (máx 5 por título)', '~83k', 'Top-5 por popularidad; afecta villancicos'),
]
print("\nPipeline completo de limpieza:")
print(f"{'Etapa':<40} {'Filas':<10} {'Decisión'}")
print("-" * 80)
for etapa, filas, decision in pipeline:
    print(f"{etapa:<40} {str(filas):<10} {decision}")

# Guardar documentación
os.makedirs('results', exist_ok=True)
with open('results/A2_limpieza_avanzada.md', 'w', encoding='utf-8') as f:
    f.write("# Limpieza Avanzada del Dataset\n\n")
    f.write("## Pipeline completo\n\n")
    f.write("| Etapa | Filas | Decisión |\n")
    f.write("|-------|-------|----------|\n")
    for etapa, filas, decision in pipeline:
        f.write(f"| {etapa} | {filas} | {decision} |\n")
    f.write("\n## Decisiones clave\n\n")
    f.write("- **feat.** se conserva: las colaboraciones son canciones originales\n")
    f.write("- **Cap en 5** versiones por título (no 1): evita sesgo hacia artistas "
            "mainstream en villancicos y clásicos\n")
    f.write("- **live/unplugged** eliminados: grabaciones en directo tienen features "
            "de audio distintas (más liveness, reverb) que distorsionarían el clustering\n")
    f.write("- **Falsos positivos potenciales**: revisar muestra de removed_non_originals.csv\n")
print("✅ Documentación guardada en results/A2_limpieza_avanzada.md")
```

---

## FASE B — SISTEMA DE RECOMENDACIÓN PERSONALIZADO

### B.1 Descripción del sistema

El sistema funciona así:

1. El usuario introduce su nombre → se carga su historial desde disco.
2. Se muestran sus **5 últimas canciones escuchadas** (más reciente primero).
3. Se calculan recomendaciones ponderando más la canción más reciente.
4. El usuario puede **añadir una nueva canción escuchada** → el historial se
   actualiza y las recomendaciones se recalculan automáticamente.
5. Los historiales **persisten en disco** entre sesiones (`user_profiles.json`).

**Pesos exactos:**

| Posición | Descripción          | Peso |
|----------|----------------------|------|
| 1ª       | Más reciente         | 1.5  |
| 2ª–5ª   | Las cuatro anteriores| 1.0  |

Vector de perfil = centroide ponderado normalizado:
`perfil = sum(w_i * features_i) / sum(w_i)`

**Algoritmo híbrido (género + KNN):**
1. Calcular centroide ponderado escalado con el `StandardScaler` del clustering.
2. Determinar el macro-género dominante (el de mayor peso acumulado).
3. KNN coseno dentro del subconjunto de ese género, excluyendo canciones
   ya escuchadas → top-10 recomendaciones principales.
4. KNN coseno en el segundo género más frecuente → 2-3 recomendaciones
   de variedad.

Usar **similitud coseno** (no euclídea): lo que importa es la dirección
del vector sonoro, no la magnitud absoluta. Documentar esta decisión.

### B.2 Estructura de ficheros a crear

```
recommendation_system/
├── user_profiles.json      # historiales (se genera automáticamente)
├── recommender.py          # clase PersonalizedRecommender
├── build_synthetic_users.py # script que crea los 8 usuarios sintéticos
├── app_usuario.py          # aplicación Streamlit
└── demo_cli.py             # demo por terminal para testear
```

---

### B.3 Fichero: `recommendation_system/recommender.py`

```python
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


class PersonalizedRecommender:

    AUDIO_FEATURES = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    WEIGHTS = [1.5, 1.0, 1.0, 1.0, 1.0]  # índice 0 = más reciente

    def __init__(self,
                 tracks_path='data/processed/tracks_clean_final.csv',
                 scaler_path='models/scaler_cluster.pkl',
                 profiles_path='recommendation_system/user_profiles.json'):

        self.df = pd.read_csv(tracks_path)
        self.df = self.df.dropna(subset=self.AUDIO_FEATURES).reset_index(drop=True)

        # Si el scaler no existe, entrenarlo ahora con el dataset limpio
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[self.AUDIO_FEATURES].fillna(0))
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler entrenado y guardado en {scaler_path}")

        self.X_scaled = self.scaler.transform(
            self.df[self.AUDIO_FEATURES].fillna(0)
        )

        # Índices por macro_genre para búsqueda rápida
        self.genre_indices = {
            genre: self.df[self.df['macro_genre'] == genre].index.tolist()
            for genre in self.df['macro_genre'].unique()
        }

        self.profiles_path = profiles_path
        os.makedirs(os.path.dirname(profiles_path), exist_ok=True)
        self.profiles = self._load_profiles()

    # ------------------------------------------------------------------
    # GESTIÓN DE PERFILES
    # ------------------------------------------------------------------

    def _load_profiles(self):
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_profiles(self):
        with open(self.profiles_path, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, ensure_ascii=False, indent=2)

    def get_or_create_user(self, username: str) -> dict:
        if username not in self.profiles:
            self.profiles[username] = {
                'username': username,
                'history': [],   # track_ids, más reciente en índice 0
                'created_at': datetime.now().isoformat()
            }
            self._save_profiles()
        return self.profiles[username]

    def get_last_n_songs(self, username: str, n: int = 5) -> pd.DataFrame:
        """Devuelve las últimas N canciones como DataFrame (índice 0 = más reciente)."""
        user = self.get_or_create_user(username)
        history_ids = user['history'][:n]
        if not history_ids:
            return pd.DataFrame()
        rows = [self.df[self.df['track_id'] == tid].iloc[0]
                for tid in history_ids
                if not self.df[self.df['track_id'] == tid].empty]
        return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()

    def add_song_to_history(self, username: str, track_id: str) -> bool:
        """
        Añade una canción al historial (al frente = más reciente).
        Evita duplicados consecutivos. Devuelve True si se añadió.
        """
        user = self.get_or_create_user(username)
        if user['history'] and user['history'][0] == track_id:
            return False
        user['history'].insert(0, track_id)
        user['last_updated'] = datetime.now().isoformat()
        self._save_profiles()
        return True

    def search_songs(self, query: str, n_results: int = 10) -> pd.DataFrame:
        """Busca por nombre de canción o artista (case-insensitive)."""
        q = query.lower().strip()
        mask = (
            self.df['track_name'].str.lower().str.contains(q, na=False) |
            self.df['artists'].str.lower().str.contains(q, na=False)
        )
        results = (self.df[mask]
                   .drop_duplicates(subset=['track_name', 'artists'])
                   .sort_values('popularity', ascending=False))
        return results[['track_id', 'track_name', 'artists',
                        'macro_genre', 'popularity']].head(n_results)

    def get_all_users(self) -> list:
        return list(self.profiles.keys())

    def get_user_stats(self, username: str) -> dict:
        user = self.get_or_create_user(username)
        history = self.get_last_n_songs(username, n=len(user['history']))
        if history.empty:
            return {'total_songs': 0}
        return {
            'total_songs': len(user['history']),
            'genero_favorito': history['macro_genre'].value_counts().index[0],
            'popularidad_media': round(history['popularity'].mean(), 1),
            'danceability_media': round(history['danceability'].mean(), 3),
            'energy_media': round(history['energy'].mean(), 3),
            'valence_media': round(history['valence'].mean(), 3),
        }

    # ------------------------------------------------------------------
    # MOTOR DE RECOMENDACIÓN
    # ------------------------------------------------------------------

    def _compute_user_profile_scaled(self, last_songs: pd.DataFrame) -> np.ndarray:
        """Centroide ponderado escalado (para KNN)."""
        n = len(last_songs)
        weights = np.array(self.WEIGHTS[:n], dtype=float)
        weights /= weights.sum()
        features = last_songs[self.AUDIO_FEATURES].fillna(0).values
        profile_raw = (features * weights[:, np.newaxis]).sum(axis=0)
        return self.scaler.transform(profile_raw.reshape(1, -1))

    def compute_user_profile_raw(self, last_songs: pd.DataFrame) -> dict:
        """Centroide ponderado en escala original (0-1), para el radar."""
        n = len(last_songs)
        weights = np.array(self.WEIGHTS[:n], dtype=float)
        weights /= weights.sum()
        features = last_songs[self.AUDIO_FEATURES].fillna(0).values
        profile = (features * weights[:, np.newaxis]).sum(axis=0)
        return dict(zip(self.AUDIO_FEATURES, profile))

    def _dominant_genres(self, last_songs: pd.DataFrame) -> list:
        """Géneros ordenados por peso acumulado (1.5 para el más reciente, 1.0 resto)."""
        n = len(last_songs)
        weights = self.WEIGHTS[:n]
        genre_weights = {}
        for i, (_, row) in enumerate(last_songs.iterrows()):
            g = row.get('macro_genre', 'otros')
            genre_weights[g] = genre_weights.get(g, 0) + weights[i]
        return sorted(genre_weights, key=genre_weights.get, reverse=True)

    def recommend(self, username: str,
                  n_main: int = 10, n_secondary: int = 3) -> dict:
        """
        Genera recomendaciones personalizadas.

        Retorna:
        {
          'last_songs': DataFrame,
          'profile_genre': str,
          'secondary_genre': str | None,
          'recommendations_main': DataFrame,
          'recommendations_secondary': DataFrame
        }
        """
        last_songs = self.get_last_n_songs(username, n=5)
        empty = {
            'last_songs': last_songs, 'profile_genre': None,
            'secondary_genre': None,
            'recommendations_main': pd.DataFrame(),
            'recommendations_secondary': pd.DataFrame()
        }
        if last_songs.empty:
            return empty

        profile_scaled = self._compute_user_profile_scaled(last_songs)
        genres = self._dominant_genres(last_songs)
        main_genre = genres[0]
        secondary_genre = genres[1] if len(genres) > 1 else None
        seen_ids = set(self.profiles[username]['history'])

        def knn_in_genre(genre, n):
            if not genre or genre not in self.genre_indices:
                return pd.DataFrame()
            idx = [i for i in self.genre_indices[genre]
                   if self.df.iloc[i]['track_id'] not in seen_ids]
            if not idx:
                return pd.DataFrame()
            sims = cosine_similarity(profile_scaled, self.X_scaled[idx])[0]
            top_local = np.argsort(sims)[::-1][:n]
            top_global = [idx[i] for i in top_local]
            result = self.df.iloc[top_global][
                ['track_id', 'track_name', 'artists', 'macro_genre', 'popularity']
            ].copy()
            result['similarity'] = sims[top_local]
            return result.reset_index(drop=True)

        return {
            'last_songs': last_songs,
            'profile_genre': main_genre,
            'secondary_genre': secondary_genre,
            'recommendations_main': knn_in_genre(main_genre, n_main),
            'recommendations_secondary': knn_in_genre(secondary_genre, n_secondary),
        }
```

---

### B.4 Fichero: `recommendation_system/build_synthetic_users.py`

Ejecutar este script UNA VEZ para crear los 8 usuarios de demo:
`python recommendation_system/build_synthetic_users.py`

```python
"""
Construye user_profiles.json con 8 usuarios sintéticos con perfiles
musicales distintos. Busca los track_ids reales en tracks_clean_final.csv.
Ejecutar una sola vez antes de arrancar la app.
"""
import sys, os
sys.path.insert(0, '.')
from recommendation_system.recommender import PersonalizedRecommender
import pandas as pd

rec = PersonalizedRecommender()

def find_and_add(username, searches):
    """
    searches: lista de (artista, título_parcial_o_None) ordenada
              de MÁS RECIENTE a MÁS ANTIGUA.
    """
    print(f"\n👤 {username}")
    added = 0
    # Añadir en orden inverso para que al terminar el historial
    # quede ordenado más reciente primero
    for artist, title in reversed(searches):
        results = rec.search_songs(artist if not title else f"{title} {artist}",
                                   n_results=5)
        # Si no hay resultado con título, buscar solo por artista
        if results.empty and title:
            results = rec.search_songs(artist, n_results=5)
        if results.empty:
            print(f"  ⚠️  No encontrado: {artist} / {title}")
            continue
        tid = results.iloc[0]['track_id']
        rec.add_song_to_history(username, tid)
        print(f"  ✅ {results.iloc[0]['track_name']} — {results.iloc[0]['artists']}")
        added += 1
    print(f"  → {added}/5 canciones añadidas")

# Marcar como sintéticos
def mark_synthetic(username, description):
    rec.profiles[username]['description'] = description
    rec.profiles[username]['is_synthetic'] = True
    rec._save_profiles()

# -----------------------------------------------------------------------
# 8 PERFILES — listas ordenadas de MÁS RECIENTE a MÁS ANTIGUA
# -----------------------------------------------------------------------

find_and_add('carlos_rdz', [
    ('Feid', 'Ferxxo'),
    ('Bad Bunny', None),
    ('J Balvin', None),
    ('KAROL G', None),
    ('Maluma', 'Hawai'),
])
mark_synthetic('carlos_rdz', 'Fan del reggaeton y urbano latino')

find_and_add('sara_mv', [
    ('Taylor Swift', 'Anti-Hero'),
    ('Harry Styles', None),
    ('Dua Lipa', 'Levitating'),
    ('Ed Sheeran', 'Shape of You'),
    ('Billie Eilish', 'bad guy'),
])
mark_synthetic('sara_mv', 'Fan del pop internacional mainstream')

find_and_add('miguel_fp', [
    ('Arctic Monkeys', 'Do I Wanna Know'),
    ('Coldplay', 'The Scientist'),
    ('Linkin Park', 'In The End'),
    ('Imagine Dragons', 'Radioactive'),
    ('Green Day', 'Boulevard'),
])
mark_synthetic('miguel_fp', 'Fan del rock alternativo')

find_and_add('laura_gs', [
    ('Calvin Harris', 'Summer'),
    ('Martin Garrix', None),
    ('Avicii', None),
    ('David Guetta', 'Titanium'),
    ('Daft Punk', None),
])
mark_synthetic('laura_gs', 'Fan de la música electrónica / EDM')

find_and_add('pablo_oc', [
    ('Drake', "God's Plan"),
    ('Kendrick Lamar', 'HUMBLE'),
    ('Eminem', 'Lose Yourself'),
    ('Kanye West', None),
    ('Jay-Z', 'Empire State'),
])
mark_synthetic('pablo_oc', 'Fan del hip-hop y rap')

find_and_add('elena_bt', [
    ('Miles Davis', None),
    ('John Coltrane', None),
    ('Bill Evans', None),
    ('Dave Brubeck', 'Take Five'),
    ('Thelonious Monk', None),
])
mark_synthetic('elena_bt', 'Fan del jazz e instrumental')

find_and_add('alex_rm', [
    ('Bad Bunny', None),       # lo último que escuchó es reggaeton
    ('Taylor Swift', None),
    ('Arctic Monkeys', None),
    ('Feid', None),
    ('Eminem', None),
])
mark_synthetic('alex_rm', 'Oyente ecléctico (mezcla de géneros)')

find_and_add('maria_lc', [
    ('SZA', None),
    ('Beyonce', 'Halo'),
    ('Frank Ocean', None),
    ('Alicia Keys', None),
    ('H.E.R.', None),
])
mark_synthetic('maria_lc', 'Fan del R&B y soul')

# -----------------------------------------------------------------------
# VERIFICACIÓN
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("VERIFICACIÓN — Recomendaciones para los 8 usuarios")
print("="*60)

import pandas as pd

verification = []
users = ['carlos_rdz','sara_mv','miguel_fp','laura_gs',
         'pablo_oc','elena_bt','alex_rm','maria_lc']

for username in users:
    result = rec.recommend(username)
    last = result['last_songs']
    recs = result['recommendations_main']

    print(f"\n{'─'*50}")
    desc = rec.profiles[username].get('description', '')
    print(f"👤 {username} — {desc}")
    if last.empty:
        print("  ❌ Sin historial")
        continue
    print("Últimas canciones:")
    for i, (_, row) in enumerate(last.iterrows()):
        peso = "★1.5x" if i == 0 else "☆1.0x"
        print(f"  [{i+1}] {peso} {row['track_name'][:38]} — {row['artists'][:22]}")
    print(f"Género dominante: {result['profile_genre']}")
    print("Top-5 recomendaciones:")
    for _, row in recs.head(5).iterrows():
        print(f"  🎵 [{row['similarity']*100:.1f}%] {row['track_name'][:38]} — {row['artists'][:22]}")

    verification.append({
        'usuario': username,
        'descripcion': desc,
        'historial_n': len(last),
        'genero_dominante': result['profile_genre'],
        'genero_secundario': result['secondary_genre'],
        'top_rec': recs.iloc[0]['track_name'] if not recs.empty else '—',
        'top_sim': f"{recs.iloc[0]['similarity']*100:.1f}%" if not recs.empty else '—',
    })

pd.DataFrame(verification).to_csv(
    'results/verificacion_usuarios_sinteticos.csv', index=False
)
print("\n✅ Tabla resumen: results/verificacion_usuarios_sinteticos.csv")
print("✅ Perfiles guardados en recommendation_system/user_profiles.json")
```

---

### B.5 Fichero: `recommendation_system/app_usuario.py`

Arrancar con: `streamlit run recommendation_system/app_usuario.py`

```python
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, '.')
from recommendation_system.recommender import PersonalizedRecommender
import plotly.graph_objects as go

st.set_page_config(
    page_title="Music Recommender · TFG",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    return PersonalizedRecommender()

rec = load_recommender()

# -----------------------------------------------------------------------
# SIDEBAR — Selector de usuario
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("🎵 Music Recommender")
    st.caption("TFG · Análisis Estadístico aplicado a la Industria Musical")
    st.divider()

    existing_users = rec.get_all_users()
    username_input = st.text_input("Nombre de usuario", placeholder="Ej: carlos_rdz")

    if existing_users:
        selected_existing = st.selectbox(
            "O elige un usuario existente",
            options=[""] + existing_users,
            index=0
        )
        if selected_existing:
            username_input = selected_existing

    username = username_input.strip()
    if not username:
        st.info("Introduce un nombre de usuario para empezar.")
        st.stop()

    rec.get_or_create_user(username)
    stats = rec.get_user_stats(username)

    st.divider()
    st.markdown(f"**Usuario activo:** `{username}`")
    if stats['total_songs'] > 0:
        st.metric("Canciones escuchadas", stats['total_songs'])
        st.metric("Género favorito", stats.get('genero_favorito', '—'))
        col1, col2, col3 = st.columns(3)
        col1.metric("Dance", f"{stats.get('danceability_media',0):.2f}")
        col2.metric("Energy", f"{stats.get('energy_media',0):.2f}")
        col3.metric("Mood", f"{stats.get('valence_media',0):.2f}")

# -----------------------------------------------------------------------
# MAIN — Dos columnas
# -----------------------------------------------------------------------
result = rec.recommend(username)
last_songs = result['last_songs']
recs_main = result['recommendations_main']
recs_secondary = result['recommendations_secondary']

col_left, col_right = st.columns([4, 6])

# ─── COLUMNA IZQUIERDA — Últimas canciones ───────────────────────────
with col_left:
    st.subheader(f"🎧 Últimas escuchadas")

    if last_songs.empty:
        st.info("Aún no has escuchado ninguna canción. Busca y añade una abajo.")
    else:
        for i, (_, row) in enumerate(last_songs.iterrows()):
            with st.container():
                peso_label = "⭐ **ÚLTIMA** · Peso 1.5×" if i == 0 else f"☆ Peso 1.0×"
                genre_color = {
                    'latino': '🟠', 'pop': '🔵', 'rock': '🔴',
                    'electronica': '🟣', 'hip-hop': '🟡', 'clasica': '⚪',
                    'jazz-blues': '🟤', 'metal': '⚫', 'folk-acustico': '🟢',
                }.get(row.get('macro_genre',''), '⚪')
                st.markdown(
                    f"**{i+1}. {row['track_name']}**  \n"
                    f"{row['artists']} · {genre_color} {row.get('macro_genre','—')}  \n"
                    f"{peso_label}"
                )
                c1, c2 = st.columns(2)
                c1.progress(float(row.get('danceability', 0)), text=f"Dance {row.get('danceability',0):.2f}")
                c2.progress(float(row.get('energy', 0)), text=f"Energy {row.get('energy',0):.2f}")
                st.divider()

    # Buscador para añadir nueva canción
    st.subheader("➕ Añadir canción escuchada")
    search_query = st.text_input("🔍 Buscar por artista o título", key="search")
    if search_query:
        search_results = rec.search_songs(search_query, n_results=8)
        if search_results.empty:
            st.warning("No se encontraron resultados.")
        else:
            for _, srow in search_results.iterrows():
                cols = st.columns([5, 2])
                cols[0].write(f"**{srow['track_name']}** — {srow['artists']} ({srow['macro_genre']})")
                if cols[1].button("+ Añadir", key=f"add_{srow['track_id']}"):
                    added = rec.add_song_to_history(username, srow['track_id'])
                    if added:
                        st.success(f"✅ Añadida: {srow['track_name']}")
                        st.rerun()
                    else:
                        st.info("Ya era tu última canción.")

# ─── COLUMNA DERECHA — Recomendaciones ──────────────────────────────
with col_right:
    st.subheader("✨ Recomendadas para ti")

    if last_songs.empty:
        st.info("Añade canciones para recibir recomendaciones personalizadas.")
    else:
        st.caption(f"Perfil sonoro → género dominante: **{result['profile_genre']}**")

        with st.expander("💡 ¿Cómo funciona esta recomendación?"):
            st.markdown(
                f"Calculamos tu **perfil sonoro** como la media ponderada de tus "
                f"últimas {len(last_songs)} canciones. Tu canción más reciente tiene "
                f"peso **1.5×** (influye un 50% más que las demás). Luego buscamos "
                f"las canciones más similares a ese perfil usando **similitud coseno** "
                f"en el espacio de 9 audio features de Spotify, dentro del género "
                f"dominante (**{result['profile_genre']}**)."
            )

        if not recs_main.empty:
            st.markdown(f"#### Porque te gusta el {result['profile_genre']}")
            display_cols = ['track_name', 'artists', 'popularity', 'similarity']
            display_df = recs_main[display_cols].copy()
            display_df['similarity'] = (display_df['similarity'] * 100).round(1).astype(str) + '%'
            display_df.columns = ['Canción', 'Artista', 'Popularidad', 'Similitud']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        if not recs_secondary.empty:
            st.markdown(f"#### Puede que también te guste · {result['secondary_genre']}")
            display_df2 = recs_secondary[display_cols].copy()
            display_df2['similarity'] = (display_df2['similarity'] * 100).round(1).astype(str) + '%'
            display_df2.columns = ['Canción', 'Artista', 'Popularidad', 'Similitud']
            st.dataframe(display_df2, use_container_width=True, hide_index=True)

# ─── RADAR DEL PERFIL SONORO ────────────────────────────────────────
if not last_songs.empty:
    st.divider()
    st.subheader("📊 Tu perfil sonoro")

    profile_raw = rec.compute_user_profile_raw(last_songs)
    features_radar = ['danceability', 'energy', 'valence',
                      'acousticness', 'speechiness', 'instrumentalness']

    # Media del género dominante para comparar
    genre_mean = (rec.df[rec.df['macro_genre'] == result['profile_genre']]
                  [features_radar].mean().to_dict())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[profile_raw.get(f, 0) for f in features_radar],
        theta=features_radar,
        fill='toself',
        name=f'Tu perfil ({username})',
        line_color='#1DB954',
        fillcolor='rgba(29,185,84,0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[genre_mean.get(f, 0) for f in features_radar],
        theta=features_radar,
        fill='toself',
        name=f'Media {result["profile_genre"]}',
        line_color='#FF6B35',
        fillcolor='rgba(255,107,53,0.15)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f'Perfil sonoro vs media del género {result["profile_genre"]}',
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Verde = tu perfil ponderado (la última canción pesa 1.5×). "
        "Naranja = media del género dominante. "
        "La diferencia entre ambas muestra en qué aspects tu gusto se desvía de la media del género."
    )
```

---

### B.6 Fichero: `recommendation_system/demo_cli.py`

Para testear sin Streamlit: `python recommendation_system/demo_cli.py`

```python
"""Demo por terminal del sistema de recomendación personalizado."""
import sys
sys.path.insert(0, '.')
from recommendation_system.recommender import PersonalizedRecommender

def main():
    rec = PersonalizedRecommender()
    users = rec.get_all_users()

    if not users:
        print("No hay usuarios. Ejecuta build_synthetic_users.py primero.")
        return

    print("\n🎵 SISTEMA DE RECOMENDACIÓN PERSONALIZADO — DEMO CLI")
    print("=" * 60)

    for username in users[:3]:  # demo con los 3 primeros usuarios
        result = rec.recommend(username)
        desc = rec.profiles[username].get('description', '')
        print(f"\n{'='*60}")
        print(f"👤 {username} — {desc}")
        print(f"{'='*60}")

        if result['last_songs'].empty:
            print("  Sin historial")
            continue

        print("Últimas canciones (más reciente primero):")
        for i, (_, row) in enumerate(result['last_songs'].iterrows()):
            peso = "★1.5x" if i == 0 else "☆1.0x"
            print(f"  [{i+1}] {peso} {row['track_name'][:40]} — {row['artists'][:25]}")

        print(f"\nGénero dominante → {result['profile_genre']}")
        print("Top-5 recomendaciones:")
        for _, row in result['recommendations_main'].head(5).iterrows():
            print(f"  🎵 [{row['similarity']*100:.1f}%] {row['track_name'][:40]} — {row['artists'][:25]}")

        if not result['recommendations_secondary'].empty:
            print(f"\nVariedad ({result['secondary_genre']}):")
            for _, row in result['recommendations_secondary'].iterrows():
                print(f"  🎶 [{row['similarity']*100:.1f}%] {row['track_name'][:40]} — {row['artists'][:25]}")

        stats = rec.get_user_stats(username)
        print(f"\nEstadísticas: {stats}")

    # Demo dinámica: añadir canción de género distinto y ver el cambio
    demo_user = users[0]
    print(f"\n{'='*60}")
    print(f"DEMO DINÁMICA — Usuario: {demo_user}")
    print(f"{'='*60}")

    result_before = rec.recommend(demo_user)
    print(f"\n[ANTES] Género dominante: {result_before['profile_genre']}")
    print("Top-3 recomendaciones:")
    for _, row in result_before['recommendations_main'].head(3).iterrows():
        print(f"  🎵 {row['track_name']} — {row['artists']}")

    # Añadir algo de género muy distinto
    genre_before = result_before['profile_genre']
    opposite = 'clasica' if genre_before != 'clasica' else 'latino'
    search_term = 'Mozart' if opposite == 'clasica' else 'Bad Bunny'
    new_song = rec.search_songs(search_term, n_results=1)
    if not new_song.empty:
        rec.add_song_to_history(demo_user, new_song.iloc[0]['track_id'])
        print(f"\n▶ Nueva canción añadida: {new_song.iloc[0]['track_name']} ({opposite})")

    result_after = rec.recommend(demo_user)
    print(f"\n[DESPUÉS] Género dominante: {result_after['profile_genre']}")
    print("Nuevas últimas 5 canciones:")
    for i, (_, row) in enumerate(result_after['last_songs'].iterrows()):
        peso = "★1.5x" if i == 0 else "☆1.0x"
        print(f"  [{i+1}] {peso} {row['track_name'][:40]}")

    recs_before = set(result_before['recommendations_main']['track_id'])
    recs_after = set(result_after['recommendations_main']['track_id'])
    print(f"\nCambios en recomendaciones: {len(recs_before - recs_after)}/10 distintas")

if __name__ == '__main__':
    main()
```

---

## TESTS DE VALIDACIÓN

Ejecutar tras completar las fases A y B:

```python
import sys
sys.path.insert(0, '.')
from recommendation_system.recommender import PersonalizedRecommender

rec = PersonalizedRecommender()

# TEST 1 — Recomendador básico
username = 'test_validation'
for q in ['Feid', 'Bad Bunny', 'J Balvin', 'KAROL G', 'Maluma']:
    r = rec.search_songs(q, 1)
    if not r.empty:
        rec.add_song_to_history(username, r.iloc[0]['track_id'])

last = rec.get_last_n_songs(username, 5)
assert len(last) <= 5, "Más de 5 canciones devueltas"
result = rec.recommend(username)
seen_ids = set(rec.profiles[username]['history'])
rec_ids = set(result['recommendations_main']['track_id'])
assert not (seen_ids & rec_ids), "¡Recomendaciones incluyen canciones ya escuchadas!"
print("✅ Test 1 OK — recomendador básico")

# TEST 2 — Persistencia entre sesiones
rec2 = PersonalizedRecommender()
assert len(rec2.profiles.get(username, {}).get('history', [])) > 0
print("✅ Test 2 OK — persistencia en disco")

# TEST 3 — Recálculo al añadir canción de otro género
recs_before = set(result['recommendations_main']['track_id'])
r = rec.search_songs('Bach', 1)
if not r.empty:
    rec.add_song_to_history(username, r.iloc[0]['track_id'])
result_after = rec.recommend(username)
recs_after = set(result_after['recommendations_main']['track_id'])
print(f"✅ Test 3 OK — {len(recs_before - recs_after)}/10 recomendaciones cambiaron al añadir Bach")

# TEST 4 — Edge cases
assert rec.recommend('usuario_inexistente_xyz')['recommendations_main'].empty
print("✅ Test 4a OK — usuario sin historial")
first_id = rec.profiles[username]['history'][0]
assert rec.add_song_to_history(username, first_id) == False
print("✅ Test 4b OK — duplicado consecutivo evitado")

# TEST 5 — Dataset limpio (solo originals)
import pandas as pd
df = pd.read_csv('data/processed/tracks_clean_final.csv')
# Verificar que no hay remixes evidentes
remix_check = df['track_name'].str.contains(
    r'(?i)\(remix\)|\(remaster', na=False
).sum()
print(f"✅ Test 5 OK — {remix_check} canciones con 'remix/remaster' en el dataset final "
      f"({'correcto: 0' if remix_check == 0 else 'REVISAR'})")
```

---

## ORDEN DE EJECUCIÓN

```
1. python -c "exec(open('fase_A_limpieza.py').read())"
   # O copiar el código de FASE A en un script y ejecutarlo

2. python recommendation_system/build_synthetic_users.py
   # Crea los 8 usuarios sintéticos con sus historiales reales

3. python recommendation_system/demo_cli.py
   # Verifica que todo funciona por terminal

4. streamlit run recommendation_system/app_usuario.py
   # Lanza la aplicación web interactiva
```

---

## ENTREGABLES FINALES

```
data/processed/
├── tracks_clean_final.csv              # dataset final (~83k canciones originales)
└── removed_non_originals.csv           # log de versiones eliminadas

recommendation_system/
├── recommender.py                      # clase PersonalizedRecommender
├── build_synthetic_users.py            # script de creación de usuarios
├── app_usuario.py                      # app Streamlit
├── demo_cli.py                         # demo por terminal
└── user_profiles.json                  # 8 usuarios sintéticos persistidos

results/
├── A2_limpieza_avanzada.md             # pipeline completo documentado
├── verificacion_usuarios_sinteticos.csv
└── E_recomendador_usuario.md           # resultados de los 5 tests + ejemplos
```

---

## NOTA PARA LA MEMORIA DEL TFG

**Limpieza:** eliminar versiones no originales mejora directamente la
diversidad del recomendador. Sin esta limpieza, un usuario que escuche
"Bohemian Rhapsody" podría recibir como "recomendación" la versión
remasterizada de la misma canción — ruido, no descubrimiento.

**Usuarios sintéticos:** permiten una validación cualitativa del sistema
(¿tienen sentido las recomendaciones para un fan de reggaeton? ¿y para un
fan de jazz?) que complementa las métricas cuantitativas (silhouette,
coherencia de género, diversidad). Esta doble validación es estándar en
la literatura de sistemas de recomendación.

**Limitaciones a mencionar:**
- Solo filtrado por contenido (content-based). Spotify combina esto con
  filtrado colaborativo (qué escuchan usuarios similares) y NLP, capas
  que no podemos replicar sin datos de usuarios reales.
- El catálogo está limitado al snapshot de ~2022 del dataset.
- El espacio de 9 features es una aproximación de muy baja dimensión
  respecto a los embeddings de alta dimensión que usa Spotify internamente.
