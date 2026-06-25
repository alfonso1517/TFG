"""FASE A — Limpieza avanzada: eliminar remixes/live/remasters y cap de versiones."""
import pandas as pd
import numpy as np
import re
import os

# ── A.1 Eliminar versiones no originales ────────────────────────────────────
df = pd.read_csv('data/processed/tracks_model.csv')
print(f"Canciones antes de limpieza avanzada: {len(df)}")

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

# Verificación: feat. NO debe marcarse como no original
feat_check = df[df['track_name'].str.contains(r'feat', case=False, na=False)]
feat_non_orig = feat_check[feat_check['is_non_original']]
print(f"Canciones con 'feat' marcadas como no originales (debe ser 0): {len(feat_non_orig)}")

print(f"Detectadas como no originales: {df['is_non_original'].sum()}")

removed_df = df[df['is_non_original']]
removed_df[['track_name', 'artists', 'macro_genre', 'popularity']].to_csv(
    'data/processed/removed_non_originals.csv', index=False
)
print("Log guardado en data/processed/removed_non_originals.csv")
print("Distribucion por macro-genero de lo eliminado:")
print(removed_df['macro_genre'].value_counts().head(10))

df_originals = df[~df['is_non_original']].copy().reset_index(drop=True)
print(f"Canciones restantes tras eliminar no originales: {len(df_originals)}")

# ── A.2 Limpieza de nombres duplicados ──────────────────────────────────────

# Capa 1: Deduplicar por (track_name + artists) exacto
before = len(df_originals)
df_originals = df_originals.sort_values('popularity', ascending=False)
df_originals = df_originals.drop_duplicates(
    subset=['track_name', 'artists'], keep='first'
).reset_index(drop=True)
print(f"\nEliminadas por (nombre+artista) duplicado exacto: {before - len(df_originals)}")

# Capa 2: Cap de 5 versiones por titulo
name_counts = df_originals['track_name'].value_counts()
crowded_names = name_counts[name_counts > 5].index
print(f"Titulos con mas de 5 versiones distintas: {len(crowded_names)}")
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

# ── A.3 Guardar ──────────────────────────────────────────────────────────────
df_clean.to_csv('data/processed/tracks_clean_final.csv', index=False)
print(f"\n[OK] Dataset guardado: data/processed/tracks_clean_final.csv")

pipeline = [
    ('Dataset original',                  114000, 'Punto de partida'),
    ('Eliminar fila nula',                113999, 'Drop fila con artista/album/nombre nulo'),
    ('Deduplicar por track_id',            89741, 'keep=first; misma cancion en varios generos'),
    ('Imputar tempo=0',                    89741, 'Mediana del genero; 157 filas corregidas'),
    ('Eliminar outliers duracion',        '~89k', '>15 min o <0.5 min'),
    ('Eliminar no originales',            len(df_originals), 'Regex en track_name; feat. conservado'),
    ('Deduplicar (track_name+artists)',   before - (before - len(df_originals)), 'keep=first por popularidad'),
    ('Cap versiones (max 5 por titulo)',  len(df_clean), 'Top-5 por popularidad; afecta villancicos'),
]

print("\nPipeline completo de limpieza:")
print(f"{'Etapa':<45} {'Filas':<12} {'Decision'}")
print("-" * 85)
for etapa, filas, decision in pipeline:
    print(f"{etapa:<45} {str(filas):<12} {decision}")

os.makedirs('results', exist_ok=True)
with open('results/A2_limpieza_avanzada.md', 'w', encoding='utf-8') as f:
    f.write("# Limpieza Avanzada del Dataset\n\n")
    f.write("## Pipeline completo\n\n")
    f.write("| Etapa | Filas | Decision |\n")
    f.write("|-------|-------|----------|\n")
    for etapa, filas, decision in pipeline:
        f.write(f"| {etapa} | {filas} | {decision} |\n")
    f.write("\n## Decisiones clave\n\n")
    f.write("- **feat.** se conserva: las colaboraciones son canciones originales\n")
    f.write("- **Cap en 5** versiones por titulo (no 1): evita sesgo hacia artistas "
            "mainstream en villancicos y clasicos\n")
    f.write("- **live/unplugged** eliminados: grabaciones en directo tienen features "
            "de audio distintas que distorsionarian el clustering\n")
    f.write("- **Falsos positivos potenciales**: revisar muestra de removed_non_originals.csv\n")

print("[OK] Documentacion guardada en results/A2_limpieza_avanzada.md")
