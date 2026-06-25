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
    searches: lista de (artista, titulo_parcial_o_None) ordenada
              de MAS RECIENTE a MAS ANTIGUA.
    """
    print(f"\n{username}")
    added = 0
    for artist, title in reversed(searches):
        results = rec.search_songs(artist if not title else f"{title} {artist}",
                                   n_results=5)
        if results.empty and title:
            results = rec.search_songs(artist, n_results=5)
        if results.empty:
            print(f"  No encontrado: {artist} / {title}")
            continue
        tid = results.iloc[0]['track_id']
        rec.add_song_to_history(username, tid)
        print(f"  OK {results.iloc[0]['track_name']} -- {results.iloc[0]['artists']}")
        added += 1
    print(f"  -> {added}/5 canciones anadidas")

def mark_synthetic(username, description):
    rec.profiles[username]['description'] = description
    rec.profiles[username]['is_synthetic'] = True
    rec._save_profiles()

# -----------------------------------------------------------------------
# 8 PERFILES
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
mark_synthetic('laura_gs', 'Fan de la musica electronica / EDM')

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
    ('Bad Bunny', None),
    ('Taylor Swift', None),
    ('Arctic Monkeys', None),
    ('Feid', None),
    ('Eminem', None),
])
mark_synthetic('alex_rm', 'Oyente eclectic (mezcla de generos)')

find_and_add('maria_lc', [
    ('SZA', None),
    ('Beyonce', 'Halo'),
    ('Frank Ocean', None),
    ('Alicia Keys', None),
    ('H.E.R.', None),
])
mark_synthetic('maria_lc', 'Fan del R&B y soul')

# -----------------------------------------------------------------------
# VERIFICACION
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("VERIFICACION -- Recomendaciones para los 8 usuarios")
print("="*60)

verification = []
users = ['carlos_rdz','sara_mv','miguel_fp','laura_gs',
         'pablo_oc','elena_bt','alex_rm','maria_lc']

for username in users:
    result = rec.recommend(username)
    last = result['last_songs']
    recs = result['recommendations_main']

    print(f"\n{'--'*25}")
    desc = rec.profiles[username].get('description', '')
    print(f"{username} -- {desc}")
    if last.empty:
        print("  Sin historial")
        continue
    print("Ultimas canciones:")
    for i, (_, row) in enumerate(last.iterrows()):
        peso = "*1.5x" if i == 0 else "1.0x"
        print(f"  [{i+1}] {peso} {str(row['track_name'])[:38]} -- {str(row['artists'])[:22]}")
    print(f"Genero dominante: {result['profile_genre']}")
    print("Top-5 recomendaciones:")
    for _, row in recs.head(5).iterrows():
        print(f"  [{row['similarity']*100:.1f}%] {str(row['track_name'])[:38]} -- {str(row['artists'])[:22]}")

    verification.append({
        'usuario': username,
        'descripcion': desc,
        'historial_n': len(last),
        'genero_dominante': result['profile_genre'],
        'genero_secundario': result['secondary_genre'],
        'top_rec': recs.iloc[0]['track_name'] if not recs.empty else '--',
        'top_sim': f"{recs.iloc[0]['similarity']*100:.1f}%" if not recs.empty else '--',
    })

pd.DataFrame(verification).to_csv(
    'results/verificacion_usuarios_sinteticos.csv', index=False
)
print("\n[OK] Tabla resumen: results/verificacion_usuarios_sinteticos.csv")
print("[OK] Perfiles guardados en recommendation_system/user_profiles.json")
