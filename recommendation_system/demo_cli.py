"""Demo por terminal del sistema de recomendacion personalizado."""
import sys
sys.path.insert(0, '.')
from recommendation_system.recommender import PersonalizedRecommender

def main():
    rec = PersonalizedRecommender()
    users = rec.get_all_users()

    if not users:
        print("No hay usuarios. Ejecuta build_synthetic_users.py primero.")
        return

    print("\nSISTEMA DE RECOMENDACION PERSONALIZADO -- DEMO CLI")
    print("=" * 60)

    for username in users[:3]:
        result = rec.recommend(username)
        desc = rec.profiles[username].get('description', '')
        print(f"\n{'='*60}")
        print(f"{username} -- {desc}")
        print(f"{'='*60}")

        if result['last_songs'].empty:
            print("  Sin historial")
            continue

        print("Ultimas canciones (mas reciente primero):")
        for i, (_, row) in enumerate(result['last_songs'].iterrows()):
            peso = "*1.5x" if i == 0 else "1.0x"
            print(f"  [{i+1}] {peso} {str(row['track_name'])[:40]} -- {str(row['artists'])[:25]}")

        print(f"\nGenero dominante -> {result['profile_genre']}")
        print("Top-5 recomendaciones:")
        for _, row in result['recommendations_main'].head(5).iterrows():
            print(f"  [{row['similarity']*100:.1f}%] {row['track_name'][:40]} -- {row['artists'][:25]}")

        if not result['recommendations_secondary'].empty:
            print(f"\nVariedad ({result['secondary_genre']}):")
            for _, row in result['recommendations_secondary'].iterrows():
                print(f"  [{row['similarity']*100:.1f}%] {row['track_name'][:40]} -- {row['artists'][:25]}")

        stats = rec.get_user_stats(username)
        print(f"\nEstadisticas: {stats}")

    # Demo dinamica
    demo_user = users[0]
    print(f"\n{'='*60}")
    print(f"DEMO DINAMICA -- Usuario: {demo_user}")
    print(f"{'='*60}")

    result_before = rec.recommend(demo_user)
    print(f"\n[ANTES] Genero dominante: {result_before['profile_genre']}")
    print("Top-3 recomendaciones:")
    for _, row in result_before['recommendations_main'].head(3).iterrows():
        print(f"  {row['track_name']} -- {row['artists']}")

    genre_before = result_before['profile_genre']
    opposite = 'clasica' if genre_before != 'clasica' else 'latino'
    search_term = 'Mozart' if opposite == 'clasica' else 'Bad Bunny'
    new_song = rec.search_songs(search_term, n_results=1)
    if not new_song.empty:
        rec.add_song_to_history(demo_user, new_song.iloc[0]['track_id'])
        print(f"\nNueva cancion anadida: {new_song.iloc[0]['track_name']} ({opposite})")

    result_after = rec.recommend(demo_user)
    print(f"\n[DESPUES] Genero dominante: {result_after['profile_genre']}")
    print("Nuevas ultimas 5 canciones:")
    for i, (_, row) in enumerate(result_after['last_songs'].iterrows()):
        peso = "*1.5x" if i == 0 else "1.0x"
        print(f"  [{i+1}] {peso} {row['track_name'][:40]}")

    recs_before = set(result_before['recommendations_main']['track_id'])
    recs_after = set(result_after['recommendations_main']['track_id'])
    print(f"\nCambios en recomendaciones: {len(recs_before - recs_after)}/10 distintas")

if __name__ == '__main__':
    main()
