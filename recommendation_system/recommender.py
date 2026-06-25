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
        self.df = self.df.dropna(subset=self.AUDIO_FEATURES + ['track_name', 'artists']).reset_index(drop=True)

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
                'history': [],
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
                   .dropna(subset=['track_name', 'artists'])
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
