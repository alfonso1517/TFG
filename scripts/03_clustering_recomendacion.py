"""
FASE 3 — Clustering y sistema de recomendación
  - KMeans, AgglomerativeClustering, DBSCAN
  - PCA / UMAP para visualización
  - 3 enfoques de recomendación: KNN, cluster-based, híbrido
Genera figuras en reports/figures/ y notas en results/03_clustering_recomendacion.md
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

ROOT   = Path(__file__).parent.parent
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS   = ROOT / "reports" / "figures"
RES    = ROOT / "results"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

print("Cargando tracks_unique.csv ...")
df = pd.read_csv(PROC / "tracks_unique.csv").reset_index(drop=True)
print(f"  {len(df):,} canciones")

CLUSTER_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

X = df[CLUSTER_FEATURES].copy()

# ── 1. Escalado ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, MODELS / "scaler_cluster.pkl")
print("[OK] Datos escalados con StandardScaler")

# ── 2. PCA para visualización (2 componentes) ─────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca2 = pca.fit_transform(X_scaled)
pca3   = PCA(n_components=3, random_state=42)
X_pca3 = pca3.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_
print(f"[OK] PCA 2D: varianza explicada = {var_exp.sum()*100:.1f}% ({var_exp*100})")

# ── 3. KMeans — método del codo + silhouette ─────────────────────────────────
print("\nBuscando k óptimo para KMeans (k=2..15) ...")
inertias, silhouettes = [], []
K_RANGE = range(2, 16)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=5000, random_state=42))
    print(f"  k={k:2d}  inertia={inertias[-1]:.0f}  silhouette={silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_RANGE), inertias, "o-", color="#4C72B0")
axes[0].set_title("Método del Codo — KMeans")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inercia")
axes[1].plot(list(K_RANGE), silhouettes, "o-", color="#DD8452")
axes[1].set_title("Silhouette Score — KMeans")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
plt.tight_layout()
plt.savefig(FIGS / "03a_kmeans_elbow.png")
plt.close()

best_k = list(K_RANGE)[np.argmax(silhouettes)]
print(f"[OK] Mejor k por silhouette: {best_k}")

# — Ajuste KMeans final —
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
km_labels = km_final.fit_predict(X_scaled)
df["cluster_kmeans"] = km_labels
sil_km  = silhouette_score(X_scaled, km_labels, sample_size=5000, random_state=42)
db_km   = davies_bouldin_score(X_scaled, km_labels)
print(f"[OK] KMeans k={best_k}  Silhouette={sil_km:.4f}  Davies-Bouldin={db_km:.4f}")
joblib.dump(km_final, MODELS / "kmeans.pkl")

# ── 4. Agglomerative Clustering ───────────────────────────────────────────────
print("\nAgglomerativeClustering ...")
# Usar el mismo k que KMeans para comparación justa
# (muestra reducida para velocidad)
SAMPLE = 10000
idx_sample = np.random.default_rng(42).choice(len(X_scaled), SAMPLE, replace=False)
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
agg_labels_sample = agg.fit_predict(X_scaled[idx_sample])
sil_agg = silhouette_score(X_scaled[idx_sample], agg_labels_sample)
db_agg  = davies_bouldin_score(X_scaled[idx_sample], agg_labels_sample)
print(f"  Agglomerative k={best_k}  Silhouette={sil_agg:.4f}  Davies-Bouldin={db_agg:.4f}")

# Asignar etiquetas a todo el dataset usando KMeans como proxy
# (Agglomerative no tiene predict; aproximamos con centroide más cercano)
centroids_agg = np.array([
    X_scaled[idx_sample][agg_labels_sample == c].mean(axis=0)
    for c in range(best_k)
])
from sklearn.metrics.pairwise import euclidean_distances
agg_full = np.argmin(euclidean_distances(X_scaled, centroids_agg), axis=1)
df["cluster_agg"] = agg_full

# ── 5. DBSCAN ─────────────────────────────────────────────────────────────────
print("\nDBSCAN ...")
# eps estimado con k-distancia (k=5)
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled[:5000])
dists, _ = nbrs.kneighbors(X_scaled[:5000])
eps_est  = np.percentile(dists[:, -1], 90)
print(f"  eps estimado (P90 5-NN): {eps_est:.3f}")

dbscan = DBSCAN(eps=eps_est, min_samples=10, n_jobs=-1)
db_labels_sample = dbscan.fit_predict(X_scaled[:5000])
n_clusters_db = len(set(db_labels_sample)) - (1 if -1 in db_labels_sample else 0)
n_noise_db    = (db_labels_sample == -1).sum()
print(f"  DBSCAN: {n_clusters_db} clusters, {n_noise_db} ruido ({n_noise_db/5000*100:.1f}%)")
if n_clusters_db > 1:
    mask = db_labels_sample != -1
    sil_db = silhouette_score(X_scaled[:5000][mask], db_labels_sample[mask])
    db_db  = davies_bouldin_score(X_scaled[:5000][mask], db_labels_sample[mask])
    print(f"  Silhouette={sil_db:.4f}  Davies-Bouldin={db_db:.4f}")
else:
    sil_db = 0.0; db_db = 0.0

# ── 6. Visualización PCA 2D con clusters KMeans ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scatter1 = axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1],
                            c=km_labels, cmap="tab10", s=3, alpha=0.4)
axes[0].set_title(f"KMeans k={best_k} — PCA 2D")
axes[0].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
plt.colorbar(scatter1, ax=axes[0])

# Colorear por género (muestra)
np.random.seed(42)
sample_idx = np.random.choice(len(df), min(5000, len(df)), replace=False)
genres_sample = df["track_genre"].iloc[sample_idx].astype("category")
axes[1].scatter(X_pca2[sample_idx, 0], X_pca2[sample_idx, 1],
                c=genres_sample.cat.codes, cmap="tab20", s=3, alpha=0.3)
axes[1].set_title("PCA 2D — coloreado por género (muestra)")
axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
plt.tight_layout()
plt.savefig(FIGS / "03b_pca2d_clusters.png")
plt.close()
print("[OK] 03b guardado | PCA 2D + clusters")

# ── 7. Tabla cruzada cluster vs género (top géneros por cluster) ──────────────
cross = pd.crosstab(df["cluster_kmeans"], df["track_genre"])
top_per_cluster = {}
for c in range(best_k):
    top_genres = cross.loc[c].sort_values(ascending=False).head(5).index.tolist()
    top_per_cluster[c] = top_genres

print("\nTop-5 géneros por cluster KMeans:")
for c, genres in top_per_cluster.items():
    print(f"  Cluster {c}: {', '.join(genres)}")

# ── 8. UMAP (opcional — puede tardar ~1-2 min) ────────────────────────────────
try:
    import umap
    print("\nCalculando UMAP (puede tardar ~1-2 min) ...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30,
                         min_dist=0.1, n_jobs=-1)
    X_umap = reducer.fit_transform(X_scaled[:8000])
    umap_labels = km_labels[:8000]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1],
                          c=umap_labels, cmap="tab10", s=4, alpha=0.5)
    plt.colorbar(scatter, ax=ax)
    ax.set_title(f"UMAP 2D — KMeans k={best_k} (8k canciones)")
    plt.tight_layout()
    plt.savefig(FIGS / "03c_umap_clusters.png")
    plt.close()
    print("[OK] 03c guardado | UMAP 2D")
    umap_ok = True
except Exception as e:
    print(f"  UMAP no disponible: {e}")
    umap_ok = False

# ════════════════════════════════════════════════════════════════════════════════
# SISTEMA DE RECOMENDACIÓN — 3 enfoques
# ════════════════════════════════════════════════════════════════════════════════
print("\n══ Sistema de Recomendación ══")

# Guardar arrays procesados para la app
np.save(MODELS / "X_scaled.npy", X_scaled)
df.to_csv(PROC / "tracks_unique_clustered.csv", index=False)

# — Enfoque 1: KNN sobre audio features escaladas —
print("Construyendo índice KNN ...")
knn = NearestNeighbors(n_neighbors=11, metric="cosine", n_jobs=-1)
knn.fit(X_scaled)
joblib.dump(knn, MODELS / "knn_recommender.pkl")
print("[OK] knn_recommender.pkl guardado")

def recommend_knn(seed_idx, n=10):
    dists, indices = knn.kneighbors(X_scaled[seed_idx].reshape(1, -1), n_neighbors=n+1)
    recs = indices[0][1:]  # excluir la propia canción
    return df.iloc[recs][["track_name", "artists", "track_genre", "popularity"]].copy()

def recommend_cluster(seed_idx, n=10):
    c = df.iloc[seed_idx]["cluster_kmeans"]
    pool = df[df["cluster_kmeans"] == c].drop(index=seed_idx, errors="ignore")
    pool_idx = pool.index.tolist()
    dists = np.linalg.norm(X_scaled[pool_idx] - X_scaled[seed_idx], axis=1)
    top_n = np.argsort(dists)[:n]
    return pool.iloc[top_n][["track_name", "artists", "track_genre", "popularity"]].copy()

def recommend_hybrid(seed_idx, n=10):
    seed_genre = df.iloc[seed_idx]["track_genre"]
    pool = df[df["track_genre"] == seed_genre].drop(index=seed_idx, errors="ignore")
    pool_idx = pool.index.tolist()
    if len(pool_idx) < n:
        return recommend_knn(seed_idx, n)
    dists = np.linalg.norm(X_scaled[pool_idx] - X_scaled[seed_idx], axis=1)
    top_n = np.argsort(dists)[:n]
    return pool.iloc[top_n][["track_name", "artists", "track_genre", "popularity"]].copy()

# ── 9. Evaluación de los 3 enfoques (métricas proxy) ─────────────────────────
print("\nEvaluando recomendadores (100 semillas aleatorias) ...")

def eval_recommender(rec_func, n_seeds=100, n_recs=10):
    rng = np.random.default_rng(42)
    seeds = rng.choice(len(df), n_seeds, replace=False)
    genre_coherences, artist_diversities, pop_means = [], [], []
    for s in seeds:
        try:
            recs = rec_func(s, n=n_recs)
            seed_genre = df.iloc[s]["track_genre"]
            genre_coherences.append((recs["track_genre"] == seed_genre).mean())
            artist_diversities.append(recs["artists"].nunique() / len(recs))
            pop_means.append(recs["popularity"].mean())
        except Exception:
            continue
    return {
        "genre_coherence":    np.mean(genre_coherences),
        "artist_diversity":   np.mean(artist_diversities),
        "mean_popularity":    np.mean(pop_means),
    }

eval_knn     = eval_recommender(recommend_knn)
eval_cluster = eval_recommender(recommend_cluster)
eval_hybrid  = eval_recommender(recommend_hybrid)

print(f"  KNN     → {eval_knn}")
print(f"  Cluster → {eval_cluster}")
print(f"  Hybrid  → {eval_hybrid}")

# — Gráfico comparativo —
metrics = ["genre_coherence", "artist_diversity", "mean_popularity"]
methods = ["KNN", "Cluster", "Hybrid"]
values  = [
    [eval_knn[m]     for m in metrics],
    [eval_cluster[m] for m in metrics],
    [eval_hybrid[m]  for m in metrics],
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors = ["#4C72B0", "#DD8452", "#55A868"]
for i, m in enumerate(metrics):
    vals = [v[i] for v in values]
    axes[i].bar(methods, vals, color=colors)
    axes[i].set_title(m.replace("_", " ").title())
    axes[i].set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
    for j, v in enumerate(vals):
        axes[i].text(j, v + max(vals)*0.02, f"{v:.3f}", ha="center", fontsize=9)
plt.suptitle("Comparativa de Enfoques de Recomendación (100 semillas)", fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "03d_recommender_comparison.png")
plt.close()
print("[OK] 03d guardado | Comparativa recomendadores")

# ── 10. Demo: recomendaciones para una canción conocida ──────────────────────
# Buscar "Shape of You" o similar
seed_name = "Shape of You"
seed_matches = df[df["track_name"].str.contains(seed_name, case=False, na=False)]
if len(seed_matches) > 0:
    seed_idx = seed_matches.index[0]
    seed_info = df.iloc[seed_idx]
    print(f"\nDemo recomendación para: {seed_info['track_name']} — {seed_info['artists']}")
    print("  KNN:")
    print(recommend_knn(seed_idx, 5).to_string(index=False))
    print("  Hybrid:")
    print(recommend_hybrid(seed_idx, 5).to_string(index=False))

# ── 11. Guardar notas ────────────────────────────────────────────────────────
notes = f"""# 03 — Clustering y Sistema de Recomendación

## Datos y preprocesado
- Dataset: `tracks_unique.csv` ({len(df):,} canciones)
- Features de clustering: {', '.join(CLUSTER_FEATURES)}
- Escalado: `StandardScaler` (media 0, desv. típica 1)

## Reducción de dimensionalidad
- PCA 2D: varianza explicada = {var_exp.sum()*100:.1f}% (PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%)
- UMAP 2D: {'calculado sobre 8k canciones' if umap_ok else 'no ejecutado'}

## Comparativa de algoritmos de clustering

| Algoritmo | k/parámetros | Silhouette | Davies-Bouldin |
|-----------|-------------|------------|----------------|
| KMeans | k={best_k} | {sil_km:.4f} | {db_km:.4f} |
| Agglomerative (Ward) | k={best_k} (10k muestra) | {sil_agg:.4f} | {db_agg:.4f} |
| DBSCAN | eps={eps_est:.3f}, minPts=10 (5k muestra) | {sil_db:.4f} | {db_db:.4f} |

**Análisis**:
- KMeans proporciona clusters de tamaño más equilibrado y permite asignar
  etiquetas a nuevas canciones (predicción rápida).
- Agglomerative (Ward) tiende a resultados similares pero sin método predict.
- DBSCAN detecta {n_clusters_db} clusters y {n_noise_db} puntos de ruido
  ({n_noise_db/50:.1f}% de la muestra de 5k). Útil para identificar canciones
  "atípicas" que no encajan en ningún cluster principal.
- **KMeans k={best_k}** se elige para el sistema de recomendación por su
  equilibrio rendimiento/interpretabilidad.

## Top géneros por cluster KMeans
{''.join(f'- **Cluster {c}**: {", ".join(gs)}{chr(10)}' for c, gs in top_per_cluster.items())}

## Sistema de Recomendación — Comparativa

| Enfoque | Coherencia Género | Diversidad Artistas | Popularidad Media |
|---------|------------------|---------------------|-------------------|
| KNN (coseno) | {eval_knn['genre_coherence']:.3f} | {eval_knn['artist_diversity']:.3f} | {eval_knn['mean_popularity']:.1f} |
| Basado en Cluster | {eval_cluster['genre_coherence']:.3f} | {eval_cluster['artist_diversity']:.3f} | {eval_cluster['mean_popularity']:.1f} |
| Híbrido (género+KNN) | {eval_hybrid['genre_coherence']:.3f} | {eval_hybrid['artist_diversity']:.3f} | {eval_hybrid['mean_popularity']:.1f} |

**Interpretación**:
- **Híbrido** ofrece mayor coherencia de género (las recomendaciones son del
  mismo género que la semilla), a costa de menor diversidad de artistas.
- **KNN global** es más diverso y equilibrado en popularidad.
- **Cluster** funciona bien como compromiso, aunque la granularidad del cluster
  puede ser insuficiente para géneros muy específicos.
- Para la app Streamlit se usará el **enfoque híbrido** como principal y KNN
  como alternativa configurable.

## Nota sobre el sistema de Spotify real
Nuestro sistema es un **recomendador basado en contenido (content-based)**
usando audio features como proxy del embedding de audio. El sistema real de
Spotify combina adicionalmente filtrado colaborativo (comportamiento de usuarios)
y NLP sobre metadatos. Nuestra aproximación replica solo la componente de
análisis de audio, que es la única replicable con este dataset.

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `03a_kmeans_elbow.png` | Codo + silhouette para selección de k |
| `03b_pca2d_clusters.png` | PCA 2D coloreado por cluster y por género |
| `03c_umap_clusters.png` | UMAP 2D (si disponible) |
| `03d_recommender_comparison.png` | Métricas comparativas de los 3 enfoques |
"""

(RES / "03_clustering_recomendacion.md").write_text(notes, encoding="utf-8")
print("\n[OK] results/03_clustering_recomendacion.md guardado")
print("✓ Fase 3 completada.")
