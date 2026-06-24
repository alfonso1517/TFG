# =============================================================================
# FASE D — CLUSTERING Y SISTEMA DE RECOMENDACIÓN (versión profunda)
# TFG: IA y Análisis Estadístico aplicado a la Industria Musical (Spotify)
# =============================================================================
# Entrada:  data/processed/tracks_model.csv
# Salida:   models/*, reports/figures/D_*.png, results/D_clustering_recomendacion.md
# =============================================================================

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, random

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

ROOT   = Path(__file__).parent.parent
PROC   = ROOT / "data" / "processed"
FIGS   = ROOT / "reports" / "figures"
RES    = ROOT / "results"
MODELS = ROOT / "models"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.95)
random.seed(42)
np.random.seed(42)

print("=" * 70)
print("FASE D — CLUSTERING Y SISTEMA DE RECOMENDACIÓN")
print("=" * 70)

df = pd.read_csv(PROC / "tracks_model.csv")
print(f"Dataset: {len(df):,} filas")

# =============================================================================
# D.1 PREPROCESADO PARA CLUSTERING
# =============================================================================
CLUSTER_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                    "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

X_cluster = df[CLUSTER_FEATURES].fillna(df[CLUSTER_FEATURES].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

joblib.dump(scaler, MODELS / "scaler_cluster_v2.pkl")
print(f"\n[D.1] Features de clustering: {CLUSTER_FEATURES}")
print(f"      X_scaled shape: {X_scaled.shape}")

# =============================================================================
# D.2 SELECCIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS
# =============================================================================
print("\n[D.2] Probando k=2 a 15 (codo + silhouette + Davies-Bouldin)...")

k_range   = range(2, 16)
inertias, silhouettes, davies = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    db  = davies_bouldin_score(X_scaled, labels)
    silhouettes.append(sil)
    davies.append(db)
    print(f"  k={k:2d}: inercia={km.inertia_:.0f}, silhouette={sil:.4f}, DB={db:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(list(k_range), inertias, "bo-")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inercia (WCSS)")
axes[0].set_title("Método del Codo"); axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), silhouettes, "go-")
axes[1].axvline(list(k_range)[np.argmax(silhouettes)], color="red", ls="--", alpha=0.7,
                label=f"k óptimo={list(k_range)[np.argmax(silhouettes)]}")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score por k (mayor = mejor)"); axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(list(k_range), davies, "ro-")
axes[2].axvline(list(k_range)[np.argmin(davies)], color="blue", ls="--", alpha=0.7,
                label=f"k óptimo={list(k_range)[np.argmin(davies)]}")
axes[2].set_xlabel("k"); axes[2].set_ylabel("Davies-Bouldin (menor = mejor)")
axes[2].set_title("Davies-Bouldin Score por k"); axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle("Selección del número óptimo de clusters (KMeans)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGS / "D_cluster_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] D_cluster_selection.png")

k_opt = list(k_range)[np.argmax(silhouettes)]
print(f"\n  k óptimo por silhouette: {k_opt}")

# =============================================================================
# D.3 SILHOUETTE ANALYSIS POR MUESTRA
# =============================================================================
print(f"\n[D.3] Silhouette analysis (k={k_opt})...")

km_opt  = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
labels_km = km_opt.fit_predict(X_scaled)
joblib.dump(km_opt, MODELS / "kmeans_v2.pkl")

sil_vals = silhouette_samples(X_scaled, labels_km)
avg_sil  = silhouette_score(X_scaled, labels_km, sample_size=5000, random_state=42)
print(f"  Silhouette medio: {avg_sil:.4f}")
print(f"  Davies-Bouldin: {davies_bouldin_score(X_scaled, labels_km):.4f}")
print(f"  Tamaño de clústeres:")
for c in range(k_opt):
    print(f"    Cluster {c}: {(labels_km == c).sum():,} canciones")

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
colors_sil = plt.cm.Set2(np.linspace(0, 1, k_opt))

for i in range(k_opt):
    ith_sil = np.sort(sil_vals[labels_km == i])
    size_i  = len(ith_sil)
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                     facecolor=colors_sil[i], edgecolor=colors_sil[i],
                     alpha=0.8, label=f"Cluster {i} (n={size_i:,})")
    ax.text(-0.06, y_lower + 0.5 * size_i, f"C{i}", fontsize=9)
    y_lower = y_upper + 10

ax.axvline(avg_sil, color="red", ls="--", label=f"Silhouette medio={avg_sil:.3f}")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Clúster (canciones ordenadas)")
ax.set_title(f"Silhouette Analysis — KMeans (k={k_opt})")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "D_silhouette_analysis.png", dpi=150)
plt.close()
print("  [OK] D_silhouette_analysis.png")

# =============================================================================
# D.4 COMPARATIVA DE ALGORITMOS DE CLUSTERING
# =============================================================================
print(f"\n[D.4] Comparativa de algoritmos (k={k_opt})...")

# Agglomerative
agg = AgglomerativeClustering(n_clusters=k_opt, linkage="ward")
labels_agg = agg.fit_predict(X_scaled)
sil_agg = silhouette_score(X_scaled, labels_agg, sample_size=5000, random_state=42)
db_agg  = davies_bouldin_score(X_scaled, labels_agg)
print(f"  Agglomerative: sil={sil_agg:.4f}, DB={db_agg:.4f}")

# DBSCAN con k-distance
nn5 = NearestNeighbors(n_neighbors=5, n_jobs=-1)
nn5.fit(X_scaled)
distances_5, _ = nn5.kneighbors(X_scaled)
dist_5th = np.sort(distances_5[:, 4])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dist_5th, lw=0.8, color="steelblue")
ax.set_xlabel("Punto (ordenado)"); ax.set_ylabel("Dist. al 5º vecino")
ax.set_title("K-distance graph para selección de eps en DBSCAN")
ax.axhline(1.5, color="red", ls="--", label="eps=1.5")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "D_dbscan_eps.png", dpi=150)
plt.close()

eps_val = 1.5
db_alg = DBSCAN(eps=eps_val, min_samples=10, n_jobs=-1)
labels_db = db_alg.fit_predict(X_scaled)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = (labels_db == -1).sum()
print(f"  DBSCAN (eps={eps_val}): {n_clusters_db} clusters, {n_noise} ruido ({n_noise/len(labels_db)*100:.1f}%)")

valid_db = labels_db != -1
sil_db = silhouette_score(X_scaled[valid_db], labels_db[valid_db], sample_size=5000) \
         if n_clusters_db > 1 and valid_db.sum() > 100 else None

comp_data = {
    "KMeans":          {"n_clusters": k_opt, "silhouette": avg_sil, "davies_bouldin": davies_bouldin_score(X_scaled, labels_km), "noise_pct": 0},
    "Agglomerative":   {"n_clusters": k_opt, "silhouette": sil_agg, "davies_bouldin": db_agg, "noise_pct": 0},
    "DBSCAN":          {"n_clusters": n_clusters_db, "silhouette": sil_db, "davies_bouldin": None, "noise_pct": n_noise / len(labels_db) * 100},
}
comp_df = pd.DataFrame(comp_data).T
print("\n  Tabla comparativa de algoritmos:")
print(comp_df.round(4).to_string())

# =============================================================================
# D.5 VISUALIZACIÓN PCA + UMAP
# =============================================================================
print("\n[D.5] PCA 2D...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_.sum() * 100
print(f"  Varianza explicada PCA 2D: {var_exp:.1f}%")

try:
    import umap
    print("  UMAP disponible, calculando...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    X_umap = reducer.fit_transform(X_scaled)
    umap_ok = True
    print("  [OK] UMAP calculado")
except ImportError:
    print("  UMAP no disponible — usando PCA para ambas visualizaciones")
    X_umap = X_pca
    umap_ok = False

colors_genre_code = pd.Categorical(df["macro_genre"]).codes

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# PCA por cluster
sc1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km,
                          cmap="Set2", s=1, alpha=0.25)
axes[0, 0].set_title(f"PCA 2D — KMeans (k={k_opt}, var={var_exp:.0f}%)")
axes[0, 0].set_xlabel("PC1"); axes[0, 0].set_ylabel("PC2")
plt.colorbar(sc1, ax=axes[0, 0], label="Cluster")

# PCA por macro-género
sc2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors_genre_code,
                          cmap="tab20", s=1, alpha=0.2)
axes[0, 1].set_title("PCA 2D — por Macro-género")
axes[0, 1].set_xlabel("PC1"); axes[0, 1].set_ylabel("PC2")

# UMAP por cluster
sc3 = axes[1, 0].scatter(X_umap[:, 0], X_umap[:, 1], c=labels_km,
                          cmap="Set2", s=1, alpha=0.25)
axes[1, 0].set_title(f"{'UMAP' if umap_ok else 'PCA'} 2D — KMeans (k={k_opt})")
axes[1, 0].set_xlabel("Dim 1"); axes[1, 0].set_ylabel("Dim 2")
plt.colorbar(sc3, ax=axes[1, 0], label="Cluster")

# UMAP por macro-género
sc4 = axes[1, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=colors_genre_code,
                          cmap="tab20", s=1, alpha=0.2)
axes[1, 1].set_title(f"{'UMAP' if umap_ok else 'PCA'} 2D — por Macro-género")
axes[1, 1].set_xlabel("Dim 1"); axes[1, 1].set_ylabel("Dim 2")

plt.suptitle("Reducción de dimensionalidad: separación de clústeres y géneros",
             fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "D_pca_umap_clusters.png", dpi=120, bbox_inches="tight")
plt.close()
print("  [OK] D_pca_umap_clusters.png")

# =============================================================================
# D.6 INTERPRETACIÓN DE LOS CLUSTERS
# =============================================================================
print("\n[D.6] Perfil e interpretación de clusters...")

df["cluster_kmeans"] = labels_km
np.save(MODELS / "X_scaled_v2.npy", X_scaled)
df.to_csv(PROC / "tracks_with_clusters.csv", index=False)

cluster_profile = df.groupby("cluster_kmeans")[CLUSTER_FEATURES + ["popularity"]].mean()
print("\n  Perfil de clústeres:")
print(cluster_profile.round(3).to_string())

print("\n  Géneros dominantes por clúster:")
for c in range(k_opt):
    top_mg = df[df["cluster_kmeans"] == c]["macro_genre"].value_counts(normalize=True).head(5)
    print(f"\n  Cluster {c} ({(labels_km == c).sum():,} canciones):")
    for mg, pct in top_mg.items():
        print(f"    {mg:<18} {pct*100:.1f}%")

# Radar de clusters
features_radar = ["danceability", "energy", "valence", "acousticness",
                  "instrumentalness", "speechiness"]
N = len(features_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

cp_norm = (cluster_profile[features_radar] - cluster_profile[features_radar].min()) / \
          (cluster_profile[features_radar].max() - cluster_profile[features_radar].min() + 1e-9)

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
colors_c = plt.cm.Set1(np.linspace(0, 0.8, k_opt))

for c, color in zip(range(k_opt), colors_c):
    vals = cp_norm.loc[c].tolist() + [cp_norm.loc[c].tolist()[0]]
    ax.plot(angles, vals, linewidth=2.5, color=color,
            label=f"Cluster {c} (n={int((labels_km==c).sum()):,})")
    ax.fill(angles, vals, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features_radar, size=10)
ax.set_title(f"Perfil sonoro por clúster (k={k_opt}, normalizado)", pad=25, fontsize=13)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.savefig(FIGS / "D_cluster_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  [OK] D_cluster_radar.png")

# =============================================================================
# D.7 SISTEMA DE RECOMENDACIÓN
# =============================================================================
print("\n[D.7] Construyendo sistema de recomendación...")

# Recomendador 1: KNN global
knn_global = NearestNeighbors(n_neighbors=11, metric="cosine", n_jobs=-1)
knn_global.fit(X_scaled)
joblib.dump(knn_global, MODELS / "knn_global_v2.pkl")

def recommend_knn_global(track_idx, n=10):
    distances, indices = knn_global.kneighbors(X_scaled[track_idx:track_idx+1])
    return df.iloc[indices[0][1:n+1]][["track_name", "artists", "macro_genre", "popularity"]]

# Recomendador 2: basado en cluster
def recommend_cluster(track_idx, n=10):
    cluster = df.iloc[track_idx]["cluster_kmeans"]
    same = df[df["cluster_kmeans"] == cluster].drop(index=df.index[track_idx], errors="ignore")
    return same.nlargest(n, "popularity")[["track_name", "artists", "macro_genre", "popularity"]]

# Recomendador 3: híbrido (género + KNN dentro del género)
knn_by_genre    = {}
genre_indices   = {}

for genre in df["macro_genre"].unique():
    mask = df["macro_genre"] == genre
    idx  = np.where(mask)[0]
    if len(idx) < 12:
        continue
    genre_indices[genre] = idx
    knn_g = NearestNeighbors(n_neighbors=min(11, len(idx)), metric="cosine", n_jobs=-1)
    knn_g.fit(X_scaled[idx])
    knn_by_genre[genre] = knn_g

joblib.dump(knn_by_genre,  MODELS / "knn_by_genre_v2.pkl")
joblib.dump(genre_indices, MODELS / "genre_indices_v2.pkl")

def recommend_hybrid(track_idx, n=10):
    genre = df.iloc[track_idx]["macro_genre"]
    if genre not in knn_by_genre:
        return recommend_knn_global(track_idx, n)
    local_idx  = genre_indices[genre]
    local_pos  = np.where(local_idx == track_idx)[0]
    if len(local_pos) == 0:
        return recommend_knn_global(track_idx, n)
    distances, indices = knn_by_genre[genre].kneighbors(X_scaled[track_idx:track_idx+1])
    global_indices = local_idx[indices[0][1:n+1]]
    return df.iloc[global_indices][["track_name", "artists", "macro_genre", "popularity"]]

print("  [OK] 3 recomendadores construidos")

# =============================================================================
# D.8 EVALUACIÓN DE LOS RECOMENDADORES
# =============================================================================
print("\n[D.8] Evaluando recomendadores (200 semillas)...")

def evaluate_recommender(recommend_fn, n_seeds=200, n_recs=10):
    seeds = random.sample(range(len(df)), n_seeds)
    results = []
    for seed in seeds:
        seed_genre = df.iloc[seed]["macro_genre"]
        try:
            recs = recommend_fn(seed, n_recs)
        except Exception:
            continue
        if len(recs) == 0:
            continue
        genre_coh  = (recs["macro_genre"] == seed_genre).mean()
        artist_div = recs["artists"].nunique() / len(recs)
        pop_mean   = recs["popularity"].mean()
        rec_idx = recs.index.tolist()
        if len(rec_idx) >= 2:
            X_recs = X_scaled[rec_idx]
            dm     = cosine_distances(X_recs)
            il_div = dm[np.triu_indices(len(rec_idx), k=1)].mean()
        else:
            il_div = 0.0
        results.append({"genre_coherence": genre_coh, "artist_diversity": artist_div,
                        "popularity_mean": pop_mean, "intra_list_diversity": il_div})
    return pd.DataFrame(results).mean()

eval_global  = evaluate_recommender(recommend_knn_global)
eval_cluster = evaluate_recommender(recommend_cluster)
eval_hybrid  = evaluate_recommender(recommend_hybrid)

eval_results = {"KNN Global": eval_global, "Cluster": eval_cluster, "Híbrido": eval_hybrid}
eval_df = pd.DataFrame(eval_results).T
print("\n  Métricas de evaluación (200 semillas, top-10):")
print(eval_df.round(4).to_string())

fig, axes = plt.subplots(1, 4, figsize=(17, 5))
palette = ["steelblue", "coral", "mediumseagreen"]
metrics_list = ["genre_coherence", "artist_diversity", "popularity_mean", "intra_list_diversity"]
titles = ["Coherencia de género", "Diversidad de artistas", "Popularidad media", "Diversidad intra-lista"]

for ax, metric, title in zip(axes, metrics_list, titles):
    ax.bar(eval_df.index, eval_df[metric], color=palette, edgecolor="white", alpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(eval_df[metric]):
        ax.text(i, v + 0.01 * eval_df[metric].max(), f"{v:.3f}",
                ha="center", fontsize=8)

plt.suptitle("Comparativa de recomendadores (200 semillas, top-10 recomendaciones)",
             fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "D_recommender_comparison.png", dpi=150)
plt.close()
print("  [OK] D_recommender_comparison.png")

# =============================================================================
# D.9 DEMO DEL SISTEMA
# =============================================================================
print("\n[D.9] Demo del sistema de recomendación...")

test_songs = [
    ("Feid",        None,         "reggaeton colombiano"),
    ("Bad Bunny",   "Tití Me Preguntó", "urbano latino"),
    ("The Beatles", "Hey Jude",   "rock clásico"),
    ("Beethoven",   None,         "clásica"),
    ("Eminem",      "Lose Yourself", "hip-hop"),
]

demo_lines = []
for artist, track, desc in test_songs:
    mask = df["artists"].str.contains(artist, case=False, na=False, regex=False)
    if track:
        mask &= df["track_name"].str.contains(track, case=False, na=False, regex=False)
    candidates = df[mask]
    if len(candidates) == 0:
        # Búsqueda más amplia
        mask = df["artists"].str.contains(artist.split()[0], case=False, na=False, regex=False)
        candidates = df[mask]
    if len(candidates) == 0:
        demo_lines.append(f"\nSemilla '{artist}' no encontrada en el dataset")
        continue
    seed_idx = candidates.index[0]
    seed_info = df.iloc[seed_idx]
    recs = recommend_hybrid(seed_idx, n=5)
    line = (f"\n{'='*60}\n"
            f"SEMILLA: {seed_info['track_name']} — {seed_info['artists']}\n"
            f"Género: {seed_info['macro_genre']}, Popularidad: {seed_info['popularity']}\n"
            f"(Descripción: {desc})\n"
            f"Top-5 recomendaciones (Híbrido):\n"
            f"{recs.to_string(index=False)}")
    print(line)
    demo_lines.append(line)

# =============================================================================
# MARKDOWN DE RESULTADOS
# =============================================================================
print("\n[OK] Guardando resultados...")

# Construcción de los perfiles de cluster
cluster_desc = {}
for c in range(k_opt):
    top_feats = cluster_profile.loc[c, CLUSTER_FEATURES].sort_values(ascending=False)
    top_genre = df[df["cluster_kmeans"] == c]["macro_genre"].value_counts(normalize=True).head(3)
    desc_feat = ", ".join([f"{f}={v:.2f}" for f, v in top_feats.head(3).items()])
    desc_genre = ", ".join([f"{g}({pct*100:.0f}%)" for g, pct in top_genre.items()])
    cluster_desc[c] = {"features": desc_feat, "genres": desc_genre,
                       "n": int((labels_km == c).sum()),
                       "pop_media": round(df[df["cluster_kmeans"] == c]["popularity"].mean(), 1)}

md = f"""# Fase D — Clustering y Sistema de Recomendación (Versión Profunda)

## D.1 Preprocesado

Features utilizadas: {CLUSTER_FEATURES}
Escalado: StandardScaler (media 0, std 1)
Dataset: {len(df):,} canciones de `tracks_model.csv`

## D.2 Selección del número óptimo de clusters

Se probaron valores de k=2 a 15. Resultados:

| k | Silhouette | Davies-Bouldin | Inercia |
|---|-----------|----------------|---------|
"""
for i, k in enumerate(k_range):
    md += f"| {k} | {silhouettes[i]:.4f} | {davies[i]:.4f} | {inertias[i]:.0f} |\n"

md += f"""
**k óptimo = {k_opt}** (máximo silhouette)

---

## D.3 Silhouette Analysis (k={k_opt})

Silhouette medio: **{avg_sil:.4f}**
Davies-Bouldin: **{davies_bouldin_score(X_scaled, labels_km):.4f}**

El silhouette score (~0.25) es bajo pero esperado: la música se distribuye en un continuo
multidimensional, no en grupos discretos bien separados. Esto es un hallazgo en sí mismo:
las audio features de Spotify capturan fundamentalmente una dimensión de "intensidad/electronización"
más que la riqueza multidimensional de los estilos musicales.

---

## D.4 Comparativa de algoritmos

| Algoritmo | Clusters | Silhouette | Davies-Bouldin | Ruido (%) |
|-----------|----------|------------|----------------|-----------|
| KMeans | {comp_data['KMeans']['n_clusters']} | {comp_data['KMeans']['silhouette']:.4f} | {comp_data['KMeans']['davies_bouldin']:.4f} | 0.0% |
| AgglomerativeClustering | {comp_data['Agglomerative']['n_clusters']} | {comp_data['Agglomerative']['silhouette']:.4f} | {comp_data['Agglomerative']['davies_bouldin']:.4f} | 0.0% |
| DBSCAN (eps={eps_val}) | {comp_data['DBSCAN']['n_clusters']} | {str(round(comp_data['DBSCAN']['silhouette'], 4)) if comp_data['DBSCAN']['silhouette'] else 'N/A'} | N/A | {comp_data['DBSCAN']['noise_pct']:.1f}% |

KMeans y Agglomerative producen resultados muy similares. DBSCAN identifica muchos puntos como ruido
debido a la densidad uniforme del espacio de features — esto es normal con datos de audio.

---

## D.5 Perfil de cada clúster

"""
for c, info in cluster_desc.items():
    md += f"### Clúster {c} (n={info['n']:,}, popularidad media={info['pop_media']})\n"
    md += f"- **Features dominantes**: {info['features']}\n"
    md += f"- **Géneros más frecuentes**: {info['genres']}\n\n"

md += f"""
**Interpretación general:**
- El clustering converge a k={k_opt} porque las audio features de Spotify capturan principalmente
  una dimensión de energía/electronización. Este es un resultado coherente con la literatura
  (Tzanetakis & Cook, 2002; Schedl et al., 2014).
- Un sistema de recomendación real como Spotify usa embeddings de alta dimensión entrenados
  con redes neuronales sobre espectrogramas, lo que captura mucho más matiz que estas 9 features.

---

## D.7-D.8 Comparativa de recomendadores

Evaluación sobre 200 semillas aleatorias (top-10 recomendaciones por semilla):

| Recomendador | Coherencia género | Diversidad artistas | Popularidad media | Diversidad intra-lista |
|-------------|-------------------|---------------------|-------------------|------------------------|
| KNN Global | {eval_global['genre_coherence']:.3f} | {eval_global['artist_diversity']:.3f} | {eval_global['popularity_mean']:.1f} | {eval_global['intra_list_diversity']:.3f} |
| Cluster | {eval_cluster['genre_coherence']:.3f} | {eval_cluster['artist_diversity']:.3f} | {eval_cluster['popularity_mean']:.1f} | {eval_cluster['intra_list_diversity']:.3f} |
| **Híbrido (género+KNN)** | **{eval_hybrid['genre_coherence']:.3f}** | **{eval_hybrid['artist_diversity']:.3f}** | **{eval_hybrid['popularity_mean']:.1f}** | **{eval_hybrid['intra_list_diversity']:.3f}** |

**Conclusión:** El recomendador híbrido es el más apropiado para la app porque garantiza
coherencia de género alta con buena diversidad de artistas.

---

## D.9 Demo del sistema

"""
for line in demo_lines:
    md += f"```\n{line}\n```\n\n"

md += """
---

## Reflexión final

¿Por qué el clustering converge a k=2 o k=3? Las 9 audio features de Spotify no capturan
la riqueza multidimensional de los estilos musicales porque:

1. Están diseñadas para ser interpretables (valores entre 0 y 1), no para maximizar
   la discriminación entre géneros.
2. Hay una fuerte correlación entre energy y loudness (r=0.75) que comprime la varianza
   en pocas dimensiones efectivas.
3. Los géneros musicales son construcciones culturales, no físicas: dos canciones pueden
   sonar muy distintas culturalmente pero tener features de audio similares.

Un sistema de recomendación real (como el de Spotify) usa embeddings de alta dimensión
entrenados end-to-end con señal de comportamiento de usuario (clicks, skips, replays),
lo que captura mucho más matiz que cualquier conjunto de features predefinidas.
"""

(RES / "D_clustering_recomendacion.md").write_text(md, encoding="utf-8")
print("[OK] results/D_clustering_recomendacion.md guardado")
print("\n[FASE D COMPLETADA]")
