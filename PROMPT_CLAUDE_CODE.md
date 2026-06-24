# Prompt para Claude Code — TFG Spotify: EDA profundo, ML y Clustering

## CONTEXTO DEL PROYECTO

Estás trabajando en el TFG de Matemáticas/Estadística titulado
"IA y Análisis Estadístico aplicado a la Industria Musical". El dataset
principal es `dataset.csv` (114.000 filas, 21 columnas), que contiene canciones
de Spotify con audio features y la variable `popularity` (0-100).

**Estado actual documentado en el informe:**
- Fase 0 (limpieza básica): hecha. Resultados: 89.740 canciones únicas en
  `tracks_unique.csv`, 1 fila nula eliminada, 157 filas con `tempo=0`
  imputadas con mediana del género, fila duplicada del segundo CSV descartada.
- Fase 1 (EDA básico): 7 figuras, correlaciones, boxplots, radar. Resultado
  clave: `instrumentalness` es la feature con mayor correlación negativa con
  popularidad (−0.127). Correlaciones en general bajas.
- Fase 2 (RF y XGBoost básico): regresión popularidad RF R²=0.472,
  XGBoost R²=0.430. Clasificación macro-género (12 clases): F1-macro ≈ 0.41
  para ambos modelos. Baseline sin agrupación F1=0.25.
- Fase 3 (clustering básico): KMeans k=2 silhouette=0.258. Recomendador
  híbrido (género + KNN) coherencia=1.0, diversidad=0.83.

**Tu tarea en esta sesión es profundizar en tres áreas concretas**,
en el orden indicado:

1. Limpieza de datos exhaustiva y EDA profundo (mejorando y ampliando lo ya hecho)
2. Modelos ML (RF y XGBoost) con búsqueda de hiperparámetros rigurosa
3. Clustering y sistema de recomendación profundizado

Trabaja en scripts Python bien comentados o notebooks. Guarda todos los
gráficos en `reports/figures/` y las métricas/tablas en `results/`. Al final
de cada fase guarda un archivo markdown con los hallazgos clave.

---

## FASE A — LIMPIEZA DE DATOS EXHAUSTIVA

Carga `data/processed/tracks_unique.csv` (89.740 canciones, una por track_id).
Si no existe aún, créalo a partir de `dataset.csv` con `drop_duplicates('track_id',
keep='first')` tras eliminar la fila nula.

### A.1 Auditoría completa de calidad

Ejecuta y documenta cada uno de estos checks:

```python
# Estructura y tipos
df.info()
df.describe(include='all')

# Nulos por columna
df.isnull().sum()

# track_id únicos vs total
assert df['track_id'].nunique() == len(df), "Hay duplicados de track_id"

# Distribución de track_genre
df['track_genre'].value_counts()  # ¿géneros con muy pocas canciones?

# Tempo = 0 (si no fue imputado aún)
print("Tempo == 0:", (df['tempo'] == 0).sum())

# Rango de cada variable numérica
num_cols = ['popularity','duration_ms','danceability','energy','loudness',
            'speechiness','acousticness','instrumentalness','liveness',
            'valence','tempo']
for col in num_cols:
    print(f"{col}: min={df[col].min():.3f}, max={df[col].max():.3f},
          mean={df[col].mean():.3f}, skew={df[col].skew():.3f}")
```

### A.2 Tratamiento de outliers (documenta cada decisión)

**duration_ms:**
- Convertir a minutos: `df['duration_min'] = df['duration_ms'] / 60000`
- Calcular IQR y límites: Q1 − 1.5·IQR, Q3 + 1.5·IQR
- Identificar canciones > 15 minutos (probables podcasts/grabaciones especiales)
  y canciones < 0.5 minutos (intros/silencio)
- Decisión recomendada: eliminar los que superen 15 minutos y los menores
  de 0.5 minutos para el conjunto de modelado. Crear versión filtrada
  `tracks_model.csv`. Documentar cuántas filas se pierden.

**loudness:**
- Rango normal: −40 a 0 dB. Outliers extremos < −40 dB (grabaciones con
  problemas o silencio) → documentar su frecuencia y excluirlos del
  conjunto de modelado.

**tempo == 0:**
- Si no está imputado: imputar con la mediana del género correspondiente
  (`df.groupby('track_genre')['tempo'].transform(lambda x: x.fillna(x.median()))`)

**popularity == 0:**
- NO son errores técnicos — son canciones con muy pocas reproducciones
  recientes. Mantenerlas. Documentar su frecuencia (ya conocida: ~10.5%).
- Para la regresión, considera si tiene sentido hacer una versión del modelo
  excluyendo popularity=0 y otra incluyéndola, para ver el impacto.

**instrumentalness > 0.9 y popularity > 60:**
- Identifica estos casos: canciones muy instrumentales pero muy populares.
  ¿Cuántas hay? ¿De qué género? Son un hallazgo interesante para la memoria.

**Correlaciones entre features de audio:**
```python
from scipy.stats import spearmanr
corr_matrix = df[num_cols].corr(method='spearman')
# ¿Algún par con correlación > 0.8? → posible multicolinealidad
high_corr = [(i,j,corr_matrix.loc[i,j])
             for i in corr_matrix.columns
             for j in corr_matrix.columns
             if i < j and abs(corr_matrix.loc[i,j]) > 0.7]
print("Pares con correlación alta:", high_corr)
```
Documentar si `energy` y `loudness` están muy correlacionadas (es esperable).

### A.3 Feature engineering

Crea estas variables adicionales y documenta su justificación:

```python
# Log-transform de variables con distribución muy sesgada
import numpy as np
df['log_instrumentalness'] = np.log1p(df['instrumentalness'])
df['log_speechiness'] = np.log1p(df['speechiness'])
df['log_acousticness'] = np.log1p(df['acousticness'])
# (log1p evita log(0))

# Variable binaria: canción popular (popularity >= 50)
df['is_popular'] = (df['popularity'] >= 50).astype(int)
# Documenta el umbral elegido y el % de canciones en cada clase

# Duración en minutos (ya calculada)
# Ratio energy/acousticness (proxy de "electronización")
df['electronic_ratio'] = df['energy'] / (df['acousticness'] + 0.01)

# Macro-género (mapa editorial de 12 categorías — usa el mismo mapa
# que aparece en el informe actual del TFG, reproducido aquí):
genre_map = {
    'acoustic': 'folk-acustico', 'singer-songwriter': 'folk-acustico',
    'songwriter': 'folk-acustico', 'folk': 'folk-acustico',
    'country': 'folk-acustico', 'bluegrass': 'folk-acustico',
    'pop': 'pop', 'dance': 'pop', 'synth-pop': 'pop',
    'edm': 'electronica', 'techno': 'electronica', 'trance': 'electronica',
    'dubstep': 'electronica', 'house': 'electronica', 'electro': 'electronica',
    'disco': 'electronica', 'minimal-techno': 'electronica', 'detroit-techno': 'electronica',
    'chicago-house': 'electronica', 'deep-house': 'electronica',
    'hip-hop': 'hip-hop', 'rap': 'hip-hop', 'trap': 'hip-hop',
    'r-n-b': 'hip-hop', 'soul': 'hip-hop', 'funk': 'hip-hop',
    'latin': 'latino', 'reggaeton': 'latino', 'salsa': 'latino',
    'cumbia': 'latino', 'bachata': 'latino', 'reggaeton-colombiano': 'latino',
    'latin-alternative': 'latino',
    'rock': 'rock', 'alternative': 'rock', 'grunge': 'rock',
    'punk': 'rock', 'emo': 'rock', 'indie': 'rock', 'psych-rock': 'rock',
    'metal': 'metal', 'heavy-metal': 'metal', 'black-metal': 'metal',
    'death-metal': 'metal', 'hard-rock': 'metal', 'metalcore': 'metal',
    'classical': 'clasica', 'opera': 'clasica', 'piano': 'clasica',
    'chamber': 'clasica', 'orchestra': 'clasica',
    'jazz': 'jazz-blues', 'blues': 'jazz-blues', 'gospel': 'jazz-blues',
    'k-pop': 'kpop-jpop', 'j-pop': 'kpop-jpop', 'j-idol': 'kpop-jpop',
    'cantopop': 'kpop-jpop', 'mandopop': 'kpop-jpop',
    'world-music': 'world', 'afrobeat': 'world', 'sertanejo': 'world',
    'pagode': 'world', 'samba': 'world', 'indian': 'world',
    'ambient': 'otros', 'new-age': 'otros', 'children': 'otros',
    'comedy': 'otros', 'show-tunes': 'otros', 'sleep': 'otros',
}
df['macro_genre'] = df['track_genre'].map(genre_map).fillna('otros')
# Verifica que no queden géneros sin mapear (o muy pocos)
print(df['macro_genre'].value_counts())
```

Guarda el dataset final limpio con feature engineering en
`data/processed/tracks_model.csv`. Documenta en
`results/A_limpieza_exhaustiva.md`:
- Número de filas en cada etapa del pipeline (114k → 89.740 → X tras outliers)
- Decisiones tomadas y justificación de cada una
- Variables creadas y su justificación
- Pares de features con alta correlación detectados

---

## FASE B — EDA PROFUNDO (ampliando lo existente)

Usa `tracks_model.csv`. Genera los siguientes gráficos (guárdalos en
`reports/figures/` con nombres descriptivos). Después de cada gráfico,
escribe en el script/notebook una celda de texto con la interpretación
del hallazgo (esto es lo que irá directamente a la memoria del TFG).

### B.1 Distribución de la variable objetivo

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Histograma con KDE
sns.histplot(df['popularity'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Distribución de Popularity')
axes[0].axvline(df['popularity'].median(), color='red', ls='--', label=f'Mediana={df["popularity"].median():.0f}')
axes[0].legend()

# Boxplot
sns.boxplot(y=df['popularity'], ax=axes[1])
axes[1].set_title('Boxplot de Popularity')

# Acumulada (ECDF)
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(df['popularity'])
axes[2].plot(ecdf.x, ecdf.y)
axes[2].set_title('ECDF de Popularity')
axes[2].axvline(50, color='red', ls='--', label='Umbral is_popular=50')
axes[2].legend()

plt.tight_layout()
plt.savefig('reports/figures/B1_popularity_distribution.png', dpi=150)
```

Además: test de normalidad de Shapiro-Wilk sobre una muestra (n=5000) y
Kolmogorov-Smirnov contra distribución normal. Documenta el resultado.

### B.2 Correlaciones profundas

```python
# Correlación de Spearman (más robusta para no lineales) con popularity
from scipy.stats import spearmanr
features = ['danceability','energy','loudness','speechiness','acousticness',
            'instrumentalness','liveness','valence','tempo','duration_min',
            'explicit']
corr_results = []
for f in features:
    r, p = spearmanr(df[f].fillna(0), df['popularity'])
    corr_results.append({'feature': f, 'spearman_r': round(r,4), 'p_value': p,
                         'significativo': p < 0.05})
corr_df = pd.DataFrame(corr_results).sort_values('spearman_r')
print(corr_df)

# Heatmap completo de correlaciones entre features
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5)
ax.set_title('Matriz de Correlación de Spearman (features de audio)')
plt.tight_layout()
plt.savefig('reports/figures/B2_correlation_heatmap.png', dpi=150)
```

### B.3 Violin plots por macro-género (mejor que boxplots)

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
features_plot = ['popularity', 'danceability', 'energy', 'valence',
                 'acousticness', 'tempo']
for ax, feat in zip(axes.flat, features_plot):
    order = df.groupby('macro_genre')[feat].median().sort_values(ascending=False).index
    sns.violinplot(data=df, x='macro_genre', y=feat, order=order,
                   palette='Set2', ax=ax, cut=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{feat} por macro-género')
plt.tight_layout()
plt.savefig('reports/figures/B3_violin_by_macrogenre.png', dpi=150)
```

### B.4 Scatter matrix de audio features coloreado por macro-género

```python
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

features_scatter = ['danceability', 'energy', 'valence', 'acousticness',
                    'instrumentalness', 'tempo']

# Muestra de 5000 por legibilidad
df_sample = df.sample(5000, random_state=42)

# Color por macro_genre (top 6 macro-géneros)
top6 = df['macro_genre'].value_counts().head(6).index
palette = {'rock': '#e41a1c', 'pop': '#377eb8', 'electronica': '#4daf4a',
           'hip-hop': '#984ea3', 'latino': '#ff7f00', 'folk-acustico': '#a65628'}
colors = df_sample['macro_genre'].map(palette).fillna('gray')

fig, axes = scatter_matrix(df_sample[features_scatter], c=colors, alpha=0.3,
                            figsize=(14,12), diagonal='kde', s=5)
plt.suptitle('Scatter matrix de audio features (coloreado por macro-género)', y=1.02)
plt.savefig('reports/figures/B4_scatter_matrix.png', dpi=120, bbox_inches='tight')
```

### B.5 Análisis de popularidad vs cada feature (scatter + regresión local)

```python
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
features_vs_pop = ['danceability','energy','loudness','speechiness',
                   'acousticness','instrumentalness','liveness','valence',
                   'tempo','duration_min','log_instrumentalness','log_acousticness']
df_s = df.sample(8000, random_state=0)

for ax, feat in zip(axes.flat, features_vs_pop):
    ax.scatter(df_s[feat], df_s['popularity'], alpha=0.05, s=2, color='steelblue')
    # Línea de tendencia (lowess o regresión lineal)
    from scipy.stats import linregress
    slope, intercept, r, p, _ = linregress(df_s[feat].dropna(), df_s['popularity'][df_s[feat].notna()])
    x_line = np.linspace(df_s[feat].min(), df_s[feat].max(), 100)
    ax.plot(x_line, slope*x_line + intercept, color='red', lw=1.5,
            label=f'r={r:.3f}')
    ax.set_xlabel(feat); ax.set_ylabel('popularity')
    ax.set_title(feat); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('reports/figures/B5_features_vs_popularity.png', dpi=120)
```

### B.6 Gráfico de radar por macro-género (perfil sonoro)

```python
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

features_radar = ['danceability','energy','valence','acousticness',
                  'instrumentalness','speechiness','tempo_norm']
# Normalizar tempo a [0,1] primero:
df['tempo_norm'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())

genre_profiles = df.groupby('macro_genre')[features_radar].mean()
# Normalizar cada feature a [0,1] para que el radar sea comparable
genre_profiles_norm = (genre_profiles - genre_profiles.min()) / (genre_profiles.max() - genre_profiles.min())

categories = features_radar
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

top_genres = ['rock','pop','electronica','hip-hop','latino','folk-acustico',
              'clasica','metal']
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
colors_radar = plt.cm.Set2(np.linspace(0, 1, len(top_genres)))

for genre, color in zip(top_genres, colors_radar):
    if genre not in genre_profiles_norm.index:
        continue
    values = genre_profiles_norm.loc[genre].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=genre, color=color)
    ax.fill(angles, values, alpha=0.05, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_title('Perfil sonoro por macro-género (normalizado)', pad=20, fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.savefig('reports/figures/B6_radar_by_macrogenre.png', dpi=150, bbox_inches='tight')
```

### B.7 Top géneros por popularidad media + intervalo de confianza

```python
genre_stats = df.groupby('macro_genre')['popularity'].agg(
    mean='mean', std='std', n='count'
).reset_index()
genre_stats['se'] = genre_stats['std'] / np.sqrt(genre_stats['n'])
genre_stats['ci95'] = 1.96 * genre_stats['se']
genre_stats = genre_stats.sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(genre_stats['macro_genre'], genre_stats['mean'],
       yerr=genre_stats['ci95'], capsize=5, color='steelblue', alpha=0.8)
ax.set_title('Popularidad media por macro-género ± IC 95%')
ax.set_xlabel('Macro-género')
ax.set_ylabel('Popularidad media')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('reports/figures/B7_popularity_by_genre_ci.png', dpi=150)
```

### B.8 Análisis de variables categóricas vs popularidad

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# explicit
sns.boxplot(data=df, x='explicit', y='popularity', ax=axes[0], palette='Set2')
axes[0].set_title('Popularity vs Explicit')

# mode (Mayor vs Menor)
df['mode_label'] = df['mode'].map({1: 'Mayor (alegre)', 0: 'Menor (triste)'})
sns.violinplot(data=df, x='mode_label', y='popularity', ax=axes[1], palette='pastel')
axes[1].set_title('Popularity vs Mode')

# key
sns.boxplot(data=df, x='key', y='popularity', ax=axes[2], palette='tab10')
axes[2].set_title('Popularity vs Key musical')

# Test Mann-Whitney para explicit
from scipy.stats import mannwhitneyu
exp_pop = df[df['explicit']==True]['popularity']
noexp_pop = df[df['explicit']==False]['popularity']
stat, p = mannwhitneyu(exp_pop, noexp_pop)
print(f"Mann-Whitney explicit vs no-explicit: stat={stat:.0f}, p={p:.4f}")
# ANOVA para key
from scipy.stats import f_oneway
groups = [df[df['key']==k]['popularity'].values for k in range(12)]
f_stat, p_anova = f_oneway(*groups)
print(f"ANOVA key: F={f_stat:.3f}, p={p_anova:.4f}")

plt.tight_layout()
plt.savefig('reports/figures/B8_categorical_vs_popularity.png', dpi=150)
```

### B.9 Análisis de artistas

```python
# Top 20 artistas por número de canciones y su popularidad media
top_artists = df.groupby('artists').agg(
    n_songs=('track_name','count'),
    pop_mean=('popularity','mean'),
    pop_max=('popularity','max')
).sort_values('n_songs', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(top_artists))
bars = ax.bar(x, top_artists['n_songs'], color='steelblue', alpha=0.7, label='Nº canciones')
ax2 = ax.twinx()
ax2.plot(x, top_artists['pop_mean'], 'ro-', lw=2, label='Popularidad media')
ax.set_xticks(x)
ax.set_xticklabels(top_artists.index, rotation=60, ha='right', fontsize=9)
ax.set_ylabel('Nº canciones en dataset')
ax2.set_ylabel('Popularidad media')
ax.set_title('Top 20 artistas: volumen vs popularidad media')
plt.tight_layout()
plt.savefig('reports/figures/B9_top_artists.png', dpi=150)
```

### B.10 Análisis de canciones con popularity = 0

```python
zero_pop = df[df['popularity'] == 0]
nonzero_pop = df[df['popularity'] > 0]

print(f"Canciones con popularity=0: {len(zero_pop)} ({len(zero_pop)/len(df)*100:.1f}%)")
print("\nDistribución por macro-género (popularity=0):")
print(zero_pop['macro_genre'].value_counts(normalize=True).head(10))

# ¿Son más instrumentales? ¿Más antiguas? ¿De géneros nicho?
print("\nComparación de features medias (pop=0 vs pop>0):")
comparison = pd.DataFrame({
    'pop=0': zero_pop[['danceability','energy','instrumentalness','acousticness','loudness']].mean(),
    'pop>0': nonzero_pop[['danceability','energy','instrumentalness','acousticness','loudness']].mean()
})
print(comparison)
```

Guarda un resumen de los 10 hallazgos más importantes del EDA en
`results/B_EDA_hallazgos.md`, con una frase interpretativa para cada gráfico.

---

## FASE C — MODELOS PREDICTIVOS RF Y XGBOOST (versión rigurosa)

### C.1 Preprocesado del pipeline de modelado

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, f1_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from category_encoders import TargetEncoder
import shap

df = pd.read_csv('data/processed/tracks_model.csv')

# Features de audio (núcleo del modelo)
AUDIO_FEATURES = ['danceability','energy','loudness','speechiness',
                  'acousticness','instrumentalness','liveness','valence',
                  'tempo','duration_min','explicit','key','mode','time_signature']

# Incluir macro_genre codificado para regresión (target encoding)
# y log-transforms
FEATURES_REG = AUDIO_FEATURES + ['log_instrumentalness','log_acousticness',
                                   'log_speechiness']
TARGET_REG = 'popularity'
TARGET_CLF = 'macro_genre'
```

### C.2 Regresión de popularidad — versión rigurosa

#### C.2.1 Split estratificado

Para regresión, la estratificación por quintiles de popularity evita que
el split aleatorio tenga distribuciones de popularidad distintas en train y test:

```python
# Estratificar por quintiles de popularity
df['pop_quintile'] = pd.qcut(df[TARGET_REG], q=5, labels=False)

X = df[FEATURES_REG].copy()
y = df[TARGET_REG].copy()
strat = df['pop_quintile']

# Target encoding para track_genre (114 categorías) — aplica ANTES del split
# pero usando solo los datos de train para evitar data leakage
# Hacer split primero, luego encodear

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=strat
)

# Target encoding: fit SOLO en train, transform en train y test
te = TargetEncoder(cols=['track_genre'])  # si track_genre está en FEATURES_REG
X_train_enc = te.fit_transform(X_train, y_train)
X_test_enc = te.transform(X_test)

# Para explicit: convertir bool a int si no está hecho
X_train_enc['explicit'] = X_train_enc['explicit'].astype(int)
X_test_enc['explicit'] = X_test_enc['explicit'].astype(int)

print(f"Train: {X_train_enc.shape}, Test: {X_test_enc.shape}")
print("Distribución quintiles en train:", y_train.describe())
print("Distribución quintiles en test:", y_test.describe())

# Verificar que la distribución de popularity es similar en train y test
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=30, alpha=0.7, label='Train', color='steelblue')
axes[0].hist(y_test, bins=30, alpha=0.7, label='Test', color='orange')
axes[0].legend(); axes[0].set_title('Distribución popularity Train vs Test')
# KDE comparativa
y_train.plot.kde(ax=axes[1], label='Train', color='steelblue')
y_test.plot.kde(ax=axes[1], label='Test', color='orange')
axes[1].legend(); axes[1].set_title('KDE popularity Train vs Test')
plt.savefig('reports/figures/C_train_test_distribution.png', dpi=150)
```

#### C.2.2 Experimento: impact del porcentaje de train

Varía el porcentaje de train de 50% a 90% y observa el impacto en R²:

```python
train_sizes_pct = [0.5, 0.6, 0.7, 0.8, 0.9]
results_split = []

for ts in train_sizes_pct:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=1-ts, random_state=42, stratify=strat
    )
    # Encoding rápido sin TE para esta prueba (o con TE si tienes tiempo)
    X_tr_num = X_tr.select_dtypes(include=[np.number]).fillna(0)
    X_te_num = X_te.select_dtypes(include=[np.number]).fillna(0)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr_num, y_tr)
    r2 = r2_score(y_te, rf.predict(X_te_num))
    results_split.append({'train_pct': ts, 'test_pct': 1-ts,
                          'n_train': len(X_tr), 'n_test': len(X_te), 'R2': r2})

df_splits = pd.DataFrame(results_split)
print(df_splits)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_splits['train_pct']*100, df_splits['R2'], 'bo-', lw=2)
ax.set_xlabel('% datos de entrenamiento')
ax.set_ylabel('R² en test')
ax.set_title('Impacto del % de train en R² (RF, 100 árboles)')
ax.grid(True, alpha=0.3)
plt.savefig('reports/figures/C_train_size_impact.png', dpi=150)
```

#### C.2.3 Learning curves (detección de over/underfitting)

```python
from sklearn.model_selection import learning_curve

rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
train_sizes, train_scores, val_scores = learning_curve(
    rf_base, X_train_enc.fillna(0), y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='r2', n_jobs=-1
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), 'b-o', label='R² train')
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.1, color='blue')
ax.plot(train_sizes, val_scores.mean(axis=1), 'r-o', label='R² validación (CV)')
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.1, color='red')
ax.set_xlabel('Tamaño conjunto de entrenamiento')
ax.set_ylabel('R²')
ax.set_title('Learning Curves — Random Forest (Regresión Popularidad)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('reports/figures/C_learning_curves_rf.png', dpi=150)
# Interpretación: si brecha grande train vs val → overfitting
# Si ambas curvas bajas → underfitting
# Si convergen → bien (más datos no ayudaría más)
```

#### C.2.4 RandomizedSearchCV para RF y XGBoost

```python
from scipy.stats import randint, uniform

# --- Random Forest ---
param_dist_rf = {
    'n_estimators': randint(200, 600),
    'max_depth': [None, 10, 15, 20, 25, 30],
    'min_samples_leaf': randint(1, 10),
    'min_samples_split': randint(2, 20),
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'bootstrap': [True, False]
}

X_train_clean = X_train_enc.fillna(0)

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist_rf,
    n_iter=30,  # 30 combinaciones aleatorias
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    refit=True
)
rf_search.fit(X_train_clean, y_train)

print("Mejores hiperparámetros RF:")
print(rf_search.best_params_)
print(f"Mejor R² en CV: {rf_search.best_score_:.4f}")

# Resultados de la búsqueda
cv_results_rf = pd.DataFrame(rf_search.cv_results_).sort_values('rank_test_score')
print(cv_results_rf[['params','mean_test_score','std_test_score','rank_test_score']].head(10))

# --- XGBoost ---
param_dist_xgb = {
    'n_estimators': randint(200, 600),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 0.3),
    'min_child_weight': randint(1, 10),
    'reg_alpha': uniform(0, 0.5),
    'reg_lambda': uniform(0.5, 2.0)
}

xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
    param_distributions=param_dist_xgb,
    n_iter=30,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    refit=True
)
xgb_search.fit(X_train_clean, y_train)

print("Mejores hiperparámetros XGBoost:")
print(xgb_search.best_params_)
print(f"Mejor R² en CV: {xgb_search.best_score_:.4f}")
```

#### C.2.5 Evaluación final en test + análisis de residuos

```python
X_test_clean = X_test_enc.fillna(0)

best_rf = rf_search.best_estimator_
best_xgb = xgb_search.best_estimator_

models = {'Random Forest': best_rf, 'XGBoost': best_xgb}
metrics_reg = {}

for name, model in models.items():
    y_pred = model.predict(X_test_clean)
    metrics_reg[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    }

metrics_df = pd.DataFrame(metrics_reg).T
print(metrics_df)

# Análisis de residuos para el mejor modelo
best_name = metrics_df['R²'].idxmax()
best_model = models[best_name]
y_pred_best = best_model.predict(X_test_clean)
residuals = y_test.values - y_pred_best

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Real vs Predicho
axes[0,0].scatter(y_test, y_pred_best, alpha=0.1, s=3)
axes[0,0].plot([0,100],[0,100],'r--')
axes[0,0].set_xlabel('Popularity real'); axes[0,0].set_ylabel('Popularity predicha')
axes[0,0].set_title(f'{best_name}: Real vs Predicho')

# Distribución de residuos
axes[0,1].hist(residuals, bins=50, edgecolor='black', color='steelblue')
axes[0,1].axvline(0, color='red', ls='--')
axes[0,1].set_title('Distribución de residuos')
axes[0,1].set_xlabel('Residuo (real - predicho)')

# Residuos vs predicho (homocedasticidad)
axes[1,0].scatter(y_pred_best, residuals, alpha=0.1, s=3)
axes[1,0].axhline(0, color='red', ls='--')
axes[1,0].set_xlabel('Popularity predicha'); axes[1,0].set_ylabel('Residuo')
axes[1,0].set_title('Residuos vs Predicho')

# Residuos por macro-género (¿el modelo falla más en algún género?)
df_test_res = df.iloc[y_test.index].copy()
df_test_res['residual'] = residuals
res_by_genre = df_test_res.groupby('macro_genre')['residual'].agg(['mean','std','count'])
axes[1,1].bar(res_by_genre.index, res_by_genre['mean'],
              yerr=res_by_genre['std']/np.sqrt(res_by_genre['count']),
              capsize=4, color='steelblue', alpha=0.8)
axes[1,1].axhline(0, color='red', ls='--')
axes[1,1].set_xticklabels(res_by_genre.index, rotation=45, ha='right')
axes[1,1].set_title('Residuo medio por macro-género')

plt.tight_layout()
plt.savefig('reports/figures/C_residual_analysis.png', dpi=150)
```

#### C.2.6 Importancia de variables — impurity + SHAP

```python
import shap

# Importancia por impurity (RF)
feat_imp_rf = pd.Series(best_rf.feature_importances_,
                         index=X_train_clean.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
feat_imp_rf.head(15).plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].invert_yaxis()
axes[0].set_title('RF: Importancia por impurity (top 15)')

# SHAP values (mucho más interpretable)
explainer = shap.TreeExplainer(best_rf)
X_shap = X_test_clean.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_shap)

shap.summary_plot(shap_values, X_shap, plot_type='bar',
                  max_display=15, show=False)
plt.title('RF: Importancia SHAP (media |SHAP value|)')
plt.savefig('reports/figures/C_shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP beeswarm plot (muestra dirección del efecto)
shap.summary_plot(shap_values, X_shap, max_display=12, show=False)
plt.savefig('reports/figures/C_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
```

### C.3 Clasificación de macro-género — versión rigurosa

#### C.3.1 Split estratificado por macro-género

```python
X_clf = df[AUDIO_FEATURES + ['log_instrumentalness','log_acousticness']].copy()
y_clf = df[TARGET_CLF].copy()

# Convertir explicit a int
X_clf['explicit'] = X_clf['explicit'].astype(int)

le = LabelEncoder()
y_clf_enc = le.fit_transform(y_clf)

# Estratificado por macro_genre (garantiza proporciones en train y test)
X_tr_clf, X_te_clf, y_tr_clf, y_te_clf = train_test_split(
    X_clf, y_clf_enc, test_size=0.20, random_state=42, stratify=y_clf_enc
)

print("Distribución de clases en train:")
unique, counts = np.unique(y_tr_clf, return_counts=True)
for u, c in zip(le.inverse_transform(unique), counts):
    print(f"  {u}: {c} ({c/len(y_tr_clf)*100:.1f}%)")
```

#### C.3.2 RandomizedSearchCV con validación cruzada estratificada

```python
param_dist_rf_clf = {
    'n_estimators': randint(200, 500),
    'max_depth': [None, 10, 15, 20],
    'min_samples_leaf': randint(1, 8),
    'max_features': ['sqrt', 'log2', 0.3],
    'class_weight': ['balanced', None]
}

rf_clf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist_rf_clf,
    n_iter=25,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2,
    random_state=42
)
rf_clf_search.fit(X_tr_clf.fillna(0), y_tr_clf)
print("Mejores params RF clasificación:", rf_clf_search.best_params_)

param_dist_xgb_clf = {
    'n_estimators': randint(200, 500),
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.02, 0.15),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 0.2),
    'min_child_weight': randint(1, 6)
}

xgb_clf_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, tree_method='hist',
                  use_label_encoder=False, eval_metric='mlogloss'),
    param_distributions=param_dist_xgb_clf,
    n_iter=25,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2,
    random_state=42
)
xgb_clf_search.fit(X_tr_clf.fillna(0), y_tr_clf)
print("Mejores params XGBoost clasificación:", xgb_clf_search.best_params_)
```

#### C.3.3 Evaluación, matriz de confusión y reporte

```python
X_te_clean = X_te_clf.fillna(0)

for name, search in [('Random Forest', rf_clf_search), ('XGBoost', xgb_clf_search)]:
    model = search.best_estimator_
    y_pred = model.predict(X_te_clean)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_te_clf, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_te_clf, y_pred, average='macro'):.4f}")
    print(f"F1-weighted: {f1_score(y_te_clf, y_pred, average='weighted'):.4f}")
    print(classification_report(y_te_clf, y_pred, target_names=le.classes_))

    # Matriz de confusión normalizada
    cm = confusion_matrix(y_te_clf, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
    ax.set_title(f'{name}: Matriz de Confusión Normalizada')
    plt.tight_layout()
    plt.savefig(f'reports/figures/C_confusion_{name.replace(" ","_")}.png', dpi=150)
    plt.close()
```

#### C.3.4 Importancia SHAP para clasificación

```python
best_rf_clf = rf_clf_search.best_estimator_
explainer_clf = shap.TreeExplainer(best_rf_clf)
X_shap_clf = X_te_clean.sample(500, random_state=42)
shap_values_clf = explainer_clf.shap_values(X_shap_clf)

# Para multiclase, shap_values es una lista de arrays (uno por clase)
# Tomar la importancia media absoluta entre todas las clases
mean_shap = np.abs(np.array(shap_values_clf)).mean(axis=0).mean(axis=0)
shap_imp = pd.Series(mean_shap, index=X_te_clean.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
shap_imp.head(12).plot(kind='barh', ax=ax, color='coral')
ax.invert_yaxis()
ax.set_title('RF Clasificación: Importancia SHAP media (todas las clases)')
plt.tight_layout()
plt.savefig('reports/figures/C_shap_classification.png', dpi=150)
```

Guarda tabla comparativa final en `results/C_modelos_predictivos.md` con:
- Tabla regresión: RF vs XGBoost → RMSE, MAE, R² + mejores hiperparámetros
- Tabla clasificación: RF vs XGBoost → Accuracy, F1-macro, F1-weighted + mejores hiperparámetros
- Top 5 features más importantes en cada tarea (SHAP)
- Análisis de residuos: ¿en qué géneros/rangos de popularidad falla más el modelo?

---

## FASE D — CLUSTERING Y SISTEMA DE RECOMENDACIÓN (versión profunda)

### D.1 Preprocesado para clustering

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import umap  # pip install umap-learn si no está instalado

df = pd.read_csv('data/processed/tracks_model.csv')

# Features para clustering: solo audio features continuas
# (no incluir genre porque queremos ver si el clustering "descubre" géneros)
CLUSTER_FEATURES = ['danceability','energy','loudness','speechiness',
                    'acousticness','instrumentalness','liveness','valence','tempo']

X_cluster = df[CLUSTER_FEATURES].fillna(df[CLUSTER_FEATURES].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Guardar el scaler para usarlo en el recomendador
import joblib
joblib.dump(scaler, 'models/scaler_cluster.pkl')
```

### D.2 Determinación del número óptimo de clusters

```python
# Probar k de 2 a 15
k_range = range(2, 16)
inertias, silhouettes, davies = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=5000))
    davies.append(davies_bouldin_score(X_scaled, labels))
    print(f"k={k}: inertia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.4f}, DB={davies[-1]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(list(k_range), inertias, 'bo-')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inercia (WCSS)')
axes[0].set_title('Método del Codo'); axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), silhouettes, 'go-')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por k'); axes[1].grid(True, alpha=0.3)

axes[2].plot(list(k_range), davies, 'ro-')
axes[2].set_xlabel('k'); axes[2].set_ylabel('Davies-Bouldin Score')
axes[2].set_title('Davies-Bouldin Score por k (menor = mejor)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Selección del número óptimo de clusters (KMeans)', y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/D_cluster_selection.png', dpi=150)
```

### D.3 Silhouette analysis por muestra (para el k óptimo)

```python
# Con el k que resulte óptimo según el codo/silhouette (probablemente k=2 o k=3)
k_opt = 3  # Ajusta según los resultados de D.2

km_opt = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
labels_km = km_opt.fit_predict(X_scaled)

# Silhouette por muestra individual (más informativo que el score agregado)
from sklearn.metrics import silhouette_samples
silhouette_vals = silhouette_samples(X_scaled, labels_km)

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
for i in range(k_opt):
    ith_silhouette = np.sort(silhouette_vals[labels_km == i])
    size_i = len(ith_silhouette)
    y_upper = y_lower + size_i
    color = plt.cm.Set2(i / k_opt)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_silhouette,
                     facecolor=color, edgecolor=color, alpha=0.7, label=f'Cluster {i}')
    ax.text(-0.05, y_lower + 0.5 * size_i, f'C{i}', fontsize=9)
    y_lower = y_upper + 10

avg_sil = silhouette_score(X_scaled, labels_km, sample_size=5000)
ax.axvline(avg_sil, color='red', ls='--', label=f'Silhouette medio={avg_sil:.3f}')
ax.set_xlabel('Silhouette coefficient'); ax.set_ylabel('Cluster')
ax.set_title(f'Silhouette Analysis (k={k_opt})')
ax.legend()
plt.savefig('reports/figures/D_silhouette_analysis.png', dpi=150)
```

### D.4 Comparativa de algoritmos de clustering

```python
# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=k_opt, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)

# DBSCAN — ajustar eps con el truco del k-nearest neighbor distance
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
dist_5th = np.sort(distances[:, 4])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dist_5th)
ax.set_xlabel('Punto (ordenado por distancia)')
ax.set_ylabel('Distancia al 5º vecino más cercano')
ax.set_title('K-distance graph para selección de eps en DBSCAN')
plt.savefig('reports/figures/D_dbscan_eps_selection.png', dpi=150)
# Busca el "codo" en esta gráfica para elegir eps

eps_val = 1.5  # Ajusta según el codo observado
db = DBSCAN(eps=eps_val, min_samples=10, n_jobs=-1)
labels_db = db.fit_predict(X_scaled)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = (labels_db == -1).sum()
print(f"DBSCAN: {n_clusters_db} clusters, {n_noise} ruido ({n_noise/len(labels_db)*100:.1f}%)")

# Tabla comparativa
# (Solo calcular silhouette para clusters válidos, no para DBSCAN con mucho ruido)
comparison_clustering = {
    'KMeans': {
        'n_clusters': k_opt,
        'silhouette': silhouette_score(X_scaled, labels_km, sample_size=5000),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels_km),
        'noise_pct': 0
    },
    'Agglomerative': {
        'n_clusters': k_opt,
        'silhouette': silhouette_score(X_scaled, labels_agg, sample_size=5000),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels_agg),
        'noise_pct': 0
    },
    'DBSCAN': {
        'n_clusters': n_clusters_db,
        'silhouette': silhouette_score(X_scaled[labels_db != -1],
                                        labels_db[labels_db != -1],
                                        sample_size=min(5000, (labels_db!=-1).sum()))
                      if n_clusters_db > 1 else None,
        'davies_bouldin': None,
        'noise_pct': n_noise/len(labels_db)*100
    }
}
print(pd.DataFrame(comparison_clustering).T)
```

### D.5 Visualización: PCA + UMAP

```python
# Reducción a 2D con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Varianza explicada PCA: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# UMAP (más potente para estructura local)
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30,
                     min_dist=0.1, metric='euclidean')
X_umap = reducer.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# PCA coloreado por cluster KMeans
scatter = axes[0,0].scatter(X_pca[:,0], X_pca[:,1], c=labels_km,
                             cmap='Set2', s=1, alpha=0.3)
axes[0,0].set_title(f'PCA 2D — KMeans (k={k_opt})')
plt.colorbar(scatter, ax=axes[0,0])

# PCA coloreado por macro-género
colors_genre = pd.Categorical(df['macro_genre']).codes
scatter2 = axes[0,1].scatter(X_pca[:,0], X_pca[:,1], c=colors_genre,
                              cmap='tab20', s=1, alpha=0.2)
axes[0,1].set_title('PCA 2D — por Macro-género')

# UMAP coloreado por cluster
scatter3 = axes[1,0].scatter(X_umap[:,0], X_umap[:,1], c=labels_km,
                              cmap='Set2', s=1, alpha=0.3)
axes[1,0].set_title(f'UMAP 2D — KMeans (k={k_opt})')
plt.colorbar(scatter3, ax=axes[1,0])

# UMAP coloreado por macro-género
scatter4 = axes[1,1].scatter(X_umap[:,0], X_umap[:,1], c=colors_genre,
                              cmap='tab20', s=1, alpha=0.2)
axes[1,1].set_title('UMAP 2D — por Macro-género')

plt.tight_layout()
plt.savefig('reports/figures/D_pca_umap_clusters.png', dpi=120)
```

### D.6 Interpretación de los clusters

```python
# Añadir etiquetas de cluster al dataset
df['cluster_kmeans'] = labels_km

# Perfil de cada cluster (media de audio features)
cluster_profile = df.groupby('cluster_kmeans')[CLUSTER_FEATURES + ['popularity']].mean()
print("Perfil de cada cluster:")
print(cluster_profile.round(3))

# Géneros dominantes en cada cluster
for c in range(k_opt):
    print(f"\nCluster {c} — géneros más frecuentes:")
    print(df[df['cluster_kmeans']==c]['macro_genre'].value_counts(normalize=True).head(5))

# Radar por cluster
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
features_radar = ['danceability','energy','valence','acousticness',
                  'instrumentalness','speechiness']
N = len(features_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

# Normalizar el perfil para el radar
cp_norm = (cluster_profile[features_radar] - cluster_profile[features_radar].min()) \
          / (cluster_profile[features_radar].max() - cluster_profile[features_radar].min())

colors_c = plt.cm.Set1(np.linspace(0, 0.8, k_opt))
for c, color in zip(range(k_opt), colors_c):
    vals = cp_norm.loc[c].tolist() + [cp_norm.loc[c].tolist()[0]]
    ax.plot(angles, vals, linewidth=2, color=color, label=f'Cluster {c}')
    ax.fill(angles, vals, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features_radar, size=10)
ax.set_title('Perfil sonoro por cluster (normalizado)', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.savefig('reports/figures/D_cluster_radar.png', dpi=150, bbox_inches='tight')
```

### D.7 Sistema de recomendación — tres enfoques + evaluación rigurosa

```python
from sklearn.neighbors import NearestNeighbors
import joblib

# Guardar datos escalados y clusters para la app
df['X_scaled_0'] = X_scaled[:,0]  # solo para guardar — mejor guardar X_scaled aparte
np.save('models/X_scaled.npy', X_scaled)
df.to_csv('data/processed/tracks_with_clusters.csv', index=False)
joblib.dump(km_opt, 'models/kmeans_model.pkl')

# === RECOMENDADOR 1: KNN global (sin filtro de género) ===
knn_global = NearestNeighbors(n_neighbors=11, metric='cosine', n_jobs=-1)
knn_global.fit(X_scaled)

def recommend_knn_global(track_idx, n=10):
    distances, indices = knn_global.kneighbors(X_scaled[track_idx:track_idx+1])
    # Excluir la propia canción (índice 0)
    return df.iloc[indices[0][1:n+1]][['track_name','artists','macro_genre','popularity']]

# === RECOMENDADOR 2: Basado en cluster ===
def recommend_cluster(track_idx, n=10):
    cluster = df.iloc[track_idx]['cluster_kmeans']
    same_cluster = df[df['cluster_kmeans'] == cluster].drop(index=df.index[track_idx])
    return same_cluster.nlargest(n, 'popularity')[['track_name','artists','macro_genre','popularity']]

# === RECOMENDADOR 3: Híbrido (género + KNN dentro del género) ===
# Para cada género, entrenar un KNN propio
knn_by_genre = {}
genre_indices = {}
for genre in df['macro_genre'].unique():
    mask = df['macro_genre'] == genre
    idx = np.where(mask)[0]
    if len(idx) < 11:
        continue
    genre_indices[genre] = idx
    knn_g = NearestNeighbors(n_neighbors=min(11, len(idx)), metric='cosine', n_jobs=-1)
    knn_g.fit(X_scaled[idx])
    knn_by_genre[genre] = knn_g

def recommend_hybrid(track_idx, n=10):
    genre = df.iloc[track_idx]['macro_genre']
    if genre not in knn_by_genre:
        return recommend_knn_global(track_idx, n)
    local_idx = genre_indices[genre]
    local_pos = np.where(local_idx == track_idx)[0]
    if len(local_pos) == 0:
        return recommend_knn_global(track_idx, n)
    distances, indices = knn_by_genre[genre].kneighbors(X_scaled[track_idx:track_idx+1])
    global_indices = local_idx[indices[0][1:n+1]]
    return df.iloc[global_indices][['track_name','artists','macro_genre','popularity']]

# Guardar modelos
joblib.dump(knn_global, 'models/knn_global.pkl')
joblib.dump(knn_by_genre, 'models/knn_by_genre.pkl')
joblib.dump(genre_indices, 'models/genre_indices.pkl')
```

### D.8 Evaluación de los recomendadores (métricas proxy)

```python
import random
random.seed(42)

def evaluate_recommender(recommend_fn, n_seeds=200, n_recs=10):
    """
    Métricas proxy (sin ground truth real):
    - genre_coherence: % recomendaciones con mismo macro-género que la semilla
    - artist_diversity: % artistas distintos en las recomendaciones
    - popularity_mean: popularidad media de las recomendaciones
    - intra_list_diversity: distancia media entre pares de recomendaciones
      (mayor = más diversidad sonora)
    """
    seeds = random.sample(range(len(df)), n_seeds)
    results = []
    for seed in seeds:
        seed_genre = df.iloc[seed]['macro_genre']
        recs = recommend_fn(seed, n_recs)
        if len(recs) == 0:
            continue
        genre_coh = (recs['macro_genre'] == seed_genre).mean()
        artist_div = recs['artists'].nunique() / len(recs)
        pop_mean = recs['popularity'].mean()
        # Diversidad intra-lista (distancias coseno entre recomendaciones)
        rec_indices = recs.index.tolist()
        if len(rec_indices) >= 2:
            X_recs = X_scaled[rec_indices]
            from sklearn.metrics.pairwise import cosine_distances
            dist_matrix = cosine_distances(X_recs)
            il_div = dist_matrix[np.triu_indices(len(rec_indices), k=1)].mean()
        else:
            il_div = 0
        results.append({'genre_coherence': genre_coh, 'artist_diversity': artist_div,
                        'popularity_mean': pop_mean, 'intra_list_diversity': il_div})
    return pd.DataFrame(results).mean()

print("\n=== Evaluación de recomendadores (200 semillas) ===")
print("\nKNN Global:")
print(evaluate_recommender(recommend_knn_global))
print("\nBasado en Cluster:")
print(evaluate_recommender(recommend_cluster))
print("\nHíbrido (género + KNN):")
print(evaluate_recommender(recommend_hybrid))

# Gráfico comparativo de métricas
eval_results = {
    'KNN Global': evaluate_recommender(recommend_knn_global),
    'Cluster': evaluate_recommender(recommend_cluster),
    'Híbrido': evaluate_recommender(recommend_hybrid)
}
eval_df = pd.DataFrame(eval_results).T

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, metric in zip(axes, ['genre_coherence','artist_diversity',
                               'popularity_mean','intra_list_diversity']):
    eval_df[metric].plot(kind='bar', ax=ax, color=['steelblue','coral','green'],
                          edgecolor='black', rot=30)
    ax.set_title(metric.replace('_',' ').title())
    ax.set_ylim(bottom=0)
plt.suptitle('Comparativa de recomendadores (200 canciones semilla, top-10 recomendaciones)')
plt.tight_layout()
plt.savefig('reports/figures/D_recommender_comparison.png', dpi=150)
```

### D.9 Demo del sistema de recomendación

```python
# Prueba con 5 canciones semilla conocidas y documenta los resultados
test_songs = [
    ('Feid', 'Ferxxo'),  # reggaeton colombiano
    ('Bad Bunny', None),  # urbano latino
    ('The Beatles', None),  # rock clásico
    ('Mozart', None),  # clásica
    ('Eminem', None)  # hip-hop
]

for artist, track in test_songs:
    mask = df['artists'].str.contains(artist, case=False, na=False)
    if track:
        mask &= df['track_name'].str.contains(track, case=False, na=False)
    if mask.sum() == 0:
        print(f"No encontrado: {artist} - {track}")
        continue
    seed_idx = df[mask].index[0]
    seed_info = df.iloc[seed_idx]
    print(f"\n{'='*60}")
    print(f"SEMILLA: {seed_info['track_name']} — {seed_info['artists']}")
    print(f"Género: {seed_info['macro_genre']}, Popularidad: {seed_info['popularity']}")
    print("\nTop-5 recomendaciones (Híbrido):")
    print(recommend_hybrid(seed_idx, n=5).to_string(index=False))
```

Guarda los resultados en `results/D_clustering_recomendacion.md` con:
- Tabla comparativa de algoritmos (KMeans, Agglomerative, DBSCAN) con métricas
- Perfiles de los clusters (qué géneros dominan cada uno, qué "suena" cada cluster)
- Tabla comparativa de recomendadores con las 4 métricas proxy
- Demo con las 5 canciones semilla (resultados reales obtenidos)
- Reflexión: ¿por qué el clustering converge a k=2 o k=3? ¿Qué dice esto
  del espacio de audio features de Spotify?

---

## ENTREGABLES MÍNIMOS DE ESTA SESIÓN

Al terminar, asegúrate de que existen:

```
data/processed/tracks_model.csv           # dataset limpio + feature engineering
models/scaler_cluster.pkl
models/kmeans_model.pkl
models/knn_global.pkl
models/knn_by_genre.pkl
models/genre_indices.pkl
models/X_scaled.npy
reports/figures/                          # todos los gráficos generados
results/A_limpieza_exhaustiva.md
results/B_EDA_hallazgos.md
results/C_modelos_predictivos.md
results/D_clustering_recomendacion.md
```

Si falta tiempo, el orden de prioridad es:
**A (limpieza) → C.2 (regresión RF/XGB) → B (EDA) → D (clustering) → C.3 (clasificación)**

La limpieza es la base de todo; la regresión de popularidad es la tarea
principal de ML; el EDA ilustra los hallazgos; el clustering es la parte
más novedosa del trabajo.

---

## NOTAS ADICIONALES PARA LA MEMORIA

- Documenta explícitamente el efecto de la estratificación: muestra la
  distribución de popularity en train vs test antes y después de estratificar.
- Para el análisis SHAP: interpreta en la memoria qué significa que
  `instrumentalness` sea la variable con mayor SHAP negativo en regresión
  (canciones sin voz → menos populares en el snapshot).
- Para el clustering: el hallazgo de k=2 es un resultado científico
  interesante, no un fracaso. Significa que las audio features de Spotify
  capturan fundamentalmente una dimensión de "intensidad/electronización"
  más que matices de estilo. Esto es coherente con la literatura sobre
  audio analysis (Tzanetakis & Cook, 2002; Schedl et al., 2014).
- En la sección de limitaciones del clustering, menciona que un sistema de
  recomendación real (como el de Spotify) usa embeddings de alta dimensión
  entrenados con redes neuronales sobre espectrogramas (no features
  predefinidas como estas), lo que explica por qué captura mucho más matiz.
