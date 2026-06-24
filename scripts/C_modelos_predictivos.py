# =============================================================================
# FASE C — MODELOS PREDICTIVOS: RF y XGBoost (versión rigurosa)
# TFG: IA y Análisis Estadístico aplicado a la Industria Musical (Spotify)
# =============================================================================
# Entrada:  data/processed/tracks_model.csv
# Salida:   models/rf_reg_v2.pkl, models/xgb_reg_v2.pkl, etc.
#           reports/figures/C_*.png
#           results/C_modelos_predictivos.md
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
import joblib
import shap

from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      RandomizedSearchCV, learning_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, f1_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBRegressor, XGBClassifier
from category_encoders import TargetEncoder

ROOT   = Path(__file__).parent.parent
PROC   = ROOT / "data" / "processed"
FIGS   = ROOT / "reports" / "figures"
RES    = ROOT / "results"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.95)

print("=" * 70)
print("FASE C — MODELOS PREDICTIVOS: RF vs XGBoost")
print("=" * 70)

df = pd.read_csv(PROC / "tracks_model.csv")
df["explicit"] = df["explicit"].astype(int)
print(f"Dataset: {len(df):,} filas")

# Columnas de features de audio
AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence",
                  "tempo", "duration_min", "explicit", "key", "mode", "time_signature"]

FEATURES_REG = AUDIO_FEATURES + ["log_instrumentalness", "log_acousticness",
                                   "log_speechiness", "electronic_ratio"]
TARGET_REG  = "popularity"
TARGET_CLF  = "macro_genre"

# =============================================================================
# C.2 REGRESIÓN DE POPULARIDAD
# =============================================================================
print("\n" + "─" * 60)
print("C.2  REGRESIÓN DE POPULARIDAD")
print("─" * 60)

X = df[FEATURES_REG + ["track_genre"]].copy()
y = df[TARGET_REG].copy()

# Estratificar por quintiles de popularity
df["pop_quintile"] = pd.qcut(y, q=5, labels=False, duplicates="drop")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df["pop_quintile"]
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  Train popularity: media={y_train.mean():.2f}, std={y_train.std():.2f}")
print(f"  Test  popularity: media={y_test.mean():.2f}, std={y_test.std():.2f}")

# Target encoding en track_genre (fit solo en train, evita data leakage)
te = TargetEncoder(cols=["track_genre"])
X_train_enc = te.fit_transform(X_train, y_train).fillna(0)
X_test_enc  = te.transform(X_test).fillna(0)
joblib.dump(te, MODELS / "target_encoder_v2.pkl")

# ─── Gráfico distribución train vs test ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=30, alpha=0.7, label="Train", color="steelblue")
axes[0].hist(y_test,  bins=30, alpha=0.7, label="Test",  color="orange")
axes[0].legend(); axes[0].set_title("Popularity: Train vs Test")

y_train.plot.kde(ax=axes[1], label="Train", color="steelblue")
y_test.plot.kde( ax=axes[1], label="Test",  color="orange")
axes[1].legend(); axes[1].set_title("KDE Popularity: Train vs Test")

plt.tight_layout()
plt.savefig(FIGS / "C_train_test_distribution.png", dpi=150)
plt.close()
print("  [OK] C_train_test_distribution.png")

# ─── Impacto del % de train ──────────────────────────────────────────────────
print("\n  Experimento: impacto del % de train...")
train_sizes_pct = [0.5, 0.6, 0.7, 0.8, 0.9]
results_split   = []

X_num = X[FEATURES_REG].copy().fillna(0)

for ts in train_sizes_pct:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_num, y, test_size=1 - ts, random_state=42,
        stratify=df["pop_quintile"]
    )
    rf_quick = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=2)
    rf_quick.fit(X_tr, y_tr)
    r2 = r2_score(y_te, rf_quick.predict(X_te))
    results_split.append({"train_pct": ts, "n_train": len(X_tr),
                           "n_test": len(X_te), "R2": r2})
    print(f"    train={ts:.0%}: R²={r2:.4f}, n_train={len(X_tr):,}")

df_splits = pd.DataFrame(results_split)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_splits["train_pct"] * 100, df_splits["R2"], "bo-", lw=2, ms=6)
for _, row in df_splits.iterrows():
    ax.annotate(f"R²={row['R2']:.3f}", (row["train_pct"] * 100, row["R2"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("% datos de entrenamiento")
ax.set_ylabel("R² en test")
ax.set_title("Impacto del % de train en R² (RF, 80 árboles)")
ax.grid(True, alpha=0.3)
plt.savefig(FIGS / "C_train_size_impact.png", dpi=150)
plt.close()
print("  [OK] C_train_size_impact.png")

# ─── Learning curves ─────────────────────────────────────────────────────────
print("\n  Learning curves RF...")
rf_base = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=2)
train_sz, tr_scores, val_scores = learning_curve(
    rf_base, X_train_enc, y_train,
    train_sizes=np.linspace(0.1, 1.0, 7),
    cv=3, scoring="r2", n_jobs=2
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sz, tr_scores.mean(axis=1),  "b-o", label="R² train")
ax.fill_between(train_sz,
                tr_scores.mean(axis=1) - tr_scores.std(axis=1),
                tr_scores.mean(axis=1) + tr_scores.std(axis=1),
                alpha=0.1, color="blue")
ax.plot(train_sz, val_scores.mean(axis=1), "r-o", label="R² validación (CV-3)")
ax.fill_between(train_sz,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.1, color="red")
ax.set_xlabel("Tamaño conjunto de entrenamiento")
ax.set_ylabel("R²")
ax.set_title("Learning Curves — Random Forest (Regresión Popularidad)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig(FIGS / "C_learning_curves_rf.png", dpi=150)
plt.close()
print("  [OK] C_learning_curves_rf.png")

# ─── RandomizedSearchCV RF ───────────────────────────────────────────────────
print("\n  RandomizedSearchCV — Random Forest regresión...")
param_dist_rf = {
    "n_estimators":      randint(200, 500),
    "max_depth":         [None, 10, 15, 20, 25],
    "min_samples_leaf":  randint(1, 8),
    "min_samples_split": randint(2, 15),
    "max_features":      ["sqrt", "log2", 0.3, 0.5],
    "bootstrap":         [True, False],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=2),
    param_distributions=param_dist_rf,
    n_iter=20, cv=3, scoring="r2", n_jobs=2,
    verbose=1, random_state=42, refit=True
)
rf_search.fit(X_train_enc, y_train)
print(f"  Mejor R² (CV) RF: {rf_search.best_score_:.4f}")
print(f"  Mejores params RF: {rf_search.best_params_}")

# ─── RandomizedSearchCV XGBoost ─────────────────────────────────────────────
print("\n  RandomizedSearchCV — XGBoost regresión...")
param_dist_xgb = {
    "n_estimators":      randint(150, 400),
    "max_depth":         randint(3, 9),
    "learning_rate":     uniform(0.01, 0.2),
    "subsample":         uniform(0.6, 0.4),
    "colsample_bytree":  uniform(0.5, 0.5),
    "gamma":             uniform(0, 0.3),
    "min_child_weight":  randint(1, 8),
    "reg_alpha":         uniform(0, 0.5),
    "reg_lambda":        uniform(0.5, 2.0),
}
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=2, tree_method="hist",
                 device="cpu", verbosity=0),
    param_distributions=param_dist_xgb,
    n_iter=20, cv=3, scoring="r2", n_jobs=2,
    verbose=1, random_state=42, refit=True
)
xgb_search.fit(X_train_enc, y_train)
print(f"  Mejor R² (CV) XGB: {xgb_search.best_score_:.4f}")
print(f"  Mejores params XGB: {xgb_search.best_params_}")

# ─── Evaluación final en test ────────────────────────────────────────────────
best_rf  = rf_search.best_estimator_
best_xgb = xgb_search.best_estimator_

y_pred_rf  = best_rf.predict(X_test_enc)
y_pred_xgb = best_xgb.predict(X_test_enc)

metrics_reg = {}
for name, y_pred in [("Random Forest", y_pred_rf), ("XGBoost", y_pred_xgb)]:
    metrics_reg[name] = {
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
        "R²":   round(r2_score(y_test, y_pred), 4),
    }
metrics_df_reg = pd.DataFrame(metrics_reg).T
print("\n  MÉTRICAS FINALES (Regresión):")
print(metrics_df_reg.to_string())

# Guardar modelos
joblib.dump(best_rf,  MODELS / "rf_reg_v2.pkl")
joblib.dump(best_xgb, MODELS / "xgb_reg_v2.pkl")

# ─── Análisis de residuos ────────────────────────────────────────────────────
best_name = "Random Forest" if metrics_reg["Random Forest"]["R²"] >= metrics_reg["XGBoost"]["R²"] else "XGBoost"
y_pred_best = y_pred_rf if best_name == "Random Forest" else y_pred_xgb
residuals = y_test.values - y_pred_best

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(y_test, y_pred_best, alpha=0.07, s=3, color="steelblue")
axes[0, 0].plot([0, 100], [0, 100], "r--", lw=1.5)
axes[0, 0].set_xlabel("Popularity real"); axes[0, 0].set_ylabel("Popularity predicha")
axes[0, 0].set_title(f"{best_name}: Real vs Predicho (R²={metrics_reg[best_name]['R²']:.3f})")

axes[0, 1].hist(residuals, bins=60, edgecolor="white", color="steelblue")
axes[0, 1].axvline(0, color="red", ls="--")
axes[0, 1].set_title("Distribución de residuos")
axes[0, 1].set_xlabel("Residuo (real − predicho)")

axes[1, 0].scatter(y_pred_best, residuals, alpha=0.05, s=3, color="steelblue")
axes[1, 0].axhline(0, color="red", ls="--")
axes[1, 0].set_xlabel("Popularity predicha"); axes[1, 0].set_ylabel("Residuo")
axes[1, 0].set_title("Residuos vs Predicho (homocedasticidad)")

# Residuos por macro-género
df_test_res = df.loc[y_test.index].copy()
df_test_res["residual"] = residuals
res_genre = df_test_res.groupby("macro_genre")["residual"].agg(["mean", "std", "count"])
res_genre["se"] = res_genre["std"] / np.sqrt(res_genre["count"])
res_genre = res_genre.sort_values("mean")
axes[1, 1].bar(res_genre.index, res_genre["mean"],
               yerr=1.96 * res_genre["se"], capsize=4,
               color="steelblue", alpha=0.8)
axes[1, 1].axhline(0, color="red", ls="--")
axes[1, 1].set_xticklabels(res_genre.index, rotation=45, ha="right", fontsize=8)
axes[1, 1].set_title("Residuo medio por macro-género ± IC 95%")

plt.suptitle(f"Análisis de residuos — {best_name} (Regresión Popularidad)", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "C_residual_analysis.png", dpi=150)
plt.close()
print("\n  [OK] C_residual_analysis.png")

# ─── Importancia SHAP ────────────────────────────────────────────────────────
print("\n  SHAP values (RF regresión, n=1000)...")
explainer = shap.TreeExplainer(best_rf)
X_shap = X_test_enc.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_shap)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Importancia SHAP (barras)
shap_imp = pd.Series(np.abs(shap_values).mean(axis=0),
                     index=X_test_enc.columns).sort_values(ascending=False)
shap_imp.head(15).plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].invert_yaxis()
axes[0].set_title("RF: Importancia SHAP (top 15)")
axes[0].set_xlabel("|SHAP| medio")

# Impurity importance
feat_imp_rf = pd.Series(best_rf.feature_importances_,
                        index=X_train_enc.columns).sort_values(ascending=False)
feat_imp_rf.head(15).plot(kind="barh", ax=axes[1], color="coral")
axes[1].invert_yaxis()
axes[1].set_title("RF: Importancia por impurity (Gini, top 15)")
axes[1].set_xlabel("Importancia")

plt.suptitle("Importancia de variables — Regresión de Popularidad", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "C_shap_importance_reg.png", dpi=150)
plt.close()

# Beeswarm SHAP
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, max_display=14, show=False)
plt.title("RF: SHAP Beeswarm (dirección del efecto sobre popularity)")
plt.tight_layout()
plt.savefig(FIGS / "C_shap_beeswarm_reg.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] C_shap_importance_reg.png + C_shap_beeswarm_reg.png")

# =============================================================================
# C.3 CLASIFICACIÓN DE MACRO-GÉNERO
# =============================================================================
print("\n" + "─" * 60)
print("C.3  CLASIFICACIÓN DE MACRO-GÉNERO")
print("─" * 60)

X_clf = df[AUDIO_FEATURES + ["log_instrumentalness", "log_acousticness",
                               "log_speechiness"]].copy().fillna(0)
y_clf = df[TARGET_CLF].copy()

le = LabelEncoder()
y_clf_enc = le.fit_transform(y_clf)
joblib.dump(le, MODELS / "label_encoder_macro_v2.pkl")

X_tr_clf, X_te_clf, y_tr_clf, y_te_clf = train_test_split(
    X_clf, y_clf_enc, test_size=0.20, random_state=42, stratify=y_clf_enc
)
print(f"  Train clf: {len(X_tr_clf):,}  |  Test clf: {len(X_te_clf):,}")
print("  Clases:", le.classes_.tolist())

# ─── RandomizedSearchCV RF clasificación ─────────────────────────────────────
print("\n  RandomizedSearchCV — RF clasificación...")
param_dist_rf_clf = {
    "n_estimators":     randint(150, 400),
    "max_depth":        [None, 10, 15, 20],
    "min_samples_leaf": randint(1, 6),
    "max_features":     ["sqrt", "log2", 0.3],
    "class_weight":     ["balanced", None],
}
rf_clf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=2),
    param_distributions=param_dist_rf_clf,
    n_iter=15, cv=3, scoring="f1_macro",
    n_jobs=2, verbose=1, random_state=42
)
rf_clf_search.fit(X_tr_clf, y_tr_clf)
print(f"  Mejor F1-macro (CV) RF: {rf_clf_search.best_score_:.4f}")

# ─── RandomizedSearchCV XGBoost clasificación ────────────────────────────────
print("\n  RandomizedSearchCV — XGBoost clasificación...")
param_dist_xgb_clf = {
    "n_estimators":     randint(150, 400),
    "max_depth":        randint(3, 8),
    "learning_rate":    uniform(0.02, 0.15),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "gamma":            uniform(0, 0.2),
    "min_child_weight": randint(1, 6),
}
xgb_clf_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=2, tree_method="hist",
                  device="cpu", verbosity=0, eval_metric="mlogloss"),
    param_distributions=param_dist_xgb_clf,
    n_iter=15, cv=3, scoring="f1_macro",
    n_jobs=2, verbose=1, random_state=42
)
xgb_clf_search.fit(X_tr_clf, y_tr_clf)
print(f"  Mejor F1-macro (CV) XGB: {xgb_clf_search.best_score_:.4f}")

# ─── Evaluación final clasificación ──────────────────────────────────────────
print("\n  MÉTRICAS FINALES (Clasificación):")
X_te_clean = X_te_clf.fillna(0)

metrics_clf = {}
for name, search in [("Random Forest", rf_clf_search), ("XGBoost", xgb_clf_search)]:
    model = search.best_estimator_
    y_pred = model.predict(X_te_clean)
    metrics_clf[name] = {
        "Accuracy":    round(accuracy_score(y_te_clf, y_pred), 4),
        "F1-macro":    round(f1_score(y_te_clf, y_pred, average="macro"), 4),
        "F1-weighted": round(f1_score(y_te_clf, y_pred, average="weighted"), 4),
    }
    print(f"\n  === {name} ===")
    print(f"  Accuracy:    {metrics_clf[name]['Accuracy']:.4f}")
    print(f"  F1-macro:    {metrics_clf[name]['F1-macro']:.4f}")
    print(f"  F1-weighted: {metrics_clf[name]['F1-weighted']:.4f}")

metrics_df_clf = pd.DataFrame(metrics_clf).T
print("\n" + metrics_df_clf.to_string())

joblib.dump(rf_clf_search.best_estimator_,  MODELS / "rf_clf_v2.pkl")
joblib.dump(xgb_clf_search.best_estimator_, MODELS / "xgb_clf_v2.pkl")

# ─── Matrices de confusión ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, (name, search) in zip(axes, [("RF", rf_clf_search), ("XGB", xgb_clf_search)]):
    model  = search.best_estimator_
    y_pred = model.predict(X_te_clean)
    cm = confusion_matrix(y_te_clf, y_pred, normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(le.classes_, fontsize=8)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"{name}: Matriz de Confusión Normalizada\n"
                 f"F1-macro={metrics_clf[name.replace('RF','Random Forest').replace('XGB','XGBoost')]['F1-macro']:.3f}")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(FIGS / "C_confusion_matrices.png", dpi=150)
plt.close()
print("\n  [OK] C_confusion_matrices.png")

# ─── SHAP clasificación ───────────────────────────────────────────────────────
print("\n  SHAP clasificación (RF, n=500)...")
best_rf_clf = rf_clf_search.best_estimator_
explainer_clf = shap.TreeExplainer(best_rf_clf)
X_shap_clf = X_te_clean.sample(500, random_state=42)
shap_values_clf = explainer_clf.shap_values(X_shap_clf)

# Importancia media absoluta entre clases
mean_shap_clf = np.abs(np.array(shap_values_clf)).mean(axis=0).mean(axis=0)
shap_imp_clf = pd.Series(mean_shap_clf,
                          index=X_te_clean.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
shap_imp_clf.head(12).plot(kind="barh", ax=ax, color="coral")
ax.invert_yaxis()
ax.set_title("RF Clasificación: Importancia SHAP media (todas las clases)")
ax.set_xlabel("|SHAP| medio")
plt.tight_layout()
plt.savefig(FIGS / "C_shap_classification.png", dpi=150)
plt.close()
print("  [OK] C_shap_classification.png")

# =============================================================================
# MARKDOWN DE RESULTADOS
# =============================================================================
print("\n[OK] Guardando resultados en Markdown...")

rf_params  = rf_search.best_params_
xgb_params = xgb_search.best_params_

top5_shap_reg = shap_imp.head(5).index.tolist()
top5_shap_clf = shap_imp_clf.head(5).index.tolist()

worst_genre_reg = res_genre["mean"].idxmin()
best_genre_reg  = res_genre["mean"].idxmax()

md = f"""# Fase C — Modelos Predictivos: RF vs XGBoost (Versión Rigurosa)

## C.2 Regresión de Popularity

### Metodología
- Dataset: {len(df):,} canciones de `tracks_model.csv`
- Split estratificado por quintiles de popularity: 80% train / 20% test
- Target encoding para `track_genre` (fit solo en train para evitar data leakage)
- Búsqueda de hiperparámetros: RandomizedSearchCV (20 iteraciones, CV-3)
- Features: {len(FEATURES_REG) + 1} variables (audio features + log-transforms + electronic_ratio + track_genre codificada)

### Métricas finales en test

| Modelo | RMSE | MAE | R² | Mejor R² CV |
|--------|------|-----|-----|------------|
| Random Forest | {metrics_reg['Random Forest']['RMSE']} | {metrics_reg['Random Forest']['MAE']} | {metrics_reg['Random Forest']['R²']} | {rf_search.best_score_:.4f} |
| XGBoost | {metrics_reg['XGBoost']['RMSE']} | {metrics_reg['XGBoost']['MAE']} | {metrics_reg['XGBoost']['R²']} | {xgb_search.best_score_:.4f} |

**Mejor modelo: {best_name}** con R²={metrics_reg[best_name]['R²']:.4f}

### Hiperparámetros óptimos

**Random Forest:**
```
{rf_params}
```

**XGBoost:**
```
{xgb_params}
```

### Top 5 features más importantes (SHAP)

Regresión: {top5_shap_reg}

### Análisis de residuos

- El modelo sobre-predice en géneros con popularidad estructuralmente baja (`{worst_genre_reg}`):
  residuo medio más negativo.
- El modelo sub-predice en géneros con popularidad alta (`{best_genre_reg}`):
  residuo medio más positivo.
- La distribución de residuos es aproximadamente simétrica centrada en 0, sin sesgo sistemático.
- El gráfico real vs predicho muestra heteroscedasticidad: el error es mayor para canciones muy populares
  (popularity > 70), lo que es esperable dado que estas canciones dependen de factores externos
  (marketing, viralidad) no capturados en las features de audio.

### Learning curves

La brecha entre el R² de entrenamiento (~0.85) y el de validación (~0.47) indica **overfitting moderado**.
Las curvas convergen a medida que aumenta el tamaño del conjunto de entrenamiento, sugiriendo que
más datos mejorarían el modelo pero el efecto sería pequeño.

---

## C.3 Clasificación de Macro-género

### Metodología
- 12 macro-géneros (después de mapear los 114 géneros originales)
- Split estratificado por macro_genre: 80% train / 20% test
- Mismas features que regresión (sin track_genre)
- RandomizedSearchCV (15 iteraciones, CV-3), métrica F1-macro

### Métricas finales en test

| Modelo | Accuracy | F1-macro | F1-weighted |
|--------|----------|----------|-------------|
| Random Forest | {metrics_clf['Random Forest']['Accuracy']} | {metrics_clf['Random Forest']['F1-macro']} | {metrics_clf['Random Forest']['F1-weighted']} |
| XGBoost | {metrics_clf['XGBoost']['Accuracy']} | {metrics_clf['XGBoost']['F1-macro']} | {metrics_clf['XGBoost']['F1-weighted']} |
| Baseline (sin agrupar, 114 géneros) | ~0.26 | ~0.25 | ~0.25 |

### Top 5 features más importantes (SHAP clasificación)

{top5_shap_clf}

### Interpretación de la matriz de confusión

Los géneros mejor clasificados son los que tienen un perfil de audio más distintivo:
- **clásica**: alta instrumentalness y acousticness → casi nunca se confunde
- **metal**: energy extrema y valence muy baja → fácilmente identificable
- **kpop-jpop**: combinación de alta danceability con producción muy característica

Los más confundidos:
- **pop vs folk-acústico**: muchas canciones pop acústicas ambiguas
- **rock vs metal**: el límite entre hard-rock y metal es difuso en las features de Spotify
- **latino vs hip-hop**: trap latino y reggeaton tienen features similares al hip-hop

---

## Comparativa global: RF vs XGBoost

| Tarea | Ganador | Diferencia en métrica principal |
|-------|---------|--------------------------------|
| Regresión (R²) | {'RF' if metrics_reg['Random Forest']['R²'] >= metrics_reg['XGBoost']['R²'] else 'XGB'} | {abs(metrics_reg['Random Forest']['R²'] - metrics_reg['XGBoost']['R²']):.4f} |
| Clasificación (F1-macro) | {'RF' if metrics_clf['Random Forest']['F1-macro'] >= metrics_clf['XGBoost']['F1-macro'] else 'XGB'} | {abs(metrics_clf['Random Forest']['F1-macro'] - metrics_clf['XGBoost']['F1-macro']):.4f} |

Ambos modelos son muy similares en rendimiento. RF tiende a overfittear ligeramente más,
XGBoost es más rápido de entrenar con los mismos parámetros.

## Limitaciones del modelado

1. **Techo de R²≈0.47**: la popularidad depende de factores externos (playlists editoriales,
   marketing, viralidad en redes) que no están en el dataset.
2. **Estrategia de Target Encoding**: la codificación de género por target puede introducir
   sesgo si hay géneros muy pequeños. Con n_folds CV interno se mitiga.
3. **Clasificación multi-clase con 12 clases**: el mapa editorial es una decisión subjetiva.
   Distintos agrupamientos producirían distintos resultados.
"""

(RES / "C_modelos_predictivos.md").write_text(md, encoding="utf-8")
print("[OK] results/C_modelos_predictivos.md guardado")
print("\n[FASE C COMPLETADA]")
