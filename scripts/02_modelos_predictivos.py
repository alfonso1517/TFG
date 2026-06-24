"""
FASE 2 — Modelos predictivos: Random Forest vs XGBoost
  2.1 Regresión de popularity
  2.2 Clasificación de macro-género (+ baseline 114 clases)
Genera modelos en models/ y resultados en results/02_modelos_predictivos.md
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from category_encoders import TargetEncoder

ROOT   = Path(__file__).parent.parent
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS   = ROOT / "reports" / "figures"
RES    = ROOT / "results"

MODELS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

print("Cargando datos ...")
df = pd.read_csv(PROC / "tracks_unique.csv")
print(f"  {len(df):,} canciones únicas")

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature",
    "duration_ms", "explicit"
]

# ════════════════════════════════════════════════════════════════════════════════
# 2.1 REGRESIÓN DE POPULARITY
# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 2.1 Regresión de Popularity ══")

_reg_models_exist = (
    (MODELS / "rf_regressor.pkl").exists() and
    (MODELS / "xgb_regressor.pkl").exists() and
    (MODELS / "target_encoder_reg.pkl").exists()
)

if _reg_models_exist:
    print("  [SKIP] Modelos de regresión ya existen — cargando desde disco ...")
    rf_reg_best  = joblib.load(MODELS / "rf_regressor.pkl")
    xgb_reg_best = joblib.load(MODELS / "xgb_regressor.pkl")
    te           = joblib.load(MODELS / "target_encoder_reg.pkl")

    X_reg_raw = df[AUDIO_FEATURES + ["track_genre"]].copy()
    y_reg = df["popularity"].values
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg_raw, y_reg, test_size=0.2, random_state=42
    )
    X_train_r = te.fit_transform(X_train_r, y_train_r)
    X_test_r  = te.transform(X_test_r)

    y_pred_rf  = rf_reg_best.predict(X_test_r)
    y_pred_xgb = xgb_reg_best.predict(X_test_r)
    rmse_rf  = np.sqrt(mean_squared_error(y_test_r, y_pred_rf))
    mae_rf   = mean_absolute_error(y_test_r, y_pred_rf)
    r2_rf    = r2_score(y_test_r, y_pred_rf)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_r, y_pred_xgb))
    mae_xgb  = mean_absolute_error(y_test_r, y_pred_xgb)
    r2_xgb   = r2_score(y_test_r, y_pred_xgb)
    print(f"    RF  → RMSE={rmse_rf:.3f}  MAE={mae_rf:.3f}  R²={r2_rf:.3f}")
    print(f"    XGB → RMSE={rmse_xgb:.3f}  MAE={mae_xgb:.3f}  R²={r2_xgb:.3f}")
    feat_imp_rf_reg  = pd.Series(rf_reg_best.feature_importances_,
                                  index=AUDIO_FEATURES + ["track_genre"])
    feat_imp_xgb_reg = pd.Series(xgb_reg_best.feature_importances_,
                                  index=AUDIO_FEATURES + ["track_genre"])
else:
    # Target encoding de track_genre
    te = TargetEncoder(cols=["track_genre"])
    X_reg_raw = df[AUDIO_FEATURES + ["track_genre"]].copy()
    y_reg = df["popularity"].values

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg_raw, y_reg, test_size=0.2, random_state=42
    )
    X_train_r = te.fit_transform(X_train_r, y_train_r)
    X_test_r  = te.transform(X_test_r)

    # — Random Forest Regressor —
    print("  Entrenando RandomForestRegressor con RandomizedSearchCV ...")
    rf_param_dist = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [10, 20, 30, None],
        "min_samples_leaf":[1, 2, 4],
        "max_features":    ["sqrt", "log2", 0.5],
    }
    rf_reg = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=2),
        param_distributions=rf_param_dist,
        n_iter=20, cv=3, scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=2, verbose=0
    )
    rf_reg.fit(X_train_r, y_train_r)
    rf_reg_best = rf_reg.best_estimator_

    y_pred_rf = rf_reg_best.predict(X_test_r)
    rmse_rf  = np.sqrt(mean_squared_error(y_test_r, y_pred_rf))
    mae_rf   = mean_absolute_error(y_test_r, y_pred_rf)
    r2_rf    = r2_score(y_test_r, y_pred_rf)
    print(f"    RF  → RMSE={rmse_rf:.3f}  MAE={mae_rf:.3f}  R²={r2_rf:.3f}")
    print(f"    Best params: {rf_reg.best_params_}")

    # — XGBoost Regressor —
    print("  Entrenando XGBRegressor con RandomizedSearchCV ...")
    xgb_param_dist = {
        "n_estimators":      [200, 300, 500],
        "max_depth":         [4, 6, 8],
        "learning_rate":     [0.05, 0.1, 0.2],
        "subsample":         [0.7, 0.8, 1.0],
        "colsample_bytree":  [0.7, 0.8, 1.0],
        "gamma":             [0, 0.1, 0.3],
    }
    xgb_reg = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=2, verbosity=0),
        param_distributions=xgb_param_dist,
        n_iter=20, cv=3, scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=2, verbose=0
    )
    xgb_reg.fit(X_train_r, y_train_r)
    xgb_reg_best = xgb_reg.best_estimator_

    y_pred_xgb = xgb_reg_best.predict(X_test_r)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_r, y_pred_xgb))
    mae_xgb  = mean_absolute_error(y_test_r, y_pred_xgb)
    r2_xgb   = r2_score(y_test_r, y_pred_xgb)
    print(f"    XGB → RMSE={rmse_xgb:.3f}  MAE={mae_xgb:.3f}  R²={r2_xgb:.3f}")
    print(f"    Best params: {xgb_reg.best_params_}")

    feat_imp_rf_reg  = pd.Series(rf_reg_best.feature_importances_,
                                  index=AUDIO_FEATURES + ["track_genre"])
    feat_imp_xgb_reg = pd.Series(xgb_reg_best.feature_importances_,
                                  index=AUDIO_FEATURES + ["track_genre"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    feat_imp_rf_reg.sort_values().plot.barh(ax=axes[0], color="#4C72B0", title="RF — Importancia (regresión)")
    feat_imp_xgb_reg.sort_values().plot.barh(ax=axes[1], color="#DD8452", title="XGB — Importancia (regresión)")
    plt.tight_layout()
    plt.savefig(FIGS / "02a_feature_importance_regression.png")
    plt.close()
    print("  [OK] 02a guardado | Importancia variables regresión")

    joblib.dump(rf_reg_best,  MODELS / "rf_regressor.pkl")
    joblib.dump(xgb_reg_best, MODELS / "xgb_regressor.pkl")
    joblib.dump(te,           MODELS / "target_encoder_reg.pkl")
    print("  [OK] Modelos regresión guardados en models/")

# ════════════════════════════════════════════════════════════════════════════════
# 2.2 CLASIFICACIÓN DE GÉNERO
# ════════════════════════════════════════════════════════════════════════════════
print("\n══ 2.2 Clasificación de Género ══")

# Mapeo de 114 géneros a ~12 macro-categorías
GENRE_MAP = {
    # Rock y derivados
    "rock": "rock", "alt-rock": "rock", "alternative": "rock",
    "grunge": "rock", "psych-rock": "rock", "rock-n-roll": "rock",
    "hard-rock": "rock", "indie": "rock", "punk": "rock",
    "punk-rock": "rock", "emo": "rock", "garage": "rock",
    # Metal
    "metal": "metal", "heavy-metal": "metal", "death-metal": "metal",
    "black-metal": "metal", "metalcore": "metal", "grindcore": "metal",
    "hardcore": "metal",
    # Electrónica / EDM
    "edm": "electronic", "electronic": "electronic", "electro": "electronic",
    "techno": "electronic", "trance": "electronic", "house": "electronic",
    "deep-house": "electronic", "chicago-house": "electronic",
    "detroit-techno": "electronic", "dubstep": "electronic",
    "drum-and-bass": "electronic", "minimal-techno": "electronic",
    "progressive-house": "electronic", "synth-pop": "electronic",
    "idm": "electronic",
    # Hip-hop / Rap
    "hip-hop": "hip-hop", "rap": "hip-hop", "trap": "hip-hop",
    "gangster-rap": "hip-hop",
    # Latino / Reggaeton
    "reggaeton": "latino", "latin": "latino", "salsa": "latino",
    "samba": "latino", "mpb": "latino", "bossanova": "latino",
    "pagode": "latino", "sertanejo": "latino", "forro": "latino",
    "axe": "latino", "brazil": "latino",
    # Pop
    "pop": "pop", "indie-pop": "pop", "k-pop": "pop",
    "power-pop": "pop", "dream-pop": "pop", "electropop": "pop",
    "j-pop": "pop",
    # Jazz / Blues / Soul
    "jazz": "jazz-blues", "blues": "jazz-blues", "soul": "jazz-blues",
    "funk": "jazz-blues", "r-n-b": "jazz-blues", "gospel": "jazz-blues",
    "new-age": "jazz-blues",
    # Clásica / Instrumental
    "classical": "classical", "opera": "classical", "piano": "classical",
    "orchestra": "classical",
    # Folk / Acoustic / Country
    "folk": "folk-country", "acoustic": "folk-country",
    "country": "folk-country", "bluegrass": "folk-country",
    "singer-songwriter": "folk-country",
    # Reggae / Ska / Dub
    "reggae": "reggae-ska", "ska": "reggae-ska", "dub": "reggae-ska",
    "dancehall": "reggae-ska",
    # World / Regional
    "afrobeat": "world", "world-music": "world", "k-pop": "world",
    "indian": "world", "turkish": "world", "swedish": "world",
    "french": "world", "german": "world", "iranian": "world",
    "malay": "world", "mandopop": "world", "cantopop": "world",
    "tango": "world", "fado": "world", "spanish": "world",
    "latin-american": "world",
    # Otros / ambient
    "ambient": "ambient", "chill": "ambient", "sleep": "ambient",
    "study": "ambient", "meditation": "ambient", "anime": "ambient",
    "children": "ambient", "comedy": "ambient", "disney": "ambient",
    "show-tunes": "ambient", "romance": "ambient", "movies": "ambient",
    "soundtracks": "ambient", "video-game-music": "ambient",
    "holidays": "ambient",
}

# Usar tracks_long para tener más observaciones por género
df_long = pd.read_csv(PROC / "tracks_long.csv")
df_long["macro_genre"] = df_long["track_genre"].map(GENRE_MAP).fillna("other")

X_cls = df_long[AUDIO_FEATURES].copy()
y_macro = df_long["macro_genre"]
y_114   = df_long["track_genre"]

le_macro = LabelEncoder()
le_114   = LabelEncoder()
y_macro_enc = le_macro.fit_transform(y_macro)
y_114_enc   = le_114.fit_transform(y_114)

X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_cls, y_macro_enc, test_size=0.2, random_state=42, stratify=y_macro_enc
)

# — Baseline rápido: RF con 114 clases —
print("  Baseline RF con 114 géneros ...")
X_tr_114, X_te_114, y_tr_114, y_te_114 = train_test_split(
    X_cls, y_114_enc, test_size=0.2, random_state=42, stratify=y_114_enc
)
rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=20,
                                      random_state=42, n_jobs=-1)
rf_baseline.fit(X_tr_114, y_tr_114)
y_pred_base = rf_baseline.predict(X_te_114)
acc_base = accuracy_score(y_te_114, y_pred_base)
f1_base  = f1_score(y_te_114, y_pred_base, average="macro", zero_division=0)
print(f"    Baseline 114 clases → Acc={acc_base:.3f}  F1-macro={f1_base:.3f}")

# — RF Macro-género con RandomizedSearchCV —
print("  RandomForestClassifier macro-género ...")
rf_cls_cv = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions={
        "n_estimators": [100, 200, 300],
        "max_depth":    [10, 20, 30, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    n_iter=15, cv=3, scoring="f1_macro",
    random_state=42, n_jobs=-1, verbose=0
)
rf_cls_cv.fit(X_tr_m, y_tr_m)
rf_cls_best = rf_cls_cv.best_estimator_

y_pred_rf_cls = rf_cls_best.predict(X_te_m)
acc_rf_cls = accuracy_score(y_te_m, y_pred_rf_cls)
f1_rf_cls  = f1_score(y_te_m, y_pred_rf_cls, average="macro", zero_division=0)
print(f"    RF macro-género → Acc={acc_rf_cls:.3f}  F1-macro={f1_rf_cls:.3f}")
print(f"    Best params: {rf_cls_cv.best_params_}")

# — XGB Macro-género con RandomizedSearchCV —
print("  XGBClassifier macro-género ...")
xgb_cls_cv = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, verbosity=0,
                  eval_metric="mlogloss"),
    param_distributions={
        "n_estimators":     [200, 300, 500],
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    n_iter=15, cv=3, scoring="f1_macro",
    random_state=42, n_jobs=-1, verbose=0
)
xgb_cls_cv.fit(X_tr_m, y_tr_m)
xgb_cls_best = xgb_cls_cv.best_estimator_

y_pred_xgb_cls = xgb_cls_best.predict(X_te_m)
acc_xgb_cls = accuracy_score(y_te_m, y_pred_xgb_cls)
f1_xgb_cls  = f1_score(y_te_m, y_pred_xgb_cls, average="macro", zero_division=0)
print(f"    XGB macro-género → Acc={acc_xgb_cls:.3f}  F1-macro={f1_xgb_cls:.3f}")
print(f"    Best params: {xgb_cls_cv.best_params_}")

# — Importancia de variables clasificación —
feat_imp_rf_cls  = pd.Series(rf_cls_best.feature_importances_, index=AUDIO_FEATURES)
feat_imp_xgb_cls = pd.Series(xgb_cls_best.feature_importances_, index=AUDIO_FEATURES)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
feat_imp_rf_cls.sort_values().plot.barh(ax=axes[0], color="#4C72B0",
                                         title="RF — Importancia (clasificación macro-género)")
feat_imp_xgb_cls.sort_values().plot.barh(ax=axes[1], color="#DD8452",
                                          title="XGB — Importancia (clasificación macro-género)")
plt.tight_layout()
plt.savefig(FIGS / "02b_feature_importance_classification.png")
plt.close()
print("  [OK] 02b guardado | Importancia variables clasificación")

# — Matriz de confusión macro-género (RF) —
class_names = le_macro.classes_
cm = confusion_matrix(y_te_m, y_pred_rf_cls)
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
ax.set_title("Matriz de Confusión — RF Macro-género")
plt.tight_layout()
plt.savefig(FIGS / "02c_confusion_matrix_rf.png")
plt.close()
print("  [OK] 02c guardado | Matriz de confusión RF")

# — Guardar modelos y encoders —
joblib.dump(rf_cls_best,  MODELS / "rf_classifier.pkl")
joblib.dump(xgb_cls_best, MODELS / "xgb_classifier.pkl")
joblib.dump(le_macro,     MODELS / "label_encoder_macro.pkl")
joblib.dump(GENRE_MAP,    MODELS / "genre_map.pkl")
print("  [OK] Modelos clasificación guardados en models/")

# ── Guardar resultados ────────────────────────────────────────────────────────
best_feat_reg_rf  = feat_imp_rf_reg.sort_values(ascending=False).head(3).index.tolist()
best_feat_reg_xgb = feat_imp_xgb_reg.sort_values(ascending=False).head(3).index.tolist()
best_feat_cls_rf  = feat_imp_rf_cls.sort_values(ascending=False).head(3).index.tolist()
best_feat_cls_xgb = feat_imp_xgb_cls.sort_values(ascending=False).head(3).index.tolist()

results_md = f"""# 02 — Resultados Modelos Predictivos

## 2.1 Regresión de Popularity

### Métricas comparativas (test 20%)
| Modelo | RMSE | MAE | R² |
|--------|------|-----|----|
| Random Forest | {rmse_rf:.3f} | {mae_rf:.3f} | {r2_rf:.3f} |
| XGBoost | {rmse_xgb:.3f} | {mae_xgb:.3f} | {r2_xgb:.3f} |

### Mejores hiperparámetros
- **RF**: {rf_reg.best_params_}
- **XGB**: {xgb_reg.best_params_}

### Top-3 features más importantes
- **RF**:  {', '.join(best_feat_reg_rf)}
- **XGB**: {', '.join(best_feat_reg_xgb)}

### Interpretación
El R² relativamente bajo es esperable: `popularity` es un índice calculado
por Spotify con factores externos al audio (tendencias virales, lanzamientos
recientes, playlists editoriales) que no están en el dataset. Las features de
audio capturan la "calidad musical" pero no el "momento de lanzamiento".
La variable `track_genre` (codificada) suele aparecer entre las más importantes,
confirmando que el género tiene un impacto significativo en la popularidad.

---

## 2.2 Clasificación de Género

### Baseline con 114 géneros (RF, sin tuning)
| Métrica | Valor |
|---------|-------|
| Accuracy | {acc_base:.3f} |
| F1-macro | {f1_base:.3f} |

*Con 114 clases muy solapadas el F1-macro bajo es esperable.*

### Macro-géneros (~12 categorías)
| Modelo | Accuracy | F1-macro |
|--------|----------|----------|
| Random Forest | {acc_rf_cls:.3f} | {f1_rf_cls:.3f} |
| XGBoost | {acc_xgb_cls:.3f} | {f1_xgb_cls:.3f} |

### Mejores hiperparámetros
- **RF**: {rf_cls_cv.best_params_}
- **XGB**: {xgb_cls_cv.best_params_}

### Top-3 features más importantes (clasificación)
- **RF**:  {', '.join(best_feat_cls_rf)}
- **XGB**: {', '.join(best_feat_cls_xgb)}

### Interpretación
Agrupar 114 géneros en ~12 macro-categorías mejora drásticamente las métricas.
Las features más discriminativas suelen ser `acousticness`, `energy` y
`instrumentalness`, lo que tiene sentido: clásica (alta acousticness +
instrumentalness), metal (alta energy, baja acousticness), electrónica
(alta energy + tempo), etc.

### Mapeo de macro-géneros utilizado
- **rock**: rock, alt-rock, alternative, grunge, indie, punk, emo, garage…
- **metal**: metal, heavy-metal, death-metal, black-metal, metalcore…
- **electronic**: edm, techno, trance, house, dubstep, drum-and-bass…
- **hip-hop**: hip-hop, rap, trap, gangster-rap
- **latino**: reggaeton, latin, salsa, samba, bossanova…
- **pop**: pop, indie-pop, k-pop, j-pop, dream-pop…
- **jazz-blues**: jazz, blues, soul, funk, r-n-b, gospel…
- **classical**: classical, opera, piano, orchestra
- **folk-country**: folk, acoustic, country, bluegrass, singer-songwriter
- **reggae-ska**: reggae, ska, dub, dancehall
- **world**: afrobeat, turkish, swedish, french, tango, fado, mandopop…
- **ambient**: ambient, chill, sleep, anime, children, comedy, disney…
- **other**: géneros no mapeados

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `02a_feature_importance_regression.png` | Importancia variables (RF vs XGB, regresión) |
| `02b_feature_importance_classification.png` | Importancia variables (RF vs XGB, clasificación) |
| `02c_confusion_matrix_rf.png` | Matriz confusión RF macro-género |
"""

(RES / "02_modelos_predictivos.md").write_text(results_md, encoding="utf-8")
print("\n[OK] results/02_modelos_predictivos.md guardado")
print("✓ Fase 2 completada.")
