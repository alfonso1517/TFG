"""
FASE 5 — Estudio de nacionalidad
Crea data/processed/artists_nationality.csv y realiza el análisis comparativo
por región: perfil de audio features, popularidad, géneros dominantes.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
RES  = ROOT / "results"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ══════════════════════════════════════════════════════════════════════════════
# 1. TABLA DE NACIONALIDADES (curada manualmente)
# ══════════════════════════════════════════════════════════════════════════════
# Formato: nombre exacto tal como aparece en el dataset → (país, región)
REGIONES = {
    "Anglophone":  "Anglophone (USA/UK/AUS/CAN)",
    "Latino-LAM":  "Latinoamérica",
    "España":      "España",
    "Asia-Pac":    "Asia-Pacífico",
    "Europa":      "Europa (no anglosajona)",
    "India":       "India/Subcontinente",
    "Africa":      "África",
    "Otros":       "Otros",
}

NATIONALITY = {
    # ── Anglophone ──────────────────────────────────────────────────────────
    "The Beatles":                  ("Reino Unido",    "Anglophone"),
    "Linkin Park":                  ("USA",            "Anglophone"),
    "Arctic Monkeys":               ("Reino Unido",    "Anglophone"),
    "CoComelon":                    ("USA",            "Anglophone"),
    "Elvis Presley":                ("USA",            "Anglophone"),
    "Glee Cast":                    ("USA",            "Anglophone"),
    "Pink Floyd":                   ("Reino Unido",    "Anglophone"),
    "Nirvana":                      ("USA",            "Anglophone"),
    "Hillsong Worship":             ("Australia",      "Anglophone"),
    "Adele":                        ("Reino Unido",    "Anglophone"),
    "Bring Me The Horizon":         ("Reino Unido",    "Anglophone"),
    "The Strokes":                  ("USA",            "Anglophone"),
    "Five Finger Death Punch":      ("USA",            "Anglophone"),
    "Red Hot Chili Peppers":        ("USA",            "Anglophone"),
    "Cigarettes After Sex":         ("USA",            "Anglophone"),
    "Billie Eilish":                ("USA",            "Anglophone"),
    "Sleeping At Last":             ("USA",            "Anglophone"),
    "keshi":                        ("USA",            "Anglophone"),
    "The Chemical Brothers":        ("Reino Unido",    "Anglophone"),
    "The Rolling Stones":           ("Reino Unido",    "Anglophone"),
    "Dan Gibson's Solitudes":       ("Canadá",         "Anglophone"),
    "Lil Peep":                     ("USA",            "Anglophone"),
    "Billy Joel":                   ("USA",            "Anglophone"),
    "John Mayer":                   ("USA",            "Anglophone"),
    "Elton John":                   ("Reino Unido",    "Anglophone"),
    "Lamb of God":                  ("USA",            "Anglophone"),
    "The Prodigy":                  ("Reino Unido",    "Anglophone"),
    "Novo Amor":                    ("Reino Unido",    "Anglophone"),
    "AC/DC":                        ("Australia",      "Anglophone"),
    "I Prevail":                    ("USA",            "Anglophone"),
    "Super Simple Songs":           ("Canadá",         "Anglophone"),
    "The Neighbourhood":            ("USA",            "Anglophone"),
    "Planetshakers":                ("Australia",      "Anglophone"),
    "Metallica":                    ("USA",            "Anglophone"),
    "Bonobo":                       ("Reino Unido",    "Anglophone"),
    "Three Days Grace":             ("Canadá",         "Anglophone"),
    "Breaking Benjamin":            ("USA",            "Anglophone"),
    "The Wiggles":                  ("Australia",      "Anglophone"),
    "Weezer":                       ("USA",            "Anglophone"),
    "Charlie Puth":                 ("USA",            "Anglophone"),
    "My Chemical Romance":          ("USA",            "Anglophone"),
    "The Kiboomers":                ("Canadá",         "Anglophone"),
    "Alter Bridge":                 ("USA",            "Anglophone"),
    "YUNGBLUD":                     ("Reino Unido",    "Anglophone"),
    "Pearl Jam":                    ("USA",            "Anglophone"),
    "Nickelback":                   ("Canadá",         "Anglophone"),
    "Boyce Avenue":                 ("USA",            "Anglophone"),
    "Greensky Bluegrass":           ("USA",            "Anglophone"),
    "Slipknot":                     ("USA",            "Anglophone"),
    "The Notorious B.I.G.":         ("USA",            "Anglophone"),
    "The Doors":                    ("USA",            "Anglophone"),
    "Blondie":                      ("USA",            "Anglophone"),
    "Shinedown":                    ("USA",            "Anglophone"),
    "Fall Out Boy":                 ("USA",            "Anglophone"),
    "Avenged Sevenfold":            ("USA",            "Anglophone"),
    "Green Day":                    ("USA",            "Anglophone"),
    "Frank Ocean":                  ("USA",            "Anglophone"),
    "Alice In Chains":              ("USA",            "Anglophone"),
    "Imagine Dragons":              ("USA",            "Anglophone"),
    "Hozier":                       ("Irlanda",        "Anglophone"),
    "JPEGMAFIA":                    ("USA",            "Anglophone"),
    "Foo Fighters":                 ("USA",            "Anglophone"),
    "Led Zeppelin":                 ("Reino Unido",    "Anglophone"),
    "Guns N' Roses":                ("USA",            "Anglophone"),
    "Bee Gees":                     ("Australia",      "Anglophone"),
    "Massive Attack":               ("Reino Unido",    "Anglophone"),
    "Napalm Death":                 ("Reino Unido",    "Anglophone"),
    "The Beach Boys":               ("USA",            "Anglophone"),
    "Alec Benjamin":                ("USA",            "Anglophone"),
    "The Score":                    ("USA",            "Anglophone"),
    "for KING & COUNTRY":           ("Australia",      "Anglophone"),
    "Jason Mraz":                   ("USA",            "Anglophone"),
    "Hillsong Young & Free":        ("Australia",      "Anglophone"),
    "Phil Wickham":                 ("USA",            "Anglophone"),
    "Chris Tomlin":                 ("USA",            "Anglophone"),
    "Burna Boy":                    ("Nigeria",        "Africa"),
    "Vybz Kartel":                  ("Jamaica",        "Anglophone"),
    "Germaine Franco":              ("USA",            "Anglophone"),
    "Sleeping At Last":             ("USA",            "Anglophone"),
    "Samuel Kim":                   ("USA",            "Anglophone"),
    "Sarah, the Illstrumentalist":  ("USA",            "Anglophone"),
    "Kathleen Madigan":             ("USA",            "Anglophone"),
    "Rain Sounds":                  ("USA",            "Anglophone"),
    "Bullet For My Valentine":      ("Reino Unido",    "Anglophone"),
    "Nusrat Fateh Ali Khan":        ("Pakistán",       "India"),
    "Eve":                          ("USA",            "Anglophone"),
    "Tove Lo":                      ("Suecia",         "Europa"),
    "Enigma":                       ("Alemania",       "Europa"),
    "OneRepublic":                  ("USA",            "Anglophone"),
    "George Jones":                 ("USA",            "Anglophone"),
    "Hank Williams":                ("USA",            "Anglophone"),
    "Hank Williams;Drifting Cowboys":("USA",           "Anglophone"),
    # ── Latinoamérica ────────────────────────────────────────────────────────
    "Bad Bunny":                    ("Puerto Rico",    "Latino-LAM"),
    "Feid":                         ("Colombia",       "Latino-LAM"),
    "KAROL G":                      ("Colombia",       "Latino-LAM"),
    "J Balvin":                     ("Colombia",       "Latino-LAM"),
    "Maluma":                       ("Colombia",       "Latino-LAM"),
    "Shakira":                      ("Colombia",       "Latino-LAM"),
    "Ozuna":                        ("Puerto Rico",    "Latino-LAM"),
    "Daddy Yankee":                 ("Puerto Rico",    "Latino-LAM"),
    "Rauw Alejandro":               ("Puerto Rico",    "Latino-LAM"),
    "Nicky Jam":                    ("Puerto Rico",    "Latino-LAM"),
    "Ricky Martin":                 ("Puerto Rico",    "Latino-LAM"),
    "Anuel AA":                     ("Puerto Rico",    "Latino-LAM"),
    "Farruko":                      ("Puerto Rico",    "Latino-LAM"),
    "Marc Anthony":                 ("Puerto Rico",    "Latino-LAM"),
    "Calle 13":                     ("Puerto Rico",    "Latino-LAM"),
    "Maelo Ruiz":                   ("Puerto Rico",    "Latino-LAM"),
    "Myke Towers":                  ("Puerto Rico",    "Latino-LAM"),
    "Juanes":                       ("Colombia",       "Latino-LAM"),
    "Camilo":                       ("Colombia",       "Latino-LAM"),
    "Sebastián Yatra":              ("Colombia",       "Latino-LAM"),
    "Sebastian Yatra":              ("Colombia",       "Latino-LAM"),
    "Manuel Turizo":                ("Colombia",       "Latino-LAM"),
    "Becky G":                      ("USA",            "Anglophone"),  # mexicano-americana, criada en USA
    "Charlie Brown Jr.":            ("Brasil",         "Latino-LAM"),
    "Henrique & Juliano":           ("Brasil",         "Latino-LAM"),
    "Sorriso Maroto":               ("Brasil",         "Latino-LAM"),
    "Os Barões Da Pisadinha":       ("Brasil",         "Latino-LAM"),
    "Criolo":                       ("Brasil",         "Latino-LAM"),
    "Ferrugem":                     ("Brasil",         "Latino-LAM"),
    "Zeca Pagodinho":               ("Brasil",         "Latino-LAM"),
    "Exaltasamba":                  ("Brasil",         "Latino-LAM"),
    "Matheus & Kauan":              ("Brasil",         "Latino-LAM"),
    "Bruno & Marrone":              ("Brasil",         "Latino-LAM"),
    "Maneva":                       ("Brasil",         "Latino-LAM"),
    "Thiaguinho":                   ("Brasil",         "Latino-LAM"),
    "Murilo Huff":                  ("Brasil",         "Latino-LAM"),
    "Legião Urbana":                ("Brasil",         "Latino-LAM"),
    "Jorge Aragão":                 ("Brasil",         "Latino-LAM"),
    "Diogo Nogueira":               ("Brasil",         "Latino-LAM"),
    "Péricles":                     ("Brasil",         "Latino-LAM"),
    "Calcinha Preta":               ("Brasil",         "Latino-LAM"),
    "Fernandinho":                  ("Brasil",         "Latino-LAM"),
    "Art Popular":                  ("Brasil",         "Latino-LAM"),
    "Kemuel":                       ("Brasil",         "Latino-LAM"),
    "Aline Barros":                 ("Brasil",         "Latino-LAM"),
    "Gerson Rufino":                ("Brasil",         "Latino-LAM"),
    "Trazendo a Arca":              ("Brasil",         "Latino-LAM"),
    "Ícaro e Gilmar":               ("Brasil",         "Latino-LAM"),
    "Roupa Nova":                   ("Brasil",         "Latino-LAM"),
    "Criolo":                       ("Brasil",         "Latino-LAM"),
    "Almafuerte":                   ("Argentina",      "Latino-LAM"),
    "Los Caligaris":                ("Argentina",      "Latino-LAM"),
    "Miranda!":                     ("Argentina",      "Latino-LAM"),
    "Carajo":                       ("Argentina",      "Latino-LAM"),
    "Almafuerte":                   ("Argentina",      "Latino-LAM"),
    "Rata Blanca":                  ("Argentina",      "Latino-LAM"),
    "Carlos Gardel":                ("Argentina",      "Latino-LAM"),
    "Hermetica":                    ("Argentina",      "Latino-LAM"),
    "Asspera":                      ("Argentina",      "Latino-LAM"),
    "Babasónicos":                  ("Argentina",      "Latino-LAM"),
    "Charly García":                ("Argentina",      "Latino-LAM"),
    "Patricio Rey y sus Redonditos de Ricota": ("Argentina", "Latino-LAM"),
    "Los Fabulosos Cadillacs":      ("Argentina",      "Latino-LAM"),
    "Soda Stereo":                  ("Argentina",      "Latino-LAM"),
    "Bizarrap":                     ("Argentina",      "Latino-LAM"),
    "Jorge Drexler":                ("Uruguay",        "Latino-LAM"),
    "Los Amigos Invisibles":        ("Venezuela",      "Latino-LAM"),
    "Joan Sebastian":               ("México",         "Latino-LAM"),
    "Panteon Rococo":               ("México",         "Latino-LAM"),
    "Cri-Cri":                      ("México",         "Latino-LAM"),
    "PXNDX":                        ("México",         "Latino-LAM"),
    "XXXTENTACION":                 ("USA",            "Anglophone"),
    # ── España ───────────────────────────────────────────────────────────────
    "Mägo de Oz":                   ("España",         "España"),
    "Alejandro Sanz":               ("España",         "España"),
    "Enrique Iglesias":             ("España",         "España"),
    "David Bisbal":                 ("España",         "España"),
    "Melendi":                      ("España",         "España"),
    "C. Tangana":                   ("España",         "España"),
    "Pablo Alborán":                ("España",         "España"),
    # ── Asia-Pacífico ────────────────────────────────────────────────────────
    "BTS":                          ("Corea del Sur",  "Asia-Pac"),
    "BLACKPINK":                    ("Corea del Sur",  "Asia-Pac"),
    "ENHYPEN":                      ("Corea del Sur",  "Asia-Pac"),
    "Stray Kids":                   ("Corea del Sur",  "Asia-Pac"),
    "TWICE":                        ("Corea del Sur",  "Asia-Pac"),
    "Pinkfong":                     ("Corea del Sur",  "Asia-Pac"),
    "Yiruma":                       ("Corea del Sur",  "Asia-Pac"),
    "Smyang Piano":                 ("Corea del Sur",  "Asia-Pac"),
    "Jay Chou":                     ("Taiwán",         "Asia-Pac"),
    "Eason Chan":                   ("Hong Kong",      "Asia-Pac"),
    "G.E.M.":                       ("Hong Kong",      "Asia-Pac"),
    "my little airport":            ("Hong Kong",      "Asia-Pac"),
    "Faye Wong":                    ("Hong Kong",      "Asia-Pac"),
    "JJ Lin":                       ("Singapur",       "Asia-Pac"),
    "YOASOBI":                      ("Japón",          "Asia-Pac"),
    "Nogizaka46":                   ("Japón",          "Asia-Pac"),
    "Eikichi Yazawa":               ("Japón",          "Asia-Pac"),
    "Seiko Matsuda":                ("Japón",          "Asia-Pac"),
    "Hiroyuki Sawano":              ("Japón",          "Asia-Pac"),
    "BOØWY":                        ("Japón",          "Asia-Pac"),
    "Mr.Children":                  ("Japón",          "Asia-Pac"),
    "Shiritsu Ebisu Chugaku":       ("Japón",          "Asia-Pac"),
    "Keyakizaka46":                 ("Japón",          "Asia-Pac"),
    "サザンオールスターズ":               ("Japón",          "Asia-Pac"),
    "RADWIMPS":                     ("Japón",          "Asia-Pac"),
    "Fujii Kaze":                   ("Japón",          "Asia-Pac"),
    "ONE OK ROCK":                  ("Japón",          "Asia-Pac"),
    # ── Europa (no anglosajona) ──────────────────────────────────────────────
    "Rammstein":                    ("Alemania",       "Europa"),
    "Scooter":                      ("Alemania",       "Europa"),
    "Enigma":                       ("Alemania",       "Europa"),
    "Boris Brejcha":                ("Alemania",       "Europa"),
    "Weißes Rauschen HD":           ("Alemania",       "Europa"),
    "Rolf Zuckowski":               ("Alemania",       "Europa"),
    "Hans Zimmer":                  ("Alemania",       "Europa"),
    "Boney M.":                     ("Alemania",       "Europa"),
    "ABBA":                         ("Suecia",         "Europa"),
    "Håkan Hellström":              ("Suecia",         "Europa"),
    "Lasse Stefanz":                ("Suecia",         "Europa"),
    "Toini & The Tomcats":          ("Suecia",         "Europa"),
    "Stam1na":                      ("Finlandia",      "Europa"),
    "Nightwish":                    ("Finlandia",      "Europa"),
    "Gojira":                       ("Francia",        "Europa"),
    "Armin van Buuren":             ("Países Bajos",   "Europa"),
    "Ludovico Einaudi":             ("Italia",         "Europa"),
    "Yanni":                        ("Grecia",         "Europa"),
    "Kato":                         ("Dinamarca",      "Europa"),
    # ── India / Subcontinente ────────────────────────────────────────────────
    "Arijit Singh":                 ("India",          "India"),
    "Sidhu Moose Wala":             ("India",          "India"),
    "Sujatha":                      ("India",          "India"),
    "Prateek Kuhad":                ("India",          "India"),
    "Yuvan Shankar Raja":           ("India",          "India"),
    "Anirudh Ravichander":          ("India",          "India"),
    "Anupam Roy":                   ("India",          "India"),
}

# ── Construir DataFrame de artistas ──────────────────────────────────────────
nat_rows = []
for artist, (pais, region) in NATIONALITY.items():
    nat_rows.append({"artist": artist, "pais": pais, "region": region,
                     "region_label": REGIONES.get(region, region)})
nat_df = pd.DataFrame(nat_rows).drop_duplicates(subset="artist")
nat_df.to_csv(PROC / "artists_nationality.csv", index=False, encoding="utf-8")
print(f"[OK] artists_nationality.csv: {len(nat_df)} artistas")
print(nat_df["region"].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 2. EXTRAER CANCIONES DEL DATASET (primer artista = artista principal)
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(PROC / "tracks_unique.csv")
# Artista principal = primer nombre antes del ";"
df["primary_artist"] = df["artists"].str.split(";").str[0].str.strip()

# Merge con tabla de nacionalidades
df_nat = df.merge(nat_df, left_on="primary_artist", right_on="artist", how="inner")
df_nat.to_csv(PROC / "tracks_nationality.csv", index=False, encoding="utf-8")
print(f"\n[OK] tracks_nationality.csv: {len(df_nat)} canciones de {df_nat['primary_artist'].nunique()} artistas")
print(df_nat["region"].value_counts().to_string())

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

# ══════════════════════════════════════════════════════════════════════════════
# 3. ANÁLISIS: POPULARIDAD POR REGIÓN
# ══════════════════════════════════════════════════════════════════════════════
region_pop = df_nat.groupby("region_label")["popularity"].agg(
    media="mean", mediana="median", std="std", n="count"
).sort_values("mediana", ascending=False)

print("\n── Popularidad por región ──")
print(region_pop.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
order = region_pop.index.tolist()
colors = sns.color_palette("Set2", len(order))

# Boxplot
sns.boxplot(data=df_nat, x="region_label", y="popularity", order=order,
            palette="Set2", ax=axes[0], fliersize=2)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=40, ha="right", fontsize=8)
axes[0].set_title("Distribución de Popularity por Región")
axes[0].set_xlabel(""); axes[0].set_ylabel("Popularity (0–100)")

# Barras con popularidad media
medias = region_pop["media"]
axes[1].barh(medias.index[::-1], medias.values[::-1],
             color=[colors[i] for i in range(len(order)-1, -1, -1)])
axes[1].set_title("Popularidad Media por Región")
axes[1].set_xlabel("Popularity media")
for i, v in enumerate(medias.values[::-1]):
    axes[1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(FIGS / "05a_popularity_by_region.png")
plt.close()
print("[OK] 05a guardado")

# ══════════════════════════════════════════════════════════════════════════════
# 4. ANÁLISIS: PERFIL DE AUDIO FEATURES POR REGIÓN
# ══════════════════════════════════════════════════════════════════════════════
# Normalizar features a [0,1] para comparación (loudness y tempo tienen escalas distintas)
from sklearn.preprocessing import MinMaxScaler
scaler01 = MinMaxScaler()
df_nat_norm = df_nat.copy()
df_nat_norm[AUDIO_FEATURES] = scaler01.fit_transform(df_nat[AUDIO_FEATURES])

region_audio = df_nat_norm.groupby("region_label")[AUDIO_FEATURES].mean()
print("\n── Perfil medio de audio features por región (normalizado 0-1) ──")
print(region_audio.round(3).to_string())

# Heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(region_audio, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.4, ax=ax, annot_kws={"size": 8})
ax.set_title("Perfil de Audio Features por Región (normalizado 0–1)")
ax.set_xlabel(""); ax.set_ylabel("")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "05b_audio_features_by_region.png")
plt.close()
print("[OK] 05b guardado")

# ── Radar por región ──────────────────────────────────────────────────────────
RADAR_FEATS = ["danceability", "energy", "valence", "acousticness", "speechiness", "liveness"]
angles = np.linspace(0, 2 * np.pi, len(RADAR_FEATS), endpoint=False).tolist()
angles += angles[:1]

regions_to_plot = region_audio.index.tolist()
colors_radar = plt.cm.Set2.colors

n_rows = (len(regions_to_plot) + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(14, n_rows * 4),
                          subplot_kw=dict(polar=True))
axes_flat = axes.flatten() if n_rows > 1 else axes

for ax, region in zip(axes_flat, regions_to_plot):
    vals = region_audio.loc[region, RADAR_FEATS].tolist()
    vals += vals[:1]
    c = colors_radar[regions_to_plot.index(region) % len(colors_radar)]
    ax.plot(angles, vals, color=c, linewidth=2)
    ax.fill(angles, vals, alpha=0.25, color=c)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_FEATS, size=7)
    ax.set_ylim(0, 1)
    ax.set_title(region[:30], size=8, pad=10)

for ax in axes_flat[len(regions_to_plot):]:
    ax.set_visible(False)

fig.suptitle("Perfil sonoro por región (audio features normalizadas)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "05c_radar_by_region.png")
plt.close()
print("[OK] 05c guardado")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ANÁLISIS: GÉNEROS DOMINANTES POR REGIÓN
# ══════════════════════════════════════════════════════════════════════════════
top_genres_by_region = {}
for region, grp in df_nat.groupby("region_label"):
    top5 = grp["track_genre"].value_counts().head(5)
    top_genres_by_region[region] = top5

print("\n── Top-5 géneros por región ──")
for region, top5 in top_genres_by_region.items():
    print(f"\n  {region}:")
    for g, n in top5.items():
        print(f"    {g}: {n}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ANÁLISIS: COMPARATIVA ESPECÍFICA Latino vs Anglophone vs España
# ══════════════════════════════════════════════════════════════════════════════
tres = ["Latinoamérica", "Anglophone (USA/UK/AUS/CAN)", "España"]
df_tres = df_nat[df_nat["region_label"].isin(tres)]

# Test ANOVA / diferencias en features clave
from scipy import stats

print("\n── Test de diferencias por región (danceability, energy, valence) ──")
for feat in ["danceability", "energy", "valence", "acousticness", "tempo"]:
    groups = [df_tres[df_tres["region_label"] == r][feat].dropna().values for r in tres]
    groups = [g for g in groups if len(g) > 5]
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {feat:20s}: F={f:.2f}  p={p:.4f} {sig}")

# Boxplots comparativos de las 3 regiones
feat_compare = ["danceability", "energy", "valence", "acousticness"]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
palette_3 = {"Latinoamérica": "#E63946",
             "Anglophone (USA/UK/AUS/CAN)": "#457B9D",
             "España": "#F4A261"}

for ax, feat in zip(axes.flatten(), feat_compare):
    sns.boxplot(data=df_tres, x="region_label", y=feat,
                order=tres, palette=palette_3, ax=ax, fliersize=2)
    ax.set_title(f"{feat.capitalize()} por región")
    ax.set_xlabel("")
    ax.set_xticklabels(["Latinoamérica", "Anglophone", "España"], fontsize=9)

plt.suptitle("Comparativa audio features: Latinoamérica vs Anglophone vs España",
             fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "05d_latam_vs_anglophone_vs_spain.png")
plt.close()
print("[OK] 05d guardado")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TOP ARTISTAS POR REGIÓN (popularidad)
# ══════════════════════════════════════════════════════════════════════════════
top_artists = (df_nat.groupby(["region_label", "primary_artist"])["popularity"]
               .mean().reset_index()
               .sort_values(["region_label", "popularity"], ascending=[True, False]))

print("\n── Top-3 artistas por popularidad media en cada región ──")
for region, grp in top_artists.groupby("region_label"):
    top3 = grp.head(3)
    print(f"\n  {region}:")
    for _, row in top3.iterrows():
        print(f"    {row['primary_artist']}: {row['popularity']:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Guardar notas
# ══════════════════════════════════════════════════════════════════════════════
n_por_region = df_nat["region_label"].value_counts()
mean_pop_region = df_nat.groupby("region_label")["popularity"].mean().sort_values(ascending=False)

notes = f"""# 05 — Estudio de Nacionalidad

## Metodología
- **Artistas incluidos**: {len(nat_df)} artistas curados manualmente
  (top-200 por score de presencia×popularidad + artistas latinos/españoles añadidos)
- **Criterio de autoría**: el "artista principal" es el **primer nombre** en el campo
  `artists` del dataset (p.ej. en "Ozuna;Feid", Ozuna es el principal)
- **Canciones analizadas**: {len(df_nat):,} canciones únicas con artista principal identificado
- **Limitación**: análisis **exploratorio sobre muestra curada y no representativa**.
  No es generalizable a toda la producción musical mundial.

## Distribución de canciones por región
{n_por_region.to_string()}

## Popularidad media por región (ranking)
{mean_pop_region.round(1).to_string()}

## Hallazgos de audio features (normalizado 0-1)
{region_audio.round(3).to_string()}

## Interpretación
- **Latinoamérica**: perfil caracterizado por alta danceability y valence
  (música alegre y bailable), baja acousticness en géneros urbanos.
- **Anglophone**: energía y diversidad alta; abarca desde rock a pop a hip-hop.
- **Asia-Pacífico**: destacan acousticness e instrumentalness (K-pop y J-pop
  tienen mucha producción instrumental); alta energía en K-pop.
- **Europa (no anglosajona)**: mayor instrumentalness (clásica, EDM sin vocals),
  diversidad estilística amplia.
- **España**: perfil intermedio entre Latino y Anglophone; mayor acousticness
  que Latinoamérica.
- **India**: alta speechiness (géneros con mucha letra), energía media.

## Géneros dominantes por región
{chr(10).join(f"- **{r}**: {', '.join(v.index[:3].tolist())}" for r, v in top_genres_by_region.items())}

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `05a_popularity_by_region.png` | Boxplot y barras de popularidad por región |
| `05b_audio_features_by_region.png` | Heatmap de audio features por región |
| `05c_radar_by_region.png` | Radares sonoros por región |
| `05d_latam_vs_anglophone_vs_spain.png` | Comparativa boxplots Latam/Anglophone/España |
"""

(RES / "05_nacionalidad.md").write_text(notes, encoding="utf-8")
print("\n[OK] results/05_nacionalidad.md guardado")
print("✓ Fase 5 completada.")
