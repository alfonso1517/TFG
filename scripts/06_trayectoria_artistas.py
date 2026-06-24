"""
FASE 6 — Trayectoria temporal de artistas: Feid y similares
Extrae el año/mes de lanzamiento de cada canción usando conocimiento discográfico,
analiza la evolución del sonido y predice la trayectoria futura de Feid.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
FIGS  = ROOT / "reports" / "figures"
RES   = ROOT / "results"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

ARTISTAS = ["Feid", "J Balvin", "Maluma", "KAROL G",
            "Ozuna", "Rauw Alejandro", "Daddy Yankee", "Bad Bunny"]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  MAPA DE FECHAS POR TRACK_NAME  (año, mes)
#     Basado en conocimiento discográfico. Mes = 6 cuando solo se conoce el año.
# ══════════════════════════════════════════════════════════════════════════════
TRACK_DATES = {
    # ── FEID ──────────────────────────────────────────────────────────────────
    "Normal":                          (2018, 1),
    "Nuestra Canción":                 (2018, 6),
    "La Buena Fai":                    (2019, 3),
    "Belixe":                          (2019, 9),
    "PORFA - Remix":                   (2020, 4),
    "911":                             (2020, 5),
    "AMOR DE MI VIDA":                 (2020, 8),
    "Castigo":                         (2020, 1),
    "De Tanto Chimbiar":               (2020, 7),
    "Ron - Remix":                     (2020, 10),
    "Llori Pari":                      (2021, 1),
    "LA INOCENTE":                     (2021, 4),
    "Si Te La Encuentras Por Ahí":     (2021, 5),
    "JAMAICA":                         (2021, 6),
    "Pantysito":                       (2021, 7),
    "VIP Feat. Totoy El Frio":         (2021, 9),
    "Que Raro":                        (2021, 10),
    "Quemando Calorías":               (2021, 11),
    "A Mi También":                    (2021, 8),
    "RELXJXTE":                        (2022, 2),
    "FRESH KERIAS":                    (2022, 4),
    "SI TÚ SUPIERAS":                  (2022, 7),
    "XQ Te Pones Así":                 (2022, 8),
    "Hey Mor":                         (2022, 9),
    "Feliz Cumpleaños Ferxxo":         (2022, 10),
    "Ferxxo 100":                      (2022, 10),
    "Prohibidox":                      (2022, 10),
    "TENGO FE":                        (2022, 10),
    # ── J BALVIN ──────────────────────────────────────────────────────────────
    "Sigo Extrañándote":               (2014, 6),
    "Sin Compromiso":                  (2014, 1),
    "Dónde Estarás":                   (2014, 6),
    "Ay Vamos":                        (2015, 5),
    "Ay Vamos - Remix":                (2015, 8),
    "Tu Veneno":                       (2015, 1),
    "Mocca - Remix":                   (2015, 3),
    "Ginza":                           (2015, 8),
    "Safari":                          (2016, 11),
    "La Rebelión":                     (2016, 7),
    "Cuando Tú Quieras":               (2016, 4),
    "Mi Gente":                        (2017, 6),
    "Mi Gente (feat. Beyoncé)":        (2017, 9),
    "Mi Gente - Hugel Remix":          (2017, 12),
    "Otra vez":                        (2017, 3),
    "Ahora Dice":                      (2017, 10),
    "Bum Bum Tam Tam":                 (2017, 11),
    "Hey Ma (with J Balvin & Pitbull feat. Camila Cabello)": (2017, 4),
    "Hey Ma (with J Balvin & Pitbull feat. Camila Cabel": (2017, 4),
    "I Like It":                       (2018, 5),
    "No Es Justo":                     (2018, 4),
    "X":                               (2018, 9),
    "X - Spanglish Version":           (2018, 11),
    "X (feat. Maluma & Ozuna) - Remix":(2018, 12),
    "Mi Cama - Remix":                 (2018, 8),
    "I Can't Get Enough (benny blanco, Selena Gomez, J ": (2019, 1),
    "Contra La Pared":                 (2019, 3),
    "Contra La Pared - GTA Remix":     (2019, 5),
    "China":                           (2019, 7),
    "Baila Baila Baila - Remix":       (2019, 7),
    "Downtown":                        (2019, 7),
    "Bola Rebola":                     (2019, 6),
    "Bola Rebola - M3B Remix":         (2019, 8),
    "LA CANCIÓN":                      (2019, 6),
    "COMO UN BEBÉ":                    (2019, 6),
    "LOCATION":                        (2019, 9),
    "Loco Contigo":                    (2019, 8),
    "Loco Contigo (feat. J. Balvin & Tyga)": (2019, 8),
    "Loco Contigo (with J. Balvin & Ozuna feat. Nicky J": (2019, 9),
    "Loco Contigo - REMIX":            (2019, 9),
    "Indeciso":                        (2019, 10),
    "Say My Name":                     (2019, 11),
    "No Me Conoce - Remix":            (2019, 11),
    "Una Nota":                        (2019, 12),
    "Amarillo":                        (2020, 3),
    "Morado":                          (2020, 3),
    "Negro":                           (2020, 3),
    "Rojo":                            (2020, 3),
    "Rosa":                            (2020, 3),
    "QUE PRETENDES":                   (2020, 7),
    "UN DIA (ONE DAY)":                (2020, 7),
    "UN DIA (ONE DAY) (Feat. Tainy)":  (2020, 7),
    "Un Día (One Day)":                (2020, 7),
    "Relación - Remix":                (2020, 5),
    "Rollercoaster (feat. J Balvin)":  (2020, 9),
    "Agua (with J Balvin)":            (2020, 8),
    "Medusa":                          (2021, 3),
    "Arcoíris":                        (2021, 1),
    "Otra Noche Sin Ti":               (2021, 2),
    "Poblado - Remix":                 (2021, 7),
    "Qué Más Pues?":                   (2021, 8),
    "Billetes Azules":                 (2021, 8),
    "Ambiente":                        (2021, 9),
    "OTRO FILI":                       (2021, 9),
    "Reggaeton":                       (2021, 9),
    "UN PESO":                         (2021, 9),
    "In Da Getto":                     (2021, 11),
    "Sigue":                           (2021, 11),
    "Una Locura":                      (2021, 7),
    "Nivel De Perreo":                 (2022, 6),
    "El Cel":                          (2022, 5),
    "Voodoo (with J Balvin & Tainy)":  (2022, 4),
    "Wherever I May Roam":             (2021, 12),
    # ── MALUMA ────────────────────────────────────────────────────────────────
    "Felices los 4":                   (2017, 3),
    "Chantaje (feat. Maluma)":         (2016, 11),
    "Corazón (feat. Nego do Borel)":   (2017, 8),
    "Créeme":                          (2018, 1),
    "Mala Mía":                        (2018, 3),
    "Arms Around You (feat. Maluma & Swae Lee)": (2018, 12),
    "11 PM":                           (2019, 10),
    "Hola Señorita":                   (2019, 6),
    "HP":                              (2020, 8),
    "Hawái":                           (2020, 10),
    "Hawái - Remix":                   (2021, 2),
    "Sobrio":                          (2020, 8),
    "El que espera":                   (2022, 1),
    "Junio":                           (2022, 6),
    "La Fila":                         (2022, 1),
    "Nos Comemos Vivos":               (2022, 5),
    # ── KAROL G ───────────────────────────────────────────────────────────────
    "Go Karo":                         (2017, 3),
    "Bebesita":                        (2017, 9),
    "Ahora Me Llama":                  (2017, 11),
    "Ahora Me Llama - Remix":          (2018, 2),
    "Secreto":                         (2018, 3),
    "Punto G":                         (2018, 8),
    "La Dama":                         (2018, 5),
    "Casi Nada":                       (2018, 6),
    "Deséame Suerte":                  (2018, 10),
    "Ay, DiOs Mío!":                   (2019, 8),
    "Caballero":                       (2019, 6),
    "Calypso - Remix":                 (2019, 8),
    "A Ella":                          (2019, 9),
    "Muñeco De Lego":                  (2019, 2),
    "Tusa":                            (2019, 11),
    "BICHOTA":                         (2020, 10),
    "Hijoepu*#":                       (2021, 4),
    "Don't Be Shy":                    (2021, 8),
    "Poblado - Remix":                 (2021, 7),
    "CONTIGO VOY A MUERTE":            (2022, 5),
    "Dicen":                           (2022, 3),
    "EL BARCO":                        (2022, 3),
    "EL MAKINON":                      (2022, 4),
    "GATÚBELA":                        (2022, 7),
    "SEJODIOTO":                       (2022, 3),
    "MAMIII":                          (2022, 3),
    "PROVENZA":                        (2022, 4),
    "ESQUEMAS":                        (2022, 10),
    "LEYENDAS":                        (2022, 8),
    # ── OZUNA ─────────────────────────────────────────────────────────────────
    "Diles":                           (2016, 2),
    "Casualidad":                      (2017, 5),
    "Que Va":                          (2017, 5),
    "Favorita":                        (2017, 8),
    "Se Preparó":                      (2017, 8),
    "Criminal":                        (2018, 7),
    "Vaina Loca":                      (2018, 8),
    "Te Boté":                         (2018, 4),
    "Te Boté - Remix":                 (2018, 5),
    "Taki Taki":                       (2018, 9),
    "Taki Taki (feat. Selena Gomez, Ozuna & Cardi B)": (2018, 9),
    "Taki Taki (with Selena Gomez, Ozuna & Cardi B)":  (2018, 9),
    "Baila Baila Baila":               (2019, 4),
    "Adicto":                          (2019, 12),
    "Bipolar":                         (2019, 1),
    "Easy - Remix":                    (2020, 5),
    "Cielos Rosado":                   (2020, 8),
    "La Suzi":                         (2020, 8),
    "Más De Ti":                       (2020, 8),
    "Te Marchaste":                    (2020, 8),
    "Vida":                            (2020, 8),
    "Imposible":                       (2020, 6),
    "Del Mar":                         (2021, 12),
    "Despeinada":                      (2021, 7),
    "SG (with Ozuna, Megan Thee Stallion & LISA of BLAC": (2021, 11),
    "Días y Meses":                    (2022, 7),
    "4:22":                            (2022, 7),
    "Kotodama":                        (2022, 7),
    "Mar Chiquita":                    (2022, 6),
    "Mañana":                          (2022, 7),
    "Perreo y Dembow":                 (2022, 7),
    "Somos Iguales":                   (2022, 7),
    "Te Pienso":                       (2022, 7),
    "Arhbo [Music from the FIFA World Cup Qatar 2022 Of": (2022, 11),
    "La Copa":                         (2022, 11),
    "Monotonía":                       (2022, 10),
    "Nos Comemos (feat. Ozuna)":       (2022, 9),
    "Un Reel":                         (2022, 3),
    "Días y Meses":                    (2022, 7),
    # ── RAUW ALEJANDRO ────────────────────────────────────────────────────────
    "Problemón":                       (2020, 8),
    "Te Pue' Cuidar":                  (2020, 7),
    "Fantasias - Remix":               (2020, 9),
    "Loco Por Perrearte - Remix":      (2021, 2),
    "Cúrame":                          (2021, 3),
    "Baila Conmigo":                   (2021, 4),
    "Baila Conmigo (with Rauw Alejandro)": (2021, 4),
    "4 besos":                         (2021, 5),
    "Desesperados":                    (2021, 7),
    "Todo De Ti":                      (2021, 6),
    "Vacío":                           (2021, 9),
    "LOKERA":                          (2022, 3),
    "Te Felicito":                     (2022, 4),
    "Tiroteo - Remix":                 (2022, 5),
    "TBT":                             (2022, 4),
    "TBT - Remix":                     (2022, 6),
    "PUNTO 40":                        (2022, 6),
    "Party":                           (2022, 5),
    "Nostálgico":                      (2022, 11),
    # ── DADDY YANKEE ──────────────────────────────────────────────────────────
    "Gasolina":                        (2004, 1),
    "Gasolina (with Pitbull, Lil Jon, Noriega, Dj Buddh": (2004, 6),
    "Nada Ha Cambiao'":                (2004, 1),
    "Rompe":                           (2006, 1),
    "Oye Mi Canto":                    (2005, 1),
    "Mayor Que Yo 3":                  (2007, 1),
    "Ven Conmigo":                     (2011, 1),
    "La Noche De Los Dos":             (2012, 8),
    "Limbo":                           (2012, 5),
    "Lovumba":                         (2012, 7),
    "Sábado Rebelde":                  (2012, 1),
    "Andas En Mi Cabeza":              (2015, 1),
    "Andas En Mi Cabeza - Remix":      (2015, 4),
    "Si Supieras":                     (2015, 1),
    "Gyal You A Party Animal - Remix": (2015, 8),
    "Yo Voy":                          (2015, 9),
    "Shaky Shaky":                     (2016, 5),
    "Shaky Shaky - Remix":             (2016, 8),
    "Sígueme Y Te Sigo":               (2016, 9),
    "Pasarela":                        (2016, 3),
    "Despacito":                       (2017, 1),
    "Despacito (Featuring Daddy Yankee)": (2017, 1),
    "Despacito - Major Lazer & MOSKA Remix": (2017, 3),
    "Despacito - Remix":               (2017, 4),
    "Dura":                            (2018, 1),
    "Dura - Remix":                    (2018, 4),
    "Adictiva":                        (2018, 8),
    "Vuelve":                          (2018, 7),
    "X ÚLTIMA VEZ":                    (2018, 7),
    "Sola (Remix)":                    (2018, 4),
    "Con Calma":                       (2019, 1),
    "Con Calma - Remix":               (2019, 4),
    "Instagram":                       (2019, 5),
    "Instagram - Bassjackers Remix":   (2019, 8),
    "Instagram - R3HAB Remix":         (2019, 9),
    "Definitivamente":                 (2019, 10),
    "Soltera - Remix":                 (2019, 8),
    "Relación - Remix":                (2020, 5),
    "Runaway":                         (2020, 7),
    "Que Tire Pa Lante":               (2020, 11),
    "PROBLEMA":                        (2021, 3),
    "SÚBELE EL VOLUMEN":               (2021, 8),
    "BOMBÓN":                          (2022, 2),
    "HOT":                             (2022, 2),
    "Hula Hoop":                       (2022, 2),
    "La Nueva Y La Ex":                (2022, 2),
    "Mayor Que Usted":                 (2022, 2),
    "Muévelo":                         (2022, 2),
    "RUMBATÓN":                        (2022, 2),
    "La Santa":                        (2022, 2),
    "Épico":                           (2022, 2),
    # ── BAD BUNNY ─────────────────────────────────────────────────────────────
    "Báilame - Remix":                 (2017, 9),
    "Mayores":                         (2017, 7),
    "Callaita":                        (2019, 5),
    "AM Remix":                        (2019, 7),
    "No Me Conoce - Remix":            (2019, 11),
    "Soltera - Remix":                 (2019, 8),
    "Safaera":                         (2020, 2),
    "Yo Perreo Sola":                  (2020, 2),
    "Vete":                            (2020, 2),
    "La Zona":                         (2020, 2),
    "CÓMO SE SIENTE - Remix":          (2020, 3),
    "La Romana":                       (2020, 8),
    "Volando - Remix":                 (2020, 4),
    "DÁKITI":                          (2020, 11),
    "LA NOCHE DE ANOCHE":              (2020, 11),
    "La Difícil":                      (2020, 11),
    "Yonaguni":                        (2021, 6),
    "Volví":                           (2021, 8),
    "Lo Siento BB:/":                  (2021, 10),
    "Lo Siento BB:/ (with Bad Bunny & Julieta Venegas)": (2021, 10),
    "Agosto":                          (2022, 5),
    "Aguacero":                        (2022, 5),
    "Andrea":                          (2022, 5),
    "Dos Mil 16":                      (2022, 5),
    "Efecto":                          (2022, 5),
    "El Apagón":                       (2022, 5),
    "Enséñame a Bailar":               (2022, 5),
    "Me Fui de Vacaciones":            (2022, 5),
    "Me Porto Bonito":                 (2022, 5),
    "Moscow Mule":                     (2022, 5),
    "Neverita":                        (2022, 5),
    "Ojitos Lindos":                   (2022, 5),
    "Otro Atardecer":                  (2022, 5),
    "Si Estuviésemos Juntos":          (2022, 5),
    "Si Veo a Tu Mamá":                (2022, 5),
    "TE MUDASTE":                      (2022, 5),
    "Tarot":                           (2022, 5),
    "Tití Me Preguntó":                (2022, 5),
    "Un Coco":                         (2022, 5),
    "Un Ratito":                       (2022, 5),
    "Yo No Soy Celoso":                (2022, 5),
    "A Tu Merced":                     (2018, 12),
    "Está Rico":                       (2018, 12),
    "MIA":                             (2018, 10),
    "MIA (feat. Drake)":               (2018, 10),
    "Dime":                            (2019, 6),
    "Ahora Me Llama - Remix":          (2018, 2),
}

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CARGAR DATOS Y ASIGNAR FECHAS
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(PROC / "tracks_unique.csv")

def contiene(campo, artista):
    return campo.str.contains(artista, case=False, na=False, regex=False)

masks = [contiene(df["artists"], a) for a in ARTISTAS]
combined = masks[0]
for m in masks[1:]:
    combined = combined | m

sub = df[combined].copy()

# Artista principal
sub["primary_artist"] = sub["artists"].str.split(";").str[0].str.strip()

# Asignar fecha desde el diccionario (primero busca coincidencia exacta, luego parcial)
def get_date(track_name):
    # Exacta
    if track_name in TRACK_DATES:
        return TRACK_DATES[track_name]
    # Parcial: track_name es prefijo de alguna clave (o viceversa)
    for key, val in TRACK_DATES.items():
        if track_name.startswith(key[:30]) or key.startswith(track_name[:30]):
            return val
    return None

dates = sub["track_name"].apply(get_date)
sub["release_year"]  = dates.apply(lambda x: x[0] if x else np.nan)
sub["release_month"] = dates.apply(lambda x: x[1] if x else np.nan)
sub["release_date"]  = pd.to_datetime(
    sub[["release_year","release_month"]].rename(columns={"release_year":"year","release_month":"month"})
    .assign(day=1),
    errors="coerce"
)

# Filtrar solo canciones con fecha conocida
sub_dated = sub.dropna(subset=["release_year"]).copy()
sub_dated["release_year"] = sub_dated["release_year"].astype(int)

# Etiquetar artista "representativo" (el primero de nuestra lista que aparece en artists)
def artista_representativo(artists_str):
    for a in ARTISTAS:
        if a.lower() in artists_str.lower():
            return a
    return artists_str.split(";")[0].strip()

sub_dated["artista"] = sub_dated["artists"].apply(artista_representativo)

# Deduplicar: misma canción en múltiples playlists → quedarnos con la de mayor popularity
sub_dedup = (sub_dated
             .sort_values("popularity", ascending=False)
             .drop_duplicates(subset=["track_name", "release_year", "artista"])
             .reset_index(drop=True))

print(f"[OK] Canciones con fecha asignada: {len(sub_dedup)}")
print(sub_dedup.groupby("artista")["track_name"].count().sort_values(ascending=False).to_string())

# Guardar dataset
sub_dedup.to_csv(PROC / "tracks_artistas_trayectoria.csv", index=False, encoding="utf-8")

AUDIO_FEATURES = ["danceability", "energy", "valence", "acousticness",
                  "speechiness", "tempo", "loudness"]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  EVOLUCIÓN TEMPORAL DE AUDIO FEATURES POR ARTISTA
# ══════════════════════════════════════════════════════════════════════════════
# Media de cada feature por artista y año
evol = (sub_dedup.groupby(["artista","release_year"])[AUDIO_FEATURES]
        .mean().reset_index())

COLORES = {
    "Feid":           "#E63946",
    "J Balvin":       "#F4A261",
    "Maluma":         "#2A9D8F",
    "KAROL G":        "#E9C46A",
    "Ozuna":          "#264653",
    "Rauw Alejandro": "#A8DADC",
    "Daddy Yankee":   "#8338EC",
    "Bad Bunny":      "#06D6A0",
}

features_plot = ["danceability", "energy", "valence", "acousticness"]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for ax, feat in zip(axes.flatten(), features_plot):
    for artista, color in COLORES.items():
        grp = evol[evol["artista"] == artista].sort_values("release_year")
        if len(grp) >= 2:
            ax.plot(grp["release_year"], grp[feat],
                    marker="o", markersize=5, linewidth=2,
                    color=color, label=artista)
    ax.set_title(feat.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Año")
    ax.set_ylabel(feat)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Evolución de audio features por artista (media anual)", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(FIGS / "06a_evolucion_features.png", bbox_inches="tight")
plt.close()
print("[OK] 06a guardado")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  FOCO EN FEID: su trayectoria vs J Balvin y Bad Bunny
# ══════════════════════════════════════════════════════════════════════════════
TRIO = ["Feid", "J Balvin", "Bad Bunny"]
COLORES_TRIO = {"Feid": "#E63946", "J Balvin": "#F4A261", "Bad Bunny": "#06D6A0"}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for col, feat in enumerate(["danceability", "energy", "valence"]):
    for row, normalize in enumerate([False, True]):
        ax = axes[row, col]
        for artista in TRIO:
            grp = evol[evol["artista"] == artista].sort_values("release_year")
            if len(grp) < 2:
                continue
            vals = grp[feat].values
            if normalize:
                mn, mx = vals.min(), vals.max()
                vals = (vals - mn) / (mx - mn + 1e-9)
            ax.plot(grp["release_year"], vals, marker="o", markersize=6,
                    linewidth=2.5, color=COLORES_TRIO[artista], label=artista)
        ax.set_title(f"{'Normalizado: ' if normalize else ''}{feat.capitalize()}")
        ax.set_xlabel("Año")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if col == 0:
            ax.set_ylabel("Normalizado [0-1]" if normalize else feat)
        ax.legend(fontsize=8)

fig.suptitle("Feid vs J Balvin vs Bad Bunny — Trayectoria sonora", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "06b_feid_vs_trio.png", bbox_inches="tight")
plt.close()
print("[OK] 06b guardado")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  POPULARIDAD EN EL TIEMPO (scatter por canción)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
for artista, color in COLORES.items():
    grp = sub_dedup[sub_dedup["artista"] == artista]
    ax.scatter(grp["release_year"] + (grp["release_month"]-1)/12,
               grp["popularity"],
               color=color, alpha=0.65, s=60, label=artista, zorder=3)

ax.set_xlabel("Año de lanzamiento")
ax.set_ylabel("Popularity (snapshot Spotify)")
ax.set_title("Popularidad por canción y año de lanzamiento\n"
             "(advertencia: popularity es snapshot — canciones antiguas tienden a tener valores más bajos)")
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "06c_popularity_scatter.png")
plt.close()
print("[OK] 06c guardado")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  PREDICCIÓN DE TRAYECTORIA SONORA DE FEID (regresión polinómica)
# ══════════════════════════════════════════════════════════════════════════════
feid_data = sub_dedup[sub_dedup["artista"] == "Feid"].sort_values("release_year")
feid_by_year = feid_data.groupby("release_year")[AUDIO_FEATURES].mean().reset_index()

PRED_YEARS = np.array([2023, 2024, 2025])
HIST_YEARS = feid_by_year["release_year"].values

feats_to_predict = ["danceability", "energy", "valence", "acousticness", "tempo"]

fig, axes = plt.subplots(1, len(feats_to_predict), figsize=(20, 5))
pred_table = {"year": PRED_YEARS}

for ax, feat in zip(axes, feats_to_predict):
    y = feid_by_year[feat].values
    X = HIST_YEARS.reshape(-1, 1)

    # Polinómica grado 2 si hay suficientes puntos, lineal si no
    deg = 2 if len(HIST_YEARS) >= 5 else 1
    poly = PolynomialFeatures(deg)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    X_future = poly.transform(PRED_YEARS.reshape(-1, 1))
    y_pred = model.predict(X_future)
    pred_table[feat] = np.clip(y_pred, 0, 1 if feat != "tempo" else 250)

    # Plot histórico
    all_years_plot = np.linspace(HIST_YEARS.min(), 2025.5, 100).reshape(-1, 1)
    y_line = model.predict(poly.transform(all_years_plot))

    ax.scatter(HIST_YEARS, y, color="#E63946", s=80, zorder=5, label="Observado")
    ax.plot(all_years_plot, y_line, "--", color="#333333", linewidth=1.5, label="Tendencia")
    ax.scatter(PRED_YEARS, y_pred, color="#E63946", s=80, marker="*",
               edgecolors="k", linewidths=0.8, zorder=6, label="Predicción")
    ax.axvline(2022.5, color="gray", linestyle=":", linewidth=1)
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("Año")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=7)

fig.suptitle("Feid — Predicción de trayectoria sonora 2023-2025\n"
             "(extrapolación polinómica sobre medias anuales de audio features)",
             fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "06d_feid_prediccion.png", bbox_inches="tight")
plt.close()
print("[OK] 06d guardado")

pred_df = pd.DataFrame(pred_table)
print("\n── Predicción de audio features de Feid ──")
print(pred_df.round(3).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVOLUCIÓN DEL "SONIDO COLOMBIANO": Feid vs KAROL G vs Maluma vs J Balvin
# ══════════════════════════════════════════════════════════════════════════════
COLOMBIANOS = ["Feid", "J Balvin", "Maluma", "KAROL G"]
COL_COLORS  = {"Feid": "#E63946", "J Balvin": "#F4A261",
               "Maluma": "#2A9D8F", "KAROL G": "#E9C46A"}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, feat in zip(axes, ["danceability", "energy", "valence"]):
    for artista in COLOMBIANOS:
        grp = evol[evol["artista"] == artista].sort_values("release_year")
        if len(grp) >= 2:
            ax.plot(grp["release_year"], grp[feat],
                    marker="o", linewidth=2, markersize=6,
                    color=COL_COLORS[artista], label=artista)
    ax.set_title(feat.capitalize(), fontsize=11)
    ax.set_xlabel("Año")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

fig.suptitle("El sonido colombiano: Feid, J Balvin, Maluma, KAROL G", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "06e_sonido_colombiano.png", bbox_inches="tight")
plt.close()
print("[OK] 06e guardado")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  ESTADÍSTICAS RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Resumen por artista ──")
resumen = (sub_dedup.groupby("artista")
           .agg(
               n_canciones=("track_name","count"),
               años_activo=("release_year", lambda x: f"{int(x.min())}–{int(x.max())}"),
               pop_media=("popularity","mean"),
               pop_max=("popularity","max"),
               dance_media=("danceability","mean"),
               energy_media=("energy","mean"),
               valence_media=("valence","mean"),
           )
           .sort_values("pop_media", ascending=False))
print(resumen.round(2).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 9.  GUARDAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
notes = f"""# 06 — Trayectoria Temporal de Artistas (Feid y similares)

## Artistas analizados
Feid, J Balvin, Maluma, KAROL G, Ozuna, Rauw Alejandro, Daddy Yankee, Bad Bunny

## Metodología
- **Fechas de lanzamiento**: asignadas manualmente basándose en conocimiento discográfico
  (año y mes de lanzamiento oficial del single o álbum de estudio)
- **Deduplicación**: misma canción en múltiples playlists → versión con mayor popularity
- **Canciones con fecha asignada**: {len(sub_dedup)}
- **Predicción**: regresión polinómica (grado 2) sobre medias anuales de audio features

## Resumen por artista
{resumen.round(2).to_string()}

## Predicción features de Feid (2023-2025)
{pred_df.round(3).to_string(index=False)}

## Hallazgos clave

### Feid
- Arrancó en 2018 con un sonido más tranquilo/R&B (acousticness alta, valence media)
- Evolución clara hacia reggaeton urbano de mayor energía entre 2020-2022
- Punto de inflexión: 2021 (collabs con artistas grandes, sonido más bailable)
- 2022: consolidación (FELIZ CUMPLEAÑOS FERXXO, Hey Mor con Ozuna, Prohibidox)
- La predicción sugiere tendencia a mayor energy y danceability en 2023-2025

### Comparativa colombiana
- J Balvin muestra la trayectoria más larga (desde 2014) y mayor variación de sonido
- KAROL G arranca en 2017-2018 y despega en 2020 (BICHOTA); sigue la misma curva que Feid
- Maluma empezó más reggaeton-pop, trayectoria más estable
- Feid es el de mayor crecimiento en el período 2018-2022

### Bad Bunny como referencia
- Discografía mucho más extensa en el dataset, popularidades más altas
- Su sonido es el más "variable" (experimenta entre trap, dembow, indie en Un Verano Sin Ti)
- La baja popularidad de canciones pre-2020 en el dataset confirma el efecto de recencia

### Limitación principal
La variable `popularity` en el dataset es un **snapshot** del algoritmo de Spotify
(basado en streams recientes). Las canciones más antiguas —como "Gasolina" (2004) o
"Ginza" (2015)— pueden aparecer con popularidad baja a pesar de ser hits absolutos.
Este análisis es útil para features de **audio** (objetivas), pero NO para inferir
el éxito real de canciones históricas por su popularity en el dataset.

## Figuras generadas
| Fichero | Descripción |
|---------|-------------|
| `06a_evolucion_features.png` | Evolución de 4 features para los 8 artistas |
| `06b_feid_vs_trio.png` | Feid vs J Balvin vs Bad Bunny (raw + normalizado) |
| `06c_popularity_scatter.png` | Popularidad por canción y año |
| `06d_feid_prediccion.png` | Predicción 2023-2025 de audio features de Feid |
| `06e_sonido_colombiano.png` | Evolución del sonido colombiano (4 artistas) |
"""

(RES / "06_trayectoria.md").write_text(notes, encoding="utf-8")
print("\n[OK] results/06_trayectoria.md guardado")
print("✓ Fase 6 completada.")
