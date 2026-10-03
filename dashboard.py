"""
dashboard.py : Dashboard Streamlit de segmentation de tumeur cérébrale (BraTS 2020)

Lancer :  streamlit run dashboard.py
À placer dans le même dossier que data_utils.py et inference.py.

Prérequis : créer un fichier .streamlit/config.toml (voir ci-dessous) :

    [theme]
    base = "light"
    primaryColor = "#0b1e3f"
    backgroundColor = "#ffffff"
    secondaryBackgroundColor = "#ffffff"
    textColor = "#0a0a0a"
    font = "serif"
"""

import os
import re
import tempfile

import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes, find_contours
from skimage.exposure import equalize_adapthist
from skimage.filters import unsharp_mask

from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, Orientationd, Spacingd, NormalizeIntensityd, CropForegroundd, EnsureTyped,
)

from data_utils import MODALITIES, LoadMultiModalImaged, wt_tc_et_to_brats_labels
from inference import build_model, keep_largest_component
from export_utils import get_affine, export_segmentation

# ------------------------------------------------------------ paramètres fixes
MODEL_PATH = "./runs/best_metric_model.pth"
PATCH = 128
POSTPROCESS = True
MASK_ALPHA = 1.0     # 1.0 = couleurs pleines (non transparentes)
STEP_3D = 1
# Dossier où une copie du masque généré est enregistrée au clic sur le bouton de téléchargement
# (créé au besoin) : test/mask_test, à côté de dashboard.py. Modifiable ici.
MASK_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "mask_test")
ENH = {"on": True, "clahe": False, "sharp": 0.8, "smooth": 0.0, "gamma": 1.0, "zoom": 3,
       "slice_window": True}

st.set_page_config(page_title="BraTS · Segmentation cérébrale", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ thème
NAVY   = "#0b1e3f"
NAVY_2 = "#112a52"
NAVY_3 = "#1e3a8a"
INK    = "#0a0a0a"
PAPER  = "#ffffff"
LINE   = "#d7dde8"
MUTED  = "#5a6478"
RED, GREEN, BLUE = "#dc2626", "#16a34a", "#2563eb"

FONT = '"Calisto MT", Calisto, "Book Antiqua", Palatino, Georgia, serif'

# Palette tumeur : rouge / jaune / bleu
COLOR_WT_EDEMA  = "#dc2626"  # Rouge  (Œdème / tumeur entière)
COLOR_TC_CORE   = "#facc15"  # Jaune  (Noyau nécrotique)
COLOR_ET_ACTIVE = "#2563eb"  # Bleu   (Tumeur rehaussée)

# Conversion RGB pour le superposeur 2D
LABEL_COLORS = {
    1: (250, 204, 21),   # Label 1 (noyau nécrotique) : Jaune
    2: (220, 38, 38),    # Label 2 (œdème) : Rouge
    4: (37, 99, 235),    # Label 4 (tumeur rehaussée) : Bleu
}
RGB = {k: tuple(c / 255 for c in v) for k, v in LABEL_COLORS.items()}
ORIENT = {0: ("P", "A", "I", "S"), 1: ("G", "D", "I", "S"), 2: ("G", "D", "P", "A")}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Calisto MT", "Book Antiqua", "Palatino Linotype", "DejaVu Serif"]

st.markdown(f"""
<style>
:root {{
  color-scheme: light !important;
  --navy: {NAVY};
  --navy-2: {NAVY_2};
  --navy-3: {NAVY_3};
  --ink: {INK};
  --paper: {PAPER};
  --line: {LINE};
  --muted: {MUTED};
}}

/* ============ base : fond blanc, texte noir, Calisto MT ============ */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], [data-testid="stBottom"], section.main {{
  background: #ffffff !important;
  background-color: #ffffff !important;
}}
.stApp, .stApp *, .stApp *::before, .stApp *::after {{
  font-family: {FONT} !important;
  color: {INK} !important;
}}
.stApp h1, .stApp h2, .stApp h3,
h1.title, h2.restitle, .brand-title {{
  font-family: {FONT} !important;
  color: {NAVY} !important;
  letter-spacing: -0.01em;
}}
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
.block-container {{ padding-top: 5rem !important; padding-bottom: 3rem !important;
                    max-width: 1320px !important; }}

/* ============ entête ============ */
.brand {{ display: flex; align-items: center; gap: 18px; margin: 0 0 6px 0; }}
.brand-mark {{
  width: 54px; height: 54px; border-radius: 14px;
  background: linear-gradient(135deg, {NAVY} 0%, {NAVY_3} 100%);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 10px 24px -10px rgba(11,30,63,.55); flex-shrink: 0;
}}
.brand-mark span {{
  color: #ffffff !important; font-family: {FONT} !important;
  font-weight: 800; font-size: 1.6rem; letter-spacing: -0.02em; line-height: 1;
}}
.brand-text {{ display: flex; flex-direction: column; }}
.brand-kicker {{
  font-size: .74rem; font-weight: 700; letter-spacing: .22em;
  text-transform: uppercase; color: {MUTED} !important; margin-bottom: 2px;
}}
.brand-title {{
  font-size: 2.05rem; font-weight: 800; line-height: 1.05; margin: 0; color: {NAVY} !important;
}}
.brand-sub {{
  font-size: .95rem; color: {MUTED} !important; margin-top: 6px; font-weight: 600;
}}
.rule {{
  height: 1px;
  background: linear-gradient(90deg, {NAVY} 0%, {NAVY} 30%, {LINE} 30%, {LINE} 100%);
  margin: 20px 0 26px 0;
}}

/* ============================================================
   SELECTBOX — fond BLANC, texte NOIR, Calisto MT, bordure navy
   ============================================================ */
.stApp [data-testid="stSelectbox"] {{ max-width: 560px !important; }}
.stApp [data-testid="stSelectbox"] > label,
.stApp [data-testid="stSelectbox"] label p {{
  color: {NAVY} !important;
  font-weight: 700 !important;
  font-size: .95rem !important;
}}
.stApp div[data-baseweb="select"] > div,
.stApp div[data-baseweb="select"] > div > div,
.stApp div[data-baseweb="select"] div[role="button"],
.stApp [data-baseweb="select"] {{
  background: #ffffff !important;
  background-color: #ffffff !important;
  border: 1.5px solid {NAVY} !important;
  border-radius: 10px !important;
  color: {INK} !important;
  min-height: 42px !important;
  box-shadow: 0 1px 2px rgba(11,30,63,.05) !important;
  font-family: {FONT} !important;
}}
.stApp div[data-baseweb="select"] > div:hover {{
  border-color: {NAVY_3} !important;
  box-shadow: 0 6px 16px -10px rgba(11,30,63,.45) !important;
}}
.stApp div[data-baseweb="select"] div,
.stApp div[data-baseweb="select"] span,
.stApp div[data-baseweb="select"] input,
.stApp div[data-baseweb="select"] p {{
  background: transparent !important;
  background-color: transparent !important;
  color: {INK} !important;
  -webkit-text-fill-color: {INK} !important;
  font-family: {FONT} !important;
  font-weight: 600 !important;
}}
.stApp div[data-baseweb="select"] svg {{
  fill: {NAVY} !important;
  color: {NAVY} !important;
}}
.stApp [data-baseweb="popover"],
.stApp [data-baseweb="popover"] > div,
.stApp [data-baseweb="popover"] ul,
.stApp [role="listbox"] {{
  background: #ffffff !important;
  background-color: #ffffff !important;
  border: 1.5px solid {NAVY} !important;
  border-radius: 10px !important;
}}
.stApp [role="option"] {{
  background: #ffffff !important;
  color: {INK} !important;
  font-family: {FONT} !important;
  font-weight: 600 !important;
}}
.stApp [role="option"]:hover {{
  background: #eef2f8 !important;
}}
.stApp [role="option"][aria-selected="true"] {{
  background: {NAVY} !important;
  color: #ffffff !important;
}}
.stApp [role="option"][aria-selected="true"] * {{
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}}

/* ============================================================
   ONGLETS — 3 BOÎTES BLEU MARINE, écriture blanche
   ============================================================ */
.stApp [role="tablist"],
.stApp [data-testid="stTabs"] [data-baseweb="tab-list"],
.stApp [data-baseweb="tab-list"] {{
  background: transparent !important;
  border: none !important;
  border-bottom: none !important;
  box-shadow: none !important;
  gap: 10px !important;
  padding: 4px 0 20px 0 !important;
  margin: 0 !important;
  width: 100% !important;
  max-width: 560px !important;
  display: flex !important;
}}
.stApp button[role="tab"],
.stApp [role="tablist"] button,
.stApp [data-testid="stTab"],
.stApp button[data-baseweb="tab"] {{
  background: {NAVY} !important;
  background-color: {NAVY} !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border: 1.5px solid {NAVY} !important;
  border-radius: 10px !important;
  padding: 0 12px !important;
  margin: 0 !important;
  flex: 1 1 0 !important;
  min-width: 0 !important;
  min-height: 42px !important;
  height: 42px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: 0 6px 16px -12px rgba(11,30,63,.55) !important;
  transition: all .18s ease !important;
}}
.stApp button[role="tab"] *,
.stApp [role="tablist"] button *,
.stApp [data-testid="stTab"] *,
.stApp button[data-baseweb="tab"] * {{
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  background: transparent !important;
  font-family: {FONT} !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: .015em !important;
}}
.stApp button[role="tab"]:hover,
.stApp [role="tablist"] button:hover,
.stApp [data-testid="stTab"]:hover {{
  background: {NAVY_2} !important;
  background-color: {NAVY_2} !important;
  border-color: {NAVY_2} !important;
  transform: translateY(-1px);
}}
.stApp button[role="tab"][aria-selected="true"],
.stApp [role="tablist"] button[aria-selected="true"],
.stApp [data-testid="stTab"][aria-selected="true"] {{
  background: {NAVY_3} !important;
  background-color: {NAVY_3} !important;
  border-color: {NAVY_3} !important;
  box-shadow: 0 10px 24px -12px rgba(30,58,138,.8) !important;
}}
.stApp [data-baseweb="tab-highlight"],
.stApp [data-baseweb="tab-border"],
.stApp [role="tablist"] hr,
.stApp [data-testid="stTabs"] hr {{
  display: none !important;
  background: transparent !important;
  border: none !important;
}}
.stApp button[role="tab"]::before,
.stApp button[role="tab"]::after,
.stApp [data-testid="stTab"]::before,
.stApp [data-testid="stTab"]::after {{
  display: none !important;
  background: transparent !important;
  border: none !important;
}}
.stApp [role="tabpanel"],
.stApp [data-baseweb="tab-panel"] {{
  background: #ffffff !important;
  padding-top: 8px !important;
  border: none !important;
}}

/* ============ champs, listes ============ */
.stApp [data-baseweb="input"],
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="base-input"],
.stApp [data-baseweb="textarea"] {{
  background: #ffffff !important;
  border: 1px solid {LINE} !important;
  border-radius: 10px !important;
}}
.stApp input, .stApp textarea {{
  color: {INK} !important;
  -webkit-text-fill-color: {INK} !important;
  background: #ffffff !important;
}}
.stApp [data-testid="stExpander"] {{
  background: #ffffff !important;
  border: 1px solid {LINE} !important;
  border-radius: 12px;
}}
.stApp [data-testid="stExpander"] details,
.stApp [data-testid="stExpander"] summary {{
  background: #ffffff !important;
}}

/* ============ cadre d'upload ============ */
.stApp [data-testid="stFileUploader"] section {{
  background: transparent !important; border: none !important; padding: 0 !important;
}}
.stApp [data-testid="stFileUploaderDropzone"] {{
  position: relative; min-height: 172px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(180deg, #ffffff 0%, #f7f9fd 100%) !important;
  border: 1.5px dashed {NAVY_3} !important;
  border-radius: 18px !important;
  cursor: pointer;
  transition: all .2s ease;
}}
.stApp [data-testid="stFileUploaderDropzone"]:hover {{
  background: linear-gradient(180deg, #f7f9fd 0%, #eef2f8 100%) !important;
  border-color: {NAVY} !important;
  box-shadow: 0 14px 34px -20px rgba(11,30,63,.35);
}}
.stApp [data-testid="stFileUploaderDropzoneInstructions"] {{ display: none !important; }}
.stApp [data-testid="stFileUploaderDropzone"]::before {{
  content: "Ajouter le dossier racine du patient BraTS";
  font-size: 1.22rem; font-weight: 700; letter-spacing: -0.005em;
  color: {NAVY}; font-family: {FONT};
}}
.stApp [data-testid="stFileUploaderDropzone"] button {{
  position: absolute !important; inset: 0; width: 100%; height: 100%;
  opacity: 0; cursor: pointer;
}}
.stApp [data-testid="stFileChip"], .stApp [data-testid="stFileUploaderFile"] {{
  background: #ffffff !important;
  border: 1px solid {LINE} !important;
  border-radius: 10px !important;
}}
.stApp [data-testid="stFileChip"] *, .stApp [data-testid="stFileUploaderFile"] *,
.stApp [data-testid*="FileChipName"], .stApp [data-testid*="FileUploaderFileName"] {{
  background: transparent !important; border: none !important;
  box-shadow: none !important; outline: none !important;
  color: {INK} !important; -webkit-text-fill-color: {INK} !important;
}}

/* ============ titres résultats ============ */
h2.restitle {{
  font-size: 1.75rem; font-weight: 800; margin: 6px 0 18px 0;
  color: {NAVY} !important; letter-spacing: -0.01em;
}}
.section-rule {{ height: 1px; background: {LINE}; margin: 34px 0 22px 0; }}

/* ============ analyse volumétrique ============ */
.stApp .vcard {{
  background: #ffffff; border: 1px solid {LINE}; border-radius: 14px;
  padding: 16px 18px 14px 18px; height: 100%;
  box-shadow: 0 10px 26px -20px rgba(11,30,63,.45);
}}
.stApp .vcard-bar {{ height: 5px; border-radius: 4px; margin-bottom: 12px; }}
.stApp .vcard-title {{ color: {NAVY} !important; font-weight: 800; font-size: 1.05rem; margin: 0 0 4px 0; }}
.stApp .vcard-vol {{ color: {NAVY} !important; font-weight: 800; font-size: 2rem; line-height: 1.1; margin: 2px 0 2px 0; }}
.stApp .vcard-pct {{ color: {MUTED} !important; font-size: .88rem; font-weight: 600; margin-bottom: 10px; }}
.stApp .vcard-def, .stApp .vcard-role {{ font-size: .92rem; line-height: 1.4; margin: 0 0 6px 0; }}
.stApp .vcard-def b, .stApp .vcard-role b {{ color: {NAVY} !important; }}
.stApp .vdisclaimer {{
  background: #ffffff; border: 1px solid {LINE}; border-radius: 12px;
  padding: 12px 18px; font-size: .9rem; line-height: 1.45; margin-top: 14px;
}}

/* expander : l'icône Material (texte parasite) est remplacée par un chevron CSS */
.stApp [data-testid="stExpander"] summary {{
  display: flex !important; align-items: center !important;
  justify-content: space-between !important; padding: 12px 18px !important;
}}
.stApp [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
.stApp [data-testid="stExpander"] summary svg {{ display: none !important; }}
.stApp [data-testid="stExpander"] summary::after {{
  content: ""; width: 8px; height: 8px; margin-left: 12px; flex-shrink: 0;
  border-right: 2px solid {NAVY}; border-bottom: 2px solid {NAVY};
  transform: rotate(45deg); transition: transform .18s ease;
}}
.stApp [data-testid="stExpander"] details[open] > summary::after {{ transform: rotate(225deg); }}
.stApp [data-testid="stExpander"] summary p {{
  font-weight: 700 !important; color: {NAVY} !important; margin: 0 !important;
}}
.stApp .veq-row {{
  display: flex; justify-content: space-between; align-items: center; gap: 16px;
  padding: 9px 2px; border-bottom: 1px solid {LINE};
}}
.stApp .veq-row:last-child {{ border-bottom: none; }}
.stApp .veq-name {{ font-weight: 800; color: {NAVY} !important; }}
.stApp .veq-def {{ color: {MUTED} !important; font-size: .88rem; }}
.stApp .veq-vol {{ font-weight: 800; color: {NAVY} !important; white-space: nowrap; }}
.stApp .veq-foot {{ color: {MUTED} !important; font-size: .85rem; margin-top: 10px; }}

/* ============ bloc d'état ============ */
.status {{
  background: #ffffff; border: 1px solid {LINE};
  border-left: 4px solid {NAVY};
  border-radius: 10px; padding: 12px 16px;
  font-weight: 700; font-size: .95rem; text-align: center;
  color: {NAVY} !important; margin: 0 0 10px 0;
}}
.status.warn {{ border-left-color: {RED}; }}

/* ============ boutons ============ */
.stApp .stButton > button {{
  background: #ffffff !important;
  border: 1.5px solid {NAVY} !important;
  border-radius: 12px !important;
  padding: 0.62rem 1.1rem !important;
  transition: all .18s ease;
}}
.stApp .stButton > button p {{
  font-family: {FONT} !important;
  font-weight: 700 !important; font-size: 1rem !important;
  color: {NAVY} !important;
}}
.stApp .stButton > button:hover {{
  background: #f2f6fc !important;
  border-color: {NAVY_3} !important;
  transform: translateY(-1px);
}}
.stApp .stButton > button:disabled {{ opacity: .45; transform: none; }}
.stApp .stButton > button:not(:disabled) {{
  background: linear-gradient(180deg, {NAVY_2} 0%, {NAVY} 100%) !important;
  border-color: {NAVY} !important;
}}
.stApp .stButton > button:not(:disabled) p,
.stApp .stButton > button:not(:disabled) * {{
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
}}
.stApp .stDownloadButton > button,
.stApp [data-testid="stDownloadButton"] > button {{
  background: linear-gradient(180deg, {NAVY_2} 0%, {NAVY} 100%) !important;
  border: 1.5px solid {NAVY} !important;
  border-radius: 12px !important;
  padding: 0.62rem 1.1rem !important;
  transition: all .18s ease;
}}
.stApp .stDownloadButton > button:hover,
.stApp [data-testid="stDownloadButton"] > button:hover {{
  border-color: {NAVY_3} !important; transform: translateY(-1px);
}}
.stApp .stDownloadButton > button *,
.stApp [data-testid="stDownloadButton"] > button * {{
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
  font-weight: 700 !important; font-size: 1rem !important;
}}

/* ============ curseurs ============ */
.stApp [data-testid="stSlider"] label p {{
  font-weight: 700 !important; font-size: .92rem !important;
  color: {NAVY} !important;
}}
.stApp [data-baseweb="slider"] [role="slider"] {{
  background: {NAVY} !important;
  border: 3px solid #ffffff !important;
  box-shadow: 0 0 0 1.5px {NAVY} !important;
}}

/* ============ captions ============ */
.stApp [data-testid="stCaptionContainer"] p,
.stApp .stCaption, .stApp small {{
  color: {MUTED} !important;
  font-size: .85rem !important;
}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- données
def parse_uploads(uploaded) -> dict:
    found = {}
    for up in uploaded or []:
        tokens = re.findall(r"(flair|t1ce|t1|t2|seg)(?=\.nii)", up.name.lower())
        if tokens and tokens[-1] in MODALITIES:
            found[tokens[-1]] = up
    return found


def save_uploads(found: dict) -> dict:
    tmp = tempfile.mkdtemp()
    files = {}
    for m, up in found.items():
        ext = ".nii.gz" if up.name.lower().endswith(".gz") else ".nii"
        p = os.path.join(tmp, f"{m}{ext}")
        with open(p, "wb") as f:
            f.write(up.getbuffer())
        files[m] = p
    return files


def get_transforms():
    keys = ["image"]
    return Compose([
        LoadMultiModalImaged(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        EnsureTyped(keys=keys),
    ])


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_model(model_path, device), device


def run_pipeline(files: dict, model_path: str, patch, postprocess: bool, patient_id: str = "patient"):
    model, device = load_model(model_path)
    item = {"image": [files[m] for m in MODALITIES]}
    sample = Dataset(data=[item], transform=get_transforms())[0]

    inputs = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = sliding_window_inference(inputs, tuple(patch), sw_batch_size=1,
                                           predictor=model, overlap=0.5)
        pred = (torch.sigmoid(out[0].float()) > 0.5).cpu().numpy().astype(np.float32)

    if postprocess:
        pred = np.stack([keep_largest_component(pred[c]) for c in range(3)])

    result = {
        "image": np.asarray(sample["image"].cpu()),
        "pred_ch": pred,
        "pred": wt_tc_et_to_brats_labels(pred),
    }

    # Exports NIfTI de la segmentation (espace d'origine + espace traité), calculés ICI,
    # avant le recadrage d'affichage ci-dessous, pour que l'affine reste exacte.
    result["exports"] = export_segmentation(
        result["pred"], get_affine(sample["image"]),
        (result["image"] != 0).any(0), files["flair"])
    result["patient_id"] = patient_id

    bb = np.argwhere((result["image"] != 0).any(0))
    sl = tuple(slice(a, b) for a, b in zip(bb.min(0), bb.max(0) + 1))
    full = (slice(None),) + sl
    result["image"], result["pred_ch"], result["pred"] = \
        result["image"][full], result["pred_ch"][full], result["pred"][sl]
    result["brain"] = (result["image"] != 0).any(0)
    return result


def window(gray):
    vals = gray[gray != 0]
    return (0.0, 1.0) if vals.size == 0 else tuple(np.percentile(vals, [1, 99.5]))


def enhance(g01, brain2d, enh):
    out = np.clip(g01, 0, 1)
    if enh["clahe"]:
        out = equalize_adapthist(out, clip_limit=0.01, kernel_size=max(32, min(out.shape) // 4))
    if enh["gamma"] != 1.0:
        out = out ** enh["gamma"]
    if enh["sharp"] > 0:
        out = unsharp_mask(out, radius=0.8 * enh["zoom"], amount=enh["sharp"])
    out = np.clip(out, 0, 1)
    out[~brain2d] = 0
    return out


def draw_slice(gray, mask, brain, axis, idx, vmin, vmax, title, enh, size=4.8):
    g = np.take(gray, idx, axis=axis).T
    m = np.take(mask, idx, axis=axis).T
    b = np.take(brain, idx, axis=axis).T

    # Fenêtrage propre à CETTE coupe : meilleur contraste, tissu bien visible
    if enh.get("slice_window") and b.sum() > 50:
        lo, hi = np.percentile(g[b], [0.5, 99.7])
        if hi - lo > 1e-6:
            vmin, vmax = lo, hi
    g01 = np.clip((g - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    g01[~b] = 0

    # Sur-échantillonnage bicubique (plus de pixels visibles) + masques à bords lisses
    z = int(enh["zoom"]) if enh["on"] and enh["zoom"] > 1 else 1
    if z > 1:
        g01 = np.clip(zoom(g01, z, order=3), 0, 1)
        b = zoom(b.astype(np.float32), z, order=1) > 0.5
        masks = {lab: gaussian_filter(zoom((m == lab).astype(np.float32), z, order=1), 0.6 * z)
                 for lab in RGB}
    else:
        masks = {lab: (m == lab).astype(np.float32) for lab in RGB}

    if enh["on"]:
        if enh.get("smooth", 0) > 0:
            g01 = gaussian_filter(g01, enh["smooth"] * z)
        if enh.get("clahe"):
            g01 = equalize_adapthist(g01, clip_limit=enh.get("clahe_clip", 0.005),
                                     kernel_size=max(32, min(g01.shape) // 6))
        if enh.get("gamma", 1.0) != 1.0:
            g01 = np.clip(g01, 0, 1) ** enh["gamma"]
        if enh["sharp"] > 0:                       # accentuation à l'échelle du voxel d'origine
            g01 = unsharp_mask(g01, radius=0.8 * z, amount=enh["sharp"])
        g01 = np.clip(g01, 0, 1)
    g01[~b] = 0

    # Recadrage serré sur le cerveau, puis mise au carré : les 3 vues ont la même taille
    coords = np.argwhere(b)
    if coords.size > 0:
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        pad = 4 * z
        r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
        r1, c1 = min(r1 + pad + 1, g01.shape[0]), min(c1 + pad + 1, g01.shape[1])
        sl = (slice(r0, r1), slice(c0, c1))
        g01, b = g01[sl], b[sl]
        masks = {k: v[sl] for k, v in masks.items()}
        h, w = g01.shape
        side = max(h, w)
        pw = ((( side - h) // 2, (side - h) - (side - h) // 2),
              (((side - w) // 2), (side - w) - (side - w) // 2))
        g01 = np.pad(g01, pw)
        masks = {k: np.pad(v, pw) for k, v in masks.items()}

    fig, ax = plt.subplots(figsize=(size, size), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("black")

    # Coupe IRM
    ax.imshow(g01, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="bilinear")

    # Masque semi-transparent (le plus interne est dessiné en dernier)
    overlay = np.zeros(g01.shape + (4,), dtype=np.float32)
    for lab in (2, 1, 4):
        if lab in RGB:
            overlay[masks[lab] > 0.5] = (*RGB[lab], MASK_ALPHA)
    ax.imshow(overlay, origin="lower", interpolation="nearest")

    # Contours nets
    for lab in (2, 1, 4):
        if lab in RGB and (masks[lab] > 0.5).any():
            ax.contour(masks[lab], levels=[0.5], colors=[RGB[lab]], linewidths=1.5, alpha=1.0)

    # Repères d'orientation (lisibles sur fond noir)
    left, right, bottom, top = ORIENT[axis]
    kw = dict(color="#e2e8f0", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.5, left, ha="left", va="center", **kw)
    ax.text(0.98, 0.5, right, ha="right", va="center", **kw)
    ax.text(0.5, 0.02, bottom, ha="center", va="bottom", **kw)
    ax.text(0.5, 0.98, top, ha="center", va="top", **kw)

    # Titre : '·' remplacé par '|' (évite le carré ⍰ de la police)
    clean_title = title.replace("·", "|").replace("TICE", "T1CE")
    ax.set_title(clean_title, color=NAVY, fontsize=11, fontweight="bold", pad=8)

    ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig


# ============================================================
# 3D — surface blanche translucide (intérieur vide) + sous-régions tumorales
# ============================================================
def iso_mesh(vol, level, color, name, opacity, step, smooth=0.0,
             ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.5,
             hoverinfo="skip", hovertemplate=None, legendgroup=None):
    """Surface 3D par marching cubes — supporte l'éclairage PBR et les infobulles."""
    vol = vol.astype(np.float32)
    if smooth > 0:
        vol = gaussian_filter(vol, smooth)
    if (vol > level).sum() < 20:
        return None
    verts, faces, _, _ = marching_cubes(np.pad(vol, 1), level, step_size=step)
    verts = (verts - 1).astype(np.float32)
    faces = faces.astype(np.int32)
    mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name,
        showlegend=False,
        hoverinfo=hoverinfo,
        lighting=dict(ambient=ambient, diffuse=diffuse,
                      specular=specular, roughness=roughness,
                      fresnel=0.1),
        lightposition=dict(x=1000, y=-1000, z=1000),
    )
    if hovertemplate is not None:
        mesh.update(hovertemplate=hovertemplate, hoverinfo="text")
    return mesh


BRAIN_WHITE = "#ffffff"          # blanc de la surface cérébrale


def brain_surface(res, step):
    """
    Surface cérébrale lisse, blanche et translucide. L'intérieur reste VIDE :
    la tumeur (et ses 3 sous-régions) se voit à travers la surface.
    """
    vol = gaussian_filter(res["brain"].astype(np.float32), sigma=1.2)   # lissage modéré
    if (vol > 0.5).sum() < 100:
        return None
    verts, faces, _, _ = marching_cubes(np.pad(vol, 1), level=0.5, step_size=max(step, 2))
    verts = (verts - 1).astype(np.float32)
    faces = faces.astype(np.int32)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=BRAIN_WHITE, opacity=0.25, flatshading=False,
        name="Cerveau", showlegend=False, hoverinfo="skip",
        lighting=dict(ambient=0.25, diffuse=0.75, specular=0.10, roughness=0.60, fresnel=0.1),
        lightposition=dict(x=1000, y=-1000, z=1200),
    )


def brain_outline(res, gap=10, level=0.12):
    """Lignes du périmètre du cerveau (contours de coupes successives), en bleu marine."""
    t1 = res["image"][MODALITIES.index("t1")].astype(np.float32)
    vol = np.pad(gaussian_filter(gaussian_filter(t1, 0.6), 0.3), 1)
    occ = vol > level
    xs, ys, zs = [], [], []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        idxs = np.where(occ.any(axis=others))[0]
        if idxs.size == 0:
            continue
        for k in range(int(idxs.min()) + gap // 2, int(idxs.max()), gap):
            for c in find_contours(np.take(vol, k, axis=axis), level):
                if len(c) < 20:
                    continue
                c = c[::2] - 1
                kk = np.full(len(c), k - 1, dtype=np.float32)
                if axis == 0:
                    px, py, pz = kk, c[:, 0], c[:, 1]
                elif axis == 1:
                    px, py, pz = c[:, 0], kk, c[:, 1]
                else:
                    px, py, pz = c[:, 0], c[:, 1], kk
                xs += px.tolist() + [None]
                ys += py.tolist() + [None]
                zs += pz.tolist() + [None]
    if not xs:
        return None
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=NAVY, width=2), opacity=0.75,
                        hoverinfo="skip", showlegend=False)


def tumor_parts(res):
    """
    Trois surfaces 3D construites sur des volumes DISJOINTS (nécrose, rehaussée) + l'enveloppe
    translucide de l'œdème : évite les surfaces superposées qui provoquaient des stries.
    """
    ch, lab = res["pred_ch"], res["pred"]
    return [
        dict(name="Œdème", vol=ch[1], level=0.35, smooth=1.4, color=COLOR_WT_EDEMA, opacity=0.18),
        dict(name="Noyau nécrotique", vol=(lab == 1), level=0.4, smooth=1.5, color=COLOR_TC_CORE, opacity=1.0),
        dict(name="Tumeur rehaussée", vol=(lab == 4), level=0.5, smooth=1.0, color=COLOR_ET_ACTIVE, opacity=1.0),
    ]


VOXEL_CM3 = 1e-3     # Spacingd rééchantillonne à 1 mm isotrope : 1 voxel = 1 mm³ = 0,001 cm³


def tumor_volumes(res) -> dict:
    """
    Volumes en cm³ (= mL). Compartiments exclusifs issus des labels BraTS
    (1 = nécrose, 2 = œdème, 4 = rehaussée) ; WT/TC/ET issus des canaux prédits (pour le détail).
    """
    ch, lab = res["pred_ch"], res["pred"]
    v = lambda a: float(np.count_nonzero(a)) * VOXEL_CM3
    out = dict(necrosis=v(lab == 1), edema=v(lab == 2), enhancing=v(lab == 4),
               wt=v(ch[1] > 0), tc=v(ch[0] > 0), et=v(ch[2] > 0))
    out["total"] = out["necrosis"] + out["edema"] + out["enhancing"]
    return out


def volume_section(res):
    """Analyse volumétrique : taille, définition et rôle des 3 compartiments de la tumeur."""
    vol = tumor_volumes(res)
    tot = vol["total"]
    pct = lambda x: f"{100.0 * x / tot:.0f} % de la lésion" if tot > 0 else "—"

    cards = [
        dict(color=COLOR_WT_EDEMA, title="Œdème péritumoral", vol=vol["edema"],
             definition="Gonflement du tissu cérébral autour de la tumeur, dû à une accumulation de liquide.",
             role="Contribue à l'effet de masse et à l'augmentation de la pression intracrânienne ; "
                  "sa taille montre l'étendue de l'impact sur le cerveau voisin."),
        dict(color=COLOR_TC_CORE, title="Noyau nécrotique", vol=vol["necrosis"],
             definition="Zone centrale de tissu tumoral mort ou non rehaussé (nécrose et tumeur non rehaussée).",
             role="Marque le cœur de la tumeur, où les cellules manquent d'oxygène ; "
                  "une nécrose étendue est souvent associée à des tumeurs de haut grade."),
        dict(color=COLOR_ET_ACTIVE, title="Tumeur rehaussée", vol=vol["enhancing"],
             definition="Zone tumorale active qui capte le produit de contraste (hypersignal T1CE), "
                        "signe d'une rupture de la barrière hémato-encéphalique.",
             role="Associée à une activité tumorale plus agressive ; "
                  "zone prioritaire pour la biopsie et la chirurgie."),
    ]

    st.markdown('<div class="section-rule"></div>'
                '<h2 class="restitle">Analyse volumétrique de la tumeur</h2>',
                unsafe_allow_html=True)

    for col, c in zip(st.columns(3), cards):
        col.markdown(
            f'<div class="vcard">'
            f'<div class="vcard-bar" style="background:{c["color"]}"></div>'
            f'<div class="vcard-title">{c["title"]}</div>'
            f'<div class="vcard-vol">{c["vol"]:.1f} cm³</div>'
            f'<div class="vcard-pct">{pct(c["vol"])}</div>'
            f'<div class="vcard-def"><b>Définition :</b> {c["definition"]}</div>'
            f'<div class="vcard-role"><b>Rôle :</b> {c["role"]}</div>'
            f'</div>',
            unsafe_allow_html=True)

    rows = [
        ("WT · Tumeur entière", "œdème + nécrose + rehaussée", vol["wt"]),
        ("TC · Noyau tumoral", "nécrose + rehaussée", vol["tc"]),
        ("ET · Tumeur rehaussée", "zone qui capte le contraste", vol["et"]),
    ]
    html = "".join(
        f'<div class="veq-row"><div><div class="veq-name">{n}</div>'
        f'<div class="veq-def">{d}</div></div><div class="veq-vol">{v:.1f} cm³</div></div>'
        for n, d, v in rows)

    with st.expander("Équivalence avec les régions BraTS (WT / TC / ET)"):
        st.markdown(f'<div class="veq">{html}</div>'
                    '<div class="veq-foot">Regroupements du challenge BraTS · 1 cm³ = 1 mL</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="vdisclaimer">Estimation automatique issue du modèle de segmentation, '
                'à but de recherche et d\'illustration. Elle dépend de la qualité de la segmentation '
                'et ne remplace pas l\'avis d\'un radiologue.</div>', unsafe_allow_html=True)


def figure_3d(res, step):
    """Surface cérébrale blanche translucide + 3 sous-régions tumorales."""
    traces = []

    # 1) surface cérébrale blanche translucide (intérieur vide)
    brain = brain_surface(res, step)
    if brain is not None:
        traces.append(brain)

    # 2) DÉSACTIVÉ : brain_outline() n'est plus tracé (plus de lignes de grille)

    # 3) 3 sous-régions tumorales, lissage modéré + PBR léger
    for p in tumor_parts(res):
        mesh = iso_mesh(
            p["vol"], level=p["level"],
            color=p["color"], name=p["name"],
            opacity=p["opacity"], step=1, smooth=p["smooth"],
            ambient=0.50, diffuse=0.80, specular=0.30, roughness=0.45,
        )
        if mesh is None:
            continue
        mesh.update(legendgroup=p["name"], hoverinfo="skip")
        traces.append(mesh)
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers", name=p["name"],
            legendgroup=p["name"], showlegend=True, hoverinfo="skip",
            marker=dict(size=9, color=p["color"], symbol="square"),
        ))

    fig = go.Figure(traces)
    ax = dict(visible=False)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=ax, yaxis=ax, zaxis=ax,
            bgcolor="white",
            camera=dict(eye=dict(x=1.0, y=-1.15, z=0.7)),
        ),
        paper_bgcolor="white",
        font=dict(color=INK, family=FONT),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode=False,
        height=640,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=0.02,
            font=dict(family=FONT, color=INK, size=14),
            bgcolor="rgba(255,255,255,0)",
        ),
    )
    return fig


def save_mask_copy(data: bytes, path: str, flag_key: str):
    """Callback du bouton de téléchargement : enregistre aussi une copie du masque sur le disque."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        st.session_state[flag_key] = path
    except Exception as e:                       # ne jamais faire planter l'application
        st.session_state[flag_key] = f"ERREUR::{e}"


def export_section(res):
    """Téléchargement du masque final (NIfTI, labels BraTS 0/1/2/4) dans la grille de l'IRM d'origine."""
    ex = res.get("exports")
    if not ex:
        return
    rid = st.session_state.get("res_id", 0)
    pid = res.get("patient_id", "patient")

    st.markdown('<div class="section-rule"></div>'
                '<h2 class="restitle">Exporter la segmentation</h2>',
                unsafe_allow_html=True)

    fname = f"{pid}_seg_pred.nii.gz"
    save_path = os.path.join(MASK_SAVE_DIR, fname)
    flag = f"saved_mask_{rid}"

    col, _ = st.columns(2)
    with col:
        if ex.get("original"):
            st.download_button("Télécharger le masque de segmentation (.nii.gz)", data=ex["original"],
                               file_name=fname, mime="application/gzip",
                               key=f"dl_orig_{rid}", use_container_width=True,
                               on_click=save_mask_copy, args=(ex["original"], save_path, flag))
            st.caption("Mêmes dimensions et même orientation que l'IRM d'origine : "
                       "se superpose directement dans ITK-SNAP ou 3D Slicer. "
                       "Une copie est aussi enregistrée dans le dossier test/mask_test.")
            done = st.session_state.get(flag)
            if done and str(done).startswith("ERREUR::"):
                st.warning("Le masque a été téléchargé, mais la copie dans test/mask_test a échoué : "
                           + str(done)[len("ERREUR::"):])
            elif done:
                st.caption(f"Copie enregistrée : {done}")
        else:
            st.info(ex.get("note") or "Export du masque indisponible.")


def show_results():
    res = st.session_state["res"]
    rid = st.session_state.get("res_id", 0)
    mask = res["pred"]

    st.markdown('<div class="section-rule"></div>'
                '<h2 class="restitle">Résultats de la segmentation</h2>',
                unsafe_allow_html=True)

    mod = st.selectbox("Modalité affichée", MODALITIES,
                       index=MODALITIES.index("t1ce") if "t1ce" in MODALITIES else 0)

    gray = res["image"][MODALITIES.index(mod)]
    vmin, vmax = window(gray)
    tumor = mask > 0
    enh = ENH

    if tumor.any():
        center = np.argwhere(tumor).mean(0).astype(int)
    else:
        center = np.array(gray.shape) // 2
        st.warning("Aucune tumeur détectée par le modèle.")

    tab2d, tab3d, tab4 = st.tabs(["Coupes segmentées", "Cerveau 3D", "4 modalités"])

    with tab2d:
        for ax_i, (col, name) in enumerate(zip(st.columns(3), ["Sagittale", "Coronale", "Axiale"])):
            idx = col.slider(name, 0, gray.shape[ax_i] - 1, int(center[ax_i]), key=f"sl{ax_i}_{rid}")
            fig = draw_slice(gray, mask, res["brain"], ax_i, idx, vmin, vmax,
                             f"{name} · {mod.upper()}", enh)
            col.pyplot(fig, clear_figure=True)
            plt.close(fig)
        st.caption("Repères : G gauche · D droite · A antérieur · P postérieur · S supérieur · I inférieur")

    with tab3d:
        if not tumor.any():
            st.info("Pas de tumeur à afficher en 3D.")
        else:
            cache = st.session_state.setdefault("fig3d", {})
            if STEP_3D not in cache:
                with st.spinner("Reconstruction du cerveau en 3D…"):
                    cache[STEP_3D] = figure_3d(res, STEP_3D)
            st.plotly_chart(cache[STEP_3D], use_container_width=True)

    with tab4:
        iz = st.slider("Coupe axiale", 0, gray.shape[2] - 1, int(center[2]), key=f"sl4_{rid}")
        enh4 = dict(enh, zoom=3, clahe=True, clahe_clip=0.008, sharp=1.4)
        for m_name, col in zip(MODALITIES, st.columns(4)):
            g = res["image"][MODALITIES.index(m_name)]
            lo, hi = window(g)
            fig = draw_slice(g, mask, res["brain"], 2, iz, lo, hi, m_name.upper(), enh4, size=4.6)
            col.pyplot(fig, clear_figure=True)
            plt.close(fig)

    # Fenêtre d'analyse volumétrique (sous les onglets)
    if tumor.any():
        volume_section(res)
        export_section(res)



# ----------------------------------------------------------------------------
# Cerveau 3D réaliste tournant (en-tête) : WebGL autonome, rotation continue ~10 s / tour
# ----------------------------------------------------------------------------
BRAIN_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin:0; height:100%; background:transparent; overflow:hidden; }
  canvas { width:100%; height:100%; display:block; }
</style></head>
<body><canvas id="c"></canvas>
<script>
(function () {
  var cv = document.getElementById('c');
  var gl = null;
  try { gl = cv.getContext('webgl', { antialias: true, alpha: true }) || cv.getContext('experimental-webgl'); } catch (e) {}
  if (!gl) return;
  var uint = gl.getExtension('OES_element_index_uint');
  var NB = uint ? 260 : 170;                      // résolution du maillage cérébral

  /* ---------- bruit de Perlin 3D ---------- */
  var perm = new Uint8Array(512);
  (function () {
    var p = [], i, s = 20240601;
    for (i = 0; i < 256; i++) p[i] = i;
    function rnd() { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }
    for (i = 255; i > 0; i--) { var j = Math.floor(rnd() * (i + 1)), t = p[i]; p[i] = p[j]; p[j] = t; }
    for (i = 0; i < 512; i++) perm[i] = p[i & 255];
  })();
  function fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
  function lerp(t, a, b) { return a + t * (b - a); }
  function grad(h, x, y, z) {
    h &= 15;
    var u = h < 8 ? x : y, v = h < 4 ? y : (h === 12 || h === 14 ? x : z);
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
  }
  function noise(x, y, z) {
    var X = Math.floor(x) & 255, Y = Math.floor(y) & 255, Z = Math.floor(z) & 255;
    x -= Math.floor(x); y -= Math.floor(y); z -= Math.floor(z);
    var u = fade(x), v = fade(y), w = fade(z);
    var A = perm[X] + Y, AA = perm[A] + Z, AB = perm[A + 1] + Z,
        B = perm[X + 1] + Y, BA = perm[B] + Z, BB = perm[B + 1] + Z;
    return lerp(w,
      lerp(v, lerp(u, grad(perm[AA], x, y, z), grad(perm[BA], x - 1, y, z)),
              lerp(u, grad(perm[AB], x, y - 1, z), grad(perm[BB], x - 1, y - 1, z))),
      lerp(v, lerp(u, grad(perm[AA + 1], x, y, z - 1), grad(perm[BA + 1], x - 1, y, z - 1)),
              lerp(u, grad(perm[AB + 1], x, y - 1, z - 1), grad(perm[BB + 1], x - 1, y - 1, z - 1))));
  }
  function sstep(a, b, x) { var t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }

  /* ---------- forme du cerveau : 2 hémisphères + circonvolutions ---------- */
  function brainPt(u, v) {
    var sv = Math.sin(v), dx = sv * Math.cos(u), dz = sv * Math.sin(u), dy = Math.cos(v);
    var ry = 0.60 * (1 - 0.38 * sstep(0, -0.9, dy));                       // base aplatie
    var rx = 0.58 * (1 - 0.10 * Math.max(dz, 0) + 0.03 * Math.max(-dz, 0)); // lobe frontal plus étroit
    var rz = 0.82;
    var bulge = 1 + 0.12 * Math.exp(-Math.pow((dy + 0.45) / 0.32, 2))
                         * Math.exp(-Math.pow((Math.abs(dx) - 0.80) / 0.30, 2))
                         * sstep(-0.8, 0.1, dz);                           // lobes temporaux
    var px = dx * 2.4, py = dy * 2.4, pz = dz * 2.9;
    var wx = noise(px * 0.6 + 11.3, py * 0.6 + 5.1, pz * 0.6 + 2.7),
        wy = noise(px * 0.6 + 7.7, py * 0.6 + 19.2, pz * 0.6 + 31.4),
        wz = noise(px * 0.6 + 23.5, py * 0.6 + 3.3, pz * 0.6 + 13.9);
    var n1 = noise(px + 1.5 * wx, py + 1.5 * wy, pz + 1.5 * wz);
    var n2 = noise(px * 1.9 + 1.0 * wy + 5.5, py * 1.9 + 1.0 * wz + 9.1, pz * 1.9 + 1.0 * wx + 1.3);
    var g1 = Math.exp(-Math.pow(n1 / W1, 2)), g2 = 0.5 * Math.exp(-Math.pow(n2 / W2, 2));
    var groove = g1 + g2 - g1 * g2;                                        // sillons
    var fis = Math.exp(-Math.pow(dx / 0.055, 2)) * sstep(-0.05, 0.35, dy); // scissure inter-hémisphérique
    var syl = Math.exp(-Math.pow((dy - (0.12 - 0.35 * dz)) / 0.06, 2))     // fissure latérale (Sylvius)
              * sstep(0.45, 0.75, Math.abs(dx)) * sstep(-0.45, -0.15, dz) * (1 - sstep(0.72, 0.92, dz));
    groove = Math.max(groove, 0.95 * syl);
    var s = 1 - 0.065 * groove - 0.30 * fis - 0.035 * syl;
    return [dx * rx * bulge * s, dy * ry * bulge * s, dz * rz * bulge * s, Math.max(groove, fis), 0, 0, 0];
  }
  var W1 = 0.14, W2 = 0.12;

  /* ---------- cervelet (stries horizontales) ---------- */
  function cerebPt(u, v) {
    var sv = Math.sin(v), dx = sv * Math.cos(u), dz = sv * Math.sin(u), dy = Math.cos(v);
    var cx = 0, cy = -0.34, cz = -0.46, y = cy + dy * 0.16;
    var gl_ = Math.pow(0.5 + 0.5 * Math.cos(y * 150), 6);
    var gm = Math.exp(-Math.pow(dx / 0.07, 2));
    var s = 1 - 0.07 * gl_ - 0.10 * gm;
    return [cx + dx * 0.36 * s, cy + dy * 0.16 * s, cz + dz * 0.26 * s, Math.max(0.9 * gl_, 0.8 * gm), cx, cy, cz];
  }

  /* ---------- tronc cérébral ---------- */
  function stemPt(u, v) {
    var yy = -0.28 - 0.44 * v, zz = -0.04 - 0.17 * v;
    var r = 0.085 * (1 - 0.18 * v) * (1 - 0.7 * sstep(0.88, 1.0, v));
    return [Math.cos(u) * r, yy, zz + Math.sin(u) * r, 0.12, 0, yy, zz];
  }

  /* ---------- construction du maillage ---------- */
  var pos = [], nrm = [], grv = [], prt = [], idx = [], OFFY = 0.07, E = 0.0025;
  function addMesh(f, nu, nv, v0, v1, part) {
    var base = pos.length / 3, i, j;
    for (j = 0; j <= nv; j++) {
      var v = v0 + (v1 - v0) * j / nv;
      for (i = 0; i <= nu; i++) {
        var u = 2 * Math.PI * i / nu;
        var a = f(u, v), b = f(u + E, v), c = f(u, v + E * (v1 - v0));
        var ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        var vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        var nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        var l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        nx /= l; ny /= l; nz /= l;
        if (nx * (a[0] - a[4]) + ny * (a[1] - a[5]) + nz * (a[2] - a[6]) < 0) { nx = -nx; ny = -ny; nz = -nz; }
        pos.push(a[0], a[1] + OFFY, a[2]); nrm.push(nx, ny, nz); grv.push(a[3]); prt.push(part);
      }
    }
    for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
      var p0 = base + j * (nu + 1) + i, p1 = p0 + 1, p2 = p0 + nu + 1, p3 = p2 + 1;
      idx.push(p0, p2, p1, p1, p2, p3);
    }
  }
  addMesh(brainPt, NB, NB, 0.03, Math.PI - 0.03, 0);
  addMesh(cerebPt, 120, 120, 0.02, Math.PI - 0.02, 1);
  addMesh(stemPt, 40, 24, 0.0, 1.0, 2);

  /* ---------- WebGL ---------- */
  var VS = 'attribute vec3 aPos; attribute vec3 aNrm; attribute float aGr; attribute float aPart;' +
           'uniform mat3 uR; uniform vec2 uS; varying vec3 vN; varying float vG; varying float vP;' +
           'void main(){ vec3 p = uR * aPos; vN = uR * aNrm; vG = aGr; vP = aPart;' +
           ' gl_Position = vec4(p.xy * uS, -p.z * 0.4, 1.0 - 0.10 * p.z); }';
  var FS = 'precision mediump float; varying vec3 vN; varying float vG; varying float vP;' +
           'void main(){' +
           ' vec3 N = normalize(vN);' +
           ' vec3 L1 = normalize(vec3(-0.45, 0.70, 0.60)); vec3 L2 = normalize(vec3(0.75, 0.15, 0.35));' +
           ' float g = clamp(vG, 0.0, 1.0);' +
           ' float amb = 0.46 + 0.12 * N.y;' +
           ' float shade = amb * (1.0 - 0.50 * g) + 0.62 * max(dot(N, L1), 0.0) * (1.0 - 0.40 * g) + 0.18 * max(dot(N, L2), 0.0);' +
           ' vec3 lit = vP < 0.5 ? vec3(0.99, 0.81, 0.74) : (vP < 1.5 ? vec3(0.90, 0.46, 0.56) : vec3(0.97, 0.82, 0.70));' +
           ' vec3 dk  = vP < 0.5 ? vec3(0.84, 0.47, 0.46) : (vP < 1.5 ? vec3(0.60, 0.22, 0.34) : vec3(0.82, 0.62, 0.52));' +
           ' vec3 base = mix(lit, dk, pow(g, 0.8) * 0.90);' +
           ' vec3 H = normalize(L1 + vec3(0.0, 0.0, 1.0));' +
           ' float sp = pow(max(dot(N, H), 0.0), 38.0) * 0.30 * (1.0 - g);' +
           ' gl_FragColor = vec4(base * shade + vec3(1.0, 0.95, 0.90) * sp, 1.0); }';
  function sh(type, src) {
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(s)); return null; }
    return s;
  }
  var vs = sh(gl.VERTEX_SHADER, VS), fs = sh(gl.FRAGMENT_SHADER, FS);
  if (!vs || !fs) return;
  var prog = gl.createProgram(); gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { console.error(gl.getProgramInfoLog(prog)); return; }
  gl.useProgram(prog);
  function buf(data, name, size) {
    var b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, name);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }
  buf(pos, 'aPos', 3); buf(nrm, 'aNrm', 3); buf(grv, 'aGr', 1); buf(prt, 'aPart', 1);
  var ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, uint ? new Uint32Array(idx) : new Uint16Array(idx), gl.STATIC_DRAW);
  var ITYPE = uint ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
  var uR = gl.getUniformLocation(prog, 'uR'), uS = gl.getUniformLocation(prog, 'uS');
  gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.clearColor(0, 0, 0, 0);

  var dpr = Math.min(window.devicePixelRatio || 1, 2), W = 0, H = 0;
  function resize() {
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = Math.max(1, Math.round(W * dpr)); cv.height = Math.max(1, Math.round(H * dpr));
    gl.viewport(0, 0, cv.width, cv.height);
  }
  resize(); window.addEventListener('resize', resize);

  /* ---------- animation : rotation continue, jamais interrompue ---------- */
  var ang = Math.PI * 1.5, TILT = 0.32, SPEED = 0.6, last = performance.now();   // 0,6 rad/s ≈ 10,5 s / tour
  function frame(now) {
    var dt = Math.min((now - last) / 1000, 0.05); last = now;
    ang += dt * SPEED;
    if (cv.clientWidth !== W || cv.clientHeight !== H) resize();
    var c = Math.cos(ang), s = Math.sin(ang), ct = Math.cos(TILT), st = Math.sin(TILT);
    gl.uniformMatrix3fv(uR, false, new Float32Array([c, st * s, -ct * s, 0, ct, st, s, -st * c, ct * c]));
    var S = Math.min(W, H) * 0.5 / 0.95;
    gl.uniform2f(uS, S / (W / 2), S / (H / 2));
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawElements(gl.TRIANGLES, idx.length, ITYPE, 0);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script></body></html>"""

# ----------------------------------------------------------------------- UI
try:
    head_l, head_r = st.columns([5, 2], vertical_alignment="center")
except TypeError:                       # anciennes versions de Streamlit
    head_l, head_r = st.columns([5, 2])

with head_l:
    st.markdown("""
<div class="brand">
  <div class="brand-mark"><span>BS</span></div>
  <div class="brand-text">
    <div class="brand-kicker">Imagerie médicale · Deep Learning</div>
    <h1 class="brand-title">Segmentation de tumeur cérébrale</h1>
    <div class="brand-sub">IRM multimodale BraTS 2020 · Réseau de segmentation 3D</div>
  </div>
</div>
""", unsafe_allow_html=True)

with head_r:
    components.html(BRAIN_HTML, height=200)

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Dossier patient", type=["nii", "gz"], accept_multiple_files=True,
                            key="patient_files", label_visibility="collapsed")

found = parse_uploads(uploaded)
missing = [m for m in MODALITIES if m not in found]
ready = bool(found) and not missing

key = None
if ready:
    sig = tuple(sorted((m, up.name, up.size) for m, up in found.items()))
    key = (sig, MODEL_PATH, PATCH, POSTPROCESS, "v14")

_, mid, _ = st.columns([2, 1.2, 2])
with mid:
    if uploaded and missing:
        st.markdown('<div class="status warn">Modalités manquantes : '
                    + ", ".join(m.upper() for m in missing) + '</div>', unsafe_allow_html=True)
    run = st.button("Lancer la segmentation", use_container_width=True, disabled=not ready)

if ready and run and st.session_state.get("res_key") != key:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable : {MODEL_PATH}")
        st.stop()
    with st.spinner("Prétraitement + inférence 3D en cours…"):
        files = save_uploads(found)
        pid = re.sub(r"_?flair\.nii(\.gz)?$", "", found["flair"].name, flags=re.I) or "patient"
        st.session_state["res"] = run_pipeline(files, MODEL_PATH, (PATCH,) * 3, POSTPROCESS, pid)
        st.session_state["res_key"] = key
        st.session_state["res_id"] = st.session_state.get("res_id", 0) + 1
        st.session_state["fig3d"] = {}

if ready and st.session_state.get("res_key") == key:
    show_results()