"""
data_utils.py
-------------
Fonctions utilitaires pour la segmentation BraTS :
  - construction de la liste des patients (datalist) à partir du dossier racine
  - transform personnalisé pour convertir les labels bruts BraTS (0,1,2,4)
    en 3 canaux binaires cliniques : WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)

Structure de données attendue :

BraTS_Dataset/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii
│   ├── BraTS20_Training_001_t1.nii
│   ├── BraTS20_Training_001_t1ce.nii
│   ├── BraTS20_Training_001_t2.nii
│   └── BraTS20_Training_001_seg.nii
├── BraTS20_Training_002/
└── ...
"""

import os
import glob
from typing import List, Dict

import numpy as np
import torch
import nibabel as nib
from monai.data import MetaTensor
from monai.transforms import MapTransform


MODALITIES = ["flair", "t1", "t1ce", "t2"]


def _extract_patient_id(filename: str) -> str:
    """
    Extrait l'identifiant patient d'un nom de fichier BraTS, ex.
    'BraTS20_Training_001_flair.nii' -> 'BraTS20_Training_001'
    """
    base = os.path.basename(filename)
    for suffix in ["_flair", "_t1ce", "_t1", "_t2", "_seg"]:
        idx = base.find(suffix)
        if idx != -1:
            return base[:idx]
    # fallback : enlève juste l'extension
    return base.split(".nii")[0]


def build_datalist(data_dir: str) -> List[Dict[str, str]]:
    """
    Construit une liste de dictionnaires {"image": [...], "label": ..., "id": ...}
    à partir des données BraTS.

    Supporte DEUX organisations de dossiers :

    A) Un dossier par patient (structure BraTS officielle) :
        data_dir/BraTS20_Training_001/BraTS20_Training_001_flair.nii, ...

    B) Fichiers à plat dans deux sous-dossiers "images" et "masks" :
        data_dir/images/BraTS20_Training_001_flair.nii, ...
        data_dir/masks/BraTS20_Training_001_seg.nii

    La fonction détecte automatiquement laquelle des deux organisations est
    utilisée en regardant si data_dir/images et data_dir/masks existent.
    """
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
        return _build_datalist_flat(images_dir, masks_dir)
    return _build_datalist_per_patient(data_dir)


def _build_datalist_flat(images_dir: str, masks_dir: str) -> List[Dict[str, str]]:
    all_image_files = glob.glob(os.path.join(images_dir, "*.nii*"))
    all_mask_files = glob.glob(os.path.join(masks_dir, "*.nii*"))

    # regrouper les fichiers image par patient
    images_by_patient: Dict[str, Dict[str, str]] = {}
    for f in all_image_files:
        pid = _extract_patient_id(f)
        base = os.path.basename(f)
        for mod in MODALITIES:
            if f"_{mod}." in base or base.endswith(f"_{mod}.nii") or f"_{mod}.nii.gz" in base:
                images_by_patient.setdefault(pid, {})[mod] = f
                break

    masks_by_patient = {_extract_patient_id(f): f for f in all_mask_files}

    if not images_by_patient:
        raise FileNotFoundError(
            f"Aucune image trouvée dans {images_dir}. Vérifie le contenu du dossier."
        )

    datalist = []
    skipped = []
    for pid, mods in sorted(images_by_patient.items()):
        if not all(m in mods for m in MODALITIES):
            skipped.append(pid)
            continue
        if pid not in masks_by_patient:
            skipped.append(pid)
            continue
        datalist.append(
            {
                "image": [mods[m] for m in MODALITIES],  # ordre fixe : flair, t1, t1ce, t2
                "label": masks_by_patient[pid],
                "id": pid,
            }
        )

    if skipped:
        print(f"[build_datalist] {len(skipped)} patient(s) ignoré(s) (fichiers manquants) : {skipped}")
    print(f"[build_datalist] {len(datalist)} patients valides trouvés (structure images/masks à plat)")

    return datalist


def _build_datalist_per_patient(data_dir: str) -> List[Dict[str, str]]:
    """
    Un patient est ignoré (avec avertissement) si un des 5 fichiers manque,
    plutôt que de faire planter tout l'entraînement.
    """
    patient_dirs = sorted(
        [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
    )
    if not patient_dirs:
        raise FileNotFoundError(
            f"Aucun sous-dossier patient trouvé dans {data_dir}, et aucun dossier "
            "images/masks non plus. Vérifie l'organisation de ton dataset."
        )

    datalist = []
    skipped = []

    for pdir in patient_dirs:
        patient_id = os.path.basename(pdir.rstrip("/"))

        image_paths = []
        missing = False
        for mod in MODALITIES:
            candidates = glob.glob(os.path.join(pdir, f"{patient_id}_{mod}.nii*"))
            if not candidates:
                missing = True
                break
            image_paths.append(sorted(candidates)[0])

        seg_candidates = glob.glob(os.path.join(pdir, f"{patient_id}_seg.nii*"))
        if missing or not seg_candidates:
            skipped.append(patient_id)
            continue

        datalist.append(
            {
                "image": image_paths,
                "label": sorted(seg_candidates)[0],
                "id": patient_id,
            }
        )

    if skipped:
        print(f"[build_datalist] {len(skipped)} patient(s) ignoré(s) (fichiers manquants) : {skipped}")
    print(f"[build_datalist] {len(datalist)} patients valides trouvés dans {data_dir}")

    return datalist


class LoadMultiModalImaged(MapTransform):
    """
    Charge explicitement les N fichiers NIfTI listés pour chaque clé et les
    empile en un tenseur multi-canal (N, H, W, D), sans dépendre du
    comportement implicite (et malheureusement version-dépendant selon les
    versions de MONAI) de LoadImaged lorsqu'on lui passe une liste de
    fichiers pour une seule clé. Ce transform garantit un empilement
    déterministe dans l'ordre exact des fichiers fournis, avec l'affine du
    premier fichier conservée pour les transforms spatiaux suivants
    (Orientationd, Spacingd).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            paths = d[key]
            if not isinstance(paths, (list, tuple)):
                paths = [paths]

            arrays = []
            affine = None
            for p in paths:
                img = nib.load(p)
                arr = np.asarray(img.get_fdata(dtype=np.float32))
                arrays.append(arr)
                if affine is None:
                    affine = img.affine

            stacked = np.stack(arrays, axis=0)  # (N, H, W, D)
            tensor = torch.as_tensor(stacked, dtype=torch.float32)
            d[key] = MetaTensor(tensor, affine=torch.as_tensor(affine, dtype=torch.float64))
        return d


class ConvertToBraTSClassesd(MapTransform):
    """
    Convertit le masque brut BraTS (valeurs 0, 1, 2, 4) en 3 canaux binaires
    empilés dans l'ordre [TC, WT, ET] (ordre standard utilisé dans les
    tutoriels MONAI officiels pour BraTS) :

      - TC (Tumor Core)      = labels {1, 4}
      - WT (Whole Tumor)     = labels {1, 2, 4}
      - ET (Enhancing Tumor) = label  {4}

    Le résultat a la forme (3, H, W, D) au lieu de (1, H, W, D).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            mask = d[key]

            # on garde une référence à la MetaTensor d'origine pour récupérer
            # l'affine/les métadonnées, qu'il ne faut surtout PAS perdre ici :
            # Orientationd et Spacingd, appliqués juste après ce transform,
            # ont besoin de l'affine réelle du NIfTI pour réorienter/rééchantillonner
            # le masque de manière cohérente avec l'image. Une perte d'affine ici
            # provoque un désalignement silencieux entre image et masque.
            is_meta = isinstance(mask, MetaTensor)
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)

            if mask_np.ndim == 4:
                mask_np = mask_np[0]

            tc = np.logical_or(mask_np == 1, mask_np == 4)
            wt = np.logical_or(tc, mask_np == 2)
            et = mask_np == 4

            out = np.stack([tc, wt, et], axis=0).astype(np.float32)
            out_tensor = torch.as_tensor(out)

            if is_meta:
                d[key] = MetaTensor(out_tensor, affine=mask.affine, meta=mask.meta)
            else:
                d[key] = out_tensor
        return d


def wt_tc_et_to_brats_labels(pred_3ch: np.ndarray) -> np.ndarray:
    """
    Opération inverse de ConvertToBraTSClassesd, utilisée à l'inférence.
    Prend une prédiction binaire (3, H, W, D) dans l'ordre [TC, WT, ET]
    et reconstruit un masque mono-canal avec les labels BraTS originaux
    (0 = fond, 1 = nécrose/non-rehaussé, 2 = œdème, 4 = tumeur rehaussée).

    Convention utilisée (standard dans la littérature BraTS) :
      - ET=1                -> label 4
      - TC=1 et ET=0         -> label 1
      - WT=1 et TC=0          -> label 2
      - sinon                -> label 0
    """
    tc, wt, et = pred_3ch[0], pred_3ch[1], pred_3ch[2]

    out = np.zeros_like(tc, dtype=np.uint8)
    out[(wt > 0) & (tc == 0)] = 2
    out[(tc > 0) & (et == 0)] = 1
    out[et > 0] = 4
    return out