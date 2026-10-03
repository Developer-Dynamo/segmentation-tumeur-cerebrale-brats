"""
export_utils.py
---------------
Export de la segmentation (labels BraTS 0, 1, 2, 4) au format NIfTI, sous deux formes :

  - "original"  : le masque est replacé dans la grille voxel de l'IRM d'origine
                  (même shape, même affine) -> il se superpose directement à l'IRM
                  dans ITK-SNAP / 3D Slicer.
  - "processed" : le masque tel que le modèle le voit (orientation RAS, 1 mm
                  isotrope, recadré sur le cerveau), avec l'affine de ce volume.

Principe du replacement : pour chaque voxel de la grille d'origine, on calcule sa
position dans le monde (affine d'origine), puis sa position dans le volume traité
(inverse de l'affine du volume traité), et on prend le label du voxel le plus proche
(interpolation "plus proche voisin", indispensable pour des labels). Cette méthode
tient compte automatiquement de la réorientation, du rééchantillonnage et du recadrage.

Sécurité : avant d'exporter dans l'espace d'origine, on vérifie que le cerveau traité
et le cerveau d'origine ont bien le même centre dans l'espace monde. Sinon (affine
manquante ou incohérente), l'export "original" est refusé plutôt que de produire un
masque décalé sans prévenir.
"""

import os
import tempfile

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform


def get_affine(tensor):
    """Affine 4x4 (voxel -> monde RAS) d'un MetaTensor MONAI ; None si indisponible."""
    aff = getattr(tensor, "affine", None)
    if aff is None:
        return None
    if hasattr(aff, "detach"):
        aff = aff.detach().cpu().numpy()
    aff = np.asarray(aff, dtype=np.float64)
    if aff.ndim == 3:            # affine d'un lot : on prend la première
        aff = aff[0]
    if aff.shape != (4, 4) or not np.isfinite(aff).all():
        return None
    return aff


def labels_to_processed_nifti(labels, affine=None):
    """Masque dans l'espace traité. Sans affine, repli sur une affine identité (1 mm)."""
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(np.asarray(labels).astype(np.uint8), affine)
    img.set_data_dtype(np.uint8)
    return img


def _world_centroid(mask, affine):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    return affine[:3, :3] @ idx.mean(axis=0) + affine[:3, 3]


def labels_to_original_nifti(labels, affine, ref_img, brain_mask=None, tol_mm=5.0):
    """
    Replace le masque dans la grille de `ref_img` (image NIfTI d'origine).
    Retourne (Nifti1Image, "") ou (None, message d'explication).
    """
    if affine is None:
        return None, ("Le volume traité n'a pas de matrice d'orientation (affine) : "
                      "l'export dans l'espace d'origine n'est pas possible.")

    ref_aff = np.asarray(ref_img.affine, dtype=np.float64)
    ref_shape = tuple(ref_img.shape[:3])

    # --- contrôle de cohérence : même centre de cerveau dans l'espace monde ---
    if brain_mask is not None:
        ref_data = np.asarray(ref_img.dataobj)
        if ref_data.ndim > 3:
            ref_data = ref_data[..., 0]
        c_ref = _world_centroid(ref_data != 0, ref_aff)
        c_proc = _world_centroid(np.asarray(brain_mask) > 0, affine)
        if c_ref is not None and c_proc is not None:
            dist = float(np.linalg.norm(c_ref - c_proc))
            if dist > tol_mm:
                return None, (f"Alignement non vérifié (écart de {dist:.1f} mm entre le cerveau "
                              "traité et le cerveau d'origine) : export dans l'espace d'origine "
                              "désactivé par sécurité.")

    # --- grille d'origine -> volume traité (plus proche voisin) ---
    m = np.linalg.inv(affine) @ ref_aff
    out = affine_transform(
        np.asarray(labels).astype(np.uint8),
        m[:3, :3], offset=m[:3, 3],
        output_shape=ref_shape, order=0, mode="constant", cval=0, prefilter=False,
    ).astype(np.uint8)

    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(None, None)
    img = nib.Nifti1Image(out, ref_aff, header=hdr)
    return img, ""


def nifti_to_bytes(img):
    """Sérialise une image NIfTI en .nii.gz (octets), pour st.download_button."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mask.nii.gz")
        nib.save(img, path)
        with open(path, "rb") as f:
            return f.read()


def export_segmentation(labels, affine, brain_mask, ref_path):
    """
    Prépare les deux fichiers pour le dashboard.
    Retourne {"processed": bytes, "original": bytes | None, "note": str}.
    """
    out = {"processed": nifti_to_bytes(labels_to_processed_nifti(labels, affine)),
           "original": None, "note": ""}
    try:
        ref_img = nib.load(ref_path)
        img, note = labels_to_original_nifti(labels, affine, ref_img, brain_mask)
        out["note"] = note
        if img is not None:
            out["original"] = nifti_to_bytes(img)
    except Exception as e:                      # ne jamais bloquer l'affichage des résultats
        out["note"] = f"Export dans l'espace d'origine impossible : {e}"
    return out