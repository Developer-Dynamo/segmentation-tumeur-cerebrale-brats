"""
inference.py
------------
À partir des 4 modalités IRM d'un patient (flair, t1, t1ce, t2), génère le
masque de segmentation de la tumeur au format BraTS (labels 0, 1, 2, 4),
en utilisant le modèle entraîné par train.py.

Usage :
    python inference.py \
        --model_path ./runs/best_metric_model.pth \
        --flair BraTS20_Training_001_flair.nii \
        --t1    BraTS20_Training_001_t1.nii \
        --t1ce  BraTS20_Training_001_t1ce.nii \
        --t2    BraTS20_Training_001_t2.nii \
        --output BraTS20_Pred_001.nii.gz

Espace du fichier de sortie (--space) :
  - original  (défaut) : le masque est replacé dans la grille de l'IRM d'origine
                         (même shape et même affine que le FLAIR) : il se superpose
                         directement à l'IRM dans ITK-SNAP / 3D Slicer.
  - processed          : le masque dans l'espace du modèle (RAS, 1 mm isotrope, recadré),
                         avec l'affine de ce volume.
Si le replacement dans l'espace d'origine ne peut pas être vérifié, le script
l'indique et enregistre le masque dans l'espace "processed".
"""

import argparse

import numpy as np
import torch
import nibabel as nib
from scipy import ndimage

from monai.transforms import (
    Compose,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    EnsureTyped,
    Activations,
    AsDiscrete,
)
from monai.data import Dataset
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference

from data_utils import wt_tc_et_to_brats_labels, LoadMultiModalImaged
from export_utils import get_affine, labels_to_original_nifti, labels_to_processed_nifti


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Post-traitement : ne garde que la plus grande composante connexe
    de chaque canal binaire, pour supprimer les faux positifs isolés.
    Réduit typiquement le bruit de segmentation et améliore le Dice.
    """
    if binary_mask.sum() == 0:
        return binary_mask
    labeled, num_features = ndimage.label(binary_mask)
    if num_features <= 1:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(binary_mask.dtype)


def build_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--flair", type=str, required=True)
    parser.add_argument("--t1", type=str, required=True)
    parser.add_argument("--t1ce", type=str, required=True)
    parser.add_argument("--t2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Chemin du .nii.gz de sortie")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument(
        "--space",
        choices=["original", "processed"],
        default="original",
        help="Espace du masque enregistré : grille de l'IRM d'origine (défaut) ou espace traité du modèle",
    )
    parser.add_argument(
        "--no_postprocess",
        action="store_true",
        help="Désactive le filtrage de la plus grande composante connexe",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_path, device)

    # Ordre fixe attendu par le modèle : flair, t1, t1ce, t2
    image_paths = [args.flair, args.t1, args.t1ce, args.t2]

    # Image d'origine (flair) : sert de référence géométrique pour l'export.
    ref_img = nib.load(args.flair)

    pre_transforms = Compose(
        [
            LoadMultiModalImaged(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
            EnsureTyped(keys=["image"]),
        ]
    )

    sample = Dataset(data=[{"image": image_paths}], transform=pre_transforms)[0]

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(threshold=0.5)

    with torch.no_grad():
        inputs = sample["image"].unsqueeze(0).to(device)
        outputs = sliding_window_inference(
            inputs, tuple(args.patch_size), sw_batch_size=1, predictor=model, overlap=0.5
        )
        pred = post_pred(post_sigmoid(outputs[0])).cpu().numpy()  # shape (3, H, W, D)

    if not args.no_postprocess:
        pred = np.stack([keep_largest_component(pred[c]) for c in range(pred.shape[0])], axis=0)

    brats_label_mask = wt_tc_et_to_brats_labels(pred)  # shape (H, W, D), valeurs {0,1,2,4}

    # Affine du volume traité (réorientation, rééchantillonnage et recadrage inclus)
    affine = get_affine(sample["image"])
    brain = (np.asarray(sample["image"].cpu()) != 0).any(0)

    out_img = None
    if args.space == "original":
        out_img, note = labels_to_original_nifti(brats_label_mask, affine, ref_img, brain)
        if out_img is None:
            print(f"[inference] ATTENTION : {note}")
            print("[inference] Le masque est enregistré dans l'espace traité (--space processed).")
    if out_img is None:
        out_img = labels_to_processed_nifti(brats_label_mask, affine)
        space_used = "processed"
    else:
        space_used = "original"

    nib.save(out_img, args.output)
    print(f"[inference] Masque sauvegardé : {args.output}  (espace : {space_used}, shape {out_img.shape})")
    print(f"[inference] Voxels tumoraux détectés (WT): {int((np.asarray(out_img.dataobj) > 0).sum())}")


if __name__ == "__main__":
    main()