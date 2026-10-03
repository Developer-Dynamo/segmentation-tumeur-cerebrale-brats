"""
train.py
--------
Entraînement d'un modèle SegResNet (MONAI) pour la segmentation de tumeurs
cérébrales sur BraTS, à partir des 4 modalités IRM (flair, t1, t1ce, t2).

Usage :
    python train.py --data_dir /chemin/vers/BraTS_Dataset/donnees --output_dir ./runs --epochs 150

Fonctionnalités incluses :
  - chargement explicite des 4 modalités (LoadMultiModalImaged) : indépendant
    des versions de MONAI, évite les bugs d'empilement en canaux
  - --cache_mode disk : met en cache les prétraitements sur le disque local
    (PersistentDataset) au lieu de la RAM. Utile avec beaucoup de patients.
  - --resume : reprend depuis output_dir/last_checkpoint.pth (modèle,
    optimiseur, scheduler, scaler, epoch, meilleur score).
  - checkpoint sauvegardé à CHAQUE epoch (écriture atomique) dans output_dir.
  - split.json : liste exacte des patients d'entraînement / validation.
"""

import os
import json
import argparse
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from monai.data import CacheDataset, PersistentDataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    EnsureTyped,
    Activations,
    AsDiscrete,
)
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from data_utils import build_datalist, ConvertToBraTSClassesd, LoadMultiModalImaged, MODALITIES  # noqa: F401


def get_transforms(patch_size):
    train_transforms = Compose(
        [
            # Chargement explicite des 4 modalités -> tenseur (4, H, W, D).
            # On n'utilise PAS LoadImaged pour "image" : son comportement
            # d'empilement implicite d'une liste de fichiers dépend de la
            # version de MONAI installée, ce qui a déjà causé un bug
            # (4 canaux attendus, 1 seul reçu).
            LoadMultiModalImaged(keys=["image"]),
            LoadImaged(keys=["label"], ensure_channel_first=False),
            EnsureChannelFirstd(keys=["label"]),
            ConvertToBraTSClassesd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            # --- tout ce qui est au-dessus est déterministe donc mis en cache ---
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            # --- Data augmentation ---
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", std=0.01, prob=0.2),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadMultiModalImaged(keys=["image"]),
            LoadImaged(keys=["label"], ensure_channel_first=False),
            EnsureChannelFirstd(keys=["label"]),
            ConvertToBraTSClassesd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


def split_data(datalist, val_fraction=0.15, seed=42):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(datalist))
    rng.shuffle(indices)
    n_val = max(1, int(len(datalist) * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    train_files = [d for i, d in enumerate(datalist) if i not in val_idx]
    val_files = [d for i, d in enumerate(datalist) if i in val_idx]
    return train_files, val_files


def save_checkpoint(path, state):
    """Écriture atomique : évite un fichier corrompu si la session est coupée pendant l'écriture."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dossier contenant images/ et masks/")
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_mode", type=str, choices=["ram", "disk"], default="ram",
                        help="ram = CacheDataset (peu de patients) ; disk = PersistentDataset (beaucoup de patients)")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="Uniquement pour --cache_mode ram")
    parser.add_argument("--cache_dir", type=str, default="/content/cache",
                        help="Dossier du cache disque (disque LOCAL, pas Drive)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reprend depuis last_checkpoint.pth s'il existe")
    args = parser.parse_args()

    set_determinism(seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[train] device = {device}")
    if not use_amp:
        print("[train] ATTENTION : aucun GPU détecté, l'entraînement sera extrêmement lent en 3D.")

    # --- Données ---
    datalist = build_datalist(args.data_dir)
    train_files, val_files = split_data(datalist, args.val_fraction, args.seed)
    print(f"[train] {len(train_files)} patients en entraînement, {len(val_files)} en validation")

    split_path = os.path.join(args.output_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(
            {
                "train": sorted(os.path.basename(str(d["label"])) for d in train_files),
                "val": sorted(os.path.basename(str(d["label"])) for d in val_files),
            },
            f,
            indent=2,
        )

    train_transforms, val_transforms = get_transforms(tuple(args.patch_size))

    if args.cache_mode == "disk":
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"[train] cache disque : {args.cache_dir} (le 1er passage est lent, les suivants rapides)")
        train_ds = PersistentDataset(
            data=train_files, transform=train_transforms, cache_dir=os.path.join(args.cache_dir, "train")
        )
        val_ds = PersistentDataset(
            data=val_files, transform=val_transforms, cache_dir=os.path.join(args.cache_dir, "val")
        )
    else:
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
        )
        val_ds = CacheDataset(
            data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # --- Modèle ---
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,  # TC, WT, ET
        dropout_prob=0.2,
    ).to(device)

    loss_function = DiceCELoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(threshold=0.5)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    best_metric = -1.0
    best_epoch = -1
    start_epoch = 0

    ckpt_path = os.path.join(args.output_dir, "last_checkpoint.pth")
    best_path = os.path.join(args.output_dir, "best_metric_model.pth")

    if args.resume:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"]
            best_metric = ckpt["best_metric"]
            best_epoch = ckpt["best_epoch"]
            print(f"[train] reprise à l'epoch {start_epoch + 1} (meilleur Dice {best_metric:.4f} à l'epoch {best_epoch})")
        else:
            print("[train] --resume demandé mais aucun checkpoint trouvé : départ de zéro.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= max(1, len(train_loader))
        writer.add_scalar("train/loss", epoch_loss, epoch)
        print(f"[epoch {epoch + 1}/{args.epochs}] loss={epoch_loss:.4f}  ({time.time() - t0:.1f}s)  lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_outputs = sliding_window_inference(
                            val_inputs, tuple(args.patch_size), sw_batch_size=1, predictor=model, overlap=0.5
                        )
                    val_outputs = [post_pred(post_sigmoid(x)) for x in decollate_batch(val_outputs)]
                    val_labels_list = decollate_batch(val_labels)
                    dice_metric(y_pred=val_outputs, y=val_labels_list)

                metric_per_class = dice_metric.aggregate()
                dice_metric.reset()
                mean_dice = metric_per_class.mean().item()

                writer.add_scalar("val/mean_dice", mean_dice, epoch)
                writer.add_scalar("val/dice_TC", metric_per_class[0].item(), epoch)
                writer.add_scalar("val/dice_WT", metric_per_class[1].item(), epoch)
                writer.add_scalar("val/dice_ET", metric_per_class[2].item(), epoch)

                print(
                    f"    -> val mean Dice={mean_dice:.4f} "
                    f"(TC={metric_per_class[0].item():.4f}, WT={metric_per_class[1].item():.4f}, ET={metric_per_class[2].item():.4f})"
                )

                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_epoch = epoch + 1
                    save_checkpoint(best_path, model.state_dict())
                    print(f"    -> nouveau meilleur modèle sauvegardé (epoch {best_epoch}, Dice={best_metric:.4f})")

        save_checkpoint(
            ckpt_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
            },
        )

    writer.close()
    print(f"[train] Entraînement terminé. Meilleur Dice moyen = {best_metric:.4f} (epoch {best_epoch})")
    print(f"[train] Modèle sauvegardé dans : {best_path}")


if __name__ == "__main__":
    main()