from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    dataset_root: str = "/home/aaditya/sandy/dataset"
    train_split: str = "train"
    val_split: str = "val"
    image_dir_name: str = "img"
    label_dir_name: str = "label"
    input_size: int = 512

    model_variant: str = "vitt"
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 25
    num_classes: int = 19
    ignore_index: int = 255
    max_queries_per_image: int = 19
    min_class_pixels: int = 16
    label_assume_bgr: bool = True
    label_max_color_distance: float = 55.0
    max_train_samples: int = 0
    max_val_samples: int = 0
    amp: bool = True
    grad_clip_norm: float = 1.0

    checkpoint_dir: str = "checkpoints_original"
    checkpoint_every: int = 1
    best_model_name: str = "best_model.pth"
    log_file: str = "log.txt"
    seed: int = 42

    @property
    def train_image_dir(self) -> Path:
        return Path(self.dataset_root) / self.train_split / self.image_dir_name

    @property
    def train_label_dir(self) -> Path:
        return Path(self.dataset_root) / self.train_split / self.label_dir_name

    @property
    def val_image_dir(self) -> Path:
        return Path(self.dataset_root) / self.val_split / self.image_dir_name

    @property
    def val_label_dir(self) -> Path:
        return Path(self.dataset_root) / self.val_split / self.label_dir_name


def parse_train_config() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Paper-exact EfficientSAM class-prompted semantic training"
    )
    parser.add_argument("--dataset-root", type=str, default=TrainConfig.dataset_root)
    parser.add_argument("--train-split", type=str, default=TrainConfig.train_split)
    parser.add_argument("--val-split", type=str, default=TrainConfig.val_split)
    parser.add_argument("--image-dir-name", type=str, default=TrainConfig.image_dir_name)
    parser.add_argument("--label-dir-name", type=str, default=TrainConfig.label_dir_name)
    parser.add_argument("--input-size", type=int, default=TrainConfig.input_size)

    parser.add_argument("--model-variant", type=str, default=TrainConfig.model_variant)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--num-classes", type=int, default=TrainConfig.num_classes)
    parser.add_argument("--ignore-index", type=int, default=TrainConfig.ignore_index)
    parser.add_argument(
        "--max-queries-per-image", type=int, default=TrainConfig.max_queries_per_image
    )
    parser.add_argument("--min-class-pixels", type=int, default=TrainConfig.min_class_pixels)
    parser.add_argument(
        "--label-assume-bgr",
        action="store_true",
        default=TrainConfig.label_assume_bgr,
    )
    parser.add_argument("--label-assume-rgb", action="store_false", dest="label_assume_bgr")
    parser.add_argument(
        "--label-max-color-distance",
        type=float,
        default=TrainConfig.label_max_color_distance,
    )
    parser.add_argument("--max-train-samples", type=int, default=TrainConfig.max_train_samples)
    parser.add_argument("--max-val-samples", type=int, default=TrainConfig.max_val_samples)
    parser.add_argument("--amp", action="store_true", default=TrainConfig.amp)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--grad-clip-norm", type=float, default=TrainConfig.grad_clip_norm)

    parser.add_argument("--checkpoint-dir", type=str, default=TrainConfig.checkpoint_dir)
    parser.add_argument(
        "--checkpoint-every", type=int, default=TrainConfig.checkpoint_every
    )
    parser.add_argument("--best-model-name", type=str, default=TrainConfig.best_model_name)
    parser.add_argument("--log-file", type=str, default=TrainConfig.log_file)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    args = parser.parse_args()
    return TrainConfig(
        dataset_root=args.dataset_root,
        train_split=args.train_split,
        val_split=args.val_split,
        image_dir_name=args.image_dir_name,
        label_dir_name=args.label_dir_name,
        input_size=args.input_size,
        model_variant=args.model_variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_queries_per_image=args.max_queries_per_image,
        min_class_pixels=args.min_class_pixels,
        label_assume_bgr=args.label_assume_bgr,
        label_max_color_distance=args.label_max_color_distance,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        amp=args.amp,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        best_model_name=args.best_model_name,
        log_file=args.log_file,
        seed=args.seed,
    )
