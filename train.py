from __future__ import annotations

import argparse
from contextlib import nullcontext
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.tiny_detector import TinyDetector
from utils.config import get_anchors, load_config
from utils.dataset import DetectionDataset, collate_fn, load_classes
from utils.loss import YoloLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch tiny YOLO detector.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--classes", default="public/classes.json")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--box_weight", type=float)
    parser.add_argument("--obj_weight", type=float)
    parser.add_argument("--noobj_weight", type=float)
    parser.add_argument("--cls_weight", type=float)
    parser.add_argument("--iou_weight", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> tuple[argparse.Namespace, list[tuple[float, float]], dict]:
    config = load_config(args.config)
    loss_weights = config["loss_weights"]
    for name in [
        "image_size",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "num_workers",
        "seed",
        "warmup_epochs",
        "label_smoothing",
    ]:
        if getattr(args, name) is None:
            setattr(args, name, config[name])
    for name in ["box_weight", "obj_weight", "noobj_weight", "cls_weight", "iou_weight"]:
        if getattr(args, name) is None:
            setattr(args, name, loss_weights[name])
    if args.no_amp:
        args.amp = False
    elif not args.amp:
        args.amp = bool(config.get("amp", True))
    return args, get_anchors(config), config["model"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: TinyDetector,
    loader: DataLoader,
    criterion: YoloLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    epoch: int = 0,
    total_epochs: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    steps = 0
    phase = "train" if training else "val"
    progress = tqdm(
        loader,
        desc=f"{phase} {epoch}/{total_epochs}" if total_epochs else phase,
        leave=False,
        dynamic_ncols=True,
    )

    for images, targets in progress:
        images = images.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            autocast_enabled = use_amp and device.type == "cuda"
            amp_context = torch.amp.autocast("cuda", enabled=True) if autocast_enabled else nullcontext()
            with amp_context:
                pred = model(images)
                loss, logs = criterion(pred, targets)
            if training:
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        steps += 1
        progress.set_postfix(
            loss=f"{logs['loss']:.3f}",
            box=f"{logs['box_loss']:.3f}",
            iou=f"{logs['iou_loss']:.3f}",
            obj=f"{logs['obj_loss']:.3f}",
            cls=f"{logs['cls_loss']:.3f}",
        )

    return {key: value / max(steps, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    args, anchors, model_config = apply_config(args)
    seed_everything(args.seed)

    classes = load_classes(args.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_set = DetectionDataset(args.train_data, args.image_dir, classes, args.image_size, augment=True)
    val_set = DetectionDataset(args.val_data, args.val_image_dir, classes, args.image_size, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = TinyDetector(num_classes=len(classes), num_anchors=len(anchors), **model_config).to(device)
    criterion = YoloLoss(
        anchors,
        image_size=args.image_size,
        num_classes=len(classes),
        box_weight=args.box_weight,
        obj_weight=args.obj_weight,
        noobj_weight=args.noobj_weight,
        cls_weight=args.cls_weight,
        iou_weight=args.iou_weight,
        label_smoothing=args.label_smoothing,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_lambda(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_logs = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            args.amp,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        with torch.no_grad():
            val_logs = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_amp=args.amp,
                epoch=epoch,
                total_epochs=args.epochs,
            )
        scheduler.step()

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_logs['loss']:.4f} "
            f"val_loss={val_logs['loss']:.4f} "
            f"box={val_logs['box_loss']:.4f} "
            f"iou={val_logs['iou_loss']:.4f} "
            f"obj={val_logs['obj_loss']:.4f} "
            f"noobj={val_logs['noobj_loss']:.4f} "
            f"cls={val_logs['cls_loss']:.4f}"
        )

        state = {
            "model": model.state_dict(),
            "classes": classes,
            "anchors": anchors,
            "image_size": args.image_size,
            "model_config": model_config,
            "loss_weights": {
                "box_weight": args.box_weight,
                "obj_weight": args.obj_weight,
                "noobj_weight": args.noobj_weight,
                "cls_weight": args.cls_weight,
                "iou_weight": args.iou_weight,
            },
            "amp": args.amp,
            "label_smoothing": args.label_smoothing,
            "epoch": epoch,
            "val_loss": val_logs["loss"],
        }
        torch.save(state, checkpoint_dir / "last.pth")
        if val_logs["loss"] < best_val:
            best_val = val_logs["loss"]
            torch.save(state, checkpoint_dir / "best.pth")
            print(f"saved best checkpoint: {checkpoint_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
