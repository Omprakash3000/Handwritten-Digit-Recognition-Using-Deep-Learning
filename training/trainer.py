"""Shared PyTorch training loop with AMP, scheduling, and checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MIN_LR,
    NUM_EPOCHS,
    USE_AMP,
    WEIGHT_DECAY,
    ensure_output_dirs,
)
from training.early_stopping import EarlyStopping
from utils import setup_logging

LOGGER = setup_logging(__name__)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _paths in tqdm(loader, leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item()
            total_acc += _accuracy(logits.detach(), labels)
            batches += 1
    return total_loss / max(batches, 1), total_acc / max(batches, 1)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    checkpoint_path: Path,
    epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    resume: bool = False,
) -> dict:
    ensure_output_dirs()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max")
    start_epoch = 1
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    if resume and checkpoint_path.exists():
        try:
            LOGGER.info("Resuming training from checkpoint: %s", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                saved_epoch = ckpt.get("epoch", 0)
                start_epoch = saved_epoch + 1
                val_acc = ckpt.get("val_accuracy", 0.0)
                stopper.best_score = val_acc
                stopper.best_state = ckpt["model_state"]
                LOGGER.info(
                    "Resumed from epoch %d with validation accuracy %.4f",
                    saved_epoch,
                    val_acc,
                )
                if "history" in ckpt and isinstance(ckpt.get("history"), dict):
                    history = {k: list(v) for k, v in ckpt["history"].items()}
        except Exception as exc:
            LOGGER.warning("Could not load checkpoint to resume (%s). Starting from epoch 1.", exc)

    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", param_group["lr"])

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=MIN_LR,
        last_epoch=start_epoch - 1 if start_epoch > 1 else -1,
    )

    if start_epoch > epochs:
        LOGGER.info(
            "Model at %s has already completed %d epochs (target: %d). Skipping further training.",
            checkpoint_path,
            start_epoch - 1,
            epochs,
        )
        if stopper.best_state is not None:
            model.load_state_dict(stopper.best_state)
        history["best_val_accuracy"] = stopper.best_score or 0.0
        return history

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, train=False
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(lr)
        improved = stopper.step(val_acc, model)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | lr=%.6f%s",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            lr,
            " | checkpoint" if improved else "",
        )
        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "history": history,
                },
                checkpoint_path,
            )
        if stopper.should_stop:
            LOGGER.info("Early stopping at epoch %d", epoch)
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    history["best_val_accuracy"] = stopper.best_score
    return history
