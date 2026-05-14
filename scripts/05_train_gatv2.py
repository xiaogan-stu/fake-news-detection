#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_train_gatv2.py
-----------------
应急备用训练脚本：当 04_train_hgnn.py 出现连续 NaN 时切换运行本脚本。

离线特征 + GATv2 训练脚本：不加载 Llama，仅使用预提取节点特征与 TAGFN 图结构。
使用纯欧氏空间的 GATv2（图注意力网络 v2）替代双曲图神经网络，避免数值稳定性问题。

用法:
  python scripts/05_train_gatv2.py --dataset politifact
  python scripts/05_train_gatv2.py --dataset gossipcop
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.tagfn_dataset import PolitifactDataset, GossipcopDataset
from src.models.gatv2_fallback import GATv2FakeNewsDetector
from src.utils.metrics import compute_metrics, EarlyStopping
from src.utils.memory_utils import cuda_memory_profiler


def parse_args():
    parser = argparse.ArgumentParser(description="GATv2 训练（离线 Llama 特征）- 应急备用方案")
    parser.add_argument("--dataset", type=str, default="politifact",
                        choices=["politifact", "gossipcop"],
                        help="数据集名称")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录（默认按 dataset 自动推断）")
    parser.add_argument("--feat_path", type=str, default=None,
                        help="预提取特征路径（默认 /mnt/workspace/features/node_features.pt）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认 /mnt/workspace/output）")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="图批大小")
    parser.add_argument("--accum_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping 耐心值")
    parser.add_argument("--gnn_dim", type=int, default=128, help="GNN 隐藏维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[警告] 未检测到 CUDA，训练将非常慢。", file=sys.stderr)

    # ── 路径配置 ──────────────────────────────────────────────────
    DATA_DIR = args.data_dir or f"/mnt/workspace/data/{args.dataset}"
    if args.feat_path:
        FEAT_PATH = args.feat_path
    elif args.dataset == "politifact":
        FEAT_PATH = "/mnt/workspace/features/node_features.pt"
    else:
        FEAT_PATH = f"/mnt/workspace/features/node_features_{args.dataset}.pt"
    OUTPUT_DIR = args.output_dir or "/mnt/workspace/output"
    MODEL_SAVE = Path(OUTPUT_DIR) / "best_model_gatv2.pt"
    LOG_PATH = Path(OUTPUT_DIR) / "training_log_gatv2.csv"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[配置] 数据集: {args.dataset}")
    print(f"[配置] 数据目录: {DATA_DIR}")
    print(f"[配置] 特征文件: {FEAT_PATH}")
    print(f"[配置] 输出目录: {OUTPUT_DIR}")
    print(f"[配置] 设备: {device}")

    # ── 加载数据 ──────────────────────────────────────────────────
    print("\n[1/4] 加载数据集...")
    assert Path(FEAT_PATH).exists(), \
        f"特征文件不存在: {FEAT_PATH}\n请先运行 03_extract_features.py"

    DatasetCls = PolitifactDataset if args.dataset == "politifact" else GossipcopDataset
    ds = DatasetCls(data_dir=DATA_DIR, feature_cache_path=FEAT_PATH)

    train_data = ds.get_split("train")
    val_data = ds.get_split("val")
    test_data = ds.get_split("test")

    assert train_data[0].x is not None, "节点特征未加载! 请检查特征文件路径"
    feat_dim = train_data[0].x.shape[1]
    print(f"  特征维度: {feat_dim}, train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # ── 初始化模型 ────────────────────────────────────────────────
    print("\n[2/4] 初始化 GATv2 模型...")
    model = GATv2FakeNewsDetector(
        in_dim=feat_dim,
        hidden_mlp=512,
        gnn_dim=args.gnn_dim,
        dropout=args.dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable:,}")
    cuda_memory_profiler("模型加载后")

    # ── 优化器和调度器 ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps = len(train_loader) * args.epochs // args.accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    # ── CSV 日志 ──────────────────────────────────────────────────
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_acc", "val_f1", "val_auc", "lr"])

    # ── 训练循环 ──────────────────────────────────────────────────
    print(f"\n[3/4] 开始训练 GATv2 (共 {args.epochs} epochs)...")
    best_val_f1 = 0.0
    best_epoch = 0
    global_step = 0
    nan_count = 0
    NAN_TOLERANCE = 3

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)

            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y.squeeze())
            loss = loss / args.accum_steps
            loss.backward()

            if torch.isnan(loss):
                nan_count += 1
                print(f"  NaN loss (第 {nan_count} 次)")
                if nan_count >= NAN_TOLERANCE:
                    print("  连续 NaN 超限，训练终止")
                    cuda_memory_profiler("abort_nan")
                    return 2
                optimizer.zero_grad()
                continue
            else:
                nan_count = 0

            total_loss += loss.item() * args.accum_steps
            n_batches += 1

            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ---- 验证 ----
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x.float(), batch.edge_index, batch.batch)
                probs = F.softmax(logits, dim=-1)[:, 1]
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        val_f1 = metrics["f1_macro"]
        val_acc = metrics["accuracy"]
        val_auc = metrics["auc"]
        cur_lr = optimizer.param_groups[0]["lr"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "model_type": "gatv2",
                "dataset": args.dataset,
            }, str(MODEL_SAVE))

        star = "*" if val_f1 == best_val_f1 else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f} | "
              f"lr={cur_lr:.2e} {star}")

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss, val_acc, val_f1, val_auc, cur_lr])

        cuda_memory_profiler(f"Epoch {epoch}")

        if early_stopping(val_f1):
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── 测试集评估 ────────────────────────────────────────────────
    print("\n[4/4] 测试集评估...")
    ckpt = torch.load(str(MODEL_SAVE), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )

    print("\n========== 测试集结果 (GATv2) ==========")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  最佳 val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
    print(f"  模型保存至: {MODEL_SAVE}")

    cuda_memory_profiler("after_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
