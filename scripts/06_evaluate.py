#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_evaluate.py
--------------
模型评估脚本：加载训练好的模型，在测试集上进行推理，输出评估指标。

支持两种模型类型：
- hyperbolic: HyperbolicFakeNewsDetector(双曲图神经网络)
- gatv2: GATv2FakeNewsDetector(GATv2图注意力网络)

用法:
  python scripts/06_evaluate.py --model_type hyperbolic --dataset politifact
  python scripts/06_evaluate.py --model_type gatv2 --dataset politifact --checkpoint /path/to/best_model_gatv2.pt
  python scripts/06_evaluate.py --model_type hyperbolic --dataset gossipcop
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.tagfn_dataset import PolitifactDataset, GossipcopDataset
from src.models.mhr_llm import HyperbolicFakeNewsDetector
from src.models.gatv2_fallback import GATv2FakeNewsDetector
from src.utils.metrics import compute_metrics
from src.utils.memory_utils import cuda_memory_profiler


def parse_args():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["hyperbolic", "gatv2"],
                        help="模型类型：hyperbolic 或 gatv2")
    parser.add_argument("--dataset", type=str, default="politifact",
                        choices=["politifact", "gossipcop"],
                        help="数据集名称")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录（默认按 dataset 自动推断）")
    parser.add_argument("--feat_path", type=str, default=None,
                        help="预提取特征路径")
    parser.add_argument("--checkpoint_path", type=Path, default=None,
                        help="模型检查点路径（默认自动推断）")
    parser.add_argument("--output_csv", type=Path, default=None,
                        help="评估结果 CSV 输出路径")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[评估] 使用设备: {device}")

    # ── 路径配置 ──────────────────────────────────────────────────
    DATA_DIR = args.data_dir or f"/mnt/workspace/data/{args.dataset}"
    if args.feat_path:
        FEAT_PATH = args.feat_path
    elif args.dataset == "politifact":
        FEAT_PATH = "/mnt/workspace/features/node_features.pt"
    else:
        FEAT_PATH = f"/mnt/workspace/features/node_features_{args.dataset}.pt"
    OUTPUT_DIR = "/mnt/workspace/output"

    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
    elif args.model_type == "hyperbolic":
        ckpt_path = Path(OUTPUT_DIR) / "best_model.pt"
    else:
        ckpt_path = Path(OUTPUT_DIR) / "best_model_gatv2.pt"

    if args.output_csv:
        csv_path = args.output_csv
    else:
        csv_path = Path(OUTPUT_DIR) / f"evaluation_results_{args.model_type}_{args.dataset}.csv"

    print(f"[评估] 数据集: {args.dataset}")
    print(f"[评估] 特征文件: {FEAT_PATH}")
    print(f"[评估] 检查点: {ckpt_path}")

    # ── 加载特征和数据 ────────────────────────────────────────────
    assert Path(FEAT_PATH).exists(), f"特征文件不存在: {FEAT_PATH}"
    assert ckpt_path.exists(), f"检查点不存在: {ckpt_path}"

    DatasetCls = PolitifactDataset if args.dataset == "politifact" else GossipcopDataset
    ds = DatasetCls(data_dir=DATA_DIR, feature_cache_path=FEAT_PATH)

    test_data = ds.get_split("test")
    assert test_data[0].x is not None, "节点特征未加载!"
    feat_dim = test_data[0].x.shape[1]
    print(f"[评估] 特征维度: {feat_dim}, 测试集: {len(test_data)} 图")

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # ── 加载模型 ──────────────────────────────────────────────────
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if args.model_type == "hyperbolic":
        curvature = float(ckpt.get("curvature", -1.0))
        num_layers = int(ckpt.get("num_layers", 2))
        model = HyperbolicFakeNewsDetector(
            in_dim=feat_dim,
            gnn_dim=128,
            num_layers=num_layers,
            dropout=0.1,
            curvature=curvature,
        ).to(device)
    else:
        model = GATv2FakeNewsDetector(
            in_dim=feat_dim,
            hidden_mlp=512,
            gnn_dim=128,
            dropout=0.1,
        ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print(f"[评估] 模型权重加载完成")

    # ── 推理 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[评估] 开始推理...")
    print("=" * 60)

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

    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )

    # ── 打印结果 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[评估] 分类报告 ({args.model_type}, {args.dataset})")
    print("=" * 60)

    from sklearn.metrics import classification_report
    print(classification_report(
        np.array(all_labels), np.array(all_preds), digits=4
    ))

    print("\n" + "=" * 60)
    print("[评估] 关键指标")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # ── 保存 CSV ──────────────────────────────────────────────────
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "值"])
        writer.writerow(["模型类型", args.model_type])
        writer.writerow(["数据集", args.dataset])
        writer.writerow(["测试样本数", len(test_data)])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.4f}"])

    print(f"\n[评估] 结果已保存至: {csv_path}")
    cuda_memory_profiler("evaluation_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
