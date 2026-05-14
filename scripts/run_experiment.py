# -*- coding: utf-8 -*-
"""
统一实验运行框架
所有基线、消融实验的统一入口脚本。

示例调用：
  # 本文方法（双曲 HGNN + Llama 特征）:
  python scripts/run_experiment.py --model_type hgnn --feat_type llama --dataset politifact --exp_name llama_hgnn_poli

  # GCN 基线（BERT 特征）:
  python scripts/run_experiment.py --model_type gcn --feat_type bert --dataset politifact --exp_name bert_gcn_poli

  # 消融-无图:
  python scripts/run_experiment.py --model_type mlp --feat_type llama --dataset politifact --exp_name ablation_nograph

  # Gossipcop 数据集:
  python scripts/run_experiment.py --model_type hgnn --feat_type llama --dataset gossipcop --exp_name llama_hgnn_gossip

  # 消融-层数:
  python scripts/run_experiment.py --model_type gcn --feat_type bert --dataset politifact --num_layers 3 --exp_name bert_gcn_poli_l3
"""

import os
import sys
import gc
import csv
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.tagfn_dataset import PolitifactDataset, GossipcopDataset
from src.utils.metrics import compute_metrics, EarlyStopping
from src.utils.memory_utils import cuda_memory_profiler


# ═══════════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='统一实验运行框架')

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['hgnn', 'gcn', 'gat', 'sage', 'bigcn', 'mlp', 'gatv2'],
                        help='模型类型')
    parser.add_argument('--feat_type', type=str, required=True,
                        choices=['llama', 'bert'],
                        help='特征类型: llama(4096d) / bert(768d)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['politifact', 'gossipcop'],
                        help='数据集')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='GNN 层数（默认 2）')
    parser.add_argument('--gnn_dim', type=int, default=128,
                        help='隐藏层维度（默认 128）')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率（默认 5e-4）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数（默认 50）')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='实验名称（用于保存结果文件名）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='图批大小（默认 4）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 概率（默认 0.1）')
    parser.add_argument('--curvature', type=float, default=-1.0,
                        help='双曲空间曲率，仅 hgnn 使用（默认 -1.0）')
    parser.add_argument('--patience', type=int, default=10,
                        help='EarlyStopping 耐心值（默认 10）')
    parser.add_argument('--accum_steps', type=int, default=8,
                        help='梯度累积步数（默认 8）')

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 模型工厂
# ═══════════════════════════════════════════════════════════════════

def build_model(model_type: str, in_dim: int, gnn_dim: int, num_layers: int,
                dropout: float, curvature: float) -> torch.nn.Module:
    """
    根据 model_type 实例化对应模型。

    所有模型的 forward 接口统一为 (x, edge_index, batch) → [B, 2]。

    参数：
        model_type:  模型类型标识
        in_dim:      输入特征维度（llama=4096, bert=768）
        gnn_dim:     GNN 隐藏维度
        num_layers:  GNN 层数
        dropout:     Dropout 概率
        curvature:   双曲曲率（仅 hgnn 使用）

    返回：
        nn.Module 实例
    """
    if model_type == 'hgnn':
        from src.models.mhr_llm import HyperbolicFakeNewsDetector
        return HyperbolicFakeNewsDetector(
            in_dim=in_dim, gnn_dim=gnn_dim, num_layers=num_layers,
            dropout=dropout, curvature=curvature,
        )

    elif model_type == 'gatv2':
        from src.models.gatv2_fallback import GATv2FakeNewsDetector
        return GATv2FakeNewsDetector(
            in_dim=in_dim, gnn_dim=gnn_dim, dropout=dropout,
        )

    elif model_type == 'gcn':
        from src.models.baselines import GCNDetector
        return GCNDetector(in_dim=in_dim, gnn_dim=gnn_dim,
                           num_layers=num_layers, dropout=dropout)

    elif model_type == 'gat':
        from src.models.baselines import GATDetector
        return GATDetector(in_dim=in_dim, gnn_dim=gnn_dim,
                           num_layers=num_layers, dropout=dropout)

    elif model_type == 'sage':
        from src.models.baselines import SAGEDetector
        return SAGEDetector(in_dim=in_dim, gnn_dim=gnn_dim,
                            num_layers=num_layers, dropout=dropout)

    elif model_type == 'bigcn':
        from src.models.baselines import BiGCNDetector
        return BiGCNDetector(in_dim=in_dim, gnn_dim=gnn_dim,
                             num_layers=num_layers, dropout=dropout)

    elif model_type == 'mlp':
        from src.models.baselines import MLPDetector
        return MLPDetector(in_dim=in_dim, gnn_dim=gnn_dim,
                           num_layers=num_layers, dropout=dropout)

    else:
        raise ValueError(f'未知模型类型: {model_type}')


# ═══════════════════════════════════════════════════════════════════
# 特征路径推断
# ═══════════════════════════════════════════════════════════════════

def get_feat_path(feat_type: str, dataset: str) -> str:
    """
    根据特征类型和数据集推断特征文件路径。

    参数：
        feat_type: llama / bert
        dataset:   politifact / gossipcop

    返回：
        特征文件绝对路径
    """
    feat_dir = '/mnt/workspace/features'
    if feat_type == 'llama':
        return os.path.join(feat_dir, 'node_features.pt')
    else:
        return os.path.join(feat_dir, f'bert_features_{dataset}.pt')


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 路径配置 ──────────────────────────────────────────────────
    DATA_DIR = f'/mnt/workspace/data/{args.dataset}'
    FEAT_PATH = get_feat_path(args.feat_type, args.dataset)
    OUTPUT_DIR = f'/mnt/workspace/output/{args.exp_name}'
    MODEL_SAVE = os.path.join(OUTPUT_DIR, 'best_model.pt')
    LOG_PATH = os.path.join(OUTPUT_DIR, 'log.csv')
    ALL_RESULTS_PATH = '/mnt/workspace/output/all_results.csv'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ALL_RESULTS_PATH), exist_ok=True)

    # ── 打印实验配置 ──────────────────────────────────────────────
    print('=' * 60)
    print(f'实验: {args.exp_name}')
    print('=' * 60)
    print(f'  模型类型:     {args.model_type}')
    print(f'  特征类型:     {args.feat_type}')
    print(f'  数据集:       {args.dataset}')
    print(f'  GNN层数:      {args.num_layers}')
    print(f'  隐藏维度:     {args.gnn_dim}')
    print(f'  学习率:       {args.lr}')
    print(f'  训练轮数:     {args.epochs}')
    print(f'  设备:         {DEVICE}')
    print(f'  特征文件:     {FEAT_PATH}')
    print(f'  输出目录:     {OUTPUT_DIR}')
    print('=' * 60)

    # ── 加载数据 ──────────────────────────────────────────────────
    print('\n[1/4] 加载数据集...')
    assert os.path.exists(FEAT_PATH), \
        f'特征文件不存在: {FEAT_PATH}\n请先运行 03_extract_features.py 或 12_extract_bert_features.py'

    DatasetCls = PolitifactDataset if args.dataset == 'politifact' else GossipcopDataset
    ds = DatasetCls(data_dir=DATA_DIR, feature_cache_path=FEAT_PATH)

    train_data = ds.get_split('train')
    val_data = ds.get_split('val')
    test_data = ds.get_split('test')

    # 验证特征已加载
    assert train_data[0].x is not None, '节点特征未加载! 请检查特征文件路径'
    feat_dim = train_data[0].x.shape[1]
    print(f'  特征维度: {feat_dim}, train={len(train_data)}, val={len(val_data)}, test={len(test_data)}')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # ── 初始化模型 ────────────────────────────────────────────────
    print('\n[2/4] 初始化模型...')
    model = build_model(
        model_type=args.model_type,
        in_dim=feat_dim,
        gnn_dim=args.gnn_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        curvature=args.curvature,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  模型: {args.model_type}, 可训练参数: {trainable:,}')
    cuda_memory_profiler('模型加载后')

    # ── 优化器和调度器 ────────────────────────────────────────────
    # hgnn 使用 RiemannianAdam，其余使用 AdamW
    if args.model_type == 'hgnn':
        try:
            import geoopt
            optimizer = geoopt.optim.RiemannianAdam(
                model.parameters(), lr=args.lr, stabilize=10
            )
            print('  优化器: RiemannianAdam (geoopt)')
        except Exception as e:
            print(f'  RiemannianAdam 不可用 ({e}), 使用 AdamW')
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print('  优化器: AdamW')

    # LR warmup + cosine decay
    total_steps = len(train_loader) * args.epochs // args.accum_steps
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    # ── CSV 训练日志 ──────────────────────────────────────────────
    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_acc', 'val_f1', 'val_auc', 'lr'])

    # ── 训练循环 ──────────────────────────────────────────────────
    print(f'\n[3/4] 开始训练 (共 {args.epochs} epochs)...')
    best_val_f1 = 0.0
    best_epoch = 0
    global_step = 0
    nan_count = 0
    NAN_TOLERANCE = 3

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            batch = batch.to(DEVICE)

            # 前向传播: [batch_size, 2]
            logits = model(batch.x.float(), batch.edge_index, batch.batch)

            # 交叉熵损失（梯度累积缩放）
            loss = F.cross_entropy(logits, batch.y.squeeze())
            loss = loss / args.accum_steps
            loss.backward()

            # NaN 检测
            if torch.isnan(loss):
                nan_count += 1
                print(f'  NaN loss (第 {nan_count} 次)')
                if nan_count >= NAN_TOLERANCE:
                    print('  连续 NaN 超限，训练终止')
                    sys.exit(1)
                optimizer.zero_grad()
                continue
            else:
                nan_count = 0

            total_loss += loss.item() * args.accum_steps
            n_batches += 1

            # 梯度累积达到指定步数后更新参数
            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                batch = batch.to(DEVICE)
                logits = model(batch.x.float(), batch.edge_index, batch.batch)
                probs = F.softmax(logits, dim=-1)[:, 1]
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        val_f1 = metrics['f1_macro']
        val_acc = metrics['accuracy']
        val_auc = metrics['auc']
        cur_lr = optimizer.param_groups[0]['lr']

        # ---- 保存最优模型 ----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'model_type': args.model_type,
                'feat_type': args.feat_type,
                'dataset': args.dataset,
                'curvature': args.curvature,
                'num_layers': args.num_layers,
            }, MODEL_SAVE)

        # ---- 打印日志 ----
        star = '*' if val_f1 == best_val_f1 else ''
        print(f'Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | '
              f'val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f} | '
              f'lr={cur_lr:.2e} {star}')

        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, val_acc, val_f1, val_auc, cur_lr])

        # ---- Early Stopping ----
        if early_stopping(val_f1):
            print(f'  Early stopping at epoch {epoch}')
            break

    # ── 测试集评估 ────────────────────────────────────────────────
    print('\n[4/4] 测试集评估...')
    ckpt = torch.load(MODEL_SAVE, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )

    # 打印测试结果
    print('\n' + '=' * 60)
    print(f'测试集结果 ({args.exp_name}):')
    print('=' * 60)
    for k, v in test_metrics.items():
        print(f'  {k:20s}: {v:.4f}')
    print(f'  最佳 val F1: {best_val_f1:.4f} (Epoch {best_epoch})')
    print(f'  模型保存至: {MODEL_SAVE}')

    # ── 追加写入全局结果文件 ──────────────────────────────────────
    header = [
        'exp_name', 'model_type', 'feat_type', 'dataset',
        'num_layers', 'gnn_dim',
        'test_accuracy', 'test_f1_macro', 'test_f1_weighted',
        'test_precision', 'test_recall', 'test_auc',
        'best_epoch', 'best_val_f1',
    ]
    row = [
        args.exp_name,
        args.model_type,
        args.feat_type,
        args.dataset,
        args.num_layers,
        args.gnn_dim,
        f"{test_metrics['accuracy']:.4f}",
        f"{test_metrics['f1_macro']:.4f}",
        f"{test_metrics['f1_weighted']:.4f}",
        f"{test_metrics['precision']:.4f}",
        f"{test_metrics['recall']:.4f}",
        f"{test_metrics['auc']:.4f}",
        best_epoch,
        f"{best_val_f1:.4f}",
    ]

    # 如果文件不存在则写入表头
    write_header = not os.path.exists(ALL_RESULTS_PATH)
    with open(ALL_RESULTS_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f'\n  结果已追加至: {ALL_RESULTS_PATH}')

    # ── 释放显存 ──────────────────────────────────────────────────
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'\n实验 {args.exp_name} 完成!')


if __name__ == '__main__':
    main()
