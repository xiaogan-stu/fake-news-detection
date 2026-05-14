
"""
HGNN 主训练脚本
前提: 03_extract_features.py 已运行完成
"""

import os, sys, gc, csv, time
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.tagfn_dataset import PolitifactDataset
from src.models.mhr_llm import HyperbolicFakeNewsDetector
from src.utils.metrics import compute_metrics, EarlyStopping
from src.utils.memory_utils import cuda_memory_profiler

# ── 配置 ──────────────────────────────────────────────────────────
DATA_DIR      = '/mnt/workspace/data/politifact'
FEAT_PATH     = '/mnt/workspace/features/node_features.pt'
OUTPUT_DIR    = '/mnt/workspace/output'
MODEL_SAVE    = os.path.join(OUTPUT_DIR, 'best_model.pt')
LOG_PATH      = os.path.join(OUTPUT_DIR, 'training_log.csv')

LR            = 5e-4
BATCH_SIZE    = 4           # 实际 batch（图数）
ACCUM_STEPS   = 8           # 梯度累积，等效 batch=32
NUM_EPOCHS    = 50
WARMUP_RATIO  = 0.1
MAX_GRAD_NORM = 1.0
PATIENCE      = 10
GNN_DIM       = 128
NUM_LAYERS    = 2
DROPOUT       = 0.1
CURVATURE     = -1.0
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
NAN_TOLERANCE = 3           # 连续NaN超过此值则退出

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"设备: {DEVICE}")
cuda_memory_profiler("训练开始前")

# ── 加载数据 ──────────────────────────────────────────────────────
print("\n[1/4] 加载数据集...")
assert os.path.exists(FEAT_PATH), f"特征文件不存在: {FEAT_PATH}\n请先运行 03_extract_features.py"

ds = PolitifactDataset(DATA_DIR, feature_cache_path=FEAT_PATH)
train_data = ds.get_split('train')  # 62 图
val_data   = ds.get_split('val')    # 31 图
test_data  = ds.get_split('test')   # 221 图

# 验证特征已加载
assert train_data[0].x is not None, "节点特征未加载!"
feat_dim = train_data[0].x.shape[1]  # 应为 4096
print(f"  特征维度: {feat_dim}, train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

# ── 初始化模型 ────────────────────────────────────────────────────
print("\n[2/4] 初始化模型...")
model = HyperbolicFakeNewsDetector(
    in_dim=feat_dim,
    gnn_dim=GNN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    curvature=CURVATURE,
).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  可训练参数: {trainable:,}")
cuda_memory_profiler("模型加载后")

# ── 优化器和调度器 ────────────────────────────────────────────────
try:
    import geoopt
    optimizer = geoopt.optim.RiemannianAdam(
        model.parameters(), lr=LR, stabilize=10
    )
    print("  优化器: RiemannianAdam (geoopt)")
except Exception as e:
    print(f"  ⚠️ RiemannianAdam 不可用 ({e}), 使用 AdamW")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

total_steps  = len(train_loader) * NUM_EPOCHS // ACCUM_STEPS
warmup_steps = int(total_steps * WARMUP_RATIO)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
early_stopping = EarlyStopping(patience=PATIENCE, mode='max')

# ── CSV 日志 ──────────────────────────────────────────────────────
with open(LOG_PATH, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'train_loss', 'val_acc', 'val_f1', 'val_auc', 'lr'])

# ── 训练循环 ──────────────────────────────────────────────────────
print(f"\n[3/4] 开始训练 (共 {NUM_EPOCHS} epochs)...")
best_val_f1 = 0.0
global_step = 0
nan_count   = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # ---- Train ----
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    n_batches  = 0

    for step, batch in enumerate(train_loader):
        batch = batch.to(DEVICE)

        # 获取每棵树的根节点特征
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        # logits: [batch_size, 2]

        loss = F.cross_entropy(logits, batch.y.squeeze())
        loss = loss / ACCUM_STEPS
        loss.backward()

        # NaN 检测
        if torch.isnan(loss):
            nan_count += 1
            print(f"  ⚠️ NaN loss (第 {nan_count} 次)")
            if nan_count >= NAN_TOLERANCE:
                print("  ❌ 连续NaN超限，请切换到 05_train_gatv2.py")
                sys.exit(1)
            optimizer.zero_grad()
            continue
        else:
            nan_count = 0

        total_loss += loss.item() * ACCUM_STEPS
        n_batches  += 1

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    avg_loss = total_loss / max(n_batches, 1)

    # ---- Validation ----
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            probs  = F.softmax(logits, dim=-1)[:, 1]
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    val_f1  = metrics['f1_macro']
    val_acc = metrics['accuracy']
    val_auc = metrics['auc']
    cur_lr  = optimizer.param_groups[0]['lr']

    # ---- 保存最优模型 ----
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'val_f1': val_f1,
            'val_acc': val_acc,
            'model_type': 'hgnn',
            'curvature': CURVATURE,
            'num_layers': NUM_LAYERS,
        }, MODEL_SAVE)

    # ---- 日志 ----
    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | loss={avg_loss:.4f} | "
          f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f} | "
          f"lr={cur_lr:.2e} {'⭐' if val_f1 == best_val_f1 else ''}")

    with open(LOG_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([epoch, avg_loss, val_acc, val_f1, val_auc, cur_lr])

    cuda_memory_profiler(f"Epoch {epoch}")

    # ---- Early Stopping ----
    if early_stopping(val_f1):
        print(f"  Early stopping triggered at epoch {epoch}")
        break

# ── 最终测试集评估 ────────────────────────────────────────────────
print("\n[4/4] 测试集评估...")
ckpt = torch.load(MODEL_SAVE)
model.load_state_dict(ckpt['model_state'])
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        probs  = F.softmax(logits, dim=-1)[:, 1]
        preds  = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.squeeze().cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
print("\n========== 测试集结果 ==========")
for k, v in metrics.items():
    print(f"  {k:20s}: {v:.4f}")
print(f"\n  最佳 val F1: {best_val_f1:.4f} (Epoch {ckpt['epoch']})")
print(f"  模型保存至: {MODEL_SAVE}")