# CLAUDE.md — Fake News Detection: Llama-3-8B + Hyperbolic GNN

## 1. 项目概述

基于大语言模型（Llama-3-8B, 4-bit 量化）与双曲图神经网络（Lorentz HGNN）的虚假新闻检测框架。核心思路：将新闻传播树中的节点文本通过冻结的 Llama-3-8B 提取为 4096 维特征，经 MLP 降维后投影到 Lorentz 双曲流形，通过多层双曲图卷积在传播树上进行消息传递，最后提取根节点特征做二分类（真/假新闻）。

**关键设计决策：**

- Llama-3-8B 仅用于离线特征提取，训练时不加载（节省显存至 < 4GB）
- 双曲空间（Lorentz 模型）比欧氏空间更好捕捉树形传播结构的层次语义
- 所有模型统一接口：`forward(x, edge_index, batch) → [batch_size, 2]`
- 分类基于每棵传播树的**根节点**（时间戳最小的节点），使用图级标签
- 上游 MHR 仓库 (guoxinyu0617/MHR) 提供流形操作的理论基础，本项目为简化实现

**数据规模：**

| 数据集 | 图数 | 节点数 | 边数 | 标签分布 |
|--------|------|--------|------|----------|
| Politifact | 314 | 41,054 | 40,740 | 真/假 |
| Gossipcop | 5,464 | 314,262 | 308,798 | 真/假 |

**项目来源：** 基于 TAGFN 数据集 (kayzliu/TAGFN on HuggingFace) 和 MHR 双曲图卷积方法 (guoxinyu0617/MHR on GitHub)。

---

## 2. 运行环境

### 2.1 Python 与核心依赖

| 项目 | 版本 | 用途 |
|------|------|------|
| Python | 3.11 | 运行时 |
| PyTorch | 2.10.0 | 深度学习框架 |
| torchvision | 0.25.0 | PyTorch 视觉扩展 |
| torchaudio | 2.10.0 | PyTorch 音频扩展 |
| CUDA | 12.8 (cu128) | GPU 加速 |
| torch-geometric | 2.7.0 | 图神经网络 (PyG) |
| pyg-lib | 0.6.0+pt210cu128 | PyG 底层库 |
| torch-scatter | 2.1.2+pt210cu128 | 稀疏张量 scatter 操作 |
| torch-sparse | 0.6.18+pt210cu128 | 稀疏张量操作 |
| torch-cluster | 1.6.3+pt210cu128 | 图聚类操作 |
| torch-spline-conv | 1.2.2+pt210cu128 | 样条卷积 |
| transformers | 4.48.3 | Llama-3-8B / BERT 加载与推理 |
| tokenizers | 0.21.1 | HuggingFace 分词器 |
| bitsandbytes | 0.45.5 | 4-bit 量化 (NF4) |
| accelerate | 1.4.0 | 分布式训练和设备映射 |
| deepspeed | 0.19.0 | 分布式训练加速 (备选) |
| geoopt | 0.5.0 | Riemannian 优化器 (RiemannianAdam) |
| scikit-learn | 1.6.1 | 评估指标 (F1, AUC, etc.) |
| numpy | 2.2.2 | 数值计算 |
| scipy | 1.15.1 | 科学计算 |
| pandas | 2.2.3 | 数据读取 (parquet) |
| datasets | 3.2.0 | HuggingFace 数据集加载 |
| tqdm | 4.67.1 | 进度条 |
| PyYAML | 6.0.2 | 配置文件解析 |
| huggingface-hub | 0.28.1 | HuggingFace 模型下载 |
| safetensors | 0.5.3 | 安全模型权重格式 |
| sentencepiece | 0.2.0 | 分词器 tokenizer |
| protobuf | 5.29.3 | 协议缓冲 |
| packaging | 24.2 | 版本号解析 |
| psutil | 6.1.1 | 系统资源监控 |
| ninja | 1.11.1.3 | JIT 编译加速 |

### 2.2 虚拟环境

```
虚拟环境路径: /mnt/workspace/venv/
激活命令:     source /mnt/workspace/venv/bin/activate
```

### 2.3 GPU 信息

目标 GPU: **NVIDIA A10 24GB**（或同等 24GB 显存 GPU）

| 阶段 | 显存占用 | 说明 |
|------|---------|------|
| Llama-3-8B (4-bit) 特征提取 | ~6-8 GB | FP16 计算，NF4 量化权重 |
| BERT-base 特征提取 | ~2-3 GB | FP32 计算，无量化 |
| HGNN 训练（不含 LLM） | < 4 GB | 仅 GNN + 分类器参数 |
| GATv2/GCN 训练 | < 2 GB | 纯欧氏空间，更少参数 |

**等效 batch size**: 32（实际 batch_size=4, gradient_accumulation_steps=8）

### 2.4 安装命令

```bash
# 方法 1: 完整安装
pip install -U pip setuptools wheel
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 方法 2: 如果 PyTorch 版本解析出错
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric==2.7.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
pip install transformers==4.48.3 bitsandbytes==0.45.5 geoopt==0.5.0
pip install scikit-learn pandas numpy tqdm PyYAML
```

---

## 3. 完整目录结构

```
e:\Desktop\fake-news-detection\
│
├── CLAUDE.md                              # 本文件 - AI 助手指南（唯一文档）
├── requirements.txt                       # Python 依赖清单（cu128 索引）
│
├── configs/
│   ├── model_config.yaml                  # 模型结构配置模板 (gnn_dim, num_layers, etc.)
│   └── train_config.yaml                  # 训练超参数配置模板 (lr, batch_size, etc.)
│
├── scripts/
│   ├── 01_clone_repos.sh                  # 克隆 MHR 上游仓库，复制 manifolds/layers 代码
│   ├── 02_download_dataset.py             # 从 HuggingFace 下载 TAGFN 数据集到本地
│   ├── 03_extract_features.py             # ⭐ Llama-3-8B (4-bit) 离线特征提取
│   ├── 04_train_hgnn.py                   # ❌ 旧版 HGNN 训练脚本（被 run_experiment.py 取代）
│   ├── 05_train_gatv2.py                  # ❌ 应急 GATv2 训练脚本（功能已被覆盖）
│   ├── 06_evaluate.py                     # 独立模型评估脚本（加载 checkpoint 推理）
│   ├── 12_extract_bert_features.py        # BERT baseline 特征提取 (768维)
│   ├── run_experiment.py                  # ⭐ 统一实验运行框架（当前主入口）
│   ├── collect_results.py                 # 结果收集/论文 LaTeX+Markdown 表格生成
│   └── run_all_experiments.sh            # 批量顺序运行所有实验的 Shell 脚本
│
├── src/
│   ├── __init__.py                        # "项目源码包" 标识
│   │
│   ├── data/
│   │   ├── __init__.py                    # 导出 PolitifactDataset, GossipcopDataset
│   │   └── tagfn_dataset.py               # ⭐ TAGFN 数据集加载器 (~269 行)
│   │       ├── FakeNewsDataset            # 基类：通用数据加载、图索引构建
│   │       ├── PolitifactDataset          # Politifact 子类 (DEFAULT_DATA_DIR 硬编码)
│   │       └── GossipcopDataset           # Gossipcop 子类 (DEFAULT_DATA_DIR 硬编码)
│   │
│   ├── models/
│   │   ├── __init__.py                    # 导出所有 7 个模型类
│   │   ├── mhr_llm.py                     # ⭐ HyperbolicFakeNewsDetector (~263 行)
│   │   ├── baselines.py                   # ⭐ 5 个基线模型 (~423 行)
│   │   │   ├── MLPDetector               # 纯文本 MLP，无图结构
│   │   │   ├── GCNDetector               # 图卷积网络 (GCNConv)
│   │   │   ├── GATDetector               # 图注意力网络 (GATConv, heads=4)
│   │   │   ├── SAGEDetector              # GraphSAGE (SAGEConv)
│   │   │   └── BiGCNDetector             # 双向 GCN (TD + BU)
│   │   ├── gatv2_fallback.py             # GATv2FakeNewsDetector (~226 行)
│   │   │                                  # 欧氏空间备用模型，HGNN NaN 时切换
│   │   ├── layers/
│   │   │   ├── __init__.py                # "网络层" 标识
│   │   │   └── hgnn_layer.py              # ⭐ 双曲图卷积层 (~136 行)
│   │   │       ├── LorentzLinear          # 流形上的线性变换
│   │   │       ├── LorentzAgg             # 流形上的邻域聚合
│   │   │       └── HyperbolicGraphConvolution  # 组合层 (Linear + Agg)
│   │   └── manifolds/
│   │       ├── __init__.py                # "流形与几何工具（由 MHR 仓库拷贝）"
│   │       └── lorentz.py                 # ⭐ Lorentz 流形简化实现 (~113 行)
│   │           └── Lorentz               # proju0/expmap0/projx/logmap0/inner
│   │
│   └── utils/
│       ├── __init__.py                    # 导出 cuda_memory_profiler, EarlyStopping, compute_metrics
│       ├── metrics.py                     # ⭐ compute_metrics + EarlyStopping (~55 行)
│       └── memory_utils.py               # ⭐ GPU 显存监控工具 (~26 行)
│
└── tests/
    ├── test_manifold.py                   # 流形数值稳定性测试 + 边界条件，CPU only (~520 行)
    └── test_local_verify.py               # ⭐ 本地验证测试套件（8 项测试），CPU only (~450 行)
```

**标注说明：**
- ⭐ = 核心文件，修改频率最高
- ❌ = 已废弃/不常用文件（功能已被 run_experiment.py 覆盖）

### 3.1 历史遗留：README.md 中描述的不存在文件

README.md（已删除，被本文档取代）曾描述了一些**实际不存在**的文件路径：

| README 描述 | 实际状态 | 说明 |
|-------------|---------|------|
| `src/models/layers/att_layers.py` | **不存在** | 注意力聚合通过 NotImplementedError 显式禁用 |
| `src/models/layers/hyp_layers.py` | **不存在** | 实际文件是 `hgnn_layer.py` |
| `src/models/manifolds/base.py` | **不存在** | 基类未单独抽取 |
| `scripts/04_train_hgnn.py` 作为主训练脚本 | **已过时** | 实际主入口是 `run_experiment.py` |

**已修复**：`mhr_llm.py` 原本试图从 `hyp_layers` 导入（永远失败的 try 分支），现已改为直接导入 `hgnn_layer.HyperbolicGraphConvolution`。详见 §7.1。

### 3.2 各文件行数统计

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/models/baselines.py` | 423 | 5 个基线模型 |
| `tests/test_manifold.py` | ~520 | 流形测试 + 边界条件 |
| `tests/test_local_verify.py` | ~450 | 本地验证测试套件 |
| `scripts/02_download_dataset.py` | 325 | 数据下载 |
| `scripts/run_experiment.py` | 449 | 统一实验框架 |
| `scripts/collect_results.py` | 373 | 结果收集 |
| `src/data/tagfn_dataset.py` | 269 | 数据集加载 |
| `src/models/mhr_llm.py` | 263 | 主模型 |
| `src/models/gatv2_fallback.py` | 226 | GATv2 备用模型 |
| `scripts/06_evaluate.py` | 190 | 独立评估脚本 |
| `scripts/05_train_gatv2.py` | 272 | 备用训练 |
| `scripts/12_extract_bert_features.py` | 227 | BERT 特征提取 |
| `scripts/03_extract_features.py` | 195 | Llama 特征提取 |
| `scripts/04_train_hgnn.py` | 222 | 旧训练脚本 |
| `src/models/layers/hgnn_layer.py` | 136 | HGNN 层 |
| `src/models/manifolds/lorentz.py` | 113 | Lorentz 流形 |
| `src/utils/metrics.py` | 55 | 指标计算 |
| `src/utils/memory_utils.py` | 26 | 显存监控 |

---

## 4. 数据流说明

### 4a) 训练前准备阶段：原始文本 → .pt 特征文件

**流程概览：**

```
┌─────────────────────────────────────────────────────────────────┐
│ TAGFN 数据集 (HuggingFace: kayzliu/TAGFN)                       │
│                                                                 │
│ 对于每个子集 (politifact / gossipcop):                            │
│   A.txt                   全局边列表 "src, dst"                   │
│   graph_labels.npy        [num_graphs] 图级标签 (0=真, 1=假)       │
│   node_graph_id.npy       [num_nodes] 每个节点所属图 ID            │
│   node_time.npy           [num_nodes] 节点时间戳                   │
│   train_idx.npy           [num_train] 训练集图索引                 │
│   val_idx.npy             [num_val] 验证集图索引                   │
│   test_idx.npy            [num_test] 测试集图索引                  │
│   raw_text/part-00000.parquet  [num_nodes] 单列 ('0') 文本        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 1: 数据加载                                                 │
│                                                                 │
│ ds = PolitifactDataset(data_dir)  # 或 GossipcopDataset         │
│   └─ _load_raw_data():                                          │
│       ├── 读取 A.txt → self.global_edges [num_edges, 2]          │
│       ├── np.load(graph_labels.npy) → self.graph_labels          │
│       ├── np.load(node_graph_id.npy) → self.node_graph_id        │
│       ├── np.load(node_time.npy) → self.node_time                │
│       ├── pd.read_parquet(raw_text/) → self.node_texts           │
│       └── torch.load(feature_cache_path) → self.node_features    │
│           形状: [num_nodes, 4096], dtype=float16                  │
│   └─ _build_graph_edge_index():                                  │
│       ├── graph_to_nodes: List[np.array]  每个图的节点 ID 列表     │
│       └── graph_to_edges: List[List]      每个图的边列表          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 2: 文本提取                                                 │
│                                                                 │
│ texts = ds.get_all_node_texts()  # List[str], 长度 = num_nodes   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 3: LLM 编码 (03_extract_features.py)                        │
│                                                                 │
│ Llama-3-8B (4-bit 量化, bitsandbytes NF4):                      │
│   tokenizer(text, max_length=256, truncation=True)               │
│   model(**inputs) → last_hidden_state [1, seq_len, 4096]         │
│   mean pooling over seq_len → [4096]                             │
│   .to(torch.float16) → 存储                                      │
│                                                                 │
│ 或 BERT-base (12_extract_bert_features.py):                     │
│   tokenizer(texts, max_length=256, padding=True)                 │
│   model(**inputs) → last_hidden_state [batch, seq_len, 768]      │
│   masked mean pooling (考虑 padding mask) → [batch, 768]         │
│   .to(torch.float16) → 存储                                      │
│                                                                 │
│ 输出文件:                                                        │
│   /mnt/workspace/features/node_features.pt                       │
│     Politifact: [41054, 4096], float16                           │
│   /mnt/workspace/features/node_features_gossipcop.pt             │
│     Gossipcop:  [314262, 4096], float16                          │
│   /mnt/workspace/features/bert_features_politifact.pt            │
│     Politifact: [41054, 768], float16                            │
│   /mnt/workspace/features/bert_features_gossipcop.pt             │
│     Gossipcop:  [314262, 768], float16                           │
└─────────────────────────────────────────────────────────────────┘
```

**断点续传机制：**

两种特征提取脚本都支持断点续传：
- 每 `save_every` 个节点保存一次中间特征文件（`features_partial_{dataset}_{idx}.pt`）
- 通过 JSON checkpoint 文件记录 `{"next_idx": N, "partial_path": "..."}`
- 重启时检测 checkpoint 并从上次中断处继续

### 4b) 训练/评估阶段：.pt 文件 → 模型输出

```
┌──────────────────────────────────────────────────────────────────┐
│ 阶段 A: 数据集加载与 PyG Data 构建                                 │
│                                                                  │
│ ds = PolitifactDataset(data_dir, feature_cache_path=FEAT_PATH)   │
│ train_data = ds.get_split('train')  # List[Data], 长度 = 62       │
│                                                                  │
│ get_graph_data(graph_idx) 内部流程:                                │
│   1. 获取全局节点 ID → global_node_ids                            │
│   2. 构建 global_to_local 映射                                     │
│   3. 转换边索引: 全局 ID → 本地 ID → COO 格式                       │
│   4. 确定根节点: argmin(node_time[global_node_ids])               │
│   5. 切片特征: self.node_features[global_node_ids]                │
│   6. 构建 PyG Data 对象                                           │
│                                                                  │
│ PyG Data 字段:                                                    │
│   x:              [num_nodes, feat_dim] float16 (从 .pt 加载)       │
│   edge_index:     [2, num_edges] int64 (COO 格式)                 │
│   y:              [1] int64 (图级标签)                             │
│   num_nodes:      int (标量)                                      │
│   root_idx:       int (根节点本地索引)                              │
│   global_node_ids:[num_nodes] int64 (全局节点 ID)                  │
│   node_texts:     List[str] (原始文本)                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 阶段 B: DataLoader 批处理                                         │
│                                                                  │
│ loader = DataLoader(train_data, batch_size=4, shuffle=True)      │
│                                                                  │
│ 每个 batch 由 PyG 自动拼接:                                        │
│   batch.x:         [total_nodes, 4096]  多个图的节点特征纵向拼接      │
│   batch.edge_index:[2, total_edges]    边索引自动偏移               │
│   batch.batch:     [total_nodes]       节点→子图映射向量             │
│                     例如 [0,0,0,0,0,1,1,1,1,1] 表示 2 个图          │
│   batch.y:         [batch_size]        图级标签                    │
│   batch.num_nodes: 各图节点数列表                                   │
│                                                                  │
│ 重要: batch.x 是 float16，需 .float() 转为 float32                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 阶段 C: 模型前向传播（以 HGNN 为例）                                │
│                                                                  │
│ logits = model(batch.x.float(), batch.edge_index, batch.batch)   │
│ 详细流程见 §5.1 的 forward 完整数据流描述                            │
│                                                                  │
│ 输出: logits [batch_size, 2]                                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 阶段 D: 损失与优化                                                │
│                                                                  │
│ loss = F.cross_entropy(logits, batch.y.squeeze())                │
│ loss = loss / accum_steps    # 梯度累积缩放                       │
│ loss.backward()                                                  │
│                                                                  │
│ if torch.isnan(loss):        # NaN 检测与容忍                     │
│     nan_count += 1                                                │
│     optimizer.zero_grad()    # 清除异常梯度                        │
│     if nan_count >= 3: sys.exit(1)  # 连续 3 次 NaN 终止          │
│     continue                                                      │
│                                                                  │
│ if (step + 1) % accum_steps == 0:  # 达到累积步数                  │
│     clip_grad_norm_(max_norm=1.0)                                 │
│     optimizer.step()         # RiemannianAdam (HGNN) 或 AdamW     │
│     scheduler.step()         # Warmup + Cosine decay              │
│     optimizer.zero_grad()                                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 阶段 E: 评估指标计算                                               │
│                                                                  │
│ probs = F.softmax(logits, dim=-1)[:, 1]  # 正类 (假新闻) 概率      │
│ preds = logits.argmax(dim=-1)            # 预测类别                │
│                                                                  │
│ compute_metrics(y_true, y_pred, y_prob):                         │
│   → {                                                            │
│       'accuracy':     float,                                      │
│       'f1_macro':     float,  # 宏平均 F1                         │
│       'f1_weighted':  float,  # 加权 F1                           │
│       'precision':    float,  # 宏平均精确率                       │
│       'recall':       float,  # 宏平均召回率                       │
│       'auc':          float,  # ROC AUC (-1.0 如果无法计算)         │
│     }                                                             │
└──────────────────────────────────────────────────────────────────┘
```

### 4c) PyG Data 对象字段详解

| 字段 | Python 类型 | 张量形状 | 说明 |
|------|------------|---------|------|
| `x` | Tensor (float16) | `[num_nodes, feat_dim]` | 预提取节点特征，训练时通过 `.float()` 转为 float32 |
| `edge_index` | Tensor (int64) | `[2, num_edges]` | COO 格式边索引：[0, :]=源节点, [1, :]=目标节点 |
| `y` | Tensor (int64) | `[1]` | 图级标签，0=真实新闻，1=虚假新闻 |
| `num_nodes` | int | 标量 | 图中节点数 |
| `root_idx` | int | 标量 | 根节点在**本地索引**中的位置（时间戳最小的节点） |
| `global_node_ids` | Tensor (int64) | `[num_nodes]` | 全局节点 ID，用于从特征矩阵切片 |
| `node_texts` | List[str] | 长度=num_nodes | 原始文本，仅在特征未提取时需要 |
| `batch` | Tensor (int64) | `[total_nodes]` | DataLoader 自动添加，值 0..B-1 表示节点所属子图 |

**根节点约定（极其重要）**：PyG Batch 拼接时保证同一子图的节点连续排列。`batch` 向量的每个值首次出现的位置 = 该子图的第一个节点 = 本地索引 0 = 根节点。`_root_node_indices()` 函数依赖这个假设。所有 7 个模型类都使用相同的根节点提取逻辑。

### 4d) 数据划分

Politifact 默认划分（由 `train_idx.npy` / `val_idx.npy` / `test_idx.npy` 定义）：
- train: ~62 图
- val: ~31 图
- test: ~221 图

Gossipcop 默认划分：
- train: ~4370 图
- val: ~547 图
- test: ~547 图

---

## 5. 所有模型的精确接口

### 5.1 HyperbolicFakeNewsDetector（主模型）

- **文件**：[src/models/mhr_llm.py](src/models/mhr_llm.py) (第 61 行, class)
- **导入名**: `HyperbolicFakeNewsDetector`
- **总参数**: ~1.8M (in_dim=4096, gnn_dim=128, num_layers=2)

**`__init__` 参数：**

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `in_dim` | int | 4096 | 输入节点特征维度（= Llama hidden dim） |
| `gnn_dim` | int | 128 | 降维后欧氏特征维度，也是 HGNN 输入/输出维度 |
| `num_layers` | int | 2 | 双曲图卷积层数 |
| `dropout` | float | 0.1 | MLP 和 HGNN 层的 dropout 概率 |
| `curvature` | float | 1.0 | 洛伦兹流形曲率 k（实际用 abs(curvature)） |
| `hidden_mlp` | int | 512 | MLP 中间隐藏层宽度 |

**注意**：`curvature` 参数语义与实际使用有差异。传入负值如 `-1.0` 是双曲空间惯例（曲率 = -1/K²），但代码中 `LorentzManifold(k=abs(curvature))` 取其绝对值传入。简化版 lorentz.py 中 `proju0` 是恒等映射，曲率值仅影响 `projx` 和 `logmap0` 的几何约束强度。

**内部架构（按 forward 调用顺序）：**

```
feat_proj: Sequential(                    # 参数: ~2.3M
    Linear(in_dim=4096, hidden_mlp=512)    # 4096×512 + 512 = 2,097,664
    → ReLU(inplace=True)
    → LayerNorm(512)
    → Dropout(dropout)
    → Linear(512, gnn_dim=128)            # 512×128 + 128 = 65,664
    → ReLU(inplace=True)
    → LayerNorm(128)
    → Dropout(dropout)
)

manifold: Lorentz(k=abs(curvature), learnable=False)  # 1 个参数 (k)

hgnn_layers: ModuleList([                 # 参数: ~66K × num_layers
    HyperbolicGraphConvolution(            # 每层结构:
        manifold=manifold,                 #   LorentzLinear: Linear(129→129)
        in_features=gnn_dim+1,  # =129     #     weight: 129×129, bias: 129
        out_features=gnn_dim+1, # =129     #     scale: 1
        use_bias=True,                     #   LorentzAgg: 无参数
        dropout=dropout,
        use_att=False,                     # 仅支持 False
        local_agg=True,
    ) × num_layers
])

classifier: Linear(gnn_dim=128, 2)         # 128×2 + 2 = 258
```

**`forward(x, edge_index, batch)` 完整数据流（含中间张量形状）：**

```python
# ═══════════ 输入 ═══════════
# x:          [num_nodes, in_dim=4096]  float32 (经过 batch.x.float())
# edge_index: [2, num_edges]            int64
# batch:      [num_nodes]               int64

# ═══════════ 步骤 1: 欧氏 MLP 降维 ═══════════
z = self.feat_proj(x)
# feat_proj 内部:
#   → Linear(4096→512):    [num_nodes, 4096] → [num_nodes, 512]
#   → ReLU:                [num_nodes, 512]
#   → LayerNorm(512):      [num_nodes, 512]
#   → Dropout:             [num_nodes, 512]
#   → Linear(512→128):     [num_nodes, 512] → [num_nodes, 128]
#   → ReLU:                [num_nodes, 128]
#   → LayerNorm(128):      [num_nodes, 128]
#   → Dropout:             [num_nodes, 128]
# 输出: z [num_nodes, 128] = [num_nodes, gnn_dim]

# ═══════════ 步骤 2: 添加时间维度 ═══════════
z_with_time = torch.cat([
    torch.zeros(z.size(0), 1, device=z.device),
    z
], dim=-1)
# 输出: z_with_time [num_nodes, 129] = [num_nodes, gnn_dim+1]
# 时间维度 (第 0 列) 初始化为 0

# ═══════════ 步骤 3: 投影到原点切空间 ═══════════
u = self.manifold.proju0(z_with_time)
# 输出: u [num_nodes, 129]
# ⚠️ 当前实现中 proju0 是恒等映射: return u
# 完整 MHR 实现中此处应执行非平凡的切空间投影

# ═══════════ 步骤 4: 指数映射到洛伦兹流形 ═══════════
h = self.manifold.expmap0(u, project=True)
# 输出: h [num_nodes, 129]
# 内部计算:
#   d = 128 (空间维度)
#   u_space = u[:, 1:]                     # [num_nodes, 128] 空间部分
#   norm = ||u_space||₂                     # [num_nodes, 1]
#   t = cosh(norm)                         # [num_nodes, 1] 时间分量
#   x = sinh(norm)/norm * u_space           # [num_nodes, 128] 空间分量
#   result = cat([t, x], dim=-1)           # [num_nodes, 129]
#   if project: result = projx(result)      # 确保在流形上

# ═══════════ 步骤 5: 构建稀疏邻接矩阵 ═══════════
num_nodes = x.size(0)
adj = _edge_index_to_sparse_adj(edge_index, num_nodes)
# 内部:
#   ei = _add_self_loops(edge_index, num_nodes)  # 添加自环
#   row = ei[1]  # target (目标节点)
#   col = ei[0]  # source (源节点)
#   values = ones(numel, dtype=float32)
#   adj = sparse_coo_tensor(stack([row,col]), values, (N,N))
#       .coalesce()
# 输出: adj [num_nodes, num_nodes] sparse float32 COO
# 注意: 使用 adj[target, source] = 1 进行聚合: h_new[target] = Σ h[source]

# ═══════════ 步骤 6: 多层双曲图卷积 ═══════════
for i, layer in enumerate(self.hgnn_layers):
    h, _ = layer((h, adj))
    # layer.forward((h, adj)) 内部:
    #   h = self.linear(h)   → LorentzLinear
    #     weight(h)           # 欧氏 Linear(129→129)
    #     时间维度: sigmoid(t) * exp(scale) + 1.1
    #     空间维度: 缩放使 t² - ||x||² = 1
    #   h = self.agg(h, adj) → LorentzAgg
    #     support = sparse.mm(adj, h)     # 邻域消息聚合
    #     denom = sqrt(|Minkowski(support, support)|)
    #     h = support / denom             # Minkowski 内积重投影
    # 输出: h [num_nodes, 129]

    h = self.manifold.projx(h)
    # 投影校正:
    #   确保 t² - ||x_space||² = k
    #   缩放空间部分: scale = sqrt((t² - k) / ||x_space||²)
    #   重算时间: t = sqrt(k + ||x_space_scaled||²)
    # 输出: h [num_nodes, 129]
# 最终 h: [num_nodes, 129]

# ═══════════ 步骤 7: 对数映射回切空间 ═══════════
t = self.manifold.logmap0(h)
# 输出: t [num_nodes, 128]  ← 注意! 去掉了时间维度
# 内部计算:
#   x_space = h[:, 1:]                     # [num_nodes, 128]
#   arg = clamp(h[:, 0] / sqrt(k), min=1+1e-8)
#   theta = arccosh(arg)                   # [num_nodes, 1]
#   norm = ||x_space||₂                    # [num_nodes, 1]
#   result = (theta / norm) * x_space      # [num_nodes, 128]
# 返回纯空间部分，丢弃时间维度

# ═══════════ 步骤 8: 提取根节点特征 ═══════════
root_ix = self._root_node_indices(batch)
# root_ix: [batch_size] 每个子图第 0 号节点的全局索引
# 算法: batch 向量中值首次变化的位置
h_root = t.index_select(0, root_ix)
# 输出: h_root [batch_size, 128] = [batch_size, gnn_dim]

# ═══════════ 步骤 9: 分类器 ═══════════
logits = self.classifier(h_root)
# 输出: logits [batch_size, 2]

return logits
```

**关键数值细节：**
- `manifold.proju0()` 当前是恒等——这是简化版与完整 MHR 的最大差异
- `manifold.expmap0()` 期望输入含时间维度（第 0 列），内部使用 `narrow(-1, 1, d)` 提取空间部分
- `manifold.logmap0()` 返回时自动丢弃时间维度，从 129 维变回 128 维
- 每层 HGNN 后的 `projx()` 是**必须的**——省略会导致特征逐渐脱离流形约束，最终 NaN

---

### 5.2 MLPDetector（纯文本基线，无图结构）

- **文件**：[src/models/baselines.py](src/models/baselines.py) (第 57 行)
- **导入名**: `MLPDetector`
- **总参数**: ~1.1M (in_dim=4096, gnn_dim=128)

**`__init__` 参数：**

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `in_dim` | int | — | 输入特征维度 (4096 或 768) |
| `gnn_dim` | int | — | 隐藏层维度 |
| `num_layers` | int | 2 | **未使用**——仅保持接口统一，MLP 固定 2 层隐藏层 |
| `dropout` | float | 0.1 | Dropout 概率 |

**内部架构：**
```
mlp: Sequential(
    Linear(in_dim, 256)       # in_dim × 256 + 256
    → ReLU(inplace=True)
    → Dropout(dropout)
    → Linear(256, gnn_dim)    # 256 × gnn_dim + gnn_dim
    → ReLU(inplace=True)
    → Dropout(dropout)
    → Linear(gnn_dim, 2)      # gnn_dim × 2 + 2
)
```

**注意**：`MLPDetector` 的中间层固定为 256，而 `HyperbolicFakeNewsDetector` 的 `feat_proj` 中间层为 512。这意味着 MLP 基线的参数量少于其他模型，不是严格可比的消融基线。

**`forward(x, edge_index, batch)` 数据流：**

```python
# x:          [num_nodes, in_dim]
# edge_index: [2, num_edges]  — 完全忽略！
# batch:      [num_nodes]

logits_all = self.mlp(x)                   # [num_nodes, in_dim] → [num_nodes, 2]
# 内部: Linear(in_dim→256) → ReLU → Dropout
#       → Linear(256→gnn_dim) → ReLU → Dropout
#       → Linear(gnn_dim→2)
root_ix = _root_node_indices(batch)        # [batch_size]
return logits_all.index_select(0, root_ix)  # [batch_size, 2]
```

**特殊性**：这是唯一**完全忽略图结构（edge_index）**的模型。作为"无图信息"的消融对照。

---

### 5.3 GCNDetector（图卷积网络基线）

- **文件**：[src/models/baselines.py](src/models/baselines.py) (第 113 行)
- **导入名**: `GCNDetector`
- **总参数**: ~560K (in_dim=4096, gnn_dim=128, num_layers=2)

**`__init__` 参数：**

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `in_dim` | int | — | 输入特征维度 |
| `gnn_dim` | int | — | GNN 隐藏维度 (输入投影 + 所有层 + 分类器) |
| `num_layers` | int | 2 | GCNConv 层数 |
| `dropout` | float | 0.1 | Dropout 概率 |

**内部架构：**
```
input_proj: Linear(in_dim, gnn_dim)
convs: ModuleList([GCNConv(gnn_dim, gnn_dim) × num_layers])
classifier: Linear(gnn_dim, 2)
```

**`forward(x, edge_index, batch)` 数据流：**

```python
h = self.input_proj(x).relu()              # [num_nodes, in_dim] → [num_nodes, gnn_dim]
for conv in self.convs:                    # num_layers 次
    h = conv(h, edge_index)                # GCN 消息传递: h_new = D̂⁻½ Â D̂⁻½ h W
    h = h.relu()                           # [num_nodes, gnn_dim]
    h = F.dropout(h, p=dropout, training=...)  # [num_nodes, gnn_dim]
root_ix = _root_node_indices(batch)        # [batch_size]
h_root = h.index_select(0, root_ix)        # [batch_size, gnn_dim]
return self.classifier(h_root)             # [batch_size, 2]
```

---

### 5.4 GATDetector（图注意力网络基线）

- **文件**：[src/models/baselines.py](src/models/baselines.py) (第 180 行)
- **导入名**: `GATDetector`
- **总参数**: ~570K (in_dim=4096, gnn_dim=128, num_layers=2)

**与 GCNDetector 的差异：**
- 使用 `GATConv(gnn_dim, gnn_dim, heads=4, concat=False, dropout=dropout)` 替代 `GCNConv`
- `heads=4, concat=False`: 4 个注意力头各自输出 `gnn_dim//4=32` 维，取平均后输出仍为 `gnn_dim=128`
- 前向流程与 GCN 完全相同：投影 → 多层 GATConv → 根节点分类
- GATConv 自带 dropout（在注意力权重上），额外的 F.dropout 为冗余

**`forward` 流程**：与 §5.3 GCNDetector 完全一致，仅卷积层类型不同。

---

### 5.5 SAGEDetector（GraphSAGE 基线）

- **文件**：[src/models/baselines.py](src/models/baselines.py) (第 250 行)
- **导入名**: `SAGEDetector`
- **总参数**: ~560K

**与 GCNDetector 的差异：**
- 使用 `SAGEConv(gnn_dim, gnn_dim)` — 默认聚合方式为 `mean`
- 前向流程与 GCN 完全相同

---

### 5.6 BiGCNDetector（双向 GCN）

- **文件**：[src/models/baselines.py](src/models/baselines.py) (第 317 行)
- **导入名**: `BiGCNDetector`
- **总参数**: ~1.1M (两套独立投影 + 两套独立 GCN + 拼接分类器)

**核心思想**：谣言传播树中，TD (自顶向下, 根→叶) 和 BU (自底向上, 叶→根) 的语义不同。两个方向各用一套独立的 GCN 建模，最后拼接根节点特征分类。

**内部架构：**
```
td_proj:  Linear(in_dim, gnn_dim)        # TD 方向投影
bu_proj:  Linear(in_dim, gnn_dim)        # BU 方向投影
td_convs: ModuleList([GCNConv × num_layers])
bu_convs: ModuleList([GCNConv × num_layers])
classifier: Linear(gnn_dim * 2, 2)       # 拼接后分类
```

**`forward(x, edge_index, batch)` 数据流：**

```python
edge_index_bu = edge_index.flip(0)        # 翻转边方向 [2, num_edges]
# flip(0) 将 [src, dst] 变为 [dst, src]——子节点→父节点的消息流

# ═══════ 自顶向下 (TD) ═══════
h_td = self.td_proj(x).relu()              # [num_nodes, in_dim] → [num_nodes, gnn_dim]
for conv in self.td_convs:                 # 使用原始 edge_index
    h_td = conv(h_td, edge_index)          # [num_nodes, gnn_dim]
    h_td = h_td.relu()
    h_td = F.dropout(h_td, p=dropout, ...)

# ═══════ 自底向上 (BU) ═══════
h_bu = self.bu_proj(x).relu()              # [num_nodes, in_dim] → [num_nodes, gnn_dim]
for conv in self.bu_convs:                 # 使用翻转的 edge_index_bu
    h_bu = conv(h_bu, edge_index_bu)       # [num_nodes, gnn_dim]
    h_bu = h_bu.relu()
    h_bu = F.dropout(h_bu, p=dropout, ...)

# ═══════ 根节点拼接 ═══════
root_ix = _root_node_indices(batch)        # [batch_size]
h_td_root = h_td.index_select(0, root_ix)  # [batch_size, gnn_dim]
h_bu_root = h_bu.index_select(0, root_ix)  # [batch_size, gnn_dim]
h_root = torch.cat([h_td_root, h_bu_root], dim=-1)  # [batch_size, gnn_dim*2]

return self.classifier(h_root)             # [batch_size, gnn_dim*2] → [batch_size, 2]
```

---

### 5.7 GATv2FakeNewsDetector（欧氏空间备用模型）

- **文件**：[src/models/gatv2_fallback.py](src/models/gatv2_fallback.py)
- **导入名**: `GATv2FakeNewsDetector`
- **总参数**: ~2.4M (in_dim=4096, gnn_dim=128, hidden_mlp=512)

**用途**：当 HGNN 出现连续 NaN 数值不稳定时，切换到此模型继续训练。与 HGNN 共享相同的 `feat_proj` 设计，但所有操作在欧氏空间完成。

**`__init__` 参数：**

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `in_dim` | int | 4096 | 输入特征维度 |
| `hidden_mlp` | int | 512 | MLP 中间层宽度 |
| `gnn_dim` | int | 128 | GNN 维度 |
| `dropout` | float | 0.1 | Dropout 概率 |
| `gat_heads` | int | 4 | 每层注意力头数 |

**内部架构：**
```
feat_proj: Sequential(                     # 与 HGNN 的 feat_proj 结构相同
    Linear(in_dim, hidden_mlp)
    → ReLU → LayerNorm → Dropout
    → Linear(hidden_mlp, gnn_dim)
    → ReLU → LayerNorm → Dropout
)
gat1: GATv2Conv(gnn_dim, gnn_dim, heads=4, concat=False, dropout=dropout)
gat2: GATv2Conv(gnn_dim, gnn_dim, heads=4, concat=False, dropout=dropout)
classifier: Linear(gnn_dim, 2)
```

**`forward(x, edge_index, batch)` 数据流：**

```python
z = self.feat_proj(x)                      # [num_nodes, 4096] → [num_nodes, 128]

# 第一层 GATv2 + 残差
h1 = self.gat1(z, edge_index)              # [num_nodes, 128] → [num_nodes, 128]
h1 = self.relu(h1)
h1 = self.dropout_layer(h1)
h1 = h1 + z                                # 残差连接

# 第二层 GATv2 + 残差
h2 = self.gat2(h1, edge_index)             # [num_nodes, 128] → [num_nodes, 128]
h2 = self.relu(h2)
h2 = self.dropout_layer(h2)
h2 = h2 + h1                               # 残差连接

root_ix = _root_node_indices(batch)        # [batch_size]
h_root = h2.index_select(0, root_ix)       # [batch_size, 128]
return self.classifier(h_root)             # [batch_size, 2]
```

**与 HyperbolicFakeNewsDetector 的关键差异：**

| 方面 | HGNN | GATv2 |
|------|------|-------|
| 空间 | 双曲 (Lorentz) | 欧氏 |
| 卷积层 | HyperbolicGraphConvolution | GATv2Conv |
| 残差连接 | 无 | 有 (每层) |
| 层数 | 可配置 `num_layers` | 固定 2 层 |
| 流形操作 | expmap0, logmap0, projx | 无 |
| 参数 | 无 `num_layers`, `curvature` | 有 `gat_heads` |

---

### 5.8 Lorentz 流形 (简化版)

- **文件**：[src/models/manifolds/lorentz.py](src/models/manifolds/lorentz.py)
- **导入名**: `Lorentz` (继承 `nn.Module`)

```python
class Lorentz(nn.Module):
    """
    Simplified Lorentz manifold: Lᵈ_K = {x ∈ Rᵈ⁺¹ : ⟨x, x⟩_L = -K, x₀ > 0}
    其中 ⟨x, y⟩_L = x₀y₀ - Σᵢ xᵢyᵢ (Minkowski 内积)
    """

    def __init__(self, k: float = 1.0, learnable: bool = False):
        # k:    曲率参数 (应为正值，对应 K = -k 的负曲率空间)
        # learnable: 是否将 k 作为可学习参数

    def proju0(self, u: Tensor) -> Tensor:
        """投影到原点切空间。输入 [*, d] → 输出 [*, d]。⚠️ 当前实现: return u (恒等)"""

    def expmap0(self, u: Tensor, project: bool = True) -> Tensor:
        """
        指数映射: 原点切空间 → 流形
        u: [*, d+1] → [*, d+1]
        期望 u 已含时间维度 (第一列)
        内部: u_space = u[:,1:]; norm = ||u_space||
              t = cosh(norm); x = sinh(norm)/norm * u_space
              若 project: return projx([t, x])
        """

    def projx(self, x: Tensor) -> Tensor:
        """
        投影到流形: 强制 t² - ||x_space||² = k, t > 0
        x: [*, d+1] → [*, d+1]
        内部: scale = sqrt((t² - k) / ||x_space||².clamp_min(1e-8))
             x_space *= scale; t = sqrt(k + ||x_space||²)
        """

    def logmap0(self, x: Tensor) -> Tensor:
        """
        对数映射: 流形 → 原点切空间
        x: [*, d+1] → [*, d]  ← 丢掉时间维度!
        内部: theta = arccosh(clamp(x₀/√k, min=1+1e-8))
             return (theta / ||x_space||) * x_space
        """

    def inner(self, x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
        """
        Minkowski 内积: x₀y₀ - Σᵢ xᵢyᵢ
        x: [*, d+1], y: [*, d+1]
        → [*] (keepdim=False) 或 [*, 1] (keepdim=True)
        """
```

**与完整 MHR 实现的差异：**
1. `proju0` 是恒等映射 — 完整版中切空间投影有实际计算
2. 缺少 `dist()`, `sqdist()`, `geodesic()`, `parallel_transport()` 等方法
3. 数值保护使用 `.clamp_min(1e-8)`, `.clamp(min=1.0+1e-8)` — 完整版可能有更精细的处理

---

### 5.9 HyperbolicGraphConvolution 层

- **文件**：[src/models/layers/hgnn_layer.py](src/models/layers/hgnn_layer.py)

```python
class HyperbolicGraphConvolution(nn.Module):
    """
    forward 输入为 tuple (x, adj):
      x:   [num_nodes, manifold_dim]  流形上的点 (含时间维)
      adj: [num_nodes, num_nodes]      稀疏 float32 COO 邻接矩阵
    """
    def __init__(self, manifold, in_features, out_features,
                 use_bias=True, dropout=0.1, use_att=False,
                 local_agg=True, nonlin=None):
        self.linear = LorentzLinear(manifold, in_features, out_features, ...)
        self.agg = LorentzAgg(manifold, out_features, ...)
```

**子组件详解：**

`LorentzLinear`: 在流形切空间（通过 `proju0` 映射后）做欧氏线性变换，然后通过 sigmoid 控制的时间维重参数化确保结果仍在流形上。

```python
# forward(x) 内部流程:
#   if nonlin: x = nonlin(x)
#   x = weight(dropout(x))              # 欧氏 Linear 变换
#   x_space = x[:, 1:]                  # 空间部分
#   time = sigmoid(x[:, 0]) * exp(scale) + 1.1  # 时间维重参数化
#   scale_factor = (time² - 1) / ||x_space||²
#   x = cat([time, x_space * sqrt(scale_factor)], dim=-1)
```

`LorentzAgg`: 无注意力的邻域聚合，对应 MHR 的 `LorentzAgg(use_att=False)`。

```python
# forward(x, adj) 内部流程:
#   if adj.is_sparse:
#       support = sparse.mm(adj, x)      # 邻域加权和
#   else:
#       support = adj @ x
#   inner_self = Minkowski(support, support)  # Minkowski 内积
#   denom = sqrt(|inner_self|).clamp_min(1e-8)
#   return support / denom               # 重投影回流形
```

**`reset_parameters` 的特殊初始化** (LorentzLinear):
```python
# 权重初始化: 均匀分布 U(-stdv, stdv), stdv = 1/sqrt(out_features)
# 然后每隔 in_features 列清零一列:
for idx in range(0, self.in_features, step):
    self.weight.weight[:, idx] = 0
```
这是 MHR 的特色做法，目的可能是为时间分量留出"干净"通道。

---

## 6. run_experiment.py 的所有命令行参数

- **文件**：[scripts/run_experiment.py](scripts/run_experiment.py) (449 行)
- **用途**：统一实验运行框架，是当前项目的主训练入口

### 6.1 完整参数列表

| 参数 | 类型 | 可选值 | 默认值 | 含义 |
|------|------|--------|--------|------|
| `--model_type` | str | `hgnn`, `gcn`, `gat`, `sage`, `bigcn`, `mlp`, `gatv2` | **必填** | 模型类型，详见 §5 |
| `--feat_type` | str | `llama`, `bert` | **必填** | 特征类型 (llama→4096d, bert→768d) |
| `--dataset` | str | `politifact`, `gossipcop` | **必填** | 数据集名称 |
| `--exp_name` | str | 任意字符串 | **必填** | 实验名称，用于输出目录和 all_results.csv |
| `--num_layers` | int | 正整数 | 2 | GNN 卷积层数 |
| `--gnn_dim` | int | 正整数 | 128 | GNN 隐藏维度（所有层统一） |
| `--lr` | float | 正浮点数 | 5e-4 | 初始学习率 |
| `--epochs` | int | 正整数 | 50 | 训练 epoch 数 |
| `--batch_size` | int | 正整数 | 4 | 每批图数量（物理 batch） |
| `--dropout` | float | [0, 1] | 0.1 | Dropout 概率 |
| `--curvature` | float | 任意 | -1.0 | 双曲空间曲率（仅 `hgnn` 使用） |
| `--patience` | int | 正整数 | 10 | EarlyStopping 耐心值 (epochs) |
| `--accum_steps` | int | 正整数 | 8 | 梯度累积步数（等效 batch = batch_size × accum_steps） |

### 6.2 10 个常用调用示例

```bash
# === 本文方法 ===
# 1. HGNN + Llama, Politifact (核心实验)
python scripts/run_experiment.py \
    --model_type hgnn --feat_type llama --dataset politifact \
    --exp_name llama_hgnn_poli

# 2. HGNN + Llama, Gossipcop (跨数据集)
python scripts/run_experiment.py \
    --model_type hgnn --feat_type llama --dataset gossipcop \
    --exp_name llama_hgnn_gossip

# === 基线对比 (BERT 特征) ===
# 3. GCN + BERT, Politifact
python scripts/run_experiment.py \
    --model_type gcn --feat_type bert --dataset politifact \
    --exp_name bert_gcn_poli

# 4. GAT + BERT, Politifact
python scripts/run_experiment.py \
    --model_type gat --feat_type bert --dataset politifact \
    --exp_name bert_gat_poli

# 5. BiGCN + BERT, Politifact
python scripts/run_experiment.py \
    --model_type bigcn --feat_type bert --dataset politifact \
    --exp_name bert_bigcn_poli

# === 消融实验 ===
# 6. 无图结构 (MLP + Llama)
python scripts/run_experiment.py \
    --model_type mlp --feat_type llama --dataset politifact \
    --exp_name ablation_nograph

# 7. 无 LLM (HGNN + BERT)
python scripts/run_experiment.py \
    --model_type hgnn --feat_type bert --dataset politifact \
    --exp_name ablation_bert_hgnn

# 8. 无双曲 (GATv2 + Llama, 欧氏对照)
python scripts/run_experiment.py \
    --model_type gatv2 --feat_type llama --dataset politifact \
    --exp_name llama_gatv2_poli

# 9. 层数消融 1 层
python scripts/run_experiment.py \
    --model_type hgnn --feat_type llama --dataset politifact \
    --num_layers 1 --exp_name ablation_1layer

# 10. 层数消融 3 层
python scripts/run_experiment.py \
    --model_type hgnn --feat_type llama --dataset politifact \
    --num_layers 3 --exp_name ablation_3layer
```

### 6.3 内部执行流程（run_experiment.py main() 详解）

```
main():
├── parse_args()
├── 路径配置
│   ├── DATA_DIR = /mnt/workspace/data/{dataset}
│   ├── FEAT_PATH = get_feat_path(feat_type, dataset)
│   │   ├── llama → /mnt/workspace/features/node_features.pt
│   │   └── bert  → /mnt/workspace/features/bert_features_{dataset}.pt
│   ├── OUTPUT_DIR = /mnt/workspace/output/{exp_name}
│   └── ALL_RESULTS_PATH = /mnt/workspace/output/all_results.csv
│
├── [1/4] 加载数据
│   ├── ds = PolitifactDataset / GossipcopDataset(data_dir, feat_path)
│   ├── train/val/test = ds.get_split('train'/'val'/'test')
│   ├── 验证 train_data[0].x is not None
│   └── DataLoader(train/val/test, batch_size=batch_size)
│
├── [2/4] 初始化模型
│   ├── build_model(model_type, in_dim=feat_dim, ...)
│   ├── .to(DEVICE)
│   └── cuda_memory_profiler()
│
├── 优化器选择
│   ├── hgnn → try: geoopt.optim.RiemannianAdam; except: AdamW
│   └── 其他 → torch.optim.AdamW (weight_decay=1e-4)
│
├── LR Scheduler
│   ├── LambdaLR: warmup (前 10% steps 线性增长) → cosine decay
│   └── EarlyStopping(patience=10, mode='max')
│
├── [3/4] 训练循环
│   for epoch in 1..epochs:
│   ├── train():
│   │   for batch in train_loader:
│   │   ├── logits = model(batch.x.float(), batch.edge_index, batch.batch)
│   │   ├── loss = CE(logits, batch.y.squeeze()) / accum_steps
│   │   ├── loss.backward()
│   │   ├── NaN 检测 (容忍 3 次, 超限则 sys.exit(1))
│   │   ├── 梯度累积: if (step+1) % accum_steps == 0:
│   │   │   ├── clip_grad_norm_(max_norm=1.0)
│   │   │   ├── optimizer.step()
│   │   │   ├── scheduler.step()
│   │   │   └── optimizer.zero_grad()
│   │   └── avg_loss = total_loss / n_batches
│   ├── val():
│   │   for batch in val_loader:
│   │   ├── logits = model(...)
│   │   ├── probs = softmax(logits)[:, 1]
│   │   └── preds = argmax(logits)
│   │   metrics = compute_metrics(...)
│   ├── 保存最优模型: if val_f1 > best_val_f1 → torch.save(checkpoint)
│   ├── CSV 日志: [epoch, avg_loss, val_acc, val_f1, val_auc, lr]
│   └── EarlyStopping: if early_stopping(val_f1) → break
│
├── [4/4] 测试集评估
│   ├── 加载最优 checkpoint
│   ├── 全量测试集推理
│   ├── compute_metrics → test_metrics
│   └── 追加写入 all_results.csv
│
└── 清理: del model; gc.collect(); torch.cuda.empty_cache()
```

### 6.4 输出文件结构

```
/mnt/workspace/output/
├── {exp_name}/
│   ├── best_model.pt          # checkpoint: epoch, model_state, val_f1, ...
│   └── log.csv                # 训练日志: epoch, loss, val_acc, val_f1, val_auc, lr
└── all_results.csv            # 全局结果汇总 (所有实验追加写入)
```

---

## 7. 已知问题与修复记录

### 7.1 导入路径 Bug — ✅ 已修复

**问题**（原 [mhr_llm.py:21](src/models/mhr_llm.py#L21)）：
```python
try:
    from src.models.layers.hyp_layers import LorentzGraphConvolution as HyperbolicGraphConvolution
    _USE_MHR_LAYER = True
except ImportError:
    ...
```

**分析**：`hyp_layers.py` **不存在**于仓库中。第一个 import 永远失败。

**已修复** (2026-05-12)：删除了整个 try/except 链，直接导入：
```python
from src.models.layers.hgnn_layer import HyperbolicGraphConvolution
```
`_USE_MHR_LAYER` 标志（无意义，永远为 False）已移除。

### 7.2 NotImplementedError

**位置**：[hgnn_layer.py:89](src/models/layers/hgnn_layer.py#L89)
```python
if use_att:
    raise NotImplementedError(
        "Current hgnn_layer only supports use_att=False to reduce dependencies."
    )
```

**原因**：MHR 的注意力聚合依赖 `att_layers.py`（不在仓库中）。如需启用，需要从 MHR 仓库复制 `att_layers.py` 并在 `LorentzAgg` 中实现注意力分支。

### 7.3 数值不稳定性

**NaN 风险点（多处）**：
- `lorentz.py` 的 `expmap0` 中 `cosh(norm)` / `sinh(norm)` — 当 `norm` 过大时溢出
- `lorentz.py` 的 `projx` 中 `sqrt(t_sq - self.k)` — 当 `t² < k` 时 NaN
- `lorentz.py` 的 `logmap0` 中 `arccosh(t/√k)` — 当参数 < 1 时 NaN
- `hgnn_layer.py` 的 `LorentzAgg` 中 `sqrt(|inner_self|)` — 当内积为 0 或 NaN 时除零

**已有防御**：
- `.clamp_min(1e-8)` 和 `.clamp(min=1.0+1e-8)` 在多处使用
- 训练循环中 NaN 检测 + 容忍机制（连续 3 次后退出）
- 每层 HGNN 后的 `manifold.projx()` 投影校正

### 7.4 简化版 Lorentz 的 proju0 是恒等映射

**位置**：[lorentz.py:29-31](src/models/manifolds/lorentz.py#L29-L31)
```python
def proju0(self, u: torch.Tensor) -> torch.Tensor:
    """Project vector to tangent space at origin."""
    return u
```

这与标准 Riemannian 优化中的切空间约束不同。完整的 MHR 实现中，`proju0` 会确保输入的欧氏向量在切空间的正确子空间中。当前简化版直接跳过此步骤，可能导致初始特征"偏离"正确的流形邻域。

**测试覆盖**：`tests/test_manifold.py` 的 `test_proju0_identity()` (测试 3) 显式验证此恒等行为，并包含与完整 MHR 实现的对比文档。`tests/test_local_verify.py` 测试 3.1 也验证此行为。

### 7.5 曲率参数语义不一致

**位置**：[mhr_llm.py:110](src/models/mhr_llm.py#L110)
```python
self.manifold = LorentzManifold(k=abs(curvature), learnable=False)
```

用户传入 `curvature=-1.0`（双曲空间惯例），但代码取 `abs()`。如果用户期望的是负曲率影响几何行为，需要深入理解 Lorentz 实现中 `k` 在 `projx` 和 `logmap0` 中如何使用。

### 7.6 权重初始化中的对角线清零

**位置**：[hgnn_layer.py:63-66](src/models/layers/hgnn_layer.py#L63-L66)
```python
for idx in range(0, self.in_features, step):
    self.weight.weight[:, idx] = 0
```

每 `step=in_features` 列（实际因为 `step=in_features`，`range(0, in_features, in_features)` 只产生 `idx=0`），将第 0 列清零。这是 MHR 的特定设计模式，修改 `in_features` 时需要注意此初始化对收敛的影响。

### 7.7 logmap0 的维度变化（有意的设计）

**位置**：[lorentz.py:79-96](src/models/manifolds/lorentz.py#L79-L96)

`logmap0(x: [*, d+1]) → [*, d]` —— 返回时自动去掉时间维度。这意味着从流形回到欧氏空间时维度从 129 变为 128。调用者（`HyperbolicFakeNewsDetector.forward` 步骤 7）依赖此行为。

### 7.8 collect_results.py 提升计算 Bug — ✅ 已修复

**问题**（原 `collect_results.py` `compute_improvement()` 函数）：当本文方法（Full Model）数据缺失时，代码仍尝试计算提升幅度。`our_auc` 为 `None`，与其他缺失数据被解析为 `-1.0` 的基线值进行数学运算，输出类似 "-215.70%" 的垃圾值。

**已修复** (2026-05-12)：在函数开头添加早期检查——如果本文方法数据缺失，直接返回警告信息，不进行数值计算。详见 [collect_results.py:224](scripts/collect_results.py#L224) 的 `compute_improvement()` 函数。

### 7.9 GATDetector "双重 Dropout" 审查 — 非 Bug

**审查结论** (2026-05-12)：`GATConv(dropout=dropout)` 的 dropout 应用在**注意力系数**上（GAT 论文标准做法），而 `F.dropout(h, p=dropout)` 应用在**输出特征**上。两者作用于不同目标，都是标准实践。

对于 GCN/SAGE/BiGCN 基线：这些卷积层（`GCNConv`, `SAGEConv`）**无内部 dropout**，forward 中的 `F.dropout` 是它们唯一的 dropout 来源——删除会完全消除正则化。因此不修改 baselines.py。

### 7.10 `.squeeze()` 使用分析

搜索结果显示代码中**未使用 `.reshape(-1)`**，全部使用 `.squeeze()` 处理标签维度。这是安全的做法——`.squeeze()` 只压缩大小为 1 的维度，如果数据形状不对会立即暴露。

---

## 8. 关键常量和路径

### 8.1 所有硬编码路径（含文件:行号）

| 路径变量 | 值 | 出现位置 (文件:行号) |
|---------|-----|---------------------|
| Politifact 数据 | `/mnt/workspace/data/politifact` | tagfn_dataset.py:196, run_experiment.py:177, 04_train_hgnn.py:21, 06_evaluate.py:71, 12_extract_bert_features.py:51, 03_extract_features.py:45 |
| Gossipcop 数据 | `/mnt/workspace/data/gossipcop` | tagfn_dataset.py:207, run_experiment.py:177, 06_evaluate.py:71 |
| 特征目录 | `/mnt/workspace/features/` | run_experiment.py:161-164, 03_extract_features.py:46-52, 12_extract_bert_features.py:52-53 |
| 输出目录 | `/mnt/workspace/output/` | run_experiment.py:179-182, 04_train_hgnn.py:23-26, 05_train_gatv2.py:84, 06_evaluate.py:78, collect_results.py:29 |
| Llama-3 模型 | `/mnt/workspace/model/Llama-3-8B` | 03_extract_features.py:44 |
| BERT 模型 | `/mnt/workspace/model/bert-base-uncased` | 12_extract_bert_features.py:50 |
| 项目根目录 | `/mnt/workspace/fake-news-detection` | 01_clone_repos.sh:4, run_all_experiments.sh:16 |
| 虚拟环境 | `/mnt/workspace/venv/bin/activate` | run_all_experiments.sh:17 |
| all_results.csv | `/mnt/workspace/output/all_results.csv` | run_experiment.py:182, collect_results.py:29 |
| paper_tables.txt | `/mnt/workspace/output/paper_tables.txt` | collect_results.py:30 |

**如果修改这些路径，需要同步的文件列表（按改动范围）：**

| 改动 | 需修改的文件 |
|------|-------------|
| 数据目录前缀 | tagfn_dataset.py, 03_extract_features.py, 04_train_hgnn.py, 05_train_gatv2.py, 06_evaluate.py, 12_extract_bert_features.py |
| 特征目录 | run_experiment.py, 03_extract_features.py, 12_extract_bert_features.py |
| 输出目录 | run_experiment.py, collect_results.py, 04_train_hgnn.py, 05_train_gatv2.py, 06_evaluate.py |
| LLM 模型路径 | 03_extract_features.py |
| BERT 模型路径 | 12_extract_bert_features.py |

### 8.2 模型维度常量及影响范围

| 常量 | 默认值 | 影响文件 | 如果修改需同步 |
|------|--------|---------|---------------|
| Llama 特征维度 | 4096 | mhr_llm.py (`in_dim` 默认), gatv2_fallback.py (`in_dim` 默认), run_experiment.py (自动推断) | 模型默认值 + 特征提取脚本 |
| BERT 特征维度 | 768 | run_experiment.py (自动推断) | 仅特征提取脚本 |
| GNN 隐藏维度 | 128 | mhr_llm.py, baselines.py 全部, gatv2_fallback.py, run_experiment.py, model_config.yaml, hgnn_layer.py (in_features) | 所有模型文件 |
| MLP 中间层 (HGNN) | 512 | mhr_llm.py (`hidden_mlp`) | 仅主模型 |
| MLP 中间层 (GATv2) | 512 | gatv2_fallback.py (`hidden_mlp`) | 仅备用模型 |
| MLP 中间层 (基线) | 256 | baselines.py (MLPDetector, 硬编码) | 仅 MLP 基线 |
| 图卷积层数 | 2 | 所有模型 + run_experiment.py + model_config.yaml | 所有训练脚本 |
| 等效 batch size | 32 | run_experiment.py (`batch_size=4, accum_steps=8`) | 训练脚本 |

**接口不一致注意事项**：
- `MLPDetector` 的中间层 256（硬编码）vs 其他模型的 512。MLP 基线参数量比其他模型少，不是严格可比的消融实验
- `configs/model_config.yaml` 和 `configs/train_config.yaml` 中的路径值均为 `""`，配置文件**未被 run_experiment.py 实际读取**——所有路径在代码中硬编码

### 8.3 训练超参数默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 5e-4 | 初始学习率 |
| `weight_decay` | 1e-4 | AdamW 权重衰减 (HGNN 无 weight_decay) |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `warmup_ratio` | 0.1 | 前 10% steps warmup |
| `warmup` → `cosine_decay` | — | LR schedule: linear warmup → cosine decay to 0 |
| `early_stopping_patience` | 10 | 验证 F1 无提升 10 epoch 则停止 |
| `nan_tolerance` | 3 | 连续 NaN 次数上限 |
| `HGNNDropout` | 0.1 | 所有 dropout |
| `optimizer` (HGNN) | RiemannianAdam(stabilize=10) | geoopt |
| `optimizer` (others) | AdamW(weight_decay=1e-4) | torch |

---

## 9. all_results.csv 的格式

### 9.1 文件路径

`/mnt/workspace/output/all_results.csv`

### 9.2 写入逻辑

由 `run_experiment.py` 的 `main()` 函数在第 429-434 行写入。首次写入时自动创建表头（检测文件是否存在）。**追加模式**写入，不覆盖已有内容。

### 9.3 所有列（14 列）

| # | 列名 | Python 类型 | 示例值 | 说明 |
|---|------|-----------|--------|------|
| 1 | `exp_name` | str | `llama_hgnn_poli` | `--exp_name` 参数值，作为实验唯一标识 |
| 2 | `model_type` | str | `hgnn` | `--model_type` 参数值 |
| 3 | `feat_type` | str | `llama` | `--feat_type` 参数值 |
| 4 | `dataset` | str | `politifact` | `--dataset` 参数值 |
| 5 | `num_layers` | int | `2` | `--num_layers` 参数值 |
| 6 | `gnn_dim` | int | `128` | `--gnn_dim` 参数值 |
| 7 | `test_accuracy` | str (4 位小数) | `"0.8523"` | 测试准确率 |
| 8 | `test_f1_macro` | str (4 位小数) | `"0.8312"` | Macro F1 |
| 9 | `test_f1_weighted` | str (4 位小数) | `"0.8501"` | Weighted F1 |
| 10 | `test_precision` | str (4 位小数) | `"0.8234"` | 宏平均精确率 |
| 11 | `test_recall` | str (4 位小数) | `"0.8412"` | 宏平均召回率 |
| 12 | `test_auc` | str (4 位小数) | `"0.9123"` | ROC AUC |
| 13 | `best_epoch` | int | `23` | 达到最佳验证 F1 的 epoch |
| 14 | `best_val_f1` | str (4 位小数) | `"0.8456"` | 最佳验证 F1 值 |

### 9.4 collect_results.py 的依赖

**文件**：[scripts/collect_results.py](scripts/collect_results.py)

`collect_results.py` 通过 `csv.DictReader` 读取，`exp_name` 作为字典键。

**依赖的列名**：
- `exp_name` — **必须**，作为字典键，重复时保留最后一次
- `test_accuracy`, `test_f1_macro`, `test_auc` — **必须**，用于生成表格和计算提升幅度
- 其他列 — 缺失时显示 `-`，不阻塞

**硬编码的实验名列表（collect_results.py 第 37-68 行）**：
```python
TABLE_A_EXPS   # 9 个 Politifact 主对比实验
TABLE_B_EXPS   # 6 个消融实验
TABLE_C_EXPS   # 3 个 Gossipcop 跨数据集实验
```

**注意事项**：
- 如果同一 `exp_name` 多次运行（追加写入），`collect_results.py` 保留最后一次
- 数值在 CSV 中以 4 位小数字符串存储，解析时需 `float()` 转换
- `ALL_EXPS` 列表需与 `run_all_experiments.sh` 中的实验名保持同步

---

## 10. 当前未完成的 TODO / 待办事项

### 高优先级（影响功能正确性）

1. ✅ **[BUG] mhr_llm.py 导入不存在的 hyp_layers.py** — 已修复 (2026-05-12)
   - 直接导入 `hgnn_layer.HyperbolicGraphConvolution`，删除 try/except 链

2. **[数值稳定性] proju0 恒等映射可能偏离标准双曲几何**
   - 位置：[lorentz.py:29-31](src/models/manifolds/lorentz.py#L29-L31)
   - 影响：特征未经切空间约束直接进入流形
   - 修复：从 MHR 仓库获取完整 `proju0` 实现
   - 已有测试覆盖：`test_manifold.py::test_proju0_identity()`, `test_local_verify.py` 测试 3.1

3. **[功能缺失] 注意力聚合不支持**
   - 位置：[hgnn_layer.py:89](src/models/layers/hgnn_layer.py#L89)
   - 影响：无法使用 `use_att=True` 进行注意力消息传递
   - 修复：从 MHR 仓库复制 `att_layers.py`

### 中优先级（影响工程可用性）

4. **[路径硬编码] 所有路径以 `/mnt/workspace/` 为前缀**
   - 至少 7 个 Python 文件 + 2 个 Shell 脚本中有硬编码路径
   - 建议：使用环境变量或统一配置文件管理，参见 §8.1

5. **[接口不一致] MLPDetector 中间层 256 vs 其他模型 512**
   - MLP 基线参数量少，消融对比不公平
   - 修复：将 `MLPDetector.mlp` 的中间层改为 512 或参数化

6. **[配置未生效] YAML 配置文件不被读取**
   - `model_config.yaml` 和 `train_config.yaml` 存在但未被任何代码读取
   - 路径值均为空字符串 `""`
   - 修复：让 `run_experiment.py` 读取 YAML 配置文件，或用环境变量替换

### 低优先级（代码质量）

7. **[代码重复] `_root_node_indices` 在 3 个文件中重复**
   - `baselines.py:33-50`, `mhr_llm.py:188-205`, `gatv2_fallback.py:27-49`
   - 建议：抽取到 `src/utils/graph_utils.py`

8. **[代码重复] 训练循环在 3 个脚本中重复**
   - `run_experiment.py`, `04_train_hgnn.py`, `05_train_gatv2.py`
   - `04_` 和 `05_` 可能可以安全删除

9. ✅ **[文档过时] README.md** — 已删除 (2026-05-12)，CLAUDE.md 为唯一文档

10. **[缺少 docstring] metrics.py compute_metrics 无返回值文档**
    - 缺少 `y_prob=None` 时 AUC 为 -1.0 的行为说明

11. ✅ **[潜在问题] GATDetector 的双重 Dropout** — 审查后确认非 Bug (2026-05-12)
    - GATConv 内部 dropout 作用于注意力系数，F.dropout 作用于输出特征
    - GCN/SAGE/BiGCN 无内部 dropout，其 F.dropout 是唯一正则化来源

12. ✅ **[测试覆盖不足] test_manifold.py** — 已增强 (2026-05-12)
    - 新增 `test_proju0_identity()` — proju0 恒等映射验证与文档
    - 新增 `test_manifold_edge_cases()` — 极大范数/零范数/单元素/空输入/最小维度/较大维度
    - 新增 `test_minkowski_constraint()` — 多 k 值 Minkowski 约束验证
    - 新增 `tests/test_local_verify.py` — 8 项完整本地验证测试

---

## 11. 给予新 AI 助手的特别注意事项

### 11.1 起源与命名约定

**类名 / 文件名与实际功能对照：**

| 代码中的名称 | 实际是什么 | 注意事项 |
|-------------|-----------|---------|
| `mhr_llm.py` | 主模型文件 | "MHR" 是上游仓库名，非本项目概念 |
| `HyperbolicFakeNewsDetector` | 项目的双曲假新闻检测器 | 注意 `FakeNews` 拼写——Fake 和 News 之间无空格 |
| `HyperbolicGraphConvolution` | 双曲图卷积层 | 代码中也用作 `LorentzGraphConvolution` 的别名 |
| `LorentzLinear` / `LorentzAgg` | 内部层，外部不直接使用 | `Lorentz` ≠ 洛伦兹变换；此处指洛伦兹流形 |
| `GATv2FakeNewsDetector` | 备用模型 | fallback 强调其定位 |
| `FakeNewsDataset` | 数据集基类 | 不要与 `FakeNewsDetector` 混淆 |
| `PolitifactDataset` / `GossipcopDataset` | 数据子类 | 继承自 `FakeNewsDataset` |
| `tagfn_dataset.py` | 数据集文件 | "TAGFN" 是 HuggingFace 数据集名 |

**文件命名约定：**
- 所有 `.py` 文件首行 `# -*- coding: utf-8 -*-`
- 注释和 docstring 使用**中文**
- 脚本编号（03_, 04_, 05_, 12_）反映开发时的执行顺序，现在 `run_experiment.py` 是唯一主入口

### 11.2 导入约定（必须遵守）

```python
# ⚠️ 规则 1: 绝不要使用相对导入 (from ..data import ...)
# 所有脚本通过 python scripts/xxx.py 直接运行，不通过 python -m

# ⚠️ 规则 2: 所有脚本开头必须添加:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ⚠️ 规则 3: 然后使用完整 src 包路径导入:
from src.data.tagfn_dataset import PolitifactDataset, GossipcopDataset
from src.models.mhr_llm import HyperbolicFakeNewsDetector
from src.models.baselines import GCNDetector, GATDetector, ...
from src.models.gatv2_fallback import GATv2FakeNewsDetector
from src.utils.metrics import compute_metrics, EarlyStopping
from src.utils.memory_utils import cuda_memory_profiler

# ⚠️ 规则 4: __init__.py 导入:
# src/models/__init__.py  → 导出所有 7 个模型类
# src/data/__init__.py    → 导出 PolitifactDataset, GossipcopDataset
# src/utils/__init__.py   → 导出 cuda_memory_profiler, EarlyStopping, compute_metrics
```

### 11.3 数据类型约定

| 存储/操作 | 数据类型 | 说明 |
|----------|---------|------|
| 预提取特征存储 (.pt) | `float16` | 节省磁盘/内存 |
| 模型权重 | `float32` | PyTorch 默认 |
| 训练输入 (batch.x) | `float16` → 手动 `.float()` | **必须**转换！否则 float16 × float32 精度不足 |
| 边索引 | `int64` (`torch.long`) | PyG 标准 |
| 标签 | `int64` (`torch.long`) | CE loss 期望 |
| 稀疏邻接矩阵 | `float32` | `_edge_index_to_sparse_adj` 显式指定 |
| LLM 计算精度 (Llama) | `bfloat16` | bitsandbytes 4-bit 量化 + bf16 compute |
| BERT 计算精度 | `float32` | 无量化 |

**最容易出错的转换**：`batch.x.float()` —— 在训练和评估循环的**每个 batch** 调用 `.float()`，忘记会导致精度问题或 CUDA 错误。

### 11.4 特殊的编码模式（含易错点）

**模式 1: 梯度累积 + NaN 检测**
```python
for step, batch in enumerate(train_loader):
    loss = F.cross_entropy(logits, batch.y.squeeze())
    loss = loss / accum_steps      # ① 先缩放
    loss.backward()                 # ② 反向传播

    if torch.isnan(loss):           # ③ NaN 检测
        nan_count += 1
        optimizer.zero_grad()       # ④ 清除异常梯度 ← 这一步容易忘!
        if nan_count >= 3: sys.exit(1)
        continue                    # ⑤ 跳过这个 batch

    if (step + 1) % accum_steps == 0:  # ⑥ 累积够了
        clip_grad_norm_(max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```
**易错点**：NaN 时跳过 `zero_grad()` 会导致异常梯度残留在 `.grad` 中，污染下一个正常 batch。

**模式 2: 根节点提取**
```python
def _root_node_indices(batch):
    change = torch.ones(batch.numel(), dtype=torch.bool)
    change[1:] = batch[1:] != batch[:-1]    # 检测 batch 向量的不连续点
    roots = torch.nonzero(change, as_tuple=True)[0]
    return roots
```
**前提假设**：PyG 的 DataLoader 保证同一子图节点连续，且每个子图第一个节点是根节点。**如果数据预处理改变了节点排列顺序，此方法会失败**。

**模式 3: 双曲层后的投影校正（强制约定）**
```python
for layer in self.hgnn_layers:
    h, _ = layer((h, adj))
    h = self.manifold.projx(h)   # ← 必须调用！省略 → NaN
```

**模式 4: 时间维度的生命周期**
```
欧氏空间 [num_nodes, 128]
  → 添加时间维 → [num_nodes, 129]
  → expmap0 → [num_nodes, 129] (流形上)
  → HGNN 传播 → [num_nodes, 129] (流形上)
  → logmap0 → [num_nodes, 128] (回到欧氏空间，时间维被丢弃)
```
**进入流形前加时间维，离开后自动去掉。这是个单向过程。**

**模式 5: LorentzLinear 的特殊 weight 初始化**
```python
nn.init.uniform_(self.weight.weight, -stdv, stdv)  # stdv = 1/sqrt(out_features)
for idx in range(0, self.in_features, step):
    self.weight.weight[:, idx] = 0   # step = in_features, 所以只清零第 0 列
```
这个模式只清零第 0 列（时间分量对应的输入维度）。如果修改 `in_features` 或 `out_features`，这个初始化逻辑不变，但效果可能不同。

### 11.5 添加新模型的 checklist

1. **创建模型文件** (`src/models/new_model.py`):
   - `__init__(self, in_dim, gnn_dim, num_layers=2, dropout=0.1)` 接口
   - `forward(self, x, edge_index, batch) → [batch_size, 2]` 接口
   - 使用 `_root_node_indices(batch)` 提取根节点（或从 `src/utils/` 导入）
   - 处理 `batch.x.float()` 输入

2. **注册模型**：
   - `src/models/__init__.py` 添加导出
   - `scripts/run_experiment.py` 的 `parse_args()` `--model_type` choices 添加
   - `scripts/run_experiment.py` 的 `build_model()` 添加分支

3. **注册实验**：
   - `scripts/run_all_experiments.sh` 添加实验命令
   - `scripts/collect_results.py` 的 `TABLE_*_EXPS` 添加实验名

### 11.6 调试技巧

- **检查 NaN 来源**：在 HGNN forward 的每个步骤后 `assert not torch.isnan(h).any(), f"NaN at step {name}"`
- **显存监控**：`cuda_memory_profiler("描述")` 任意位置打印 GPU 显存
- **验证流形约束**：`manifold.inner(h, h)` 应接近 `-manifold.k`（Lorentz 模型上 Minkowski 内积）
- **特征范围**：进入 `expmap0` 前，空间部分范数不应过大（否则 cosh/sinh 溢出）
- **检查 feature_cache_path**：`.pt` 文件的 `shape[1]` 决定了 `in_dim` (llama→4096, bert→768)
- **复现最佳模型**：checkpoint 中保存了 `curvature`, `num_layers`, `model_type`，可用于精确复现

### 11.7 未列出的脚本文件说明

**scripts/02_download_dataset.py** (325 行)：
- 从 HuggingFace `kayzliu/TAGFN` 下载数据集
- 支持自动检测 repo_id（`kayzliu/TAGFN` vs `kayzliu/tagfn`）
- 加载所有 config/split，统计全局节点数、边数、标签分布
- 使用 `ds.save_to_disk()` 保存为 HuggingFace DatasetDict 格式
- 参数: `--output` (默认 `/mnt/workspace/data/tagfn`), `--cache-dir`, `--stats-max-graphs`
- 数据格式与 `FakeNewsDataset` 加载的格式**不同**——02 下载的是 HF Dataset 格式，实际数据由子类直接从原始文件加载

**scripts/01_clone_repos.sh** (61 行)：
- 克隆 `guoxinyu0617/MHR` 仓库到 `/tmp/MHR`
- 尝试多个可能的目录结构复制 manifolds/ 和 layers/ 到项目
- 用于获取完整 MHR 流形实现（当前项目使用简化版 lorentz.py）

**scripts/run_all_experiments.sh** (85 行)：
- 按 4 个阶段顺序运行所有实验
- 阶段 1: Politifact BERT 基线 (5 个实验)
- 阶段 2: Politifact Llama 基线 (3 个实验, HGNN 被注释)
- 阶段 3: 消融实验 (3 个实验)
- 阶段 4: Gossipcop 跨数据集 (3 个实验)
- 结束后自动调用 `collect_results.py`
- 使用 `set -e` 遇错停止，通过 `tee -a` 记录时间线
- 硬编码: 工作目录 `/mnt/workspace/fake-news-detection`, 虚拟环境 `/mnt/workspace/venv/`

### 11.8 运行完整实验的正确顺序

```bash
# 前提: 激活虚拟环境
source /mnt/workspace/venv/bin/activate

# 步骤 1: (可选) 克隆 MHR 仓库获取完整流形代码
bash scripts/01_clone_repos.sh

# 步骤 2: (可选) 下载原始 TAGFN 数据集
# 注意: 实际训练使用的数据由 FakeNewsDataset 直接从原始文件加载，
# 不依赖此脚本的输出格式
python scripts/02_download_dataset.py

# 步骤 3: 提取 Llama-3-8B 特征 (耗时最长)
python scripts/03_extract_features.py --dataset politifact
python scripts/03_extract_features.py --dataset gossipcop

# 步骤 4: 提取 BERT 特征 (基线用)
python scripts/12_extract_bert_features.py --dataset politifact
python scripts/12_extract_bert_features.py --dataset gossipcop

# 步骤 5: 运行实验 (单个或批量)
# 单个实验:
python scripts/run_experiment.py --model_type hgnn --feat_type llama \
    --dataset politifact --exp_name llama_hgnn_poli

# 批量运行:
bash scripts/run_all_experiments.sh

# 步骤 6: 收集结果生成论文表格
python scripts/collect_results.py
# 输出: /mnt/workspace/output/paper_tables.txt
# 同时打印到 stdout
```

### 11.9 常见错误排查

| 错误信息 | 可能原因 | 解决方法 |
|---------|---------|---------|
| `特征文件不存在` | 未运行特征提取脚本 | 先运行 `03_extract_features.py` 或 `12_extract_bert_features.py` |
| `节点特征未加载!` | `feature_cache_path` 路径错误 | 检查 `.pt` 文件路径 |
| `NaN loss (第 N 次)` | 双曲空间数值不稳定 | 切换到 `--model_type gatv2`；降低 `--lr`；检查特征范围 |
| `continu NaN超限，训练终止` | 连续 3 次 NaN | 切换到欧氏模型或调整 curvature |
| `无法导入 HyperbolicGraphConvolution` | hgnn_layer.py 缺失 | 检查 `src/models/layers/hgnn_layer.py` 存在 |
| `RiemannianAdam 不可用` | geoopt 未安装 | `pip install geoopt==0.5.0`；会自动 fallback 到 AdamW |
| `CUDA out of memory` | batch 太大 | 降低 `--batch_size`；增加 `--accum_steps` |
| `特征维度不匹配` | 用 BERT 特征但选了 llama feat_type | 检查 `--feat_type` 与特征文件匹配 |

---

## 12. 当前实验结果记录 (2026-05-12)

### 12.1 实验完成状态

数据来源：`/mnt/workspace/output/all_results.csv`（通过 `collect_results.py` 收集）

| 实验名称 | 状态 | 说明 |
|----------|------|------|
| `bert_mlp_poli` | DONE | |
| `bert_gcn_poli` | DONE | |
| `bert_gat_poli` | DONE | |
| `bert_sage_poli` | DONE | |
| `bert_bigcn_poli` | DONE | |
| `llama_mlp_poli` | **MISSING** | 不在 `run_all_experiments.sh` 中 |
| `llama_gcn_poli` | DONE | |
| `llama_gatv2_poli` | DONE | |
| `llama_hgnn_poli` | **MISSING** | 在 `run_all_experiments.sh` 中被注释 (第 54 行) |
| `ablation_bert_hgnn` | DONE | |
| `ablation_1layer` | DONE | |
| `ablation_3layer` | DONE | |
| `bert_gcn_gossip` | DONE | |
| `bert_gat_gossip` | DONE | |
| `bert_hgnn_gossip` | DONE | |

已完成: 13/15，缺失: 2

### 12.2 表 A: Politifact 主对比结果

| Method | Accuracy | F1-macro | AUC |
|--------|----------|----------|-----|
| GCN (LLM) | 0.8643 | 0.8638 | 0.9688 |
| GATv2 (LLM) | 0.8462 | 0.8452 | 0.9491 |
| BiGCN (BERT) | 0.8552 | 0.8552 | 0.9213 |
| MLP (BERT) | 0.8054 | 0.8031 | 0.8762 |
| GraphSAGE (BERT) | 0.7964 | 0.7926 | 0.8631 |
| GCN (BERT) | 0.7285 | 0.7284 | 0.8030 |
| GAT (BERT) | 0.7104 | 0.7021 | 0.8002 |
| MLP (LLM) | - | - | - |
| **HGNN (LLM, Ours)** | - | - | - |

**关键发现：**
- 最强基线是 GCN (LLM): Accuracy 0.8643, AUC 0.9688
- LLM 特征 (4096d) 显著优于 BERT (768d)，在相同 GNN 架构下提升约 6-16 个百分点
- BiGCN 双向设计有效，BERT 特征下即达 0.8552

### 12.3 表 B: 消融实验结果

| Method | Accuracy | F1-macro | AUC |
|--------|----------|----------|-----|
| w/o Multi-layer (1 layer) | 0.8371 | 0.8346 | 0.9639 |
| 3 layers | 0.8643 | 0.8641 | 0.9568 |
| w/o Hyperbolic (LLM + GATv2) | 0.8462 | 0.8452 | 0.9491 |
| w/o LLM (BERT + HGNN) | 0.8281 | 0.8275 | 0.9188 |
| Full Model | - | - | - |
| w/o Graph (LLM + MLP) | - | - | - |

**关键发现：**
- 1 层 HGNN 已有竞争力 (AUC 0.9639)，3 层反而略降至 0.9568
- 去掉 LLM (BERT+HGNN) AUC 下降到 0.9188，说明 LLM 特征贡献大
- 去掉双曲空间 (LLM+GATv2) AUC 下降到 0.9491，说明双曲几何有益但不巨大

### 12.4 表 C: Gossipcop 跨数据集结果

| Method | Accuracy | F1-macro | AUC |
|--------|----------|----------|-----|
| **HGNN (BERT, Ours)** | 0.7480 | 0.7480 | 0.8257 |
| GAT (BERT) | 0.7457 | 0.7425 | 0.8219 |
| GCN (BERT) | 0.7488 | 0.7453 | 0.8212 |

**关键发现：**
- Gossipcop 上 HGNN 优势极小：AUC 0.8257 vs GCN 0.8212 (+0.4%)
- 三种方法表现接近，Gossipcop 任务可能更难或 BERT 特征不足以区分类别
- 缺少 LLM 特征的 Gossipcop 实验

### 12.5 待完成的关键实验

1. **`llama_hgnn_poli`** — 本文完整方法 (HGNN + LLM + Politifact)，最重要的结果
2. **`llama_mlp_poli`** — 纯文本基线 (MLP + LLM)，验证图结构是否必要
3. **`llama_hgnn_gossip`** — LLM + HGNN on Gossipcop，验证双曲方法是否能跨数据集泛化
4. **`llama_mlp_gossip`** — LLM + MLP on Gossipcop 纯文本对照

### 12.6 服务器部署下一步

需要在服务器上运行（从项目根目录 `/mnt/workspace/fake-news-detection/`）：

```bash
source /mnt/workspace/venv/bin/activate

# 最关键的两个缺失实验
python scripts/run_experiment.py --model_type hgnn --feat_type llama \
    --dataset politifact --exp_name llama_hgnn_poli

python scripts/run_experiment.py --model_type mlp --feat_type llama \
    --dataset politifact --exp_name llama_mlp_poli

# 重新生成论文表格
python scripts/collect_results.py
```
