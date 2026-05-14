# -*- coding: utf-8 -*-
"""
离线 BERT 特征提取脚本
用 bert-base-uncased 对所有节点文本做 mean pooling
输出: [num_nodes, 768] float16 tensor
  - politifact: [41054, 768]
  - gossipcop:  [314262, 768]
支持断点续传

用法:
  python scripts/12_extract_bert_features.py --dataset politifact
  python scripts/12_extract_bert_features.py --dataset gossipcop --batch_size 64
"""

import os
import sys
import gc
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.tagfn_dataset import PolitifactDataset, GossipcopDataset


def parse_args():
    parser = argparse.ArgumentParser(description='BERT 特征提取')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['politifact', 'gossipcop'],
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录（默认按 dataset 自动推断）')
    parser.add_argument('--output_path', type=str, default=None,
                        help='保存路径（默认 /mnt/workspace/features/bert_features_{dataset}.pt）')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小（默认 32，BERT 比 Llama 小得多）')
    parser.add_argument('--max_length', type=int, default=256,
                        help='文本截断长度（默认 256）')
    parser.add_argument('--save_every', type=int, default=2000,
                        help='每 N 个节点保存一次检查点（默认 2000）')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 路径配置 ──────────────────────────────────────────────────
    MODEL_PATH = '/mnt/workspace/model/bert-base-uncased'
    DATA_DIR = args.data_dir or f'/mnt/workspace/data/{args.dataset}'
    OUTPUT_DIR = '/mnt/workspace/features'
    FEAT_PATH = args.output_path or os.path.join(OUTPUT_DIR, f'bert_features_{args.dataset}.pt')
    CKPT_PATH = os.path.join(OUTPUT_DIR, f'bert_extract_checkpoint_{args.dataset}.json')
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    SAVE_EVERY = args.save_every
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────────
    print('=' * 60)
    print('步骤 1: 加载数据集')
    DatasetCls = PolitifactDataset if args.dataset == 'politifact' else GossipcopDataset
    ds = DatasetCls(data_dir=DATA_DIR)
    texts = ds.get_all_node_texts()
    N = len(texts)
    print(f'  数据集: {args.dataset}, 总节点数: {N}')

    # ── 断点续传检查 ───────────────────────────────────────────────
    start_idx = 0
    features_buffer = []

    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            ckpt = json.load(f)
        start_idx = ckpt['next_idx']
        print(f'  发现检查点，从第 {start_idx} 个节点继续')
        partial_path = os.path.join(OUTPUT_DIR, f'bert_features_partial_{args.dataset}_{start_idx}.pt')
        if os.path.exists(partial_path):
            features_buffer = [torch.load(partial_path, map_location='cpu')]
            print(f'  已加载 {start_idx} 个节点的特征')
    else:
        print('  未找到检查点，从头开始')

    # ── 加载 BERT ──────────────────────────────────────────────────
    print(f'\n步骤 2: 加载 bert-base-uncased (float32)')
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # 冻结所有参数
    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        print(f'  模型加载完成, 显存占用: {allocated:.2f} GB')

    # ── 批量特征提取函数 ───────────────────────────────────────────
    def encode_batch(text_batch: list) -> torch.Tensor:
        """
        批量文本 → [batch_size, 768] float16 特征
        流程: tokenize → BERT forward → last hidden state → mean pooling

        参数：
            text_batch: 字符串列表，长度 = batch_size

        返回：
            [batch_size, 768] float16 tensor（CPU）
        """
        # 空文本兜底
        text_batch = [t if t and t.strip() else '[PAD]' for t in text_batch]

        inputs = tokenizer(
            text_batch,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            # last_hidden_state: [batch_size, seq_len, 768]
            hidden = outputs.last_hidden_state

            # mean pooling: 对 seq_len 维度取平均 → [batch_size, 768]
            # 注意：需要考虑 padding mask，只对非 padding 位置取平均
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [batch_size, seq_len, 1]
            masked_hidden = hidden * attention_mask                    # [batch_size, seq_len, 768]
            lengths = attention_mask.sum(dim=1).clamp(min=1e-9)       # [batch_size, 1]
            feat = masked_hidden.sum(dim=1) / lengths                 # [batch_size, 768]

        feat = feat.to(torch.float16).cpu()
        del inputs, outputs, hidden
        return feat  # [batch_size, 768]

    # ── 主提取循环 ────────────────────────────────────────────────
    print(f'\n步骤 3: 开始特征提取 ({start_idx} → {N})')
    print(f'  设备: {DEVICE}, batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}')

    current_batch = []

    for idx in tqdm(range(start_idx, N), desc='提取节点特征', unit='node'):
        current_batch.append(texts[idx])

        # 凑够一个 batch 或到达末尾时编码
        if len(current_batch) == BATCH_SIZE or idx == N - 1:
            feat = encode_batch(current_batch)  # [len(current_batch), 768]
            features_buffer.append(feat)
            current_batch = []

        # 定期保存检查点
        if (idx + 1) % SAVE_EVERY == 0 or (idx + 1) == N:
            all_feats = torch.cat(features_buffer, dim=0)  # [已处理数, 768]
            partial_path = os.path.join(
                OUTPUT_DIR, f'bert_features_partial_{args.dataset}_{idx + 1}.pt'
            )
            torch.save(all_feats, partial_path)

            # 更新检查点
            with open(CKPT_PATH, 'w') as f:
                json.dump({'next_idx': idx + 1, 'partial_path': partial_path}, f)

            # 清理旧检查点文件（只保留最新）
            for old_idx_offset in range(max(0, idx + 1 - SAVE_EVERY * 3), idx + 1, SAVE_EVERY):
                if old_idx_offset == idx + 1:
                    continue
                old_path = os.path.join(
                    OUTPUT_DIR, f'bert_features_partial_{args.dataset}_{old_idx_offset}.pt'
                )
                if os.path.exists(old_path) and old_path != partial_path:
                    os.remove(old_path)

            # 打印进度
            mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
            print(f'\n  [检查点] 已完成 {idx + 1}/{N}, 显存: {mem:.2f}GB')

    # ── 最终保存 ──────────────────────────────────────────────────
    print('\n步骤 4: 合并并保存最终特征')
    all_features = torch.cat(features_buffer, dim=0)  # [N, 768]
    print(f'  最终特征形状: {all_features.shape}')
    assert all_features.shape[0] == N, f'节点数不匹配! {all_features.shape[0]} vs {N}'
    assert all_features.shape[1] == 768, f'特征维度不匹配! {all_features.shape[1]} vs 768'

    torch.save(all_features, FEAT_PATH)
    file_size_mb = os.path.getsize(FEAT_PATH) / 1024 ** 2
    print(f'  已保存: {FEAT_PATH}')
    print(f'  文件大小: {file_size_mb:.1f} MB')

    # 清理检查点
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(f'bert_features_partial_{args.dataset}_'):
            os.remove(os.path.join(OUTPUT_DIR, f))

    # ── 统计信息 ──────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('特征提取统计:')
    print(f'  数据集:       {args.dataset}')
    print(f'  节点数:       {N}')
    print(f'  特征维度:     768')
    print(f'  特征类型:     float16')
    print(f'  文件路径:     {FEAT_PATH}')
    print(f'  文件大小:     {file_size_mb:.1f} MB')
    print(f'  NaN 数量:     {torch.isnan(all_features).sum().item()}')
    print(f'  特征范围:     [{all_features.min().item():.4f}, {all_features.max().item():.4f}]')
    print('=' * 60)

    # ── 释放显存 ──────────────────────────────────────────────────
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'  BERT 已从显存卸载, 剩余: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB')

    print('\nBERT 特征提取完成!')


if __name__ == '__main__':
    main()
