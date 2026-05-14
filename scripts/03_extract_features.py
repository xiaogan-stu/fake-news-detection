"""
离线 LLM 特征提取脚本
用 Llama-3-8B (4-bit量化) 对所有节点文本做 mean pooling
输出: [num_nodes, 4096] float16
  - politifact: /mnt/workspace/features/node_features.pt
  - gossipcop:  /mnt/workspace/features/node_features_gossipcop.pt
支持断点续传

用法:
  python scripts/03_extract_features.py --dataset politifact
  python scripts/03_extract_features.py --dataset gossipcop
"""

import os, sys, gc, json, argparse, torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.tagfn_dataset import PolitifactDataset, GossipcopDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Llama-3 特征提取')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['politifact', 'gossipcop'],
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录（默认按 dataset 自动推断）')
    parser.add_argument('--output_path', type=str, default=None,
                        help='保存路径（默认 /mnt/workspace/features/node_features[_gossipcop].pt）')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批大小（默认 1，A10 24G + 4-bit 最稳）')
    parser.add_argument('--max_length', type=int, default=256,
                        help='文本截断长度（默认 256）')
    parser.add_argument('--save_every', type=int, default=200,
                        help='每 N 个节点保存一次检查点（默认 200）')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 路径配置 ──────────────────────────────────────────────────
    MODEL_PATH   = '/mnt/workspace/model/Llama-3-8B'
    DATA_DIR     = args.data_dir or f'/mnt/workspace/data/{args.dataset}'
    OUTPUT_DIR   = '/mnt/workspace/features'
    if args.output_path:
        FEAT_PATH = args.output_path
    elif args.dataset == 'politifact':
        FEAT_PATH = os.path.join(OUTPUT_DIR, 'node_features.pt')
    else:
        FEAT_PATH = os.path.join(OUTPUT_DIR, f'node_features_{args.dataset}.pt')
    CKPT_PATH    = os.path.join(OUTPUT_DIR, f'llama_extract_checkpoint_{args.dataset}.json')
    BATCH_SIZE   = args.batch_size
    MAX_LENGTH   = args.max_length
    SAVE_EVERY   = args.save_every
    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────────
    print("=" * 60)
    print("步骤 1: 加载数据集")
    DatasetCls = PolitifactDataset if args.dataset == 'politifact' else GossipcopDataset
    ds = DatasetCls(data_dir=DATA_DIR)
    texts = ds.get_all_node_texts()
    N = len(texts)
    print(f"  数据集: {args.dataset}, 总节点数: {N}")

    # ── 断点续传检查 ───────────────────────────────────────────────
    start_idx = 0
    features_buffer = []

    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            ckpt = json.load(f)
        start_idx = ckpt['next_idx']
        print(f"  发现检查点，从第 {start_idx} 个节点继续")
        partial_path = os.path.join(OUTPUT_DIR, f'features_partial_{args.dataset}_{start_idx}.pt')
        if os.path.exists(partial_path):
            features_buffer = [torch.load(partial_path, map_location='cpu')]
            print(f"  已加载 {start_idx} 个节点的特征")
    else:
        print("  未找到检查点，从头开始")

    # ── 加载 Llama-3 ──────────────────────────────────────────────
    print("\n步骤 2: 加载 Llama-3-8B (4-bit 量化)")
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"  模型加载完成, 显存占用: {allocated:.2f} GB")

    # ── 特征提取函数 ───────────────────────────────────────────────
    def encode_text(text: str) -> torch.Tensor:
        if not text or not text.strip():
            text = "[PAD]"

        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)
            hidden = outputs.last_hidden_state
            feat = hidden[0].mean(dim=0)
            feat = feat.to(torch.float16).cpu()

        del inputs, outputs, hidden
        return feat

    # ── 主提取循环 ────────────────────────────────────────────────
    print(f"\n步骤 3: 开始特征提取 ({start_idx} → {N})")
    print(f"  设备: {DEVICE}, batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}")

    current_batch = []

    for idx in tqdm(range(start_idx, N), desc="提取节点特征", unit="node"):
        feat = encode_text(texts[idx])
        current_batch.append(feat)

        if (idx + 1) % SAVE_EVERY == 0 or (idx + 1) == N:
            batch_tensor = torch.stack(current_batch)
            features_buffer.append(batch_tensor)
            current_batch = []

            all_feats = torch.cat(features_buffer, dim=0)
            partial_path = os.path.join(OUTPUT_DIR, f'features_partial_{args.dataset}_{idx+1}.pt')
            torch.save(all_feats, partial_path)

            with open(CKPT_PATH, 'w') as f:
                json.dump({'next_idx': idx + 1, 'partial_path': partial_path}, f)

            for old_idx in [idx + 1 - SAVE_EVERY]:
                old_path = os.path.join(OUTPUT_DIR, f'features_partial_{args.dataset}_{old_idx}.pt')
                if os.path.exists(old_path) and old_path != partial_path:
                    os.remove(old_path)

            mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"\n  [检查点] 已完成 {idx+1}/{N}, 显存: {mem:.2f}GB")

    # ── 最终保存 ──────────────────────────────────────────────────
    print("\n步骤 4: 合并并保存最终特征")
    all_features = torch.cat(features_buffer, dim=0)
    print(f"  最终特征形状: {all_features.shape}")
    assert all_features.shape[0] == N, f"节点数不匹配! {all_features.shape[0]} vs {N}"

    torch.save(all_features, FEAT_PATH)
    print(f"  已保存: {FEAT_PATH}")
    print(f"  文件大小: {os.path.getsize(FEAT_PATH) / 1024**2:.1f} MB")

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(f'features_partial_{args.dataset}_'):
            os.remove(os.path.join(OUTPUT_DIR, f))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  Llama-3 已从显存卸载, 剩余: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    print(f"\n特征提取完成! 输出: {FEAT_PATH}")


if __name__ == '__main__':
    main()
