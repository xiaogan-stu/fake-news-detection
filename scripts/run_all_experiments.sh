#!/bin/bash
# ================================================================
# run_all_experiments.sh
# 按照实验顺序依次运行所有实验，记录每个实验的开始/结束时间。
#
# 用法:
#   bash scripts/run_all_experiments.sh
#
# 前提:
#   1. 03_extract_features.py 已运行（Llama 特征）
#   2. 12_extract_bert_features.py 已运行（BERT 特征）
#   3. 01_clone_repos.sh 已运行（MHR 流形代码）
# ================================================================

set -e  # 任何命令失败立即停止
cd /mnt/workspace/fake-news-detection
source /mnt/workspace/venv/bin/activate

BASE_CMD="python scripts/run_experiment.py"
LOG_FILE="/mnt/workspace/output/experiment_timeline.log"
mkdir -p /mnt/workspace/output

# ── 辅助函数：带时间戳运行实验 ──────────────────────────────────
run_exp() {
    local exp_name=$1
    shift
    echo "[$(date '+%H:%M:%S')] 开始: $exp_name" | tee -a $LOG_FILE
    $BASE_CMD "$@" --exp_name $exp_name
    echo "[$(date '+%H:%M:%S')] 完成: $exp_name" | tee -a $LOG_FILE
    echo "---" | tee -a $LOG_FILE
}

echo "========== 实验开始 ==========" | tee $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE

# ========== 阶段1：Politifact 基线对比（BERT特征）==========
echo "" | tee -a $LOG_FILE
echo "[阶段1] Politifact 基线对比 (BERT 特征)..." | tee -a $LOG_FILE

run_exp "bert_mlp_poli"   --model_type mlp   --feat_type bert --dataset politifact
run_exp "bert_gcn_poli"   --model_type gcn   --feat_type bert --dataset politifact
run_exp "bert_gat_poli"   --model_type gat   --feat_type bert --dataset politifact
run_exp "bert_sage_poli"  --model_type sage  --feat_type bert --dataset politifact
run_exp "bert_bigcn_poli" --model_type bigcn --feat_type bert --dataset politifact

# ========== 阶段2：Politifact LLM增强基线（Llama特征）==========
echo "" | tee -a $LOG_FILE
echo "[阶段2] Politifact LLM增强基线 (Llama 特征)..." | tee -a $LOG_FILE

run_exp "llama_mlp_poli"   --model_type mlp   --feat_type llama --dataset politifact
run_exp "llama_gcn_poli"   --model_type gcn   --feat_type llama --dataset politifact
run_exp "llama_gatv2_poli" --model_type gatv2 --feat_type llama --dataset politifact
# 本文方法（若已单独跑过，可取消注释跳过）
# run_exp "llama_hgnn_poli" --model_type hgnn --feat_type llama --dataset politifact

# ========== 阶段3：Politifact 消融实验 ==========
echo "" | tee -a $LOG_FILE
echo "[阶段3] 消融实验..." | tee -a $LOG_FILE

# 消融1：无双曲空间（llama_gatv2_poli 可直接复用）
# 消融2：无LLM（BERT特征 + HGNN）
run_exp "ablation_bert_hgnn" --model_type hgnn --feat_type bert  --dataset politifact
# 消融3：无图结构（llama_mlp_poli 可直接复用）
# 消融4：层数消融
run_exp "ablation_1layer"    --model_type hgnn --feat_type llama --dataset politifact --num_layers 1
run_exp "ablation_3layer"    --model_type hgnn --feat_type llama --dataset politifact --num_layers 3

# ========== 阶段4：Gossipcop 跨数据集实验 ==========
echo "" | tee -a $LOG_FILE
echo "[阶段4] Gossipcop 跨数据集..." | tee -a $LOG_FILE

run_exp "bert_gcn_gossip"  --model_type gcn  --feat_type bert --dataset gossipcop
run_exp "bert_gat_gossip"  --model_type gat  --feat_type bert --dataset gossipcop
run_exp "bert_hgnn_gossip" --model_type hgnn --feat_type bert --dataset gossipcop

echo "" | tee -a $LOG_FILE
echo "========== 全部实验完成 ==========" | tee -a $LOG_FILE
echo "结束时间: $(date)" | tee -a $LOG_FILE

# ── 自动收集结果 ──────────────────────────────────────────────
echo "" | tee -a $LOG_FILE
echo "生成论文表格..." | tee -a $LOG_FILE
python scripts/collect_results.py
echo "[$(date '+%H:%M:%S')] 结果收集完成" | tee -a $LOG_FILE
