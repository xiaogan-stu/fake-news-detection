# -*- coding: utf-8 -*-
"""
collect_results.py
------------------
读取所有实验结果，生成论文格式的表格（Markdown + LaTeX）。

输入: /mnt/workspace/output/all_results.csv（由 run_experiment.py 追加写入）
输出: /mnt/workspace/output/paper_tables.txt

生成三张表:
  表A - 主对比表（Politifact，各方法对比）
  表B - 消融实验表
  表C - 跨数据集表（Gossipcop）

用法:
  python scripts/collect_results.py
"""

import os
import csv
import sys
from collections import OrderedDict


# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════

RESULTS_PATH = '/mnt/workspace/output/all_results.csv'
OUTPUT_PATH = '/mnt/workspace/output/paper_tables.txt'

# 本文方法标识（用于加 ** 标注）
OUR_METHOD_KEYWORDS = ['hgnn']

# ── 表A：主对比表（Politifact 基线对比）──────────────────────────
# 实验名 → 论文中的方法名
TABLE_A_EXPS = OrderedDict([
    ('bert_mlp_poli',        'MLP (BERT)'),
    ('bert_gcn_poli',        'GCN (BERT)'),
    ('bert_gat_poli',        'GAT (BERT)'),
    ('bert_sage_poli',       'GraphSAGE (BERT)'),
    ('bert_bigcn_poli',      'BiGCN (BERT)'),
    ('llama_mlp_poli',       'MLP (LLM)'),
    ('llama_gcn_poli',       'GCN (LLM)'),
    ('llama_gatv2_poli',     'GATv2 (LLM)'),
    ('llama_hgnn_poli',      '**HGNN (LLM, Ours)**'),
])

# ── 表B：消融实验表 ──────────────────────────────────────────────
TABLE_B_EXPS = OrderedDict([
    ('llama_hgnn_poli',      'Full Model'),
    ('ablation_bert_hgnn',   'w/o LLM (BERT + HGNN)'),
    ('llama_gatv2_poli',     'w/o Hyperbolic (LLM + GATv2)'),
    ('llama_mlp_poli',       'w/o Graph (LLM + MLP)'),
    ('ablation_1layer',      'w/o Multi-layer (1 layer)'),
    ('ablation_3layer',      '3 layers'),
])

# ── 表C：跨数据集表（Gossipcop）──────────────────────────────────
TABLE_C_EXPS = OrderedDict([
    ('bert_gcn_gossip',      'GCN (BERT)'),
    ('bert_gat_gossip',      'GAT (BERT)'),
    ('bert_hgnn_gossip',     '**HGNN (BERT, Ours)**'),
])

# ── 所有实验（用于检查完成状态）──────────────────────────────────
ALL_EXPS = list(TABLE_A_EXPS.keys()) + list(TABLE_B_EXPS.keys()) + list(TABLE_C_EXPS.keys())
ALL_EXPS = list(dict.fromkeys(ALL_EXPS))  # 去重保序


# ═══════════════════════════════════════════════════════════════════
# 数据读取
# ═══════════════════════════════════════════════════════════════════

def load_results(path: str) -> dict:
    """
    读取 all_results.csv，返回 {exp_name: {col: value}} 字典。

    参数：
        path: CSV 文件路径

    返回：
        以 exp_name 为键的字典
    """
    results = {}
    if not os.path.exists(path):
        print(f'[警告] 结果文件不存在: {path}')
        return results

    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('exp_name', '').strip()
            if name:
                # 如果同一实验出现多次，保留最后一次
                results[name] = row

    print(f'[信息] 读取到 {len(results)} 条实验结果')
    return results


# ═══════════════════════════════════════════════════════════════════
# 表格生成
# ═══════════════════════════════════════════════════════════════════

def _get_metric(results: dict, exp_name: str, key: str, default: str = '-') -> str:
    """安全获取指标值"""
    if exp_name not in results:
        return default
    val = results[exp_name].get(key, default)
    return val.strip() if val else default


def _is_our_method(display_name: str) -> bool:
    """判断是否为本文方法（用于加粗标注）"""
    return '**' in display_name


def generate_markdown_table(results: dict, exps: OrderedDict, title: str) -> str:
    """
    生成 Markdown 格式表格。

    参数：
        results: 实验结果字典
        exps:    {exp_name: display_name} 有序字典
        title:   表格标题

    返回：
        Markdown 格式字符串
    """
    lines = []
    lines.append(f'### {title}')
    lines.append('')
    lines.append('| Method | Accuracy | F1-macro | AUC |')
    lines.append('|--------|----------|----------|-----|')

    # 收集数据并按 AUC 降序排列
    rows = []
    for exp_name, display_name in exps.items():
        acc = _get_metric(results, exp_name, 'test_accuracy')
        f1 = _get_metric(results, exp_name, 'test_f1_macro')
        auc = _get_metric(results, exp_name, 'test_auc')
        try:
            auc_val = float(auc)
        except (ValueError, TypeError):
            auc_val = -1.0
        rows.append((display_name, acc, f1, auc, auc_val))

    # 按 AUC 降序排列
    rows.sort(key=lambda r: r[4], reverse=True)

    for display_name, acc, f1, auc, _ in rows:
        lines.append(f'| {display_name} | {acc} | {f1} | {auc} |')

    lines.append('')
    return '\n'.join(lines)


def generate_latex_table(results: dict, exps: OrderedDict, title: str, label: str) -> str:
    """
    生成 LaTeX 格式表格。

    参数：
        results: 实验结果字典
        exps:    {exp_name: display_name} 有序字典
        title:   表格标题
        label:   LaTeX label

    返回：
        LaTeX 格式字符串
    """
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(f'  \\caption{{{title}}}')
    lines.append(f'  \\label{{{label}}}')
    lines.append(r'  \begin{tabular}{lccc}')
    lines.append(r'    \toprule')
    lines.append(r'    Method & Accuracy & F1-macro & AUC \\')
    lines.append(r'    \midrule')

    # 收集数据并按 AUC 降序排列
    rows = []
    for exp_name, display_name in exps.items():
        acc = _get_metric(results, exp_name, 'test_accuracy')
        f1 = _get_metric(results, exp_name, 'test_f1_macro')
        auc = _get_metric(results, exp_name, 'test_auc')
        try:
            auc_val = float(auc)
        except (ValueError, TypeError):
            auc_val = -1.0
        # LaTeX 中去掉 ** 标记，改用 \textbf
        clean_name = display_name.replace('**', '')
        is_ours = _is_our_method(display_name)
        rows.append((clean_name, acc, f1, auc, auc_val, is_ours))

    rows.sort(key=lambda r: r[4], reverse=True)

    for clean_name, acc, f1, auc, _, is_ours in rows:
        if is_ours:
            lines.append(f'    \\textbf{{{clean_name}}} & \\textbf{{{acc}}} & \\textbf{{{f1}}} & \\textbf{{{auc}}} \\\\')
        else:
            lines.append(f'    {clean_name} & {acc} & {f1} & {auc} \\\\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')
    return '\n'.join(lines)


def compute_improvement(results: dict, exps: OrderedDict) -> str:
    """
    计算本文方法相对最强基线的提升幅度。

    参数：
        results: 实验结果字典
        exps:    {exp_name: display_name} 有序字典

    返回：
        提升信息字符串
    """
    # 先检查本文方法数据是否存在
    our_exp_name = None
    for exp_name, display_name in exps.items():
        if _is_our_method(display_name):
            our_exp_name = exp_name
            break
    if our_exp_name and our_exp_name not in results:
        return '\n'.join([
            '### 本文方法相对最强基线的提升',
            '',
            '- ⚠️ 本文方法尚未运行，无提升数据',
            f'- 缺失实验: `{our_exp_name}`',
            '',
        ])

    our_auc = None
    best_baseline_auc = -1.0
    best_baseline_name = ''
    our_acc = None
    best_baseline_acc = -1.0
    our_f1 = None
    best_baseline_f1 = -1.0

    for exp_name, display_name in exps.items():
        is_ours = _is_our_method(display_name)
        try:
            auc = float(_get_metric(results, exp_name, 'test_auc', '-1'))
            acc = float(_get_metric(results, exp_name, 'test_accuracy', '-1'))
            f1 = float(_get_metric(results, exp_name, 'test_f1_macro', '-1'))
        except (ValueError, TypeError):
            continue

        if is_ours:
            our_auc = auc
            our_acc = acc
            our_f1 = f1
        else:
            if auc > best_baseline_auc:
                best_baseline_auc = auc
                best_baseline_name = display_name.replace('**', '')
            if acc > best_baseline_acc:
                best_baseline_acc = acc
            if f1 > best_baseline_f1:
                best_baseline_f1 = f1

    lines = []
    lines.append('### 本文方法相对最强基线的提升')
    lines.append('')

    if our_auc is not None and best_baseline_auc > 0:
        imp_auc = (our_auc - best_baseline_auc) / best_baseline_auc * 100
        imp_acc = (our_acc - best_baseline_acc) / best_baseline_acc * 100 if best_baseline_acc > 0 else 0
        imp_f1 = (our_f1 - best_baseline_f1) / best_baseline_f1 * 100 if best_baseline_f1 > 0 else 0
        lines.append(f'- 最强基线: {best_baseline_name}')
        lines.append(f'- Accuracy 提升: +{imp_acc:.2f}% ({best_baseline_acc:.4f} → {our_acc:.4f})')
        lines.append(f'- F1-macro 提升: +{imp_f1:.2f}% ({best_baseline_f1:.4f} → {our_f1:.4f})')
        lines.append(f'- AUC 提升:      +{imp_auc:.2f}% ({best_baseline_auc:.4f} → {our_auc:.4f})')
    else:
        lines.append('- 无法计算提升（本文方法或基线结果缺失）')

    lines.append('')
    return '\n'.join(lines)


def check_experiment_status(results: dict) -> str:
    """
    检查所有实验的完成状态。

    参数：
        results: 实验结果字典

    返回：
        状态信息字符串
    """
    lines = []
    lines.append('### 实验完成状态')
    lines.append('')
    lines.append('| 实验名称 | 状态 |')
    lines.append('|----------|------|')

    done_count = 0
    missing_count = 0

    for exp_name in ALL_EXPS:
        if exp_name in results:
            status = 'DONE'
            done_count += 1
        else:
            status = 'MISSING'
            missing_count += 1
        lines.append(f'| {exp_name} | {status} |')

    lines.append('')
    lines.append(f'已完成: {done_count}/{len(ALL_EXPS)}, 缺失: {missing_count}')
    lines.append('')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('收集实验结果，生成论文表格')
    print('=' * 60)

    results = load_results(RESULTS_PATH)

    if not results:
        print('[警告] 没有实验结果，生成空表格模板')

    output_lines = []

    # ── 标题 ──────────────────────────────────────────────────────
    output_lines.append('# 实验结果表格')
    output_lines.append('')
    output_lines.append(f'数据来源: {RESULTS_PATH}')
    output_lines.append(f'已收集实验数: {len(results)}')
    output_lines.append('')

    # ── 实验完成状态 ──────────────────────────────────────────────
    output_lines.append(check_experiment_status(results))

    # ── 表A：主对比表 ─────────────────────────────────────────────
    output_lines.append('---')
    output_lines.append('')
    output_lines.append('## 表A：主对比表 (Politifact)')
    output_lines.append('')
    output_lines.append(generate_markdown_table(results, TABLE_A_EXPS, '表A: Politifact 基线对比'))
    output_lines.append(generate_latex_table(results, TABLE_A_EXPS, 'Baseline Comparison on Politifact', 'tab:main_comparison'))
    output_lines.append(compute_improvement(results, TABLE_A_EXPS))

    # ── 表B：消融实验表 ──────────────────────────────────────────
    output_lines.append('---')
    output_lines.append('')
    output_lines.append('## 表B：消融实验表')
    output_lines.append('')
    output_lines.append(generate_markdown_table(results, TABLE_B_EXPS, '表B: 消融实验'))
    output_lines.append(generate_latex_table(results, TABLE_B_EXPS, 'Ablation Study', 'tab:ablation'))

    # ── 表C：跨数据集表 ──────────────────────────────────────────
    output_lines.append('---')
    output_lines.append('')
    output_lines.append('## 表C：跨数据集表 (Gossipcop)')
    output_lines.append('')
    output_lines.append(generate_markdown_table(results, TABLE_C_EXPS, '表C: Gossipcop 跨数据集'))
    output_lines.append(generate_latex_table(results, TABLE_C_EXPS, 'Cross-dataset Evaluation on Gossipcop', 'tab:cross_dataset'))
    output_lines.append(compute_improvement(results, TABLE_C_EXPS))

    # ── 写入文件 ──────────────────────────────────────────────────
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    content = '\n'.join(output_lines)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    # 同时打印到控制台
    print(content)
    print(f'\n表格已保存至: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
