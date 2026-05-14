# -*- coding: utf-8 -*-
"""
本地验证测试套件 (CPU only)
---------------------------
在拷贝到服务器之前，用本脚本在本地验证所有代码修改的正确性。

依赖: pip install torch numpy scikit-learn

运行: python tests/test_local_verify.py

测试覆盖:
  1. 所有模型 import + __init__
  2. 所有模型 forward 形状验证
  3. Lorentz 流形数值稳定性
  4. 边索引 → 邻接矩阵转换
  5. _root_node_indices 各场景
  6. compute_metrics / EarlyStopping
  7. collect_results compute_improvement 缺失数据
  8. 各模型参数量统计
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# 项目根目录加入 path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

PASS, FAIL, SKIP = 0, 0, 0
DEVICE = torch.device("cpu")


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  FAILED: {detail}")


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════
# 测试 1: 所有模型的 import + __init__
# ═══════════════════════════════════════════════════════════════════

def test_imports():
    section("测试 1: 模型导入与初始化")

    # 主模型
    try:
        from src.models.mhr_llm import HyperbolicFakeNewsDetector
        m = HyperbolicFakeNewsDetector(in_dim=4096, gnn_dim=128, num_layers=2,
                                        dropout=0.1, curvature=-1.0)
        check("HyperbolicFakeNewsDetector import+init", True,
              f"params={sum(p.numel() for p in m.parameters()):,}")
    except Exception as e:
        check("HyperbolicFakeNewsDetector import+init", False, str(e))

    # GATv2 备用模型
    try:
        from src.models.gatv2_fallback import GATv2FakeNewsDetector
        m = GATv2FakeNewsDetector(in_dim=4096, hidden_mlp=512, gnn_dim=128, dropout=0.1)
        check("GATv2FakeNewsDetector import+init", True,
              f"params={sum(p.numel() for p in m.parameters()):,}")
    except Exception as e:
        check("GATv2FakeNewsDetector import+init", False, str(e))

    # 5 个基线模型
    try:
        from src.models.baselines import MLPDetector
        m = MLPDetector(in_dim=4096, gnn_dim=128)
        check("MLPDetector import+init", True,
              f"params={sum(p.numel() for p in m.parameters()):,}")
    except Exception as e:
        check("MLPDetector import+init", False, str(e))

    try:
        from src.models.baselines import GCNDetector
        m = GCNDetector(in_dim=4096, gnn_dim=128, num_layers=2)
        check("GCNDetector import+init", True,
              f"params={sum(p.numel() for p in m.parameters()):,}")
    except Exception as e:
        check("GCNDetector import+init", False, str(e))

    try:
        from src.models.baselines import GATDetector
        m = GATDetector(in_dim=4096, gnn_dim=128, num_layers=2)
        check("GATDetector import+init", True,
              f"params={sum(p.numel() for p in m.parameters()):,}")
    except Exception as e:
        check("GATDetector import+init", False, str(e))

    try:
        from src.models.baselines import SAGEDetector
        m = SAGEDetector(in_dim=4096, gnn_dim=128, num_layers=2)
        check("SAGEDetector import+init", True,
              f"params={sum(p.numel() for p in m.parameters()):,}")
    except Exception as e:
        check("SAGEDetector import+init", False, str(e))

    try:
        from src.models.baselines import BiGCNDetector
        m = BiGCNDetector(in_dim=4096, gnn_dim=128, num_layers=2)
        check("BiGCNDetector import+init", True,
              f"params={sum(p.numel() for p in m.parameters()):,}")
    except Exception as e:
        check("BiGCNDetector import+init", False, str(e))

    # 流形
    try:
        from src.models.manifolds.lorentz import Lorentz
        mf = Lorentz(k=1.0, learnable=False)
        check("Lorentz manifold import+init", True)
    except Exception as e:
        check("Lorentz manifold import+init", False, str(e))
        mf = None

    # HGNN 层
    try:
        from src.models.layers.hgnn_layer import HyperbolicGraphConvolution
        mf2 = Lorentz(k=1.0) if mf is None else mf
        layer = HyperbolicGraphConvolution(
            manifold=mf2, in_features=129, out_features=129,
            use_bias=True, dropout=0.1, use_att=False, local_agg=True,
        )
        check("HyperbolicGraphConvolution import+init", True)
    except Exception as e:
        check("HyperbolicGraphConvolution import+init", False, str(e))

    # utils
    try:
        from src.utils.metrics import compute_metrics, EarlyStopping
        check("compute_metrics + EarlyStopping import", True)
    except Exception as e:
        check("compute_metrics + EarlyStopping import", False, str(e))

    try:
        from src.utils.memory_utils import cuda_memory_profiler, clear_gpu_memory
        check("memory_utils import", True)
    except Exception as e:
        check("memory_utils import", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 测试 2: 所有模型的 forward 形状验证
# ═══════════════════════════════════════════════════════════════════

def test_forward_shapes():
    section("测试 2: 模型 forward 形状验证")

    # 构造模拟输入: 2 个图，每个图 5 个节点
    B = 2
    N = 10
    in_dim = 4096

    x = torch.randn(N, in_dim)           # [num_nodes=10, 4096]
    edge_index = torch.tensor([          # [2, 8]  稀疏边
        [0, 1, 2, 3, 5, 6, 7, 8],
        [1, 2, 3, 4, 6, 7, 8, 9],
    ], dtype=torch.long)
    batch = torch.tensor(                # [10]  图0: 5节点, 图1: 5节点
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long
    )

    models_to_test = []

    # 主模型
    try:
        from src.models.mhr_llm import HyperbolicFakeNewsDetector
        models_to_test.append((
            "HyperbolicFakeNewsDetector",
            HyperbolicFakeNewsDetector(in_dim=in_dim, gnn_dim=128, num_layers=2,
                                        dropout=0.1, curvature=-1.0)
        ))
    except Exception:
        pass

    # GATv2
    try:
        from src.models.gatv2_fallback import GATv2FakeNewsDetector
        models_to_test.append((
            "GATv2FakeNewsDetector",
            GATv2FakeNewsDetector(in_dim=in_dim, hidden_mlp=512, gnn_dim=128, dropout=0.1)
        ))
    except Exception:
        pass

    # 基线
    try:
        from src.models.baselines import MLPDetector, GCNDetector, GATDetector, SAGEDetector, BiGCNDetector
        models_to_test.append(("MLPDetector", MLPDetector(in_dim=in_dim, gnn_dim=128)))
        models_to_test.append(("GCNDetector", GCNDetector(in_dim=in_dim, gnn_dim=128, num_layers=2)))
        models_to_test.append(("GATDetector", GATDetector(in_dim=in_dim, gnn_dim=128, num_layers=2)))
        models_to_test.append(("SAGEDetector", SAGEDetector(in_dim=in_dim, gnn_dim=128, num_layers=2)))
        models_to_test.append(("BiGCNDetector", BiGCNDetector(in_dim=in_dim, gnn_dim=128, num_layers=2)))
    except Exception:
        pass

    for name, model in models_to_test:
        model.eval()
        try:
            with torch.no_grad():
                logits = model(x, edge_index, batch)
            shape_ok = logits.shape == (B, 2)
            nan_ok = not torch.isnan(logits).any()
            inf_ok = not torch.isinf(logits).any()
            all_ok = shape_ok and nan_ok and inf_ok
            detail = f"shape={logits.shape}" if not shape_ok else ""
            if not nan_ok:
                detail += " NaN!"
            if not inf_ok:
                detail += " Inf!"
            check(f"{name} forward", all_ok, detail)
        except Exception as e:
            check(f"{name} forward", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 测试 3: Lorentz 流形数值稳定性
# ═══════════════════════════════════════════════════════════════════

def test_lorentz_manifold():
    section("测试 3: Lorentz 流形操作")

    try:
        from src.models.manifolds.lorentz import Lorentz
        manifold = Lorentz(k=1.0, learnable=False)
    except Exception as e:
        check("Lorentz 导入", False, str(e))
        return

    torch.manual_seed(42)
    B, D = 32, 128

    # 测试 3.1: proju0 恒等映射
    u_in = torch.randn(B, D)
    u_out = manifold.proju0(u_in)
    check("proju0 恒等映射", torch.allclose(u_in, u_out),
          f"max diff={(u_in - u_out).abs().max():.6f}")

    # 测试 3.2: expmap0 + logmap0 往返 (传入含时间维的张量)
    u = torch.randn(B, D + 1)  # 含时间维度 (第一列)
    h = manifold.expmap0(u, project=True)
    check("expmap0 无 NaN", not torch.isnan(h).any())
    check("expmap0 无 Inf", not torch.isinf(h).any())
    check("expmap0 形状", h.shape == (B, D + 1), f"got {h.shape}")

    # 测试 3.3: projx 幂等性
    h_proj = manifold.projx(h)
    h_proj2 = manifold.projx(h_proj)
    check("projx 幂等性", torch.allclose(h_proj, h_proj2, atol=1e-5),
          f"max diff={(h_proj - h_proj2).abs().max():.6f}")

    # 测试 3.4: logmap0 形状（应该去掉时间维度）
    t = manifold.logmap0(h)
    check("logmap0 去掉时间维度", t.shape == (B, D),
          f"got {t.shape}")
    check("logmap0 无 NaN", not torch.isnan(t).any())
    check("logmap0 无 Inf", not torch.isinf(t).any())

    # 测试 3.5: Minkowski 内积 (流形上的点应满足 <x,x>_L = -k)
    inner_self = manifold.inner(h, h, keepdim=False)
    expected = -torch.ones(B) * manifold.k
    # 由于 projx 投影，内积应接近 -k
    error = (inner_self - expected).abs().mean()
    check(f"Minkowski 内积约束 (error={error:.6f})", error < 0.1,
          f"mean error={error:.6f}")

    # 测试 3.6: logmap0 的 cosh 溢出保护
    u_big = torch.randn(B, D + 1) * 10.0  # 大方差输入
    h_big = manifold.expmap0(u_big, project=True)
    check("大方差 expmap0 无 NaN", not torch.isnan(h_big).any())
    check("大方差 expmap0 无 Inf", not torch.isinf(h_big).any())

    # 测试 3.7: 边界情况 — 单元素
    u_one = torch.randn(1, D + 1)
    h_one = manifold.expmap0(u_one, project=True)
    t_one = manifold.logmap0(h_one)
    check("单元素往返 形状", t_one.shape == (1, D))
    check("单元素往返 无 NaN", not torch.isnan(t_one).any())

    # 测试 3.8: learnable=True 曲率可学习
    mf_learn = Lorentz(k=1.0, learnable=True)
    check("learnable 曲率 requires_grad", mf_learn.k.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# 测试 4: 边索引 → 邻接矩阵转换
# ═══════════════════════════════════════════════════════════════════

def test_edge_utils():
    section("测试 4: 边索引 → 邻接矩阵")

    try:
        from src.models.mhr_llm import _add_self_loops, _edge_index_to_sparse_adj
    except Exception as e:
        check("导入 edge utils", False, str(e))
        return

    N = 5
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    # 4.1 添加自环
    ei_with_sl = _add_self_loops(ei, N)
    check("添加自环后边数", ei_with_sl.shape[1] == ei.shape[1] + N)

    # 4.2 稀疏邻接矩阵形状
    adj = _edge_index_to_sparse_adj(ei, N)
    check("邻接矩阵形状", adj.shape == (N, N))

    # 4.3 邻接矩阵包含自环
    diag_vals = torch.tensor([adj[i, i] for i in range(N)])
    check("邻接矩阵对角线=1 (自环)", torch.allclose(diag_vals, torch.ones(N)))

    # 4.4 邻接矩阵语义: adj[target, source] = 1
    for s, t in ei.t().tolist():
        check(f"边 {s}→{t}: adj[{t},{s}]=1", adj[t, s].item() == 1.0)
    check("adj[source, target]=0 (非对称)", adj[0, 1].item() == 0.0,
          detail=f"adj[0,1]={adj[0,1].item()}")


# ═══════════════════════════════════════════════════════════════════
# 测试 5: _root_node_indices
# ═══════════════════════════════════════════════════════════════════

def test_root_indices():
    section("测试 5: _root_node_indices")

    try:
        from src.models.baselines import _root_node_indices
    except Exception as e:
        check("导入 _root_node_indices", False, str(e))
        return

    # 5.1 标准多图场景
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    roots = _root_node_indices(batch)
    check("2图场景: 根节点数=2", len(roots) == 2)
    check("2图场景: 根0=索引0", roots[0].item() == 0)
    check("2图场景: 根1=索引5", roots[1].item() == 5)

    # 5.2 单图场景
    batch_one = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    roots_one = _root_node_indices(batch_one)
    check("单图场景: 根节点数=1", len(roots_one) == 1)
    check("单图场景: 根=索引0", roots_one[0].item() == 0)

    # 5.3 每个图 1 个节点
    batch_tiny = torch.tensor([0, 1, 2], dtype=torch.long)
    roots_tiny = _root_node_indices(batch_tiny)
    check("3单节点图: 根数=3", len(roots_tiny) == 3)
    check("3单节点图: 根=[0,1,2]", torch.equal(roots_tiny, torch.tensor([0, 1, 2])))

    # 5.4 空 batch
    batch_empty = torch.tensor([], dtype=torch.long)
    roots_empty = _root_node_indices(batch_empty)
    check("空batch: 返回空张量", roots_empty.numel() == 0)

    # 5.5 不连续图ID（不应该出现但验证鲁棒性）
    batch_skip = torch.tensor([0, 0, 2, 2, 2], dtype=torch.long)
    roots_skip = _root_node_indices(batch_skip)
    check("非连续图ID: 根数=2", len(roots_skip) == 2,
          f"got {roots_skip.tolist()}")


# ═══════════════════════════════════════════════════════════════════
# 测试 6: compute_metrics / EarlyStopping
# ═══════════════════════════════════════════════════════════════════

def test_metrics():
    section("测试 6: compute_metrics / EarlyStopping")

    for lib in ["sklearn"]:
        try:
            __import__(lib)
        except ImportError:
            global SKIP
            SKIP += 1
            check(f"sklearn 未安装, 跳过指标测试", False, "pip install scikit-learn")
            return

    try:
        from src.utils.metrics import compute_metrics, EarlyStopping
    except Exception as e:
        check("导入 metrics", False, str(e))
        return

    # 6.1 完美预测
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.9])
    r = compute_metrics(y_true, y_pred, y_prob)
    check("完美预测 accuracy=1", abs(r['accuracy'] - 1.0) < 1e-6, f"got {r['accuracy']}")
    check("完美预测 f1=1", abs(r['f1_macro'] - 1.0) < 1e-6, f"got {r['f1_macro']}")
    check("完美预测 auc=1", abs(r['auc'] - 1.0) < 1e-6, f"got {r['auc']}")

    # 6.2 随机预测
    y_pred_rand = np.array([0, 1, 0, 1, 0, 1])
    r2 = compute_metrics(y_true, y_pred_rand, y_prob)
    check("随机预测 accuracy<1", r2['accuracy'] < 1.0)
    check("随机预测 所有指标非负", all(v >= 0 for v in r2.values()))

    # 6.3 无概率输入
    r3 = compute_metrics(y_true, y_pred)
    check("无概率时 auc=-1", r3['auc'] == -1.0, f"got {r3['auc']}")

    # 6.4 EarlyStopping mode='max'
    es = EarlyStopping(patience=3, mode='max')
    should_stop = [es(0.8), es(0.85), es(0.82), es(0.81), es(0.80)]
    check("EarlyStopping patience=3 stop", should_stop == [False, False, False, False, True],
          f"got {should_stop}")

    # 6.5 EarlyStopping mode='min'
    es2 = EarlyStopping(patience=2, mode='min')
    stop2 = [es2(1.0), es2(0.8), es2(0.9), es2(0.85)]
    check("EarlyStopping mode=min stop", stop2 == [False, False, False, True],
          f"got {stop2}")


# ═══════════════════════════════════════════════════════════════════
# 测试 7: collect_results compute_improvement 缺失数据处理
# ═══════════════════════════════════════════════════════════════════

def test_compute_improvement():
    section("测试 7: compute_improvement 缺失数据")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "collect_results",
            str(_ROOT / "scripts" / "collect_results.py")
        )
        cr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cr)
    except Exception as e:
        check("导入 collect_results", False, str(e))
        return

    from collections import OrderedDict

    # 7.1 本文方法缺失时应输出提示（不应输出垃圾值）
    results_missing = {
        "bert_gcn_poli": {
            "test_accuracy": "0.7285", "test_f1_macro": "0.7284",
            "test_auc": "0.8030", "exp_name": "bert_gcn_poli",
        },
    }
    exps_with_ours = OrderedDict([
        ("bert_gcn_poli",  "GCN (BERT)"),
        ("llama_hgnn_poli", "**HGNN (LLM, Ours)**"),
    ])
    output = cr.compute_improvement(results_missing, exps_with_ours)
    check("缺失本文方法: 包含警告", "⚠️" in output or "尚未运行" in output or "缺失" in output)
    check("缺失本文方法: 不含垃圾负值", "-215%" not in output,
          f"output snippet: {output[:200]}")

    # 7.2 本文方法存在时应正常计算
    results_full = {
        "bert_gcn_poli": {
            "test_accuracy": "0.7285", "test_f1_macro": "0.7284",
            "test_auc": "0.8030", "exp_name": "bert_gcn_poli",
        },
        "llama_hgnn_poli": {
            "test_accuracy": "0.8643", "test_f1_macro": "0.8638",
            "test_auc": "0.9688", "exp_name": "llama_hgnn_poli",
        },
    }
    output2 = cr.compute_improvement(results_full, exps_with_ours)
    check("存在本文方法: 含提升数据", "提升" in output2)
    check("存在本文方法: 不含警告", "尚未运行" not in output2)

    # 7.3 _is_our_method
    check("_is_our_method **标记**", cr._is_our_method("**HGNN (LLM, Ours)**"))
    check("_is_our_method 普通", not cr._is_our_method("GCN (BERT)"))


# ═══════════════════════════════════════════════════════════════════
# 测试 8: 参数统计
# ═══════════════════════════════════════════════════════════════════

def test_param_counts():
    section("测试 8: 各模型参数量 (in_dim=4096, gnn_dim=128)")

    models = {}
    try:
        from src.models.mhr_llm import HyperbolicFakeNewsDetector
        models["HGNN (2 layer)"] = HyperbolicFakeNewsDetector(
            in_dim=4096, gnn_dim=128, num_layers=2, dropout=0.1, curvature=-1.0)
    except Exception:
        pass
    try:
        from src.models.gatv2_fallback import GATv2FakeNewsDetector
        models["GATv2 (2 layer)"] = GATv2FakeNewsDetector(
            in_dim=4096, hidden_mlp=512, gnn_dim=128, dropout=0.1)
    except Exception:
        pass
    try:
        from src.models.baselines import (MLPDetector, GCNDetector,
                                           GATDetector, SAGEDetector, BiGCNDetector)
        models["MLP"]       = MLPDetector(in_dim=4096, gnn_dim=128)
        models["GCN (2)"]   = GCNDetector(in_dim=4096, gnn_dim=128, num_layers=2)
        models["GAT (2)"]   = GATDetector(in_dim=4096, gnn_dim=128, num_layers=2)
        models["SAGE (2)"]  = SAGEDetector(in_dim=4096, gnn_dim=128, num_layers=2)
        models["BiGCN (2)"] = BiGCNDetector(in_dim=4096, gnn_dim=128, num_layers=2)
    except Exception:
        pass

    for name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:20s}: {total:>10,} params  (trainable: {trainable:>10,})")
        check(f"{name} has params", total > 0)


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    global PASS, FAIL, SKIP

    print("\n" + "=" * 70)
    print("  本地验证测试套件 (CPU only)")
    print(f"  PyTorch: {torch.__version__}, 设备: {DEVICE}")
    print("=" * 70)
    print("\n  此测试用于在拷贝到服务器前验证代码正确性。")
    print("  只需 pip install torch numpy scikit-learn 即可运行。")

    test_imports()
    test_forward_shapes()
    test_lorentz_manifold()
    test_edge_utils()
    test_root_indices()
    test_metrics()
    test_compute_improvement()
    test_param_counts()

    # 汇总
    total = PASS + FAIL + SKIP
    print(f"\n{'=' * 70}")
    print(f"  测试汇总")
    print(f"{'=' * 70}")
    print(f"  通过: {PASS}/{total}")
    if FAIL > 0:
        print(f"  失败: {FAIL}/{total}")
    if SKIP > 0:
        print(f"  跳过: {SKIP}/{total}")

    if FAIL == 0:
        print(f"\n  ✅ 所有测试通过！可以拷贝到服务器。")
        return 0
    else:
        print(f"\n  ❌ {FAIL} 个测试失败，请修复后再拷贝。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
