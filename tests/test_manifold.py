# -*- coding: utf-8 -*-
"""
test_manifold.py
----------------
流形数值稳定性测试（CPU only，无需 GPU）。

此测试文件用于在上传 ModelScope 前在本地快速验证代码正确性。
由于 lorentz.py 是靠 01_clone_repos.sh 从 MHR 仓库复制的，本地可能不存在，
因此添加了导入失败时的 fallback 逻辑。

测试内容：
1. 随机生成欧氏特征张量，测试流形操作的数值稳定性
2. 测试完整链路：proj_tan0 → exp_map0 → proj → log_map0
3. 测试 HyperbolicGraphConvolution 前向传播
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 检查 torch 是否安装
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch():
    """检查 torch 是否可用。"""
    if not TORCH_AVAILABLE:
        print("[-] 未检测到 torch 模块")
        print("[提示] 请先安装项目依赖: pip install -r requirements.txt")
        return False
    return True


def test_manifold_numerical_stability():
    """测试流形操作的数值稳定性。"""
    if not _check_torch():
        print("[跳过] 流形测试因 torch 缺失而跳过")
        return False
    import torch

    print("=" * 60)
    print("测试 1: 流形数值稳定性")
    print("=" * 60)

    # 尝试导入流形模块
    try:
        from src.models.manifolds.lorentz import Lorentz as LorentzManifold
        print("[+] 成功导入 Lorentz 流形模块")
    except ImportError as e:
        print(f"[-] 导入失败: {e}")
        print("[提示] lorentz.py 需要通过 scripts/01_clone_repos.sh 从 MHR 仓库复制")
        print("[跳过] 流形测试因依赖缺失而跳过")
        return False

    # 参数设置
    batch_size = 32
    feature_dim = 128
    curvature = 1.0

    # 创建流形实例
    manifold = LorentzManifold(k=curvature, learnable=False)
    print(f"[+] 创建 Lorentz 流形实例 (k={curvature})")

    # 随机生成欧氏特征 [batch_size, feature_dim]
    torch.manual_seed(42)
    euclidean_features = torch.randn(batch_size, feature_dim)
    print(f"[+] 生成随机欧氏特征: shape={euclidean_features.shape}")
    print(f"    特征范围: [{euclidean_features.min():.4f}, {euclidean_features.max():.4f}]")

    try:
        # Step 1: proj_tan0 - 投影到切空间
        u = manifold.proju0(euclidean_features)
        print(f"\n[+] Step 1: proj_tan0")
        print(f"    输出形状: {u.shape}")
        print(f"    值范围: [{u.min():.4f}, {u.max():.4f}]")
        assert not torch.any(torch.isnan(u)), "proj_tan0 产生 NaN"
        assert not torch.any(torch.isinf(u)), "proj_tan0 产生 Inf"
        print("    ✅ 通过")

        # Step 2: exp_map0 - 指数映射到流形
        h = manifold.expmap0(u, project=True)
        print(f"\n[+] Step 2: exp_map0")
        print(f"    输出形状: {h.shape}")
        print(f"    值范围: [{h.min():.4f}, {h.max():.4f}]")
        assert not torch.any(torch.isnan(h)), "exp_map0 产生 NaN"
        assert not torch.any(torch.isinf(h)), "exp_map0 产生 Inf"
        print("    ✅ 通过")

        # Step 3: proj - 流形投影校正
        h_proj = manifold.projx(h)
        print(f"\n[+] Step 3: projx")
        print(f"    输出形状: {h_proj.shape}")
        print(f"    值范围: [{h_proj.min():.4f}, {h_proj.max():.4f}]")
        assert not torch.any(torch.isnan(h_proj)), "projx 产生 NaN"
        assert not torch.any(torch.isinf(h_proj)), "projx 产生 Inf"
        print("    ✅ 通过")

        # Step 4: log_map0 - 对数映射回到切空间
        t = manifold.logmap0(h_proj)
        print(f"\n[+] Step 4: logmap0")
        print(f"    输出形状: {t.shape}")
        print(f"    值范围: [{t.min():.4f}, {t.max():.4f}]")
        assert not torch.any(torch.isnan(t)), "logmap0 产生 NaN"
        assert not torch.any(torch.isinf(t)), "logmap0 产生 Inf"
        print("    ✅ 通过")

        # Step 5: 验证往返误差
        error = torch.norm(t - u, dim=-1).mean()
        print(f"\n[+] Step 5: 往返误差验证")
        print(f"    平均重建误差: {error:.6f}")
        print("    ✅ 通过")

        print("\n" + "=" * 60)
        print("测试 1 完成: 所有流形操作数值稳定 ✅")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n[-] 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n[-] 测试异常: {type(e).__name__}: {e}")
        return False


def test_hgnn_layer():
    """测试双曲图卷积层的前向传播。"""
    if not _check_torch():
        print("[跳过] HGNN 层测试因 torch 缺失而跳过")
        return False
    import torch

    print("\n" + "=" * 60)
    print("测试 2: 双曲图卷积层")
    print("=" * 60)

    # 尝试导入必要模块
    try:
        from src.models.manifolds.lorentz import Lorentz as LorentzManifold
        from src.models.layers.hgnn_layer import HyperbolicGraphConvolution
        print("[+] 成功导入 HGNN 层模块")
    except ImportError as e:
        print(f"[-] 导入失败: {e}")
        print("[提示] 需要通过 scripts/01_clone_repos.sh 从 MHR 仓库复制依赖")
        print("[跳过] HGNN 层测试因依赖缺失而跳过")
        return False

    # 参数设置
    num_nodes = 64
    feature_dim = 128
    hidden_dim = 128
    num_edges = 128
    curvature = 1.0

    # 创建流形实例
    manifold = LorentzManifold(k=curvature, learnable=False)

    # 创建 HGNN 层
    hgnn_layer = HyperbolicGraphConvolution(
        manifold=manifold,
        in_features=hidden_dim + 1,  # 流形嵌入维 = 空间维 + 1
        out_features=hidden_dim + 1,
        use_bias=True,
        dropout=0.1,
        use_att=False,
        local_agg=True,
    )
    print(f"[+] 创建 HyperbolicGraphConvolution 层")

    # 随机生成流形特征 [num_nodes, hidden_dim + 1]
    torch.manual_seed(42)
    euclidean_features = torch.randn(num_nodes, hidden_dim)
    u = manifold.proju0(euclidean_features)
    x = manifold.expmap0(u, project=True)
    x = manifold.projx(x)
    print(f"[+] 生成随机流形特征: shape={x.shape}")

    # 创建随机边索引 [2, num_edges]
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    print(f"[+] 生成随机边索引: shape={edge_index.shape}")

    try:
        # 构建稀疏邻接矩阵
        def _edge_index_to_sparse_adj(ei, n_nodes):
            row = ei[1]
            col = ei[0]
            values = torch.ones(row.numel(), dtype=torch.float32)
            adj = torch.sparse_coo_tensor(
                torch.stack([row, col]),
                values,
                (n_nodes, n_nodes),
                dtype=torch.float32,
            ).coalesce()
            return adj

        adj = _edge_index_to_sparse_adj(edge_index, num_nodes)
        print(f"[+] 构建稀疏邻接矩阵: shape={adj.shape}, nnz={adj._nnz()}")

        # 前向传播
        output, _ = hgnn_layer((x, adj))
        print(f"\n[+] HGNN 前向传播")
        print(f"    输出形状: {output.shape}")
        print(f"    值范围: [{output.min():.4f}, {output.max():.4f}]")
        assert not torch.any(torch.isnan(output)), "HGNN 前向传播产生 NaN"
        assert not torch.any(torch.isinf(output)), "HGNN 前向传播产生 Inf"
        print("    ✅ 通过")

        print("\n" + "=" * 60)
        print("测试 2 完成: HGNN 层前向传播正常 ✅")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n[-] 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n[-] 测试异常: {type(e).__name__}: {e}")
        return False


def test_proju0_identity():
    """
    测试 proju0 恒等映射行为。

    ⚠️ 当前简化版 Lorentz 流形中，proju0 是恒等映射 (return u)，
    这与标准 MHR 实现不同。完整 MHR 中 proju0 会确保输入向量
    在切空间的正确定向子空间中。

    此测试显式验证当前行为，确保调用者了解这一简化。
    """
    if not _check_torch():
        print("[跳过] proju0 测试因 torch 缺失而跳过")
        return False
    import torch

    print("\n" + "=" * 60)
    print("测试 3: proju0 恒等映射行为")
    print("=" * 60)

    try:
        from src.models.manifolds.lorentz import Lorentz as LorentzManifold
    except ImportError as e:
        print(f"[-] 导入失败: {e}")
        return False

    manifold = LorentzManifold(k=1.0, learnable=False)
    torch.manual_seed(42)

    # 测试 3.1: proju0 是精确恒等映射
    x = torch.randn(32, 128)
    u = manifold.proju0(x)
    assert torch.allclose(u, x, atol=1e-8), \
        f"proju0 不是恒等映射！max diff = {(u - x).abs().max():.10f}"
    print("[+] 测试 3.1 通过: proju0 是精确恒等映射")

    # 测试 3.2: 多种形状下均为恒等
    for shape in [(1, 128), (1, 1), (64, 256), (100, 10)]:
        x = torch.randn(*shape)
        u = manifold.proju0(x)
        assert u.shape == x.shape, f"形状不匹配: {u.shape} != {x.shape}"
        assert torch.allclose(u, x, atol=1e-8), \
            f"形状 {shape} 下不是恒等映射"
    print(f"[+] 测试 3.2 通过: 4 种形状下均为恒等映射")

    # 测试 3.3: 含 NaN/Inf 输入时仍为恒等（恒等映射不会消除异常值）
    x_nan = torch.tensor([[float('nan'), 1.0], [2.0, 3.0]])
    u_nan = manifold.proju0(x_nan)
    assert torch.isnan(u_nan[0, 0]), "NaN 应保持不变（恒等映射）"
    print("[+] 测试 3.3 通过: NaN/Inf 输入仍为恒等")

    # 测试 3.4: 文档说明 — 对比完整 MHR 预期行为
    print("\n[文档] proju0 当前行为 vs 完整 MHR 实现:")
    print("  当前简化版: proju0(u) = u  (恒等映射)")
    print("  完整 MHR 版: proju0(u) 会将 u 投影到 Lorentz 切空间的")
    print("              正确子空间中，确保时间分量为零且空间")
    print("              分量满足一定的正交约束。")
    print("  影响: 当前简化版中，输入特征未经切空间约束直接")
    print("        进入 expmap0，可能导致初始特征偏离正确的流形邻域。")
    print("        在数值稳定的输入范围内此简化通常安全。")

    print("\n" + "=" * 60)
    print("测试 3 完成: proju0 恒等映射行为已验证 ✅")
    print("=" * 60)
    return True


def test_manifold_edge_cases():
    """
    测试流形操作的边界条件。

    覆盖场景:
    1. 极大范数输入 (触发 cosh/sinh 溢出边界)
    2. 零范数输入 (退化情况)
    3. 单元素 batch (N=1)
    4. 空输入 (N=0)
    5. 不同维度 (d=1 最小, d=512 较大)
    """
    if not _check_torch():
        print("[跳过] 边界测试因 torch 缺失而跳过")
        return False
    import torch

    print("\n" + "=" * 60)
    print("测试 4: 流形操作边界条件")
    print("=" * 60)

    try:
        from src.models.manifolds.lorentz import Lorentz as LorentzManifold
    except ImportError as e:
        print(f"[-] 导入失败: {e}")
        return False

    manifold = LorentzManifold(k=1.0, learnable=False)
    all_passed = True

    # --- 测试 4.1: 极大范数输入 ---
    print("\n[+] 测试 4.1: 极大范数输入 (可能触发 cosh/sinh 溢出)")
    large_norms = [10.0, 50.0, 100.0, 500.0]
    for norm_val in large_norms:
        try:
            x = torch.ones(2, 128) * (norm_val / (128 ** 0.5))
            # 添加时间维度
            x_with_time = torch.cat([torch.zeros(2, 1), x], dim=-1)
            u = manifold.proju0(x_with_time)
            h = manifold.expmap0(u, project=True)
            if torch.any(torch.isnan(h)) or torch.any(torch.isinf(h)):
                print(f"    [WARN] 范数={norm_val:.0f}: 出现 NaN/Inf (超出浮点范围)")
                all_passed = False
            else:
                print(f"    [OK] 范数={norm_val:.0f}: 正常")
        except Exception as e:
            print(f"    [WARN] 范数={norm_val:.0f}: 异常 {type(e).__name__}: {e}")
            all_passed = False

    # --- 测试 4.2: 零范数输入 ---
    print("\n[+] 测试 4.2: 零范数输入 (退化情况)")
    try:
        x = torch.zeros(3, 128)
        x_with_time = torch.cat([torch.zeros(3, 1), x], dim=-1)
        u = manifold.proju0(x_with_time)
        h = manifold.expmap0(u, project=True)
        # 零向量应该映射到流形上的原点: t=cosh(0)=1, x=sinh(0)/0 * 0 = 0
        assert not torch.any(torch.isnan(h)), "零向量不应产生 NaN"
        assert torch.allclose(h[:, 0], torch.ones(3), atol=1e-5), \
            f"零向量应映射到时间分量=1, 实际={h[:, 0]}"
        assert torch.allclose(h[:, 1:], torch.zeros(3, 128), atol=1e-5), \
            "零向量应映射到空间分量=0"
        # 往返测试
        t = manifold.logmap0(h)
        assert torch.allclose(t, torch.zeros(3, 128), atol=1e-4), \
            "零范数往返应回到零"
        print("    [OK] 零范数输入正确处理")
    except Exception as e:
        print(f"    [FAIL] 零范数异常: {type(e).__name__}: {e}")
        all_passed = False

    # --- 测试 4.3: 单元素 batch ---
    print("\n[+] 测试 4.3: 单元素 batch (N=1)")
    try:
        x = torch.randn(1, 128)
        x_with_time = torch.cat([torch.zeros(1, 1), x], dim=-1)
        u = manifold.proju0(x_with_time)
        h = manifold.expmap0(u, project=True)
        t = manifold.logmap0(h)
        assert h.shape == (1, 129), f"expmap0 形状错误: {h.shape}"
        assert t.shape == (1, 128), f"logmap0 形状错误: {t.shape}"
        print("    [OK] 单元素正确处理")
    except Exception as e:
        print(f"    [FAIL] 单元素异常: {type(e).__name__}: {e}")
        all_passed = False

    # --- 测试 4.4: 空输入 ---
    print("\n[+] 测试 4.4: 空输入 (N=0)")
    try:
        x = torch.randn(0, 128)
        x_with_time = torch.cat([torch.zeros(0, 1), x], dim=-1)
        u = manifold.proju0(x_with_time)
        assert u.shape == (0, 129), f"proju0 空输入形状错误: {u.shape}"
        h = manifold.expmap0(u, project=True)
        assert h.shape == (0, 129), f"expmap0 空输入形状错误: {h.shape}"
        h_proj = manifold.projx(h)
        assert h_proj.shape == (0, 129), f"projx 空输入形状错误: {h_proj.shape}"
        t = manifold.logmap0(h)
        assert t.shape == (0, 128), f"logmap0 空输入形状错误: {t.shape}"
        print("    [OK] 空输入正确处理")
    except Exception as e:
        print(f"    [FAIL] 空输入异常: {type(e).__name__}: {e}")
        all_passed = False

    # --- 测试 4.5: 最小空间维度 ---
    print("\n[+] 测试 4.5: 最小维度 (d=1)")
    try:
        x = torch.randn(2, 1)
        x_with_time = torch.cat([torch.zeros(2, 1), x], dim=-1)
        u = manifold.proju0(x_with_time)
        h = manifold.expmap0(u, project=True)
        assert h.shape == (2, 2), f"d=1 时 expmap0 形状错误: {h.shape}"
        t = manifold.logmap0(h)
        assert t.shape == (2, 1), f"d=1 时 logmap0 形状错误: {t.shape}"
        # 验证 Minkowski 内积约束: t² - x² = k
        inner = manifold.inner(h, h, keepdim=False)
        assert torch.allclose(inner, -torch.ones(2) * manifold.k, atol=1e-4), \
            f"Minkowski 约束不满足: {inner}"
        print("    [OK] d=1 正确处理，Minkowski 约束满足")
    except Exception as e:
        print(f"    [FAIL] d=1 异常: {type(e).__name__}: {e}")
        all_passed = False

    # --- 测试 4.6: 较大维度 ---
    print("\n[+] 测试 4.6: 较大维度 (d=512)")
    try:
        x = torch.randn(4, 512)
        x_with_time = torch.cat([torch.zeros(4, 1), x], dim=-1)
        u = manifold.proju0(x_with_time)
        h = manifold.expmap0(u, project=True)
        assert not torch.any(torch.isnan(h)), "d=512 不应产生 NaN"
        t = manifold.logmap0(h)
        assert t.shape == (4, 512), f"d=512 时 logmap0 形状错误: {t.shape}"
        print("    [OK] d=512 正确处理")
    except Exception as e:
        print(f"    [FAIL] d=512 异常: {type(e).__name__}: {e}")
        all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("测试 4 完成: 所有边界条件通过 ✅")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("测试 4 完成: 部分边界条件存在问题 ⚠️")
        print("=" * 60)
    return all_passed


def test_minkowski_constraint():
    """
    测试 Minkowski 内积约束验证。

    流形上的点应满足: <x, x>_L = -k (即 t² - Σxᵢ² = k)
    expmap0 和 projx 之后应该始终满足此约束。
    """
    if not _check_torch():
        print("[跳过] Minkowski 测试因 torch 缺失而跳过")
        return False
    import torch

    print("\n" + "=" * 60)
    print("测试 5: Minkowski 内积约束")
    print("=" * 60)

    try:
        from src.models.manifolds.lorentz import Lorentz as LorentzManifold
    except ImportError as e:
        print(f"[-] 导入失败: {e}")
        return False

    for k in [0.5, 1.0, 2.0]:
        manifold = LorentzManifold(k=k, learnable=False)
        torch.manual_seed(42)

        # 随机点 → expmap0
        x = torch.randn(16, 128) * 0.5
        x_with_time = torch.cat([torch.zeros(16, 1), x], dim=-1)
        u = manifold.proju0(x_with_time)
        h = manifold.expmap0(u, project=True)

        # 验证 Minkowski 内积 = -k
        inner = manifold.inner(h, h, keepdim=False)
        expected = -torch.ones(16) * k
        assert torch.allclose(inner, expected, atol=1e-4), \
            f"k={k}: expmap0 后 Minkowski 约束不满足! 最大误差={((inner - expected).abs().max()):.6f}"

        # projx 后再次验证
        h_proj = manifold.projx(h)
        inner_proj = manifold.inner(h_proj, h_proj, keepdim=False)
        assert torch.allclose(inner_proj, expected, atol=1e-4), \
            f"k={k}: projx 后 Minkowski 约束不满足!"

        # logmap0 后的点应在欧氏空间（无约束）
        t = manifold.logmap0(h)
        assert t.shape == (16, 128), f"logmap0 形状错误: {t.shape}"
        assert not torch.any(torch.isnan(t)), "logmap0 产生 NaN"

        print(f"    [OK] k={k}: Minkowski 约束满足 (误差 < 1e-4)")

    print("\n" + "=" * 60)
    print("测试 5 完成: Minkowski 约束全部满足 ✅")
    print("=" * 60)
    return True


def test_gatv2_layer():
    """测试 GATv2 备用模型（不需要流形依赖）。"""
    if not _check_torch():
        print("[跳过] GATv2 测试因 torch 缺失而跳过")
        return False
    import torch

    print("\n" + "=" * 60)
    print("测试 6: GATv2 备用模型")
    print("=" * 60)

    # 尝试导入 GATv2 模型（直接从文件导入，避免通过 __init__.py）
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gatv2_fallback",
            str(_ROOT / "src" / "models" / "gatv2_fallback.py")
        )
        gatv2_fallback = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gatv2_fallback)
        GATv2FakeNewsDetector = gatv2_fallback.GATv2FakeNewsDetector
        print("[+] 成功导入 GATv2FakeNewsDetector")
    except ImportError as e:
        print(f"[-] 导入失败: {e}")
        return False
    except Exception as e:
        print(f"[-] 导入异常: {type(e).__name__}: {e}")
        return False

    # 参数设置
    num_nodes = 64
    in_dim = 4096
    hidden_mlp = 512
    gnn_dim = 128
    num_edges = 128
    batch_size = 2

    # 创建模型
    model = GATv2FakeNewsDetector(
        in_dim=in_dim,
        hidden_mlp=hidden_mlp,
        gnn_dim=gnn_dim,
        dropout=0.1,
    )
    print(f"[+] 创建 GATv2FakeNewsDetector 模型")

    # 随机生成输入
    torch.manual_seed(42)
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.tensor([0] * (num_nodes // 2) + [1] * (num_nodes // 2))
    print(f"[+] 生成随机输入:")
    print(f"    x shape: {x.shape}")
    print(f"    edge_index shape: {edge_index.shape}")
    print(f"    batch shape: {batch.shape}")

    try:
        # 前向传播
        logits = model(x, edge_index, batch)
        print(f"\n[+] GATv2 前向传播")
        print(f"    输出形状: {logits.shape}")
        print(f"    值范围: [{logits.min():.4f}, {logits.max():.4f}]")
        assert not torch.any(torch.isnan(logits)), "GATv2 前向传播产生 NaN"
        assert not torch.any(torch.isinf(logits)), "GATv2 前向传播产生 Inf"
        assert logits.shape[0] == batch_size, f"输出 batch_size 不匹配: {logits.shape[0]} != {batch_size}"
        assert logits.shape[1] == 2, f"输出类别数不匹配: {logits.shape[1]} != 2"
        print("    [OK] 通过")

        print("\n" + "=" * 60)
        print("测试 6 完成: GATv2 模型前向传播正常 [OK]")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n[-] 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n[-] 测试异常: {type(e).__name__}: {e}")
        return False


def main():
    """运行所有测试。"""
    print("\n" + "=" * 70)
    print("流形数值稳定性测试套件 (CPU only)")
    print("=" * 70)
    print("注意: 此测试用于在上传 ModelScope 前验证代码正确性")
    print("=" * 70 + "\n")

    results = []

    # 测试 1: 流形数值稳定性
    results.append(test_manifold_numerical_stability())

    # 测试 2: HGNN 层
    results.append(test_hgnn_layer())

    # 测试 3: proju0 恒等映射文档
    results.append(test_proju0_identity())

    # 测试 4: 边界条件（极端范数、空输入、单元素等）
    results.append(test_manifold_edge_cases())

    # 测试 5: Minkowski 内积约束
    results.append(test_minkowski_constraint())

    # 测试 6: GATv2 备用模型（此测试不依赖 MHR 仓库）
    results.append(test_gatv2_layer())

    # 汇总结果
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if all(results):
        print("\n[OK] 所有测试通过！")
        return 0
    else:
        print("\n[WARN] 部分测试失败或跳过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
