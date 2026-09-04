"""
Step 1: 弱敏感子空间诊断实验
=============================

目标：验证 anti-heretic 方案的核心前提假设——
在 Heretic 会编辑的模块（o_proj / down_proj）附近，
是否存在"对最近下游读取者弱敏感"的子空间，可以容纳诱饵扰动。

具体做什么：
1. 提取每层 o_proj、down_proj、up_proj、gate_proj、q/k/v_proj 的权重矩阵
2. 对下游读取矩阵做 SVD，分析 singular value 衰减情况
3. 计算 refusal direction 与下游矩阵 singular vectors 的对齐程度
4. 量化"弱敏感子空间"的有效维度和容量

核心回答的问题：
- W_up / W_gate 是否存在足够大的弱敏感子空间？
- refusal direction 落在下游矩阵的哪些 singular vector 上？
- 如果在弱敏感子空间里注入扰动，理论上能承载多大幅度？
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rich.table import Table

from shared_utils import (
    RESULTS_DIR,
    get_args,
    get_layers,
    get_layer_module,
    get_residuals_batched,
    load_model,
    load_prompts_simple,
    print,
    resolve_device,
)


def get_step1_svd_settings() -> tuple[str, int, int, int]:
    """Return Step1 SVD settings from environment variables.

    The paper Table 3 path uses exact SVD by default. Large follow-up models
    can opt into deterministic low-rank SVD via STEP1_SVD_METHOD=lowrank.
    """
    method = os.environ.get("STEP1_SVD_METHOD", "full").strip().lower()
    if method not in {"full", "lowrank"}:
        raise ValueError(
            f"STEP1_SVD_METHOD must be 'full' or 'lowrank', got {method!r}"
        )
    q = int(os.environ.get("STEP1_SVD_LOWRANK_Q", "64"))
    niter = int(os.environ.get("STEP1_SVD_LOWRANK_NITER", "4"))
    seed = int(os.environ.get("STEP1_SVD_LOWRANK_SEED", "42"))
    return method, q, niter, seed


def compute_step1_svd(weight: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SVD factors for Step1 diagnostics.

    Full SVD is preserved as the default. Low-rank SVD is intended for large
    engineering targets such as Qwen3.8-27B, where exact CPU SVD over many
    projection matrices is prohibitively slow.
    """
    W = weight.float().detach().cpu()
    method, q, niter, seed = get_step1_svd_settings()
    if method == "full":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return U, S, Vh

    q = max(1, min(q, min(W.shape)))
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        U, S, V = torch.svd_lowrank(W, q=q, niter=niter)
    finally:
        torch.random.set_rng_state(rng_state)
    print(f"  Low-rank Step1 SVD: {name} q={q} niter={niter} seed={seed}")
    return U, S, V.T


def compute_step1_singular_values(weight: torch.Tensor, name: str) -> torch.Tensor:
    method, _, _, _ = get_step1_svd_settings()
    if method == "full":
        return torch.linalg.svdvals(weight.float().detach().cpu())
    _, S, _ = compute_step1_svd(weight, name)
    return S


def analyze_svd(weight: torch.Tensor, name: str) -> dict:
    """对权重矩阵做 SVD 分析。"""
    W = weight.float().detach().cpu()
    method, q, niter, seed = get_step1_svd_settings()

    # SVD: W = U @ diag(S) @ Vh
    U, S, Vh = compute_step1_svd(weight, name)
    S_np = S.numpy()

    # 归一化 singular values
    S_normalized = S_np / S_np[0]

    # 有效秩：用不同阈值截断
    thresholds = [0.01, 0.05, 0.1, 0.2]
    effective_ranks = {}
    for t in thresholds:
        effective_ranks[f"rank_above_{t}"] = int(np.sum(S_normalized > t))

    # 能量分布
    energy = (S_np**2) / (S_np**2).sum()
    cumulative_energy = np.cumsum(energy)

    # 找到覆盖 90%, 95%, 99% 能量需要的维度
    energy_dims = {}
    for target in [0.9, 0.95, 0.99]:
        energy_dims[f"dims_for_{int(target*100)}pct_energy"] = int(
            np.searchsorted(cumulative_energy, target) + 1
        )

    return {
        "name": name,
        "shape": list(W.shape),
        "svd_method": method,
        "svd_lowrank_q": q if method == "lowrank" else None,
        "svd_lowrank_niter": niter if method == "lowrank" else None,
        "svd_lowrank_seed": seed if method == "lowrank" else None,
        "computed_rank": len(S_np),
        "singular_values": S_np.tolist()[:50],  # 前 50 个
        "effective_ranks": effective_ranks,
        "energy_dims": energy_dims,
        "total_rank": len(S_np),
        "condition_number": float(S_np[0] / S_np[-1]) if S_np[-1] > 1e-10 else float("inf"),
        "top10_sv_ratio": float(S_np[:10].sum() / S_np.sum()),
        # 返回 U 和 Vh 用于后续对齐分析
        "_U": U,
        "_Vh": Vh,
        "_S": S,
    }


def compute_refusal_direction_alignment(
    refusal_dir: torch.Tensor,
    svd_result: dict,
) -> dict:
    """
    计算 refusal direction 与 SVD 各 singular vector 的对齐程度。
    关键洞察：如果 refusal direction 与低 singular value 的 vector 对齐度高，
    说明诱饵可以利用这个方向而不被下游矩阵放大。
    """
    # Vh 的每行是一个右奇异向量（输入空间的基），shape (min(m,n), n)
    # U 的每列是一个左奇异向量（输出空间的基），shape (m, min(m,n))
    # 对于 W_up (intermediate_size, hidden_size):
    #   输入空间 = hidden_size（这是残差流经 LN 后进入的维度）
    #   所以应该看 Vh（输入空间的基）与 refusal direction 的对齐

    Vh = svd_result["_Vh"]  # (min(m,n), n) = (hidden, hidden) 或 (inter, hidden)
    S = svd_result["_S"]

    r = refusal_dir.float().cpu()
    r = F.normalize(r, dim=0)

    # 计算 refusal direction 在每个右奇异向量上的投影
    # Vh: (k, hidden_dim), r: (hidden_dim,)
    n_components = min(Vh.shape[0], len(r))
    Vh_truncated = Vh[:n_components, :len(r)]
    projections = (Vh_truncated @ r).abs()  # (k,)

    S_np = S[:n_components].numpy()
    projections_np = projections.numpy()

    # 加权投影：projection * singular_value = 该方向对最终输出的贡献
    weighted = projections_np * S_np

    # 分段统计
    n = len(projections_np)
    segments = {
        "top_10pct": slice(0, max(1, n // 10)),
        "mid_40pct": slice(n // 10, n // 2),
        "bottom_50pct": slice(n // 2, n),
    }

    alignment_stats = {}
    for seg_name, s in segments.items():
        alignment_stats[f"proj_energy_{seg_name}"] = float(
            (projections_np[s] ** 2).sum()
        )
        alignment_stats[f"weighted_energy_{seg_name}"] = float(
            (weighted[s] ** 2).sum()
        )

    # 核心指标：refusal direction 有多少投影和能量落在"弱 singular value"区域
    # 弱区域定义：singular value < 0.1 * max
    weak_mask = S_np < 0.1 * S_np[0]
    alignment_stats["proj_energy_in_weak_zone"] = float(
        (projections_np[weak_mask] ** 2).sum()
    )
    alignment_stats["n_weak_dimensions"] = int(weak_mask.sum())

    return alignment_stats


def main():
    args = get_args("Step 1: 弱敏感子空间诊断")
    device = resolve_device(args.device)
    svd_method, svd_q, svd_niter, svd_seed = get_step1_svd_settings()
    print(
        f"Step1 SVD method: {svd_method}"
        + (
            f" (q={svd_q}, niter={svd_niter}, seed={svd_seed})"
            if svd_method == "lowrank"
            else ""
        )
    )
    model, tokenizer = load_model(args.model, device)

    # 加载数据并提取残差
    good_prompts, bad_prompts = load_prompts_simple(
        args.n_good,
        args.n_bad,
        source=args.data_source,
    )

    print("\nExtracting residuals...")
    good_residuals = get_residuals_batched(model, tokenizer, good_prompts, args.batch_size)
    bad_residuals = get_residuals_batched(model, tokenizer, bad_prompts, args.batch_size)

    # 计算逐层 refusal direction（与 Heretic 完全一致）
    good_means = good_residuals.mean(dim=0)  # (n_layers+1, hidden_dim)
    bad_means = bad_residuals.mean(dim=0)
    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    layers = get_layers(model)
    n_layers = len(layers)

    # 选择分析的目标层：默认中后层；若 pipeline 请求 covN，则从同一
    # coverage window 的起点开始，保证后续显式 target layer 都有 Step1 prior。
    coverage_layers_raw = os.environ.get("HERETIC_COVERAGE_LAYERS") or os.environ.get("COVERAGE_LAYER_COUNT")
    coverage_layers = int(coverage_layers_raw) if coverage_layers_raw else 0
    if coverage_layers > 0:
        target_start = max(0, (n_layers - 2) - coverage_layers + 1)
        print(
            f"\nStep1 coverage window: HERETIC_COVERAGE_LAYERS={coverage_layers} "
            f"→ analyzing from layer {target_start}"
        )
    else:
        target_start = n_layers // 2

    target_layers = list(range(target_start, n_layers))
    print(f"\nAnalyzing layers {target_layers[0]}-{target_layers[-1]}...")

    all_results = {}

    # =========================================
    # 分析 1: o_proj → 下游 up_proj / gate_proj
    # =========================================
    print("\n[bold]Analysis 1: o_proj → up_proj / gate_proj pathway[/]")

    table = Table(title="o_proj → MLP input: 弱敏感子空间分析")
    table.add_column("Layer", justify="right")
    table.add_column("W_up dim", justify="right")
    table.add_column("W_up weak dims\n(sv<10%max)", justify="right")
    table.add_column("Refusal in\nweak zone", justify="right")
    table.add_column("W_gate weak dims", justify="right")
    table.add_column("99% energy\ndims", justify="right")

    for l_idx in target_layers:
        layer = layers[l_idx]
        refusal_dir = refusal_directions[l_idx + 1]  # +1 因为 index 0 是 embedding

        # 分析 up_proj
        up_proj = get_layer_module(layer, "mlp.up_proj")
        gate_proj = get_layer_module(layer, "mlp.gate_proj")

        layer_result = {"layer": l_idx}

        if up_proj is not None:
            up_svd = analyze_svd(up_proj.weight, f"layer_{l_idx}_up_proj")
            up_alignment = compute_refusal_direction_alignment(refusal_dir, up_svd)
            layer_result["up_proj_svd"] = {
                k: v for k, v in up_svd.items() if not k.startswith("_")
            }
            layer_result["up_proj_alignment"] = up_alignment

        if gate_proj is not None:
            gate_svd = analyze_svd(gate_proj.weight, f"layer_{l_idx}_gate_proj")
            gate_alignment = compute_refusal_direction_alignment(refusal_dir, gate_svd)
            layer_result["gate_proj_svd"] = {
                k: v for k, v in gate_svd.items() if not k.startswith("_")
            }
            layer_result["gate_proj_alignment"] = gate_alignment

        all_results[f"layer_{l_idx}"] = layer_result

        # 打印表格行
        up_weak = up_alignment["n_weak_dimensions"] if up_proj else "N/A"
        up_refusal_weak = (
            f'{up_alignment["proj_energy_in_weak_zone"]:.4f}' if up_proj else "N/A"
        )
        gate_weak = gate_alignment["n_weak_dimensions"] if gate_proj else "N/A"
        up_99 = (
            up_svd["energy_dims"]["dims_for_99pct_energy"] if up_proj else "N/A"
        )

        table.add_row(
            str(l_idx),
            str(up_svd["shape"]) if up_proj else "N/A",
            str(up_weak),
            str(up_refusal_weak),
            str(gate_weak),
            str(up_99),
        )

    print(table)

    # =========================================
    # 分析 2: down_proj → 下层 q/k/v
    # =========================================
    print("\n[bold]Analysis 2: down_proj → next-layer q/k/v pathway[/]")

    table2 = Table(title="down_proj → next q/k/v: 弱敏感子空间分析")
    table2.add_column("Layer", justify="right")
    table2.add_column("W_q weak dims", justify="right")
    table2.add_column("W_k weak dims", justify="right")
    table2.add_column("W_v weak dims", justify="right")
    table2.add_column("Refusal in Wq weak", justify="right")

    for l_idx in target_layers[:-1]:  # 排除最后一层（没有下一层）
        refusal_dir = refusal_directions[l_idx + 1]
        next_layer = layers[l_idx + 1]

        q_proj = get_layer_module(next_layer, "attn.q_proj")
        k_proj = get_layer_module(next_layer, "attn.k_proj")
        v_proj = get_layer_module(next_layer, "attn.v_proj")

        q_weak = k_weak = v_weak = q_refusal_weak = "N/A"

        if q_proj is not None:
            q_svd = analyze_svd(q_proj.weight, f"layer_{l_idx+1}_q_proj")
            q_alignment = compute_refusal_direction_alignment(refusal_dir, q_svd)
            q_weak = q_alignment["n_weak_dimensions"]
            q_refusal_weak = f'{q_alignment["proj_energy_in_weak_zone"]:.4f}'
            all_results[f"layer_{l_idx}"]["next_q_alignment"] = q_alignment

        if k_proj is not None:
            k_svd = analyze_svd(k_proj.weight, f"layer_{l_idx+1}_k_proj")
            k_alignment = compute_refusal_direction_alignment(refusal_dir, k_svd)
            k_weak = k_alignment["n_weak_dimensions"]

        if v_proj is not None:
            v_svd = analyze_svd(v_proj.weight, f"layer_{l_idx+1}_v_proj")
            v_alignment = compute_refusal_direction_alignment(refusal_dir, v_svd)
            v_weak = v_alignment["n_weak_dimensions"]

        table2.add_row(str(l_idx), str(q_weak), str(k_weak), str(v_weak), str(q_refusal_weak))

    print(table2)

    # =========================================
    # 可视化
    # =========================================
    print("\nGenerating visualizations...")

    model_short = os.environ.get("MODEL_SLUG") or args.model.replace("/", "_")

    # 图 1: singular value 衰减曲线（代表性层）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    representative_layers = [
        target_layers[0],
        target_layers[len(target_layers) // 2],
        target_layers[-1],
    ]

    for ax, l_idx in zip(axes, representative_layers):
        layer = layers[l_idx]
        for comp_name, color in [
            ("mlp.up_proj", "tab:blue"),
            ("mlp.gate_proj", "tab:orange"),
        ]:
            mod = get_layer_module(layer, comp_name)
            if mod is not None:
                S = compute_step1_singular_values(mod.weight, f"layer_{l_idx}_{comp_name}").numpy()
                ax.semilogy(S / S[0], label=comp_name, color=color, alpha=0.8)

        ax.set_title(f"Layer {l_idx}")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Normalized singular value (log)")
        ax.legend()
        ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="10% threshold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Singular Value Decay of Downstream Matrices ({args.model})")
    fig.tight_layout()
    sv_path = RESULTS_DIR / f"step1_sv_decay_{model_short}.png"
    fig.savefig(sv_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: [bold]{sv_path}[/]")

    # 图 2: refusal direction 在各 singular vector 上的投影分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, l_idx in zip(axes, representative_layers):
        layer = layers[l_idx]
        refusal_dir = refusal_directions[l_idx + 1]

        up_proj = get_layer_module(layer, "mlp.up_proj")
        if up_proj is not None:
            U, S, Vh = compute_step1_svd(up_proj.weight, f"layer_{l_idx}_up_proj_plot")
            r = F.normalize(refusal_dir.float().cpu(), dim=0)
            n = min(Vh.shape[0], len(r))
            projections = (Vh[:n, :len(r)] @ r).abs().numpy()

            ax.bar(range(len(projections)), projections, alpha=0.6, width=1.0)
            ax.set_title(f"Layer {l_idx} — W_up")
            ax.set_xlabel("Singular vector index")
            ax.set_ylabel("|projection of refusal dir|")
            ax.set_xlim(0, min(100, len(projections)))

    fig.suptitle(f"Refusal Direction Alignment with W_up Singular Vectors ({args.model})")
    fig.tight_layout()
    align_path = RESULTS_DIR / f"step1_alignment_{model_short}.png"
    fig.savefig(align_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: [bold]{align_path}[/]")

    # 图 3: 逐层可分性 (Fisher 判别比) — 用于对比 Step 2 的效果
    print("\nComputing per-layer Fisher discriminant ratio (baseline)...")
    fisher_ratios = []
    for l_idx in range(n_layers + 1):
        good_r = good_residuals[:, l_idx, :]  # (n_good, hidden_dim)
        bad_r = bad_residuals[:, l_idx, :]

        # 简化的 Fisher 比：||μ_b - μ_g||² / (tr(Σ_b) + tr(Σ_g))
        mean_diff = (bad_r.mean(0) - good_r.mean(0))
        between_var = (mean_diff**2).sum().item()

        within_var_good = good_r.var(dim=0).sum().item()
        within_var_bad = bad_r.var(dim=0).sum().item()
        within_var = within_var_good + within_var_bad + 1e-10

        fisher_ratios.append(between_var / within_var)

    all_results["fisher_ratios_baseline"] = fisher_ratios

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(fisher_ratios)), fisher_ratios, "o-", markersize=3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Fisher discriminant ratio")
    ax.set_title(f"Per-layer Separability (Baseline) — {args.model}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fisher_path = RESULTS_DIR / f"step1_fisher_baseline_{model_short}.png"
    fig.savefig(fisher_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: [bold]{fisher_path}[/]")

    # 保存结果
    # 移除不可序列化的内部字段
    serializable_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            serializable_results[k] = {
                kk: vv for kk, vv in v.items() if not isinstance(vv, dict) or not any(
                    kk2.startswith("_") for kk2 in (vv.keys() if isinstance(vv, dict) else [])
                )
            }
        else:
            serializable_results[k] = v

    result_path = RESULTS_DIR / f"step1_results_{model_short}.json"
    with open(result_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"\n  Full results saved to: [bold]{result_path}[/]")

    # =========================================
    # 结论摘要
    # =========================================
    print("\n" + "=" * 60)
    print("[bold]Step 1 诊断结论[/]")
    print("=" * 60)

    # 统计所有层的弱敏感子空间维度
    weak_dims_up = []
    refusal_in_weak = []
    for l_idx in target_layers:
        key = f"layer_{l_idx}"
        if key in all_results and "up_proj_alignment" in all_results[key]:
            weak_dims_up.append(all_results[key]["up_proj_alignment"]["n_weak_dimensions"])
            refusal_in_weak.append(
                all_results[key]["up_proj_alignment"]["proj_energy_in_weak_zone"]
            )

    if weak_dims_up:
        avg_weak = np.mean(weak_dims_up)
        avg_refusal_weak = np.mean(refusal_in_weak)
        total_dims = all_results[f"layer_{target_layers[0]}"]["up_proj_svd"]["total_rank"]

        print(f"\n  W_up 弱敏感子空间平均维度: {avg_weak:.0f} / {total_dims}")
        print(f"  Refusal direction 在弱区域的投影能量: {avg_refusal_weak:.4f}")

        if avg_weak > total_dims * 0.3:
            print("  [green]✅ 弱敏感子空间较大，理论上可以容纳诱饵扰动[/]")
        elif avg_weak > total_dims * 0.1:
            print("  [yellow]⚠️ 弱敏感子空间中等，诱饵容量有限但可能可行[/]")
        else:
            print("  [red]❌ 弱敏感子空间太小，诱饵方案在当前模块可能不可行[/]")

        if avg_refusal_weak > 0.3:
            print("  [green]✅ Refusal direction 有较多分量在弱区域，有利于诱饵设计[/]")
        elif avg_refusal_weak > 0.1:
            print("  [yellow]⚠️ 部分 refusal direction 在弱区域，但不多[/]")
        else:
            print("  [red]❌ Refusal direction 主要在强 singular value 方向，弱敏感子空间策略可能无效[/]")

    print(f"\n  查看可视化:")
    print(f"    - SV衰减: {sv_path}")
    print(f"    - 对齐分析: {align_path}")
    print(f"    - Fisher基线: {fisher_path}")
    print()


if __name__ == "__main__":
    main()
