# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253, ANN001, ANN201, ANN202
"""Correctness and speed check for the fused DPA4 SO(2) convolution operator.

The reference is a dense PyTorch transcription of the documented math with a
block-diagonal Wigner-D, which is what the model builds. Configurations cover
every shape in the production model zoo.
"""

from __future__ import (
    annotations,
)

import argparse
import ctypes
import os
import site
import sys
from pathlib import (
    Path,
)

import torch

# The repository root, resolved from this file so the script runs unchanged on
# every machine.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _REPO_ROOT)


def op_library() -> str:
    """Path of the operator library to load.

    ``DP_OP_LIB`` overrides everything; otherwise the freshest of the local
    build trees and the installed package wins, so a rebuild is picked up
    without reinstalling.
    """
    override = os.environ.get("DP_OP_LIB")
    if override:
        return override
    candidates = [
        *Path(_REPO_ROOT).glob("build/*/op/pt/libdeepmd_op_pt.so"),
        *(Path(p) / "deepmd/lib/libdeepmd_op_pt.so" for p in site.getsitepackages()),
    ]
    found = [c for c in candidates if c.is_file()]
    if not found:
        raise FileNotFoundError("no libdeepmd_op_pt.so found; set DP_OP_LIB")
    return str(max(found, key=lambda c: c.stat().st_mtime))


# lmax, focus width, focus streams, layers, mixer rank, heads
ZOO = {
    "nano": (1, 32, 1, 3, 0, 1),
    "mini": (2, 32, 1, 3, 1, 1),
    "neo": (3, 32, 2, 3, 1, 1),
    "air": (3, 64, 1, 4, 1, 1),
    "plus": (4, 64, 1, 4, 2, 1),
    "pro": (5, 64, 2, 4, 2, 1),
}


def load_ops() -> None:
    ctypes.CDLL(op_library(), mode=ctypes.RTLD_GLOBAL)
    import deepmd.pt_expt.kernels.cuda.dpa4.so2_conv as binding

    binding.ensure_registered()


def red_wigner_row(lmax: int, r: int) -> int:
    if r <= lmax:
        return r * r + r
    if r <= 2 * lmax:
        l = r - lmax
        return l * l + l - 1
    l = r - 2 * lmax
    return l * l + l + 1


def run_tables(lmax: int, device) -> tuple:
    from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv import (
        wigner_run_tables,
    )

    return tuple(x.to(device) for x in wigner_run_tables(lmax))


def selected_rows(quat: torch.Tensor, lmax: int, tables: tuple) -> torch.Tensor:
    """Reduced Wigner rows from the fitted run tables, shape (E, RED, DIM)."""
    from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv import (
        _monomial_exponents,
        _monomials,
    )

    exps = _monomial_exponents(2 * lmax).to(quat.device)
    run = _monomials(quat, exps) @ tables[0].t()  # (E, NW)
    dim = (lmax + 1) ** 2
    red = 3 * lmax + 1
    dsel = quat.new_zeros(quat.shape[0], red, dim)
    off = 0
    for r in range(red):
        l = r if r <= lmax else (r - lmax if r <= 2 * lmax else r - 2 * lmax)
        width = 2 * l + 1
        dsel[:, r, l * l : l * l + width] = run[:, off : off + width]
        off += width
    return dsel


def make_case(
    n_node: int,
    deg: int,
    lmax: int,
    cf: int,
    n_focus: int,
    n_layers: int,
    rank: int,
    n_head: int,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    dim = (lmax + 1) ** 2
    red = 3 * lmax + 1
    m0, m1, gate = (lmax + 1) * cf, 2 * lmax * cf, lmax * cf
    ksz = (lmax + 1) ** 2 + lmax**2
    c_wide = n_focus * cf
    n_edge = n_node * deg
    kc_len = (lmax + 1) * c_wide if rank == 0 else ksz * rank
    return {
        "x": torch.randn(n_node, dim, c_wide, device=device) * 0.5,
        "src": torch.randint(0, n_node, (n_edge,), device=device).to(torch.long),
        "dst": torch.arange(n_node, device=device)
        .repeat_interleave(deg)
        .to(torch.long),
        "quat": torch.nn.functional.normalize(
            torch.randn(n_edge, 4, device=device), dim=1
        ),
        "tables": run_tables(lmax, device),
        "kc": torch.randn(n_edge, kc_len, device=device) * 0.5,
        "cb": (
            torch.randn(max(rank, 1), c_wide, device=device) * 0.5 + 1.0
            if rank
            else torch.zeros(1, device=device)
        ),
        "w0": torch.randn(n_layers, n_focus, m0, m0, device=device) / m0**0.5,
        "w1": torch.randn(n_layers, n_focus, m1, m1, device=device) / m1**0.5,
        "gw": torch.randn(n_layers - 1, n_focus, cf, gate, device=device) / cf**0.5,
        "q": torch.randn(n_node, c_wide, device=device) * 0.5,
        "k": torch.randn(n_node, c_wide, device=device) * 0.5,
        "logit_w": torch.randn(n_focus, cf, n_head, device=device) * 0.3,
        "null_logit": torch.randn(n_focus, n_head, device=device) * 0.5,
        "env": torch.rand(n_edge, device=device) * 0.9,
        "rad0": torch.randn(n_edge, c_wide, device=device) * 0.5,
        "fscale": (
            torch.rand(n_edge, n_focus, device=device) + 0.5
            if n_focus > 1
            else torch.empty(0, device=device)
        ),
        "head_gate": torch.rand(n_node, n_focus, n_head, device=device),
        "rescale": torch.rand(dim, device=device) + 0.5,
        "lmax": lmax,
        "cf": cf,
        "rank": rank,
        "n_focus": n_focus,
        "n_layers": n_layers,
        "n_head": n_head,
        "red": red,
        "dim": dim,
    }


def reference_alpha(c: dict) -> torch.Tensor:
    """Envelope-gated segment softmax with a null mass, dense transcription."""
    n_edge = c["src"].shape[0]
    n_node = c["x"].shape[0]
    n_focus, n_head = c["n_focus"], c["n_head"]
    cf, head_dim = c["cf"], c["cf"] // c["n_head"]
    q = c["q"].reshape(n_node, n_focus, n_head, head_dim)
    k = c["k"].reshape(n_node, n_focus, n_head, head_dim)
    logits = (q[c["dst"]] * k[c["src"]]).sum(-1) * head_dim**-0.5
    rad = c["rad0"].reshape(n_edge, n_focus, cf)
    logits = logits + torch.einsum("efi,fih->efh", rad, c["logit_w"])
    env = c["env"]
    eff = torch.where(
        (env > 0).view(n_edge, 1, 1),
        logits + 2.0 * torch.log(env.clamp_min(1e-30)).view(n_edge, 1, 1),
        torch.full_like(logits, float("-inf")),
    )
    null = c["null_logit"].view(1, n_focus, n_head)
    group_max = null.expand(n_node, n_focus, n_head).clone()
    idx = c["dst"].view(n_edge, 1, 1).expand_as(eff)
    group_max = torch.scatter_reduce(
        group_max, 0, idx, eff, reduce="amax", include_self=True
    )
    edge_exp = torch.exp(eff - group_max[c["dst"]])
    denom = torch.zeros_like(group_max).scatter_add_(0, idx, edge_exp)
    denom = denom + torch.exp(null - group_max)
    alpha = edge_exp / denom[c["dst"]]
    if c["fscale"].numel() > 0:
        alpha = alpha * c["fscale"].unsqueeze(-1)
    return alpha


def reference(c: dict) -> torch.Tensor:
    lmax, cf, rank = c["lmax"], c["cf"], c["rank"]
    n_focus, n_head = c["n_focus"], c["n_head"]
    dim, red = c["dim"], c["red"]
    ndeg, m0 = lmax + 1, (lmax + 1) * cf
    k0 = ndeg * ndeg
    x, src, dst = c["x"], c["src"], c["dst"]
    n_node, c_wide = x.shape[0], x.shape[2]
    n_edge = src.shape[0]
    dsel = selected_rows(c["quat"], lmax, c["tables"])  # (E, RED, DIM)
    x_local = torch.bmm(dsel, x[src])  # (E, RED, C_wide)

    kc = c["kc"]
    if rank == 0:
        rad = kc.reshape(n_edge, ndeg, c_wide)
        keff0 = torch.zeros(n_edge, ndeg, ndeg, c_wide, device=x.device)
        for o in range(ndeg):
            keff0[:, o, o, :] = rad[:, o, :]
        keff1 = torch.zeros(n_edge, lmax, lmax, c_wide, device=x.device)
        for o in range(lmax):
            keff1[:, o, o, :] = rad[:, o + 1, :]
    else:
        flat = kc.reshape(n_edge, -1, rank)
        eff = torch.einsum("esr,rc->esc", flat, c["cb"])  # (E, KSZ, C_wide)
        keff0 = eff[:, :k0, :].reshape(n_edge, ndeg, ndeg, c_wide)
        keff1 = eff[:, k0:, :].reshape(n_edge, lmax, lmax, c_wide)

    mixed = torch.zeros_like(x_local)
    for o in range(ndeg):
        mixed[:, o, :] = sum(keff0[:, i, o, :] * x_local[:, i, :] for i in range(ndeg))
    for o in range(lmax):
        mixed[:, ndeg + o, :] = sum(
            keff1[:, i, o, :] * x_local[:, ndeg + i, :] for i in range(lmax)
        )
        mixed[:, ndeg + lmax + o, :] = sum(
            keff1[:, i, o, :] * x_local[:, ndeg + lmax + i, :] for i in range(lmax)
        )

    # Focus-major flat activation, one row per focus stream.
    u = (
        mixed.reshape(n_edge, red, n_focus, cf)
        .permute(0, 2, 1, 3)
        .reshape(n_edge, n_focus, -1)
    )
    for layer in range(c["n_layers"]):
        z0 = torch.einsum("efi,fio->efo", u[:, :, :m0], c["w0"][layer])
        z1 = torch.einsum("efi,fio->efo", u[:, :, m0:], c["w1"][layer])
        if layer < c["n_layers"] - 1:
            sig = torch.sigmoid(
                torch.einsum("efi,fio->efo", z0[:, :, :cf], c["gw"][layer])
            )
            add0 = torch.cat(
                [z0[:, :, :cf] * torch.sigmoid(z0[:, :, :cf]), z0[:, :, cf:] * sig], -1
            )
            add1 = z1 * sig.repeat(1, 1, 2)
            u = u + torch.cat([add0, add1], -1)
        else:
            u = u + torch.cat([z0, z1], -1)

    u_red = (
        u.reshape(n_edge, n_focus, red, cf)
        .permute(0, 2, 1, 3)
        .reshape(n_edge, red, c_wide)
    )
    rb = torch.bmm(dsel.transpose(1, 2), u_red)  # (E, DIM, C_wide)
    head_dim = cf // n_head
    weight = (
        reference_alpha(c)
        .reshape(n_edge, n_focus, n_head, 1)
        .expand(n_edge, n_focus, n_head, head_dim)
        .reshape(n_edge, 1, c_wide)
    )
    pre = torch.zeros(n_node, dim, c_wide, device=x.device, dtype=x.dtype)
    pre.index_add_(0, dst, rb * weight)
    pre = pre * c["rescale"].view(1, -1, 1)
    gate = (
        c["head_gate"]
        .reshape(n_node, n_focus, n_head, 1)
        .expand(n_node, n_focus, n_head, head_dim)
        .reshape(n_node, 1, c_wide)
    )
    return pre * gate


def edge_csr(key, n_node):
    order = torch.argsort(key, dim=0, stable=True)
    counts = torch.bincount(key, minlength=n_node)
    row_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
    return order, row_ptr


def fused(c: dict) -> tuple:
    n = c["x"].shape[0]
    csr = edge_csr(c["dst"], n) + edge_csr(c["src"], n)
    runs = torch.ops.deepmd.dpa4_wigner_runs(
        c["quat"], c["tables"][0], c["tables"][2], c["lmax"]
    )
    return torch.ops.deepmd.dpa4_so2_conv(
        c["x"],
        c["src"],
        c["dst"],
        *csr,
        runs,
        c["kc"],
        c["cb"],
        c["w0"],
        c["w1"],
        c["gw"],
        c["q"],
        c["k"],
        c["logit_w"],
        c["null_logit"],
        c["env"],
        c["rad0"],
        c["fscale"],
        c["head_gate"],
        c["rescale"],
        c["lmax"],
        c["cf"],
        c["rank"],
    )


def fused_grads(c: dict, leaves: tuple, g_out: torch.Tensor) -> dict:
    """Drive the registered autograd of the operator end to end."""
    import deepmd.pt_expt.kernels.cuda.dpa4.so2_conv as binding

    binding.ensure_registered()
    tensors = {name: c[name].detach().requires_grad_(True) for name in leaves}
    saved = {name: c[name] for name in leaves}
    c.update(tensors)
    out = fused(c)[0]
    grads = torch.autograd.grad(out, [tensors[n] for n in leaves], g_out)
    c.update(saved)
    return dict(zip(leaves, grads, strict=True))


def block_mask(lmax: int, device) -> torch.Tensor:
    dim = (lmax + 1) ** 2
    m = torch.zeros(dim, dim, device=device)
    for l in range(lmax + 1):
        b, n = l * l, 2 * l + 1
        m[b : b + n, b : b + n] = 1.0
    return m


def check(name: str, cfg: tuple, nodes: int, deg: int) -> float:
    lmax, cf, n_focus, n_layers, rank, n_head = cfg
    c = make_case(nodes, deg, lmax, cf, n_focus, n_layers, rank, n_head)
    worst = 0.0
    ref = reference(c)
    got, got_alpha = fused(c)[0], fused(c)[1]
    rel = ((got - ref).abs().max() / max(ref.abs().max().item(), 1e-30)).item()
    a_ref = reference_alpha(c)
    a_rel = (
        (got_alpha - a_ref).abs().max() / max(a_ref.abs().max().item(), 1e-30)
    ).item()
    worst = max(worst, rel, a_rel)
    print(f"  [{name}] fwd rel={rel:.3e} alpha={a_rel:.1e}", end="")

    leaves = ["x", "quat", "kc", "q", "k", "env", "rad0", "head_gate"]
    if c["fscale"].numel() > 0:
        leaves.append("fscale")
    for kname in leaves:
        c[kname] = c[kname].detach().requires_grad_(True)
    out = reference(c)
    g_out = torch.randn_like(out)
    want = dict(
        zip(
            leaves,
            torch.autograd.grad(out, [c[kname] for kname in leaves], g_out),
            strict=True,
        )
    )
    for kname in leaves:
        c[kname] = c[kname].detach()
    got_g = fused_grads(c, tuple(leaves), g_out)
    for label in leaves:
        a, b = got_g[label], want[label]
        r = ((a - b).abs().max() / max(b.abs().max().item(), 1e-30)).item()
        worst = max(worst, r)
        print(f"  g_{label}={r:.1e}", end="")
    print()
    return worst


def timed(fn, iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s0, s1 = torch.cuda.Event(True), torch.cuda.Event(True)
    s0.record()
    for _ in range(iters):
        fn()
    s1.record()
    torch.cuda.synchronize()
    return s0.elapsed_time(s1) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ZOO))
    ap.add_argument("--nodes", type=int, default=8000)
    ap.add_argument("--deg", type=int, default=158)
    ap.add_argument("--check-nodes", type=int, default=29)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    load_ops()
    torch.backends.cuda.matmul.allow_tf32 = False

    if not args.skip_check:
        print("[check] against the dense reference")
        worst = 0.0
        for name in args.models:
            for deg in (7, 17, 33):
                worst = max(
                    worst, check(f"{name}/d{deg}", ZOO[name], args.check_nodes, deg)
                )
        print(f"[check] worst relative error {worst:.3e}")

    if args.skip_bench:
        return
    print("[bench] production shape")
    for name in args.models:
        lmax, cf, n_focus, n_layers, rank, n_head = ZOO[name]
        c = make_case(args.nodes, args.deg, lmax, cf, n_focus, n_layers, rank, n_head)
        torch.cuda.reset_peak_memory_stats()
        ms_f = timed(lambda c=c: fused(c))
        import deepmd.pt_expt.kernels.cuda.dpa4.so2_conv as binding

        binding.ensure_registered()
        for kname in ("x", "quat", "kc", "q", "k", "env", "rad0", "head_gate"):
            c[kname] = c[kname].detach().requires_grad_(True)
        out = fused(c)[0]
        g_out = torch.randn_like(out)
        leaves = [c[kname] for kname in ("x", "quat", "kc", "q", "k")]

        def run_bwd(out=out, leaves=leaves, g_out=g_out):
            torch.autograd.grad(out, leaves, g_out, retain_graph=True)

        ms_b = timed(run_bwd)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"  [{name:5s}] lmax={lmax} cf={cf} F={n_focus} L={n_layers} r={rank}"
            f"   fwd {ms_f:7.3f} ms  bwd {ms_b:7.3f} ms  peak {peak:5.2f} GiB"
        )
        del c, out, g_out, leaves
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    main()
