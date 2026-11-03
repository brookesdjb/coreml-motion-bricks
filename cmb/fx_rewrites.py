"""FX graph rewrites applied to NVIDIA's captured PyTorch model before conversion (cmb/convert.py).

Each one replaces a pattern with the same maths written in ops that coremltools converts and the Neural Engine
runs correctly; convert.py checks the rewritten graph against eager PyTorch. Why each one exists:
experiments/direct/RESULTS.md and docs/findings.md.
"""
import torch


def _put_last_dim(x, idx, values):
    """x[..., idx] = values, written as a one-hot matmul and a mask blend (exact; no scatter op)."""
    select = (idx.long().unsqueeze(-1) == torch.arange(x.shape[-1], device=x.device)).to(x.dtype)  # (k, D)
    placed = torch.matmul(values.to(x.dtype), select)        # (..., D): values at idx, 0 elsewhere
    keep = 1 - select.sum(0).clamp(max=1)                     # (D,): 0 at idx, 1 elsewhere
    return x * keep + placed


def rewrite_index_put(gm):
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target != torch.ops.aten.index_put.default:
            continue
        x, idx, vals = n.args[:3]
        acc = n.args[3] if len(n.args) > 3 else False
        rank = len(n.meta["val"].shape)
        if acc or len(idx) != rank or any(i is not None for i in idx[:-1]) or idx[-1] is None:
            raise NotImplementedError(f"unexpected index_put pattern at {n}")
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_put_last_dim, (x, idx[-1], vals))
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new)
        gm.graph.erase_node(n)
        n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def _scatter_value(x, dim, index, value):
    return torch.scatter(x, dim, index, torch.full(index.shape, value, dtype=x.dtype, device=x.device))


def rewrite_scatter_value(gm):
    """scatter(x, dim, index, <python scalar>) -> scatter with a tensor of x's dtype (coremltools typed it int32)."""
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target == torch.ops.aten.scatter.value:
            x, dim, index, value = n.args[:4]
            with gm.graph.inserting_before(n):
                new = gm.graph.call_function(_scatter_value, (x, dim, index, value))
            new.meta = dict(n.meta)
            n.replace_all_uses_with(new)
            gm.graph.erase_node(n)
            n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def _int_scatter(x, dim, index, src):
    # Small integer data (token counts <= 64): exact in fp16/fp32, and avoids coremltools' int/float mix-up.
    src = src if torch.is_tensor(src) else torch.full(index.shape, src, device=x.device)
    return torch.scatter(x.float(), dim, index, src.float()).to(x.dtype)


def rewrite_int_scatter(gm):
    n_done = 0
    for n in list(gm.graph.nodes):
        fn = n.target
        if fn not in (torch.ops.aten.scatter.src, torch.ops.aten.scatter.value, _scatter_value):
            continue
        if n.meta.get("val") is not None and n.meta["val"].is_floating_point():
            continue
        x, dim, index, src = n.args[:4]
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_int_scatter, (x, dim, index, src))
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new)
        gm.graph.erase_node(n)
        n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def positive_cumsum_axes(gm):
    """coremltools rejects valid negative cumsum axes; rewrite them as positive."""
    n_done = 0
    for n in gm.graph.nodes:
        if n.target == torch.ops.aten.cumsum.default and isinstance(n.args[1], int) and n.args[1] < 0:
            rank = len(n.args[0].meta["val"].shape)
            n.args = (n.args[0], n.args[1] + rank, *n.args[2:])
            n_done += 1
    gm.recompile()
    return n_done


def finite_masks(gm, value=-1.0e4):
    """-inf attention-mask fills -> a large finite negative (same softmax result, fp16-safe on the Neural Engine)."""
    import math
    n_done = 0
    for n in gm.graph.nodes:
        if n.target in (torch.ops.aten.masked_fill.Scalar, torch.ops.aten.full.default):
            new_args = tuple(value if isinstance(a, float) and math.isinf(a) and a < 0 else a for a in n.args)
            if new_args != n.args:
                n.args = new_args
                n_done += 1
    gm.recompile()
    return n_done


def _static_full(shape, value, dtype):
    return torch.full(shape, value, dtype=dtype)


def _sort_as_topk(x, dim=-1, descending=False):
    return torch.topk(x, x.shape[dim], dim=dim, largest=descending, sorted=True)


def static_fills_and_sort(gm):
    """zeros_like/full_like/ones_like -> constant-shape full (shapes are static here); sort -> full-length topk.
    The Neural Engine compiler rejected the whole program with these (everything fell back to the CPU)."""
    fills = {torch.ops.aten.zeros_like.default: 0.0, torch.ops.aten.ones_like.default: 1.0}
    n_fill = n_sort = 0
    for n in list(gm.graph.nodes):
        if n.target in fills or n.target == torch.ops.aten.full_like.default:
            val = n.meta["val"]
            value = fills.get(n.target, n.args[1] if len(n.args) > 1 else None)
            with gm.graph.inserting_before(n):
                new = gm.graph.call_function(_static_full, (tuple(val.shape), value, val.dtype))
            new.meta = dict(n.meta)
            n.replace_all_uses_with(new); gm.graph.erase_node(n); n_fill += 1
        elif n.target == torch.ops.aten.sort.default:
            dim = n.args[1] if len(n.args) > 1 else -1
            desc = n.args[2] if len(n.args) > 2 else False
            with gm.graph.inserting_before(n):
                new = gm.graph.call_function(_sort_as_topk, (n.args[0], dim, desc))
            new.meta = dict(n.meta)
            n.replace_all_uses_with(new); gm.graph.erase_node(n); n_sort += 1
    gm.graph.lint(); gm.recompile()
    return n_fill, n_sort


def _scatter_one_as_float_mask(x, dim, index, src):
    """scatter writing one position along `dim`, as a blend with a position mask: same values, no scatter op
    (scatter_along_axis made the Neural Engine compiler reject the whole program). The position test is done in
    x's float type (positions <= 64 are exact in fp16): an integer modulo/compare/tile version ran on the CPU and
    handed (B, 64, 512) masks back to the Neural Engine. Integer x is computed in float and cast back."""
    xf = x if x.is_floating_point() else x.float()
    d = dim % x.dim(); n = x.shape[d]
    shape = [1] * x.dim(); shape[d] = n
    pos = index.to(xf.dtype)
    pos = pos + n * (pos < 0).to(xf.dtype)
    m = (torch.arange(n, dtype=xf.dtype, device=x.device).reshape(shape) == pos).to(xf.dtype)
    # x*(1-m) + src*m: exact for a 0/1 mask and finite x, and broadcasts as plain mul/add (coremltools tiles
    # every select input to the full (B, 64, C) shape)
    out = xf * (1 - m) + src.to(xf.dtype) * m
    return out if x.is_floating_point() else out.to(x.dtype)


def scatters_as_masks(gm):
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target not in (torch.ops.aten.scatter.src, _int_scatter):
            continue
        x, dim, index, src = n.args[:4]
        if not torch.is_tensor(src.meta.get("val")) if hasattr(src, "meta") else True:
            continue
        d = dim % len(x.meta["val"].shape)
        if index.meta["val"].shape[d] != 1:
            continue
        # the index is usually a position repeated across channels; compare against the un-repeated one
        while index.target in (torch.ops.aten.repeat.default, torch.ops.aten.expand.default) and \
                index.args[0].meta["val"].dim() == index.meta["val"].dim():
            index = index.args[0]
        if hasattr(src, "target") and src.target in (torch.ops.aten.repeat.default, torch.ops.aten.expand.default) \
                and src.args[0].meta["val"].dim() == src.meta["val"].dim():
            src = src.args[0]
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_scatter_one_as_float_mask, (x, dim, index, src))
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new); gm.graph.erase_node(n); n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def _gather_as_onehot(x, dim, index):
    """gather(x, dim, index) for float x as a one-hot select-and-sum along `dim`: same values, no gather op.
    A data-dependent gather (heights at the chosen clip's last frames) made the Neural Engine compiler reject
    the whole program."""
    d = dim % x.dim()
    xm, im = x.movedim(d, -1), index.long().movedim(d, -1)        # (..., N), (..., K)
    onehot = (im.unsqueeze(-1) == torch.arange(xm.shape[-1], device=x.device)).to(x.dtype)  # (..., K, N)
    out = (onehot * xm.unsqueeze(-2)).sum(-1)                      # (..., K)
    return out.movedim(-1, d)


def gathers_as_onehot(gm):
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target != torch.ops.aten.gather.default or not n.meta["val"].is_floating_point():
            continue
        x, dim, index = n.args[:3]
        if x.meta["val"].shape[:dim % x.meta["val"].dim()] != index.meta["val"].shape[:dim % x.meta["val"].dim()]:
            continue  # only the plain "same leading dims" gathers used here
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_gather_as_onehot, (x, dim, index))
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new); gm.graph.erase_node(n); n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def _embedding_as_onehot(w, idx):
    """embedding(w, idx) as a one-hot (float compare) times the table. At batch 2 the Neural Engine miscomputed
    the gather -> reshape -> linear chain that follows the token embedding."""
    pos = idx.to(w.dtype).unsqueeze(-1)
    onehot = (pos == torch.arange(w.shape[0], dtype=w.dtype, device=w.device)).to(w.dtype)
    return onehot @ w


def embeddings_as_onehot(gm):
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target != torch.ops.aten.embedding.default or len(n.args) > 2 and n.args[2] != -1:
            continue  # padding_idx only matters for the backward pass, but keep to the plain form
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_embedding_as_onehot, n.args[:2])
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new); gm.graph.erase_node(n); n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def _attn_out_proj_batch_first(b, y, w, rows):
    """addmm(b, y.permute(2, 0, 1, 3).reshape(rows, c), w) for attention output y (B, H, T, D), computed
    batch-first: (B, T, H*D) @ w + b, then transposed to nn.MultiheadAttention's (T, B, E) and flattened."""
    z = y.permute(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1) @ w + b
    return z.transpose(0, 1).reshape(rows, w.shape[-1])


def attn_out_proj_batch_first(gm):
    """nn.MultiheadAttention flattens its (B, H, T, D) output seq-first, as (T * B, H * D) rows, into the output
    projection. At batch 4 the Neural Engine compiler rejected that transpose -> reshape -> linear (and so dropped
    the whole program to the CPU); batch-first, the transpose moves after the linear and cancels against the
    transpose back to batch-first that follows."""
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target != torch.ops.aten.addmm.default or n.kwargs:
            continue
        b, v, w = n.args
        if v.target != torch.ops.aten.view.default:
            continue
        x = v.args[0]
        p = x.args[0] if x.target == torch.ops.aten.clone.default else x  # .contiguous() after the transpose
        if p.target != torch.ops.aten.permute.default or list(p.args[1]) != [2, 0, 1, 3]:
            continue
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_attn_out_proj_batch_first, (b, p.args[0], w, v.meta["val"].shape[0]))
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new); gm.graph.erase_node(n); n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def _softmax_zero_if_all_masked(x, dim):
    """aten._safe_softmax: softmax, but 0 for rows whose entries are all masked. Its decomposition is an int
    fill_like and bool any/logical_not per attention layer, which can't run on the Neural Engine. Same values,
    float-only: masked entries are <= -1e4 after finite_masks (or -inf), real scores are far above that."""
    keep = (x > -5.0e3).to(x.dtype).amax(dim, keepdim=True)
    return torch.softmax(x, dim) * keep


def safe_softmax_as_float(gm):
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target != torch.ops.aten._safe_softmax.default:
            continue
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_softmax_zero_if_all_masked, (n.args[0], n.args[1]))
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new); gm.graph.erase_node(n); n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def _conv1d_batch_folded(x, w, b, stride, padding, dilation, groups):
    """Conv1d over (B, C, L) computed as a Conv2d over (1, C, B, L) with a 1-tall kernel: same maths, but the
    Neural Engine sees one image instead of a batch (the published ONNX route does the same at batch > 4)."""
    y = torch.nn.functional.conv2d(x.permute(1, 0, 2).unsqueeze(0), w.unsqueeze(2), b, (1, stride[0]),
                                   (0, padding[0]), (1, dilation[0]), groups)
    return y.squeeze(0).permute(1, 0, 2)


def fold_conv1d_batch(gm):
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target != torch.ops.aten.convolution.default or len(n.meta["val"].shape) != 3:
            continue
        x, w, b, stride, padding, dilation, transposed, _, groups = n.args
        if transposed:
            continue
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_conv1d_batch_folded, (x, w, b, stride, padding, dilation, groups))
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new); gm.graph.erase_node(n); n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


def _repeat_last(x, k):
    """Nearest-neighbour upsampling by an integer factor along the last axis, as expand + reshape."""
    return x.unsqueeze(-1).expand(*x.shape, k).reshape(*x.shape[:-1], x.shape[-1] * k)


def nearest_upsample_as_repeat(gm):
    """upsample_nearest1d arrives decomposed as _unsafe_index(x, [None, None, floor(arange * scale)]), a gather.
    For an integer factor it is each element repeated k times; write it so (checked by the FX-vs-eager test)."""
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target != torch.ops.aten._unsafe_index.Tensor:
            continue
        x, idx = n.args
        lin, lout = x.meta["val"].shape[-1], n.meta["val"].shape[-1]
        if any(i is not None for i in idx[:-1]) or len(idx) != x.meta["val"].dim() or lout % lin:
            continue
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(_repeat_last, (x, lout // lin))
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new); gm.graph.erase_node(n); n_done += 1
    gm.graph.eliminate_dead_code(); gm.graph.lint(); gm.recompile()
    return n_done


def _slice_scatter_as_cat(x, src, dim, start, end):
    d = dim % x.dim(); n = x.shape[d]
    start = 0 if start is None else (start + n if start < 0 else min(start, n))
    end = n if end is None else (end + n if end < 0 else min(end, n))
    parts = [x.narrow(d, 0, start)] if start > 0 else []
    parts.append(src.to(x.dtype))
    if end < n:
        parts.append(x.narrow(d, end, n - end))
    return torch.cat(parts, d) if len(parts) > 1 else parts[0]


def _select_scatter_as_cat(x, src, dim, index):
    d = dim % x.dim()
    return _slice_scatter_as_cat(x, src.unsqueeze(d), d, index % x.shape[d], index % x.shape[d] + 1)


def scatters_as_cat(gm):
    """slice_scatter / select_scatter (functionalized in-place slice writes) -> the source itself when it covers
    the whole tensor, else a concat of the untouched pieces and the source. coremltools lowered them to scatters on
    the flattened tensor, which the Neural Engine can't run (at batch 8: six (8*64*366,) scatters on the CPU)."""
    n_done = 0
    for n in list(gm.graph.nodes):
        if n.target == torch.ops.aten.slice_scatter.default:
            x, src = n.args[:2]
            dim, start, end, step = (list(n.args[2:]) + [0, None, None, 1][len(n.args) - 2:])[:4]
            if step != 1:
                continue
            if tuple(src.meta["val"].shape) == tuple(x.meta["val"].shape):
                n.replace_all_uses_with(src); gm.graph.erase_node(n); n_done += 1
                continue
            fn, args = _slice_scatter_as_cat, (x, src, dim, start, end)
        elif n.target == torch.ops.aten.select_scatter.default:
            fn, args = _select_scatter_as_cat, tuple(n.args[:4])
        else:
            continue
        with gm.graph.inserting_before(n):
            new = gm.graph.call_function(fn, args)
        new.meta = dict(n.meta)
        n.replace_all_uses_with(new); gm.graph.erase_node(n); n_done += 1
    gm.graph.lint(); gm.recompile()
    return n_done


# (name, rewrite, batch sizes it applies to), in the order applied. finite_masks must precede
# safe_softmax_as_float. Above batch 4 the Neural Engine also needs Conv1d batches folded (and convert.py keeps
# tensors under 128 elements in fp32).
REWRITES = [
    ("index_put", rewrite_index_put, None),
    ("int scatter", rewrite_int_scatter, None),
    ("scatter value", rewrite_scatter_value, None),
    ("cumsum axes", positive_cumsum_axes, None),
    ("finite masks", finite_masks, None),
    ("fills and sort", static_fills_and_sort, None),
    ("scatter -> mask", scatters_as_masks, None),
    ("gather -> one-hot", gathers_as_onehot, None),
    ("Conv1d batch fold", fold_conv1d_batch, lambda b: b > 4),
    ("upsample -> repeat", nearest_upsample_as_repeat, lambda b: b > 4),
    ("slice/select scatter -> cat", scatters_as_cat, None),
    ("safe softmax -> float", safe_softmax_as_float, None),
    ("embedding -> one-hot", embeddings_as_onehot, None),
    ("attention out-projection batch-first", attn_out_proj_batch_first, None),
]


def apply(gm, batch):
    """Apply every rewrite that applies at this batch size; returns {name: count}."""
    return {name: fn(gm) for name, fn, applies in REWRITES if applies is None or applies(batch)}
