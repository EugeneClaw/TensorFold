//! MLA attention math (absorb, attendDense|Sparse|Blocks, selectKeys, attendIndexed), split from forward.zig.
const std = @import("std");
const mtl = @import("metal");
const wts = @import("weights.zig");
const cfg = @import("config.zig");
const wide = @import("wide.zig");
const Ref = wts.Ref;

const fwd = @import("forward.zig");
const Seg = fwd.Seg;
const ScoreArgs = fwd.ScoreArgs;
const SelectArgs = fwd.SelectArgs;
const bind = fwd.bind;
const shape = fwd.shape;
const size = fwd.size;

/// Every row's 64 heads of q_nope (`qp`, rows qrProj apart) into the latent: qvm on kv_b's key half, or embed_q's qmv.
pub fn absorb(x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Mla, qp: Ref, ql: Ref, rows: u32) void {
    const c = x.c;
    if (w.absorbed) |pair| return wide.perHead(&x.k.wide.?, e, pair[0], c.mla_heads, .{ .in = qp, .ldx = c.qrProj(), .out = ql, .ldo = c.mla_heads * c.kv_lora, .rows = rows }, c.nope, c.kv_lora);
    e.setPipeline(x.k.absorb);
    bind(e, 0, .{ w.kv_b.w, w.kv_b.s, w.kv_b.b, qp, ql });
    e.setValue(c.qrProj(), 5);
    e.setValue(c.mla_heads, 6);
    e.dispatchGroups(size(1, c.kv_lora / 64, rows * c.mla_heads), size(64, 1, 1));
}

/// Every row's 64 heads' latent outputs to values [rows, heads * v]: qmv_fast on kv_b's value half, or unembed_out's.
pub fn unabsorb(x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Mla, att: Ref, vals: Ref, rows: u32) void {
    const c = x.c;
    if (w.absorbed) |pair| return wide.perHead(&x.k.wide.?, e, pair[1], c.mla_heads, .{ .in = att, .ldx = c.mla_heads * c.kv_lora, .out = vals, .ldo = c.mla_heads * c.v_dim, .rows = rows }, c.kv_lora, c.v_dim);
    e.setPipeline(x.k.unabsorb);
    bind(e, 0, .{ w.kv_b.w, w.kv_b.s, w.kv_b.b, att, vals });
    e.setValue(c.mla_heads, 5);
    e.dispatchGroups(size(1, c.v_dim / 4, rows * c.mla_heads), size(32, 1, 1));
}

/// One row's 64 heads over keys [0, n) as one 64-row matrix against the latent keys: scores, a precise softmax, values.
pub fn attendDense(x: *const fwd.Ctx, e: mtl.ComputeEncoder, keys: Ref, q: Ref, scores: Ref, probs: Ref, out: Ref, n: u32) void {
    const k = x.k;
    const H = x.c.mla_heads;
    e.setPipeline(k.latent_scores);
    bind(e, 0, .{ q, keys, scores });
    e.setValue([2]i32{ @intCast(n), @intFromBool(n < 256) }, 3); // fewer than 256 keys: the 512 dims in two halves
    e.dispatchGroups(size((n + 31) / 32, 1, 1), size(2 * H, 1, 1)); // a simdgroup each 16 heads
    e.setPipeline(k.softmax);
    bind(e, 0, .{scores});
    shape(e, 1, .{n});
    bind(e, 2, .{probs});
    e.dispatchGroups(size(H, 1, 1), size(((n + 3) / 4 + 31) / 32 * 32, 1, 1));
    e.setPipeline(k.latent_values);
    bind(e, 0, .{ probs, keys, out });
    e.setValue([2]i32{ @intCast(n), if (n > 1024) 1024 else 0 }, 3); // past 1,024 keys: the first 1,024 apart
    e.dispatchGroups(size(x.c.kv_lora / 32, 1, 1), size(2 * H, 1, 1));
}

/// Stream rows [first, rows) past index_topk keys: fp32 block scores, best blocks in block order + tail, sparse kernel.
pub fn attendSparse(x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, g: Seg, first: u32) void {
    const c = x.c;
    const sc = x.sc;
    const n = g.rows - first;
    const r: usize = g.row0 + first; // the first sparse row's place in the window
    selectKeys(x, e, mi, sc.qp.at((r * c.qrProj() + c.mla_heads * c.nope) * 2), c.qrProj(), sc.iw.at(r * c.i_heads * 2), sc.sscore, sc.indices, n, g.pos + first);
    const ql = sc.ql.at(r * c.mla_heads * c.kv_lora * 2);
    const att = sc.att.at(r * c.mla_heads * c.kv_lora * 2);
    if (c.tp > 1) return attendIndexed(x, e, mi, ql, sc.indices, att, n, g.pos + g.rows);
    attendBlocks(x, e, mi, ql, sc.indices, att, n, g.pos + g.rows);
}

/// `n` rows' 64 heads over listed keys: each block of attn_span entries its own softmax, blocks in order.
pub fn attendBlocks(x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, ql: Ref, indices: Ref, out: Ref, n: u32, key_length: u32) void {
    const c = x.c;
    const k = x.k;
    const blocks = c.attnBlocks();
    const vals = @as(usize, n) * blocks * c.mla_heads * c.kv_lora * 4;
    const marks = @as(usize, n) * blocks * c.mla_heads * 4;
    const part = x.sc.apart;
    e.setPipeline(k.sparse_split);
    bind(e, 0, .{ ql, x.s.mla[mi].keys, indices });
    e.setValue(@as(f32, 1.0 / 16.0), 3);
    e.setValue([4]i32{ @intCast(c.keyWidth()), @intCast(key_length), cfg.attn_span, 0 }, 4);
    bind(e, 5, .{ part, part.at(vals), part.at(vals + marks) });
    e.dispatchGroups(size(c.mla_heads / 16, blocks, n), size(128, 1, 1));
    e.setPipeline(k.sparse_combine);
    bind(e, 0, .{ part, part.at(vals), part.at(vals + marks) });
    e.setValue(@as(i32, @intCast(blocks)), 3);
    bind(e, 4, .{out});
    e.dispatchGroups(size(c.mla_heads, n, 1), size(128, 1, 1));
}

/// Key lists for `n` rows at positions p0.. past index_topk keys (indexer queries `iq` a row `q_stride` apart).
pub fn selectKeys(x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, iq: Ref, q_stride: u32, iw: Ref, scores: Ref, indices: Ref, n: u32, p0: u32) void {
    const c = x.c;
    const k = x.k;
    const s_stride = x.s.cap / c.kpool + 1;
    const most = (p0 + n) / c.kpool; // the last row's blocks
    e.setPipeline(k.index_decode); // a row's heads as the tensor op's rows (glm_index_nax.metal)
    bind(e, 0, .{ iq, iw, x.s.mla[mi].pool, scores });
    e.setValue(ScoreArgs{ .p0 = p0, .q_stride = q_stride, .w_stride = c.i_heads, .s_stride = s_stride, .rows = n }, 4);
    e.dispatchGroups(size((most + 127) / 128, n, 1), size(128, 1, 1));
    e.setPipeline(k.index_select);
    bind(e, 0, .{ scores, indices });
    e.setValue(SelectArgs{ .p0 = p0, .top = c.i_topk / c.kpool, .width = c.keyWidth(), .s_stride = s_stride, .i_stride = c.keyWidth() }, 2);
    e.dispatchGroups(size(n, 1, 1), size(1024, 1, 1));
}

/// The sparse kernel for `n` rows' 64 heads over their listed keys (`key_length` keys written so far).
pub fn attendIndexed(x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, ql: Ref, indices: Ref, out: Ref, n: u32, key_length: u32) void {
    const c = x.c;
    e.setPipeline(if (c.tp > 1) x.k.sparse_attention_tp else x.k.sparse_attention);
    bind(e, 0, .{ ql, x.s.mla[mi].keys, indices });
    e.setValue(@as(f32, 1.0 / 16.0), 3);
    e.setValue(@as(i32, @intCast(key_length)), 4);
    bind(e, 5, .{out});
    e.dispatchThreads(size(1024, n * c.mla_heads, 1), size(1024, 1, 1));
}
