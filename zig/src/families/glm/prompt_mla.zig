//! MLA prompt-chunk code: projection, absorb/unabsorb, key selection, attention, o_proj; split for the 600-line rule.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const kernels = @import("kernels.zig");
const wide = @import("wide.zig");
const Kernels = kernels.Kernels;
const Ref = wts.Ref;
const Prompt = @import("prompt.zig").Prompt;
const bind = fwd.bind;
const size = fwd.size;

/// Rows one select pass holds: 256 (v1.0.12); GLM_SELECT_WIDE=1 gives 2048, host-only so bit-exact. DEFAULT OFF.
pub fn selectRows() u32 {
    if (std.c.getenv("GLM_SELECT_WIDE")) |v| {
        if (v[0] != 0 and v[0] == '1') return 2048;
    }
    return 256;
}

/// y [M, N] = x [M, K] W^T for a 4-bit matrix (64x64 tiles; N rounded up to the weights' padded rows, y's row pitch).
pub fn qmm(p: *const Prompt, e: mtl.ComputeEncoder, x: Ref, q: wts.Affine, y: Ref, M: u32) void {
    if (!wide.native(q)) return qmmAt(p, e, x, q.k, q, y, M);
    affine_mm.rowSums(e, p.mm.mm_bf16, 64, x, p.sums, M, q.k);
    affine_mm.dense(e, p.mm.mm_bf16, x, p.sums, q, y, M);
}

/// `qmm` over x's rows `ld` elements apart (a slice of a wider row).
pub fn qmmAt(p: *const Prompt, e: mtl.ComputeEncoder, x: Ref, ld: u32, q: wts.Affine, y: Ref, M: u32) void {
    const pitch = std.mem.alignForward(u32, q.n, 64);
    if (!wide.native(q)) return wide.denseMm(
        &p.mm.wide.?,
        e,
        q,
        .{ .in = x, .ldx = ld, .out = y, .ldo = pitch, .rows = M },
        p.sums,
    );
    const mm = p.mm.mm_bf16;
    affine_mm.rowSumsAt(e, mm, 64, x, p.sums, M, q.k, ld);
    affine_mm.denseBatch(e, mm, x, p.sums, q, y, M, 1, .{ .x_row = @intCast(ld), .y_row = @intCast(pitch) });
}

/// `qmm` with fp32 out (TP2: one Mac's partial of a row-split projection).
pub fn qmmF32(p: *const Prompt, e: mtl.ComputeEncoder, x: Ref, q: wts.Affine, y: Ref, M: u32) void {
    affine_mm.rowSums(e, p.mm.mm_bf16, 64, x, p.sums, M, q.k);
    affine_mm.dense(e, p.mm.mm_f32, x, p.sums, q, y, M);
}

/// Every row's 64 heads' latent outputs to values on the tensor units: each head's value half of kv_b, dense.
pub fn unabsorb(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Mla, M: u32) void {
    const c = x.c;
    const H = c.mla_heads;
    const K = c.kv_lora;
    const per = c.nope + c.v_dim; // kv_b's rows a head: its key half, then its value half
    if (w.absorbed) |pair| return wide.headsBatch(
        &p.mm.wide.?,
        e,
        pair[1],
        H,
        .{ .in = p.att, .ldx = H * K, .out = p.vals, .ldo = H * c.v_dim, .rows = M },
        K,
        p.sums,
    );
    var q = w.kv_b;
    q.w = q.w.at(@as(usize, c.nope) * K / 2);
    q.s = q.s.at(@as(usize, c.nope) * (K / 64) * 2);
    q.b = q.b.at(@as(usize, c.nope) * (K / 64) * 2);
    q.n = c.v_dim;
    affine_mm.rowSums(e, p.mm.mm_bf16, 64, p.att, p.sums, M * H, K);
    affine_mm.denseBatch(e, p.mm.mm_bf16, p.att, p.sums, q, p.vals, M, H, .{ .x_row = @intCast(H * K), .y_row = @intCast(H * c.v_dim), .sums_row = @intCast(H), .x_batch = @intCast(K), .y_batch = @intCast(c.v_dim), .sums_batch = 1, .w_batch = @intCast(per) });
}

/// `n` rows' key lists from index scores over every pooled block they read (positions p0..), on the tensor units.
pub fn selectKeys(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, iq: Ref, iw: Ref, indices: Ref, n: u32, p0: u32) void {
    const c = x.c;
    const s_stride = x.s.cap / c.kpool + 1;
    const most = (p0 + n) / c.kpool; // the last row's blocks
    e.setPipeline(x.k.index_scores_nax);
    bind(e, 0, .{ iq, iw, x.s.mla[mi].pool, p.sscore });
    e.setValue([5]u32{ p0, c.qrProj(), c.i_heads, s_stride, n }, 4);
    e.dispatchGroups(size((most + 63) / 64, (n + 63) / 64, 1), size(128, 1, 1));
    e.setPipeline(x.k.index_select);
    bind(e, 0, .{ p.sscore, indices });
    e.setValue([5]u32{ p0, c.i_topk / c.kpool, c.keyWidth(), s_stride, c.keyWidth() }, 2);
    e.dispatchGroups(size(n, 1, 1), size(1024, 1, 1));
}

/// MLA layer `mi` on M rows at pos..: projections, cache writes, key list (sparse); B4 `mode` gates halves.
pub const MlaMode = enum { full, proj, attn };

pub fn mla(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, w: *const wts.Mla, x_in: Ref, M: u32, pos: u32, mode: MlaMode) void {
    const c = x.c;
    const XP: u32 = @intCast(std.mem.alignForward(usize, c.xProj(), 64));
    const width = c.keyWidth();
    const run_proj = mode != .attn;
    const run_attn = mode != .proj;
    if (run_proj and fwd.on(x, "mla_proj")) {
        qmm(p, e, x_in, w.x_proj, p.xp, M);
        fwd.rms(x, e, p.xp, w.q_norm, p.qr, M, c.q_lora, XP, c.q_lora, c.eps);
        qmm(p, e, p.qr, w.qr_proj, p.qp, M);
    }
    if (run_attn and fwd.on(x, "mla_cache")) fwd.mlaCache(x, e, mi, w, x_in, p.xp, XP, p.iw, M, pos);
    if (run_proj) {
        if (fwd.on(x, "mla_absorb") and w.absorbed != null) {
            wide.headsBatch(
                &p.mm.wide.?,
                e,
                w.absorbed.?[0],
                c.mla_heads,
                .{ .in = p.qp, .ldx = c.qrProj(), .out = p.ql, .ldo = c.mla_heads * c.kv_lora, .rows = M },
                c.nope,
                p.sums,
            ); // mlx-lm's embed_q
        } else if (fwd.on(x, "mla_absorb")) {
            e.setPipeline(if (c.tp > 1) x.k.absorb_nax_tp else x.k.absorb_nax); // row kernel's qvm, weights in three bf16 parts
            bind(e, 0, .{ w.kv_b.w, w.kv_b.s, w.kv_b.b, p.qp, p.ql });
            e.setValue([2]i32{ @intCast(M), @intCast(c.qrProj()) }, 5);
            e.dispatchGroups(size(c.kv_lora / 64, (M + 63) / 64, c.mla_heads), size(128, 1, 1));
        }
    }
    var dense: u32 = 0;
    while (dense < M and pos + dense + 1 <= c.i_topk) dense += 1;
    if (run_attn and fwd.on(x, "mla_select")) {
        if (dense > 0) {
            e.setPipeline(x.k.dense_indices);
            bind(e, 0, .{p.indices});
            e.setValue([4]u32{ width, pos, dense, 0 }, 1);
            e.dispatchThreads(size(width, dense, 1), size(256, 1, 1));
        }
        var r0 = dense;
        const pass = selectRows();
        while (r0 < M) : (r0 += pass) {
            const sb = @min(pass, M - r0);
            selectKeys(p, x, e, mi, p.qp.at((@as(usize, r0) * c.qrProj() + c.mla_heads * c.nope) * 2), p.iw.at(@as(usize, r0) * c.i_heads * 2), p.indices.at(@as(usize, r0) * width * 4), sb, pos + r0);
        }
    }
    if (run_attn and fwd.on(x, "mla_attn")) {
        if (p.sparse_nax) {
            e.setPipeline(if (c.tp > 1) x.k.sparse_nax_tp else x.k.sparse_nax);
            bind(e, 0, .{ p.ql, x.s.mla[mi].keys, p.indices });
            e.setValue(@as(f32, 1.0 / 16.0), 3);
            e.setValue([4]i32{ @intCast(width), @intCast(pos + M), @intCast(pos), 0 }, 4);
            bind(e, 5, .{p.att});
            e.dispatchGroups(size(c.mla_heads / 16, M, 1), size(128, 1, 1));
        } else fwd.attendIndexed(x, e, mi, p.ql, p.indices, p.att, M, pos + M);
    }
    if (run_proj and fwd.on(x, "mla_unabs")) unabsorb(p, x, e, w, M);
    if (run_proj and fwd.on(x, "mla_out")) {
        if (c.tp == 1) return qmm(p, e, p.vals, w.o_proj, p.streams.branch, M);
        qmmF32(p, e, p.vals, w.o_proj, p.part, M); // TP2: this Mac's heads' partial, summed with the peer's
        x.ep.?.reduce(e, p.part, p.streams.branch, M);
    }
}
