//! Prompt chunks: matmuls on the M5 tensor units (core/affine_mm.zig); the rest on row kernels.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const pk = @import("../nemotron/prefill_kernels.zig");
const ep_mod = @import("ep.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const hc = @import("../../core/hc.zig");
const kernels = @import("kernels.zig");
const wide = @import("wide.zig");
const Kernels = kernels.Kernels;
const moe_route = @import("../../core/moe_route.zig");
const Ref = wts.Ref;
const bind = fwd.bind;
const size = fwd.size;

/// Chunk heights a load picks from (tallest that fits this Mac's buffer limit); expert parallel keeps one exchange's.
pub const heights = [_]u32{ 16384, 8192, 6144, 4096, 2048 };
pub const max_rows = heights[0];
pub const mla_mod = @import("prompt_mla.zig");

/// The chunk's buffers beyond the decode scratch's (whose stream fields `streams` points at prompt-sized ones).
pub const Prompt = struct {
    k: pk.Kernels,
    mm: *const Kernels, // the engine's: core/affine_mm.zig's pipelines among them
    streams: st.Scratch, // the decode scratch with x, h, normed, branch, post, comb, inv, mixes, raw, hidden resized
    proj: Ref,
    y: Ref,
    xp: Ref,
    qr: Ref,
    qp: Ref,
    ql: Ref,
    att: Ref,
    vals: Ref,
    iw: Ref,
    indices: Ref,
    sscore: Ref,
    logits_r: Ref,
    pick: Ref,
    wts: Ref,
    counts: Ref,
    starts: Ref,
    offsets: Ref,
    order: Ref,
    sorted: Ref,
    act: Ref,
    yp: Ref,
    yf: Ref, // fp32 [rows * topk, hidden]: down partials by (row, slot); one Mac: `buffers` lists the planes
    part: Ref, // fp32 [rows, hidden]: TP2's partial of a row-split projection
    sgu: Ref,
    sact: Ref,
    ys: Ref,
    gu: Ref,
    actd: Ref,
    m_emb: Ref,
    m_eh: Ref,
    m_x: Ref,
    m_xn: Ref,
    m_out: Ref,
    sums: Ref, // fp32 group sums of a matmul's rows (affine_mm's bias operand)
    sparse_nax: bool, // the sparse MLA attention on the tensor units (GLM_SPARSE_NAX=0: the row kernel)
    rows: u32, // a chunk's rows at most (the buffers' height)
    kda_rows: u32, // the rows a KDA block's planes hold (half a chunk)

    pub fn deinit(p: *Prompt) void {
        p.k.deinit();
    }
};

/// The chunk buffers of `rows`-row chunks for `cap`-token caches (selection passes read cap / 4 block scores a row).
pub fn init(gpa: std.mem.Allocator, arena: *st.Arena, device: mtl.Device, c: *const cfg.Config, decode: *const st.Scratch, engine_kernels: *const Kernels, cap: u32, rows: u32) !Prompt {
    var p: Prompt = undefined;
    p.k = try pk.load(gpa, device);
    errdefer p.k.deinit();
    p.mm = engine_kernels;
    try buffers(&p, arena, c, decode, cap, rows);
    return p;
}

/// The prompt's sparse MLA on the tensor units (GLM_SPARSE_NAX=0: the row kernel, its output beside its queries).
fn sparseNax() bool {
    return if (platform.c.getenv("GLM_SPARSE_NAX")) |v| v[0] != '0' else true;
}

/// The bytes `init` takes from the arena for `rows`-row chunks.
pub fn chunkBytes(gpa: std.mem.Allocator, c: *const cfg.Config, decode: *const st.Scratch, cap: u32, rows: u32) usize {
    var dry: st.Arena = .{ .device = undefined, .gpa = gpa, .dry = true };
    var p: Prompt = undefined;
    buffers(&p, &dry, c, decode, cap, rows) catch unreachable;
    return dry.bytes;
}

fn buffers(p: *Prompt, arena: *st.Arena, c: *const cfg.Config, decode: *const st.Scratch, cap: u32, rows: u32) !void {
    const R: usize = rows;
    const D: usize = c.hidden;
    const n: usize = R * c.topk;
    const blocks = (n + 255) / 256;
    const up = struct {
        fn to(v: usize) usize {
            return std.mem.alignForward(usize, v, 256);
        }
    }.to;
    p.rows = rows;
    p.kda_rows = @max(rows / 2, 1);
    p.sparse_nax = sparseNax();
    p.streams = decode.*;
    const big = struct {
        fn of(a: *st.Arena, bytes: usize) !Ref {
            return a.buffer(bytes);
        }
    };
    const xs = try big.of(arena, R * 4 * D * 2); // one buffer; expands write the streams over themselves (core/hc.zig)
    p.streams.x = .{ xs, xs };
    p.streams.normed = try big.of(arena, R * D * 2);
    p.streams.branch = try big.of(arena, R * D * 2);
    p.streams.post = try big.of(arena, R * 4 * 4);
    p.streams.comb = try big.of(arena, R * 16 * 4);
    p.streams.inv = try big.of(arena, R * 4);
    p.streams.mixes = try big.of(arena, R * hc.partBytes(.{ .width = @intCast(D), .sinkhorn = 0, .eps_e9 = 0 }));
    p.sscore = try big.of(arena, mla_mod.selectRows() * (@as(usize, cap) / c.kpool + 1) * 4);
    p.pick = try big.of(arena, n * 4);
    p.wts = try big.of(arena, n * 4);
    p.counts = try big.of(arena, blocks * c.experts * 4);
    p.starts = try big.of(arena, blocks * c.experts * 4);
    p.offsets = try big.of(arena, c.experts * 4);
    p.order = try big.of(arena, n * 4);
    p.sorted = try big.of(arena, n * 4);
    p.sums = try big.of(arena, @max(R * 16384, n * D) / 64 * 4); // widest dense K (MLA's out-projection) or a gather's
    // by layer three regions; proj: KDA projection, MLA queries then key lists then values, expert planes, gate+up
    const pitch = std.mem.alignForward(usize, c.kdaProj(), 64); // the matmul's padded pitch
    const qp = up(R * c.qrProj() * 2);
    const act = up(n * c.moe_inter * 2);
    const ys = up(R * D * 2);
    p.proj = try big.of(arena, @max(@max(R * pitch * 2, qp + R * c.keyWidth() * 4), @max(act + ys + R * c.experts * 4, R * 2 * c.dense_inter * 2)));
    p.qp = p.proj;
    p.indices = p.proj.at(qp);
    p.vals = p.proj; // written once the attention has read the queries and key lists
    p.sgu = p.proj; // done before the routed experts write their activations (one Mac) or after their down (by rows)
    p.sact = p.proj.at(up(R * 2 * c.moe_inter * 2));
    p.act = p.proj;
    p.ys = p.proj.at(act);
    p.logits_r = p.ys.at(ys);
    p.gu = p.proj;
    // y: KDA's output; MLA's input projection, normed queries and index weights
    const xp = up(R * std.mem.alignForward(usize, c.xProj(), 64) * 2);
    const qr = up(R * c.q_lora * 2);
    p.y = try big.of(arena, @max(R * c.kdaWidth() * 2, xp + qr + R * c.i_heads * 2));
    p.xp = p.y;
    p.qr = p.y.at(xp);
    p.iw = p.y.at(xp + qr);
    // yf: embeddings, KDA planes, MLA latent queries and attention, expert outputs, dense activations, final rows
    const plane = @as(usize, p.kda_rows) * c.kdaWidth() * 2;
    const lat = R * c.mla_heads * c.kv_lora * 2;
    const part = up(R * c.dense_inter * 2);
    const row = up(R * D * 2);
    const one = @max(@max(7 * plane + @as(usize, p.kda_rows) * c.kda_heads * 4, if (p.sparse_nax) lat else 2 * lat), @max(@max(n * D * 2, part + R * D * 4), 4 * row));
    p.yf = try big.of(arena, if (c.byRows()) @max(one, n * D * 4) else one);
    p.streams.h = p.yf;
    p.streams.raw = p.yf; // the final mean, long after the embedding rows were read
    p.streams.hidden = p.yf.at(row);
    p.ql = p.yf;
    p.att = if (p.sparse_nax) p.yf else p.yf.at(lat); // the tensor-unit kernel reads a row's queries before it writes over them
    p.yp = p.yf;
    p.actd = p.yf;
    p.part = p.yf.at(part);
    p.m_eh = p.yf.at(2 * row);
    p.m_emb = xs; // the MTP head's rows in the streams (the backbone is done)
    p.m_x = xs.at(row);
    p.m_xn = xs.at(2 * row);
    p.m_out = xs.at(3 * row);
}

/// The int32 parameter array the NAX and sort kernels read (16 entries, zero-padded).
fn params(e: mtl.ComputeEncoder, index: usize, values: anytype) void {
    var v: [16]i32 = @splat(0);
    inline for (values, 0..) |x, i| v[i] = @intCast(x);
    e.setBytes(std.mem.asBytes(&v), index);
}

/// MLX's threadgroup for a grid of threads: the group, but no larger than the grid.
fn run(e: mtl.ComputeEncoder, grid: [3]usize, group: [3]usize) void {
    e.dispatchThreads(size(grid[0], grid[1], grid[2]), size(@min(group[0], grid[0]), @min(group[1], grid[1]), @min(group[2], grid[2])));
}

/// Routed experts' gate, up and SwiGLU on the M rows of `x_in` its pairs read (expert order) into `act` ([n, gate.n]).
fn gateUp(p: *const Prompt, mm: affine_mm.Pipes, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32, n: u32) void {
    affine_mm.rowSums(e, mm, 64, x_in, p.sums, M, w.gate.k);
    affine_mm.gatherGlu(e, mm, x_in, p.sums, w.gate, w.up, p.offsets, p.order, p.act, n, x.c.experts, x.c.topk, x.c.swiglu_limit);
    affine_mm.rowSums(e, mm, 64, p.act, p.sums, n, w.down.k);
}

fn swiglu(x: *const fwd.Ctx, e: mtl.ComputeEncoder, gu: Ref, act: Ref, rows: u32, width: u32) void {
    e.setPipeline(x.k.swiglu);
    bind(e, 0, .{ gu, act });
    e.setValue(fwd.Rows{ .rows = @intCast(rows), .width = @intCast(width), .x_stride = @intCast(2 * width), .y_stride = @intCast(width), .eps = x.c.swiglu_limit }, 2);
    e.dispatchThreads(size(width, rows, 1), size(256, 1, 1));
}

fn add(x: *const fwd.Ctx, e: mtl.ComputeEncoder, a: Ref, b: Ref, out: Ref, n: u32) void {
    e.setPipeline(x.k.add);
    bind(e, 0, .{ a, b, out });
    e.setValue(n, 3);
    e.dispatchThreads(size(n, 1, 1), size(256, 1, 1));
}

/// The shared expert on M rows into `ys`.
fn shared(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32) void {
    mla_mod.qmm(p, e, x_in, w.sh_gate_up, p.sgu, M);
    swiglu(x, e, p.sgu, p.sact, M, x.c.moe_inter);
    mla_mod.qmm(p, e, p.sact, w.sh_down, p.ys, M);
}

/// MoE block on M rows: shared expert, route, experts by expert then by rows, this Mac's halves summed in slot order.
fn moe(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32, ep: ?*ep_mod.Ep) void {
    const c = x.c;
    const k = x.k;
    const D = c.hidden;
    const E = c.experts;
    const n = M * c.topk;
    const out = p.streams.branch;
    const s = x.skip; // classes a profile leaves out
    const Class = fwd.Class;
    if (ep) |t| {
        {
            var g = fwd.SpanGuard.begin(x, &e, .exchange);
            defer g.end(x, &e);
            if (s & Class.exchange == 0) t.begin();
        }
    } else {
        {
            var g = fwd.SpanGuard.begin(x, &e, .shared);
            defer g.end(x, &e);
            if (s & Class.shared == 0) shared(p, x, e, w, x_in, M);
        }
    }
    {
        var g = fwd.SpanGuard.begin(x, &e, .route);
        defer g.end(x, &e);
        if (s & Class.route == 0) route(p, x, e, w, x_in, M);
    }
    if (ep) |t| {
        {
            var g = fwd.SpanGuard.begin(x, &e, .routed);
            defer g.end(x, &e);
            if (s & Class.routed == 0) { // down's fp32 partials put back in (row, slot) order for the decode's combine
                gateUp(p, p.mm.mm_bf16, x, e, w, x_in, M, n);
                affine_mm.gatherTo(e, p.mm.mm_f32, p.act, p.sums, w.down, p.offsets, p.yf, p.order, n, E);
            }
        }
        {
            var g = fwd.SpanGuard.begin(x, &e, .exchange);
            defer g.end(x, &e);
            if (s & Class.exchange == 0) t.sendRows(e, p.yf, p.wts, M);
        }
        {
            var g = fwd.SpanGuard.begin(x, &e, .shared);
            defer g.end(x, &e);
            if (s & Class.shared == 0) shared(p, x, e, w, x_in, M);
        }
        {
            var g = fwd.SpanGuard.begin(x, &e, .combine);
            defer g.end(x, &e);
            if (s & Class.combine == 0) t.receiveRows(e, p.ys, out, M);
        }
        return;
    }
    const path = wide.routed(w.gate, w.up, w.down);
    {
        var gsp = fwd.SpanGuard.begin(x, &e, .routed);
        defer gsp.end(x, &e);
        if (s & Class.routed == 0 and path == .rows) { // each expert's sorted pairs at its width
            wide.bySorted(
                &p.mm.wide.?,
                e,
                .{ .w = w.gate, .up = w.up, .experts = E, .repeat = c.topk, .lim = c.swiglu_limit },
                x_in,
                p.offsets,
                p.order,
                p.act,
                c.moe_inter,
                n,
            );
            wide.bySorted(
                &p.mm.wide.?,
                e,
                .{ .w = w.down, .experts = E, .repeat = 1 },
                p.act,
                p.offsets,
                p.order,
                p.yp,
                D,
                n,
            );
        } else if (s & Class.routed == 0 and path == .gather) {
            const gg = wide.gather;
            gg.encode(e, p.mm.wide.?.gather[0], w.gate, w.up, .{ .x = x_in, .offsets = p.offsets, .order = p.order, .out = p.act }, n, E, c.topk, c.moe_inter, c.swiglu_limit);
            gg.encode(e, p.mm.wide.?.gather[1], w.down, null, .{ .x = p.act, .offsets = p.offsets, .order = p.order, .out = p.yp }, n, E, 1, D, 0);
        } else if (s & Class.routed == 0) { // down's rows put back in (row, slot) order for the combine
            const mm = if (path == .mm4) p.mm.mm_bf16 else p.mm.wide.?.mm2;
            gateUp(p, mm, x, e, w, x_in, M, n);
            affine_mm.gatherTo(e, mm, p.act, p.sums, w.down, p.offsets, p.yp, p.order, n, E);
        }
    }
    {
        var gsp = fwd.SpanGuard.begin(x, &e, .combine);
        defer gsp.end(x, &e);
        if (s & Class.combine != 0) return;
        e.setPipeline(k.moe_combine);
        bind(e, 0, .{ p.ys, p.yp, p.wts });
        fwd.shape(e, 3, .{ M, c.topk });
        bind(e, 4, .{out});
        e.dispatchThreads(size(M * D, 1, 1), size(256, 1, 1));
    }
}

/// The route on M rows: the decode's router and top-k, pairs sorted by expert (`order`: each sorted pair's index).
fn route(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32) void {
    const c = x.c;
    const k = x.k;
    const E = c.experts;
    const n = M * c.topk;
    moe_route.logits(e, k.route_logits, kernels.route_shape, x_in, w.router, p.logits_r, M); // decode's bits, any rows
    e.setPipeline(k.route_rows);
    bind(e, 0, .{ p.logits_r, w.bias });
    e.setValue(c.routed_scale, 2);
    bind(e, 3, .{ p.pick, p.wts });
    e.setValue(M, 5);
    e.dispatchThreads(size(32 * M, 1, 1), size(256, 1, 1));
    const blocks = (n + 255) / 256;
    e.setPipeline(p.k.get("custom_kernel_tf_sort_count_uint32_t_int32_t_int32_t"));
    bind(e, 0, .{p.pick});
    params(e, 1, .{ n, E });
    bind(e, 2, .{p.counts});
    run(e, .{ blocks * 256, 1, 1 }, .{ 256, 1, 1 });
    e.setPipeline(p.k.get("custom_kernel_tf_sort_starts_int32_t_int32_t_int32_t_int32_t"));
    bind(e, 0, .{p.counts});
    params(e, 1, .{ n, E });
    bind(e, 2, .{ p.starts, p.offsets });
    run(e, .{ E, 1, 1 }, .{ E, 1, 1 });
    e.setPipeline(p.k.get("custom_kernel_tf_sort_place_uint32_t_int32_t_int32_t_uint32_t_uint32_t"));
    bind(e, 0, .{ p.pick, p.starts });
    params(e, 2, .{ n, E });
    bind(e, 3, .{ p.order, p.sorted });
    run(e, .{ blocks * 256, 1, 1 }, .{ 256, 1, 1 });
}

/// KDA layer `ki` on a chunk (glm_kda_prompt.metal) in `kda_rows` blocks: f_b/g_b on tensor units, prep, scan, norm.
fn kdaChunk(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, ki: usize, w: *const wts.Kda, M: u32) void {
    const c = x.c;
    const k = x.k;
    const L = &x.s.kda[ki];
    const cur = L.cur;
    const H = c.kda_heads;
    const W = c.kdaWidth();
    const tp = c.tp > 1; // TP2: this Mac's heads
    const pitch: u32 = std.mem.alignForward(u32, c.kdaProj(), 64);
    const fa = 3 * W; // the projection row: q, k, v, then f_a, g_a and beta
    var r0: u32 = 0;
    while (r0 < M) : (r0 += p.kda_rows) {
        const m = @min(p.kda_rows, M - r0);
        const plane = @as(usize, m) * W * 2; // [m, heads * dim] bf16
        const q = p.yf;
        const kk = q.at(plane);
        const v = kk.at(plane);
        const sy = v.at(plane);
        const g = sy.at(plane); // fp32
        const gate = g.at(2 * plane);
        const beta = gate.at(plane);
        if (fwd.on(x, "kda_pre")) {
            const at = @as(usize, r0) * pitch * 2; // the block's projection rows
            mla_mod.qmmAt(p, e, p.proj.at(at + @as(usize, fa) * 2), pitch, w.f_b, sy, m); // f_b rows into sy's plane, read before scan writes
            mla_mod.qmmAt(p, e, p.proj.at(at + @as(usize, fa + c.kda_dim) * 2), pitch, w.g_b, gate, m);
            e.setPipeline(if (tp) k.kda_prep_tp else k.kda_prep);
            bind(e, 0, .{p.proj});
            fwd.shape(e, 1, .{ M, pitch, r0 });
            bind(e, 2, .{ L.cs[cur], w.conv_w, sy, w.a, w.dt_bias });
            e.setValue(c.lower_bound, 7);
            bind(e, 8, .{ q, kk, v, g, beta });
            e.dispatchGroups(size(m, H / 4, 1), size(128, 1, 1));
        }
        if (fwd.on(x, "kda_scan")) {
            e.setPipeline(if (tp) k.kda_scan_tp else k.kda_scan);
            bind(e, 0, .{ q, kk, v, g, beta, L.st[if (r0 == 0) cur else 1 - cur], L.st[1 - cur], sy });
            e.setValue(@as(i32, @intCast(m)), 8);
            e.dispatchGroups(size(c.kda_dim / 64, H, 1), size(512, 1, 1));
        }
        if (!fwd.on(x, "kda_post")) continue;
        e.setPipeline(if (tp) k.kda_post_tp else k.kda_post);
        bind(e, 0, .{ sy, gate, w.o_norm });
        e.setValue(c.eps, 3);
        bind(e, 4, .{ p.y.at(@as(usize, r0) * W * 2), p.proj });
        fwd.shape(e, 6, .{ M, pitch }); // every block writes the next chunk's conv window, the same rows
        bind(e, 7, .{ L.cs[cur], L.cs[1 - cur] });
        e.dispatchGroups(size(m, H, 1), size(32, 1, 1));
    }
}

/// The backbone over a chunk (tokens in `ids`) at positions pos..: final-normed rows into `streams.hidden`.
pub fn backbone(p: *const Prompt, x: *fwd.Ctx, e_in: mtl.ComputeEncoder, ids: Ref, M: u32, pos: u32) void {
    var e = e_in;
    const c = x.c;
    const ss = &p.streams;
    std.debug.assert(x.sc == ss and M <= p.rows);
    const s = x.skip; // classes a profile leaves out
    const Class = fwd.Class;
    {
        var g = fwd.SpanGuard.begin(x, &e, .ends);
        defer g.end(x, &e);
        if (s & Class.ends == 0) fwd.embed(x, e, ids, M, pos);
    }
    const plane = @as(usize, M) * c.hidden * 2; // a trace's capture points: the decode backbone's
    fwd.snap(x, e, ss.h, plane);
    var pending = false;
    var ki: usize = 0;
    var mi: usize = 0;
    for (0..c.run) |li| {
        const L = &x.w.layers[li];
        const hcs = L.hc.?;
        {
            var g = fwd.SpanGuard.begin(x, &e, .hc);
            defer g.end(x, &e);
            if (s & Class.hc == 0) fwd.boundary(x, e, M, pending, hcs[0], L.in_norm);
        }
        fwd.snap(x, e, ss.normed, plane);
        switch (L.attn) {
            .kda => |*a| {
                // B4 split: kda_proj = in_proj + o_proj GEMMs, kda_attn = the recurrence, `kda` skips all.
                const do_proj = s & Class.kda == 0 and s & Class.kda_proj == 0;
                const do_attn = s & Class.kda == 0 and s & Class.kda_attn == 0;
                {
                    var g = fwd.SpanGuard.begin(x, &e, .kda);
                    defer g.end(x, &e);
                    if (do_proj or do_attn) {
                        if (do_proj and fwd.on(x, "kda_in")) mla_mod.qmm(p, e, ss.normed, a.in_proj, p.proj, M);
                        if (do_attn) kdaChunk(p, x, e, ki, a, M);
                        if (do_proj) {
                            if (c.tp > 1) { // TP2: this Mac's heads' out-projection partial, summed with the peer's
                                mla_mod.qmmF32(p, e, p.y, a.o_proj, p.part, M);
                                x.ep.?.reduce(e, p.part, ss.branch, M);
                            } else if (fwd.on(x, "kda_out")) mla_mod.qmm(p, e, p.y, a.o_proj, ss.branch, M);
                        }
                    }
                }
                ki += 1;
            },
            .mla => |*a| {
                // B4: mla_proj = x_proj/qr_proj/kv_b/o_proj, mla_attn = cache writes, selectKeys, scores; `mla` skips.
                const do_proj = s & Class.mla == 0 and s & Class.mla_proj == 0;
                const do_attn = s & Class.mla == 0 and s & Class.mla_attn == 0;
                if (do_proj or do_attn) {
                    const mode: mla_mod.MlaMode = if (do_proj and do_attn) .full else if (do_proj) .proj else .attn;
                    if (mode == .full) {
                        var g = fwd.SpanGuard.begin(x, &e, .mla);
                        defer g.end(x, &e);
                    }
                    mla_mod.mla(p, x, e, mi, a, ss.normed, M, pos, mode);
                }
                mi += 1;
            },
        }
        fwd.snap(x, e, ss.branch, plane);
        {
            var g = fwd.SpanGuard.begin(x, &e, .hc);
            defer g.end(x, &e);
            if (s & Class.hc == 0) fwd.boundary(x, e, M, true, hcs[1], L.post_norm);
        }
        fwd.snap(x, e, ss.normed, plane);
        switch (L.mlp) {
            .dense => |*d| {
                var g = fwd.SpanGuard.begin(x, &e, .dense);
                defer g.end(x, &e);
                if (s & Class.dense == 0) {
                    mla_mod.qmm(p, e, ss.normed, d.gate_up, p.gu, M);
                    swiglu(x, e, p.gu, p.actd, M, c.dense_inter);
                    if (c.tp > 1) { // TP2: this Mac's half, summed with the peer's
                        mla_mod.qmmF32(p, e, p.actd, d.down, p.part, M);
                        x.ep.?.reduce(e, p.part, ss.branch, M);
                    } else mla_mod.qmm(p, e, p.actd, d.down, ss.branch, M);
                }
            },
            .moe => |*m| moe(p, x, e, m, ss.normed, M, if (c.byRows()) x.ep else null),
        }
        fwd.snap(x, e, ss.branch, plane);
        pending = true;
    }
    {
        var g = fwd.SpanGuard.begin(x, &e, .hc);
        defer g.end(x, &e);
        if (s & Class.hc == 0) fwd.boundary(x, e, M, true, null, null);
    }
    {
        var g = fwd.SpanGuard.begin(x, &e, .ends);
        defer g.end(x, &e);
        if (s & Class.ends != 0) return;
        e.setPipeline(x.k.stream_mean);
        bind(e, 0, .{ ss.x[x.xi], ss.raw });
        e.setValue([2]u32{ c.hidden, M }, 2);
        e.dispatchThreads(size(c.hidden, M, 1), size(256, 1, 1));
        fwd.rms(x, e, ss.raw, x.w.norm, ss.hidden, M, c.hidden, c.hidden, c.hidden, c.eps);
    }
    fwd.snap(x, e, ss.hidden, plane);
}

/// The MTP head over M prompt rows at head positions pos..: cache entries only (no one reads the block's output).
pub fn mtp(p: *const Prompt, x: *const fwd.Ctx, e_in: mtl.ComputeEncoder, h: Ref, next: Ref, M: u32, pos: u32) void {
    var e = e_in;
    var g = fwd.SpanGuard.begin(x, &e, .mtp);
    defer g.end(x, &e);
    const c = x.c;
    const D = c.hidden;
    const L = &x.w.layers[c.layers];
    const m = x.w.mtp.?;
    fwd.embedRows(x, e, next, p.m_emb, M);
    fwd.inject(x, e, p.m_emb, M, pos + 1); // the head reads each row's next token: an image's rows one place on
    fwd.rms(x, e, p.m_emb, m.enorm, p.m_eh, M, D, D, 2 * D, c.eps);
    fwd.rms(x, e, h, m.hnorm, p.m_eh.at(@as(usize, D) * 2), M, D, D, 2 * D, c.eps);
    // GLM_EHPROJ_NAX=0 = scalar wide path (diagnostic); default tensor-unit kernel matches it, 2.6-2.7x faster.
    const eh_nax = if (std.c.getenv("GLM_EHPROJ_NAX")) |v| v[0] != '0' else true;
    if (m.eh_bf16) |eh| {
        if (eh_nax) wide.bf16Nax(&p.mm.wide.?, e, eh, D, 2 * D, .{ .in = p.m_eh, .ldx = 2 * D, .out = p.m_x, .ldo = D, .rows = M }) else wide.bf16(&p.mm.wide.?, e, eh, D, 2 * D, .{ .in = p.m_eh, .ldx = 2 * D, .out = p.m_x, .ldo = D, .rows = M });
    } else mla_mod.qmm(p, e, p.m_eh, m.eh_proj, p.m_x, M);
    fwd.rms(x, e, p.m_x, L.in_norm, p.m_xn, M, D, D, D, c.eps);
    mla_mod.qmm(p, e, p.m_xn, L.attn.mla.x_proj, p.xp, M);
    fwd.mlaCache(x, e, c.countKind(.mla), &L.attn.mla, p.m_xn, p.xp, @intCast(std.mem.alignForward(usize, c.xProj(), 64)), p.iw, M, pos);
}
