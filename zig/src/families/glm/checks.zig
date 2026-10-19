//! GLM-5.3-Flash's dev-time checks on a loaded engine: captures, traces, teacher forcing and knock-out profiles.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const mtp = @import("mtp.zig");
const prompt_mod = @import("prompt.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const Engine = @import("engine.zig").Engine;
const counters_class = @import("counters_class.zig");

/// The first window of `prompt`: each sublayer's input and output, then the logits, raw bf16 in glm_ref.py's order.
pub fn capture(e: *Engine, prompt: []const u32, path: []const u8) !void {
    const c = &e.c;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const n: u32 = @intCast(@min(prompt.len, st.max_rows));
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, n), prompt[0..n]);
    const plane = @as(usize, n) * c.hidden * 2;
    const bytes = plane * (2 + 4 * @as(usize, c.run)) + @as(usize, n) * c.vocab * 2;
    const dump = try e.arena.buffer(bytes);
    var x = e.ctx();
    x.dump = dump;
    const b = e.begin();
    fwd.backbone(&x, b.enc, e.prompt_ids, n, 0);
    fwd.head(&x, b.enc, e.sc.hidden, dump.at(x.dump_at), e.sc.picks, n);
    try e.finish(b.cb, b.enc);
    fwd.flipKda(&x);
    const file = platform.c.fopen(try std.fmt.allocPrintSentinel(e.gpa, "{s}", .{path}, 0), "wb") orelse return error.OpenFailed;
    defer _ = platform.c.fclose(file);
    if (platform.c.fwrite(dump.addr(), 1, bytes, file) != bytes) return error.WriteFailed;
    std.debug.print("captured {d} rows ({d} bytes) in {s}\n", .{ n, bytes, path });
}

/// Each call of a plain reply (prompt windows, then `steps` rows of `reply`), sublayer by sublayer: glm_ref.py --trace's twin.
pub fn trace(e: *Engine, prompt: []const u32, reply: []const u32, steps: usize, path: []const u8) !void {
    const c = &e.c;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const P: u32 = @intCast(prompt.len);
    const total: u32 = P + @as(u32, @intCast(@min(steps, reply.len)));
    if (total + 1 > e.s.cap) return error.ContextFull;
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, P), prompt);
    @memcpy(Engine.u32s(e.prompt_ids.at(@as(usize, P) * 4), total - P), reply[0 .. total - P]);
    const most = @as(usize, st.max_rows) * c.hidden * 2 * (2 + 4 * @as(usize, c.run)) + @as(usize, st.max_rows) * c.vocab * 2;
    const dump = try e.arena.buffer(most);
    const file = platform.c.fopen(try std.fmt.allocPrintSentinel(e.gpa, "{s}", .{path}, 0), "wb") orelse return error.OpenFailed;
    defer _ = platform.c.fclose(file);
    var x = e.ctx();
    var at: u32 = 0;
    var calls: usize = 0;
    while (at < total) : (calls += 1) {
        const n: u32 = if (at < P) @min(st.max_rows, P - at) else 1;
        x.dump = dump;
        x.dump_at = 0;
        const b = e.begin();
        fwd.backbone(&x, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
        fwd.head(&x, b.enc, e.sc.hidden, dump.at(x.dump_at), e.sc.picks, n);
        try e.finish(b.cb, b.enc);
        fwd.flipKda(&x);
        const bytes = x.dump_at + @as(usize, n) * c.vocab * 2;
        if (platform.c.fwrite(dump.addr(), 1, bytes, file) != bytes) return error.WriteFailed;
        at += n;
    }
    std.debug.print("traced {d} calls ({d} prompt rows, {d} steps) in {s}\n", .{ calls, P, total - P, path });
}

/// Teacher-forced agreement: the greedy picks with `reply`'s tokens fed back; how many equal them, the first that differs.
pub fn forced(e: *Engine, prompt: []const u32, reply: []const u32) !struct { same: usize, first: ?usize } {
    const c = &e.c;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const P: u32 = @intCast(prompt.len);
    if (reply.len == 0) return .{ .same = 0, .first = null };
    const total: u32 = P + @as(u32, @intCast(reply.len - 1));
    if (total + 1 > e.s.cap) return error.ContextFull;
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, P), prompt);
    @memcpy(Engine.u32s(e.prompt_ids.at(@as(usize, P) * 4), total - P), reply[0 .. total - P]);
    var x = e.ctx();
    var at: u32 = 0;
    var same: usize = 0;
    var first: ?usize = null;
    while (at < total) {
        const n: u32 = if (at < P) @min(st.max_rows, P - at) else 1;
        const b = e.begin();
        fwd.backbone(&x, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
        fwd.head(&x, b.enc, e.sc.hidden.at(@as(usize, n - 1) * c.hidden * 2), e.sc.logits, e.sc.picks, 1);
        try e.finish(b.cb, b.enc);
        fwd.flipKda(&x);
        at += n;
        if (at >= P) { // this call's last row predicts reply[at - P]
            const i = at - P;
            if (Engine.u32s(e.sc.picks, 1)[0] == reply[i]) same += 1 else if (first == null) first = i;
        }
    }
    return .{ .same = same, .first = first };
}

/// Knock-out profile: a `depth`-draft round replayed `reps` times with each launch class left out in turn, by median GPU time.
pub fn profile(e: *Engine, depth: u32, reps: usize, only: bool, parts: bool) !void {
    const c = &e.c;
    const D = c.hidden;
    const d: u32 = if (e.w.mtp == null) 0 else @min(depth, st.max_rows - 1);
    const wide = if (platform.c.getenv("GLM_PROFILE_ROWS")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1 else 1; // a copy window's rows, without the MTP head
    const R = if (d == 0) std.math.clamp(wide, 1, st.max_rows) else d + 1;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (e.s.pos + R + 1 > e.s.cap) return error.ContextFull;
    e.sync();
    const ids = Engine.u32s(e.sc.ids, R);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + 37 * i);
    const n_class = if (parts) fwd.Part.names.len else fwd.Class.names.len;
    var masks_buf: [34]u32 = undefined;
    masks_buf[0] = 0;
    for (0..n_class) |i| masks_buf[i + 1] = @as(u32, 1) << @intCast(i);
    masks_buf[n_class + 1] = 0xffff_ffff; // every class left out: what remains
    masks_buf[n_class + 2] = 0;
    const masks = masks_buf[0 .. n_class + 3];
    const times = try e.gpa.alloc(f64, reps);
    defer e.gpa.free(times);
    var full: f64 = 0;
    const all: u32 = (@as(u32, 1) << @intCast(n_class)) - 1;
    for (masks, 0..) |mask, mi| {
        var x = e.ctx();
        const m = if (mask == 0xffff_ffff) all else if (only and mask != 0) all & ~mask else mask;
        if (parts) x.pskip = m else x.skip = m;
        for (0..reps + 1) |rep| {
            const b = e.begin();
            if (d > 0 and x.skip & fwd.Class.mtp == 0) {
                mtp.run(&x, b.enc, e.sc.hidden, e.sc.picks, R, e.s.mtp_pos, e.sc.ids.at(4));
                for (1..d) |j| mtp.chain(&x, b.enc, if (j == 1) e.sc.m_x.at(@as(usize, R - 1) * D * 2) else e.sc.m_x, e.sc.ids.at(j * 4), e.s.mtp_pos + R + @as(u32, @intCast(j)) - 1, e.sc.ids.at((j + 1) * 4));
            }
            fwd.backbone(&x, b.enc, e.sc.ids, R, e.s.pos);
            fwd.head(&x, b.enc, e.sc.hidden, e.sc.logits, e.sc.picks, R);
            try e.finish(b.cb, b.enc);
            if (rep > 0) times[rep - 1] = (e.gpu[1] - e.gpu[0]) * 1e3; // the first run warms the class's state
        }
        std.mem.sort(f64, times, {}, std.sort.asc(f64));
        const med = times[reps / 2];
        if (mi == 0) full = med;
        const name = if (mask == 0) (if (mi == 0) "full" else "full again") else if (mask == 0xffff_ffff) "none" else if (parts) fwd.Part.names[@ctz(mask)] else fwd.Class.names[@ctz(mask)];
        std.debug.print("profile {d} rows{s}: {s:<10} {d:7.3} ms (min {d:.3}, max {d:.3}){s}", .{ R, if (only) " only" else "", name, med, times[0], times[reps - 1], if (mask == 0 or only or mask == 0xffff_ffff) "\n" else "" });
        if (mask != 0 and !only and mask != 0xffff_ffff) std.debug.print("  class {d:6.3} ms {d:5.1}%\n", .{ full - med, 100 * (full - med) / full });
    }
}

/// The chunk path's matmul against the decode's row kernel on layer 0's real KDA in-projection input: the share that differs.
pub fn checkMatmul(e: *Engine, prompt: []const u32) !void {
    const pr = &(e.pr orelse return error.NoPromptPath);
    const rows: u32 = @intCast(@min(prompt.len, st.max_rows));
    const L = &e.w.layers[0];
    const a = switch (L.attn) {
        .kda => |*a| a,
        .mla => return error.NotKda,
    };
    if (!@import("wide.zig").native(a.in_proj)) return error.UnsupportedWidth; // the 4/64 product's comparison only
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, rows), prompt[0..rows]);
    const N = a.in_proj.n;
    const out = try e.arena.buffer(@as(usize, rows) * N * 2);
    var x = e.ctx();
    const b = e.begin();
    fwd.embed(&x, b.enc, e.prompt_ids, rows, 0);
    fwd.boundary(&x, b.enc, rows, false, L.hc.?[0], L.in_norm);
    fwd.qmv(&x, b.enc, e.k.qmv_kda_in, e.sc.normed, a.in_proj, out, rows);
    affine_mm.rowSums(b.enc, e.k.mm_bf16, 64, e.sc.normed, pr.sums, rows, a.in_proj.k);
    affine_mm.dense(b.enc, e.k.mm_bf16, e.sc.normed, pr.sums, a.in_proj, pr.proj, rows);
    try e.finish(b.cb, b.enc);
    const want: [*]const u16 = @ptrCast(@alignCast(out.addr()));
    const got: [*]const u16 = @ptrCast(@alignCast(pr.proj.addr()));
    var differ: usize = 0;
    var worst: u32 = 0;
    for (0..@as(usize, rows) * N) |i| if (want[i] != got[i]) {
        differ += 1;
        const d = @as(i32, @as(i16, @bitCast(want[i]))) - @as(i32, @as(i16, @bitCast(got[i])));
        worst = @max(worst, @abs(d));
    };
    std.debug.print("matmul check (layer 0 KDA in-projection, {d} rows x {d}): {d} of {d} bf16 outputs differ ({d:.3}%), at most {d} steps\n", .{ rows, N, differ, @as(usize, rows) * N, 100 * @as(f64, @floatFromInt(differ)) / @as(f64, @floatFromInt(@as(usize, rows) * N)), worst });
}

/// One reply's per-round counter-span rollup (GLM_COUNTER_SPANS=1): total wall, sum of per-class spans, BUBBLES = wall − Σspan − encoding.
/// Prints the report inline; the table itself lives on the engine (`e.span_table`).
pub fn counterSpans(e: *Engine) !void {
    const table = if (e.span_table) |t| t else return;
    var sum_per_class: [counters_class.Kind.count]f64 = @splat(0);
    var sum_calls: [counters_class.Kind.count]u64 = @splat(0);
    var sum_span_count: usize = 0;
    var sum_wall: f64 = 0;
    var sum_encode: f64 = 0;
    var sum_classes: f64 = 0;
    var n_rows: usize = 0;
    for (table.items) |rep| {
        sum_wall += rep.wall_seconds;
        sum_encode += rep.encode_seconds;
        sum_span_count += rep.span_count;
        inline for (0..counters_class.Kind.count) |i| {
            sum_per_class[i] += rep.per_class_seconds[i];
            sum_calls[i] += rep.per_class_calls[i];
        }
        n_rows += 1;
    }
    inline for (0..counters_class.Kind.count) |i| {
        if (i != @backingInt(counters_class.Kind._round)) sum_classes += sum_per_class[i];
    }
    const bubbles = (sum_wall - sum_classes - sum_encode) * 1e3;
    std.debug.print("counterSpans: {d} reps, {d} spans total, Σclass={d:.3} ms, Σround span={d:.3} ms, Σwall={d:.3} ms, Σencode={d:.3} ms, BUBBLES={d:.3} ms\n", .{
        n_rows,
        sum_span_count,
        sum_classes * 1e3,
        sum_per_class[@backingInt(counters_class.Kind._round)] * 1e3,
        sum_wall * 1e3,
        sum_encode * 1e3,
        bubbles,
    });
    inline for (0..counters_class.Kind.count) |i| {
        std.debug.print("  {s:<10} {d:.3} ms (calls={d})\n", .{ counters_class.Kind.names[i], sum_per_class[i] * 1e3, sum_calls[i] });
    }
}

/// Run `reps` rounds at `rows` × `depth` and record per-class spans. Each round = one rep in the span table.
/// Mirrors the GLM_PROFILE shaping but uses GLM_COUNTER_SPANS data; emits a per-rep table inline.
/// Per rep: a whole-round span (Kind._round) wraps mtp+backbone+head; wall is the CPU time begin→wait-done,
/// encoding the CPU time begin→commit (engine.committed), Σspan the 12 class spans' GPU sum.
/// BUBBLES = wall − Σspan − encoding (STALL-CENSUS-PLAN formula); gpu ms and gpu−Σspan print beside as the
/// GPU-timeline cross-check (the synchronous drive's GPU idles only between spans).
pub fn counterSpanDrive(e: *Engine, rows: u32, depth: u32, reps: usize) !void {
    const c = &e.c;
    const D = c.hidden;
    const d: u32 = if (e.w.mtp == null) 0 else @min(depth, st.max_rows - 1);
    const wide = if (std.c.getenv("GLM_PROFILE_ROWS")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch rows else rows;
    const R: u32 = if (d == 0) std.math.clamp(wide, 1, st.max_rows) else d + 1;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (e.s.pos + R + 1 > e.s.cap) return error.ContextFull;
    e.sync();
    const ids = Engine.u32s(e.sc.ids, R);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + 37 * i);
    const start = e.span_table.?.items.len;
    var x = e.ctx();
    for (0..reps + 1) |rep| {
        const t_begin = std.c.mach_absolute_time();
        const b = e.begin();
        var enc = b.enc;
        var round_guard = fwd.SpanGuard.begin(&x, &enc, ._round);
        if (d > 0 and x.skip & fwd.Class.mtp == 0) {
            mtp.run(&x, enc, e.sc.hidden, e.sc.picks, R, e.s.mtp_pos, e.sc.ids.at(4));
            for (1..d) |j| mtp.chain(&x, enc, if (j == 1) e.sc.m_x.at(@as(usize, R - 1) * D * 2) else e.sc.m_x, e.sc.ids.at(j * 4), e.s.mtp_pos + R + @as(u32, @intCast(j)) - 1, e.sc.ids.at((j + 1) * 4));
        }
        fwd.backbone(&x, enc, e.sc.ids, R, e.s.pos);
        fwd.head(&x, enc, e.sc.hidden, e.sc.logits, e.sc.picks, R);
        round_guard.end(&x, &enc);
        try e.finish(b.cb, enc);
        const t_end = std.c.mach_absolute_time();
        // Skip the first rep's classes (state-warmup); the rest feed the table.
        if (rep == 0) {
            e.span_table.?.items.len = start; // discard the warmup's data
            continue;
        }
        // Fill this rep's wall/encoding (the engine's read path only has the GPU numbers).
        const last = &e.span_table.?.items[e.span_table.?.items.len - 1];
        last.wall_seconds = @as(f64, @floatFromInt(t_end - t_begin)) / 24e6;
        last.encode_seconds = @as(f64, @floatFromInt(e.committed - t_begin)) / 24e6;
    }
    // Print a per-rep table: 12 class columns, whole-round span, wall, encoding, Σspan, BUBBLES (both forms).
    const tbl = e.span_table.?;
    const end = tbl.items.len;
    std.debug.print("counterSpanDrive {d} reps at {d} rows × depth {d}:\n", .{ end - start, R, d });
    inline for (0..counters_class.Kind.count) |i| std.debug.print("  {s:<11} ", .{counters_class.Kind.names[i]});
    std.debug.print("  round ms     wall ms    enc ms     Σspan ms   BUBBLES ms  gpu ms     gpu-Σspan ms  spans\n", .{});
    for (tbl.items[start..end], 0..) |rep, ri| {
        var sum_classes: f64 = 0;
        inline for (0..counters_class.Kind.count) |i| {
            if (i != @backingInt(counters_class.Kind._round)) sum_classes += rep.per_class_seconds[i];
            std.debug.print("  {d:7.3}     ", .{rep.per_class_seconds[i] * 1e3});
        }
        const bubbles = (rep.wall_seconds - sum_classes - rep.encode_seconds) * 1e3;
        const gpu_bubbles = (rep.gpu_seconds_whole - sum_classes) * 1e3;
        std.debug.print("  {d:8.3}     {d:7.3}    {d:7.3}    {d:7.3}    {d:7.3}     {d:7.3}   {d:7.3}      {d:>5}\n", .{
            rep.per_class_seconds[@backingInt(counters_class.Kind._round)] * 1e3,
            rep.wall_seconds * 1e3,
            rep.encode_seconds * 1e3,
            sum_classes * 1e3,
            bubbles,
            rep.gpu_seconds_whole * 1e3,
            gpu_bubbles,
            rep.span_count,
        });
        _ = ri;
    }
}

/// A prompt chunk's GPU time by class (median of `reps`): whole, then each class left out (alone with `only`); a pair runs it together.
/// B4 enhancement: the class list now separates `mla_proj`/`mla_attn` and `kda_proj`/`kda_attn`
/// so the profile can isolate attention-MATH (scores/softmax/values kernels, KDA recurrence) from
/// PROJECTIONS (x_proj/qr_proj/kv_b absorb/unabsorb/o_proj, KDA in/out GEMMs). The meta-classes
/// `mla` and `kda` are retained so the legacy grouping still works.
pub fn profilePrompt(e: *Engine, prompt: []const u32, reps: usize, only: bool, parts: bool) !void {
    profilePromptAt(e, prompt, reps, only, parts, 0, e.pr.?.rows);
}

/// A prompt chunk's GPU time by class at position `pos` (rows `M`). When `pos == 0` this is the
/// first chunk (vanilla profilePrompt); for `pos >= 100000` the prompt path warms state to `pos`
/// before timing, so the profiled chunk sees the late-context cache shape.
/// Default M = e.pr.?.rows (engine's chunk size).
pub fn profilePromptAt(e: *Engine, prompt: []const u32, reps: usize, only: bool, parts: bool, pos: u32, M_arg: u32) !void {
    const pr = &(e.pr orelse return error.NoPromptPath);
    const C = fwd.Class;
    const M: u32 = if (M_arg == 0) pr.rows else M_arg;
    if (M + 1 > e.s.cap) return error.ContextFull;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.sync();
    @memcpy(Engine.u32s(e.prompt_ids, prompt.len), prompt[0..prompt.len]);
    const total: u32 = @intCast(prompt.len);
    // B4 late-chunk warmup: when pos > 0, fill state from at=0 up to at=pos in chunk_rows steps
    // (the engine's normal prefill path), then profile the chunk at `pos` itself.
    if (pos > 0) {
        var at: u32 = 0;
        while (at < pos and at < total) {
            const n: u32 = @min(@as(u32, pr.rows), pos - at, total - at);
            var wx = e.ctx();
            wx.sc = &pr.streams;
            const wb = e.begin();
            prompt_mod.backbone(pr, &wx, wb.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
            try e.finish(wb.cb, wb.enc);
            fwd.flipKda(&wx);
            at += n;
        }
    }
    const P = fwd.Part;
    // B4: 14 profile classes (the original 10 + 4 finer splits). Trailing zeros pad to 16 so
    // the (classes.len + 3) mask walk below reaches the all-set and full-again sentinels.
    const classes = if (parts) [_]u32{ P.bit("mla_proj"), P.bit("mla_cache"), P.bit("mla_absorb"), P.bit("mla_select"), P.bit("mla_attn"), P.bit("mla_unabs"), P.bit("mla_out"), P.bit("kda_in"), P.bit("kda_pre"), P.bit("kda_scan"), P.bit("kda_post"), P.bit("kda_out"), 0, 0, 0, 0 } else [_]u32{ C.hc, C.kda, C.mla, C.dense, C.route, C.routed, C.shared, C.exchange, C.combine, C.ends, C.mla_proj, C.mla_attn, C.kda_proj, C.kda_attn, 0, 0 };
    var all: u32 = 0;
    for (classes) |m| {
        if (m != 0) all |= m;
    }
    const times = try e.gpa.alloc(f64, reps);
    defer e.gpa.free(times);
    var full: f64 = 0;
    for (0..classes.len + 3) |mi| {
        const mask: u32 = if (mi == 0 or mi == classes.len + 2) 0 else if (mi == classes.len + 1) all else classes[mi - 1];
        if (mask == 0 and mi > 0 and mi <= classes.len) continue;
        var x = e.ctx();
        x.sc = &pr.streams;
        const m = if (mask == all) all else if (only and mask != 0) all & ~mask else mask;
        if (parts) x.pskip = m else x.skip = m;
        for (0..reps + 1) |rep| {
            // B4: pos==0 resets state per rep, pos>0 replays the chunk cache-write [pos, pos+M); flipKda keeps the KDA buffers coherent.
            if (pos == 0) e.s.reset();
            const b = e.begin();
            prompt_mod.backbone(pr, &x, b.enc, e.prompt_ids.at(@as(usize, pos) * 4), M, pos);
            try e.finish(b.cb, b.enc);
            fwd.flipKda(&x);
            if (rep > 0) times[rep - 1] = (e.gpu[1] - e.gpu[0]) * 1e3;
        }
        std.mem.sort(f64, times, {}, std.sort.asc(f64));
        const med = times[reps / 2];
        if (mi == 0) full = med;
        const name = if (mask == 0) (if (mi == 0) "full" else "full again") else if (mask == all) "none" else if (parts) fwd.Part.names[@ctz(mask)] else fwd.Class.names[@ctz(mask)];
        std.debug.print("prompt profile {d} rows{s} @ pos {d}: {s:<10} {d:9.2} ms (min {d:.2}, max {d:.2}){s}", .{ M, if (only) " only" else "", pos, name, med, times[0], times[reps - 1], if (mask == 0 or only or mask == all) "\n" else "" });
        if (mask != 0 and !only and mask != all) std.debug.print("  class {d:8.2} ms {d:5.1}%{s}\n", .{ full - med, 100 * (full - med) / full, if (mask == (C.mla_proj | C.mla_attn) or mask == C.mla_proj or mask == C.mla_attn or mask == (C.kda_proj | C.kda_attn) or mask == C.kda_proj or mask == C.kda_attn) "  ← B4 split" else "" });
    }
}
