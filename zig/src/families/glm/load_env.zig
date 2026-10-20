//! GLM-5.3-Flash's load-time environment: the GLM_* reads that set the engine's knobs, and the warm-up they gate.
const std = @import("std");
const platform = @import("sys");
const cfg = @import("config.zig");
const st = @import("state.zig");
const mtl = @import("metal");
const fwd = @import("forward.zig");
const adaptive = @import("adaptive.zig");
const prompt_mod = @import("prompt.zig");
const ep_mod = @import("ep.zig");
const eng = @import("engine.zig");
const Engine = eng.Engine;

/// Every GLM_* flag read once at load onto the engine's fields (before any device work).
pub fn apply(e: *Engine) !void {
    e.fused_route = if (platform.c.getenv("GLM_ROUTE")) |v| v[0] != '0' else true;
    e.draft_vocab = if (platform.c.getenv("GLM_DRAFT_VOCAB")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0 else 0;
    e.cut = if (platform.c.getenv("GLM_CUTS")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0 else 0;
    e.copy_min = if (platform.c.getenv("GLM_COPY")) |v| std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0, 0, 8) else 0;
    e.rank_log = if (platform.c.getenv("GLM_RANKS")) |v| v[0] == '1' else false;
    e.draft_depth = if (platform.c.getenv("GLM_DRAFT_DEPTH")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0 else 0;
    e.adaptive_depth = if (platform.c.getenv("GLM_ADAPTIVE_DEPTH")) |v| v[0] == '1' else true;
    e.adaptive = adaptive.Config.fromEnv();
    e.sampling = try fwd.envSampling(platform.c.getenv);
    if (platform.c.getenv("GLM_CHUNK")) |v| e.chunk_rows = std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch prompt_mod.max_rows, st.max_rows + 1, prompt_mod.max_rows);
}

/// Model-trimming reads after the parsed config: GLM_LAYERS=N (first N layers); GLM_MTP=0 keeps the ~4 GB head on disk.
pub fn subset(e: *Engine) !void {
    if (platform.c.getenv("GLM_LAYERS")) |v| try cfg.subset(&e.c, std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.BadLayerCount);
    if (platform.c.getenv("GLM_MTP")) |v| if (v[0] == '0') { // no MTP head: no drafts, its ~4 GB left on disk
        e.c.mtp = 0;
    };
}

/// GLM_PROMPT=0: every prompt row in a 16-row decode window instead of the tensor units.
pub fn chunked() bool {
    return if (platform.c.getenv("GLM_PROMPT")) |v| v[0] != '0' else true;
}

/// GLM_DRY: stop the load once the plan is measured, before any weight is read.
pub fn dry() bool {
    return std.c.getenv("GLM_DRY") != null;
}

/// The peer Mac's link settings read from `sp` (a link settings file), and the config splits they imply.
pub fn link(e: *Engine, sp: []const u8) !ep_mod.Settings {
    const sf = try mtl.MappedFile.open(try e.ep_arena.allocator().dupeSentinel(u8, sp, 0));
    defer sf.deinit();
    const s = try ep_mod.readSettings(e.ep_arena.allocator(), sf.bytes[0..sf.size]);
    const by_rows = if (platform.c.getenv("GLM_EP_SPLIT")) |v| !std.mem.eql(u8, std.mem.span(v), "experts") else true;
    if (by_rows) try cfg.splitRows(&e.c, s.rank, 2) else try cfg.split(&e.c, s.rank, 2);
    const tp = if (platform.c.getenv("GLM_TP")) |v| v[0] != '0' else true; // TP2: each Mac half the KDA heads (GLM_TP=0: off)
    if (by_rows and tp) try cfg.splitHeads(&e.c, s.rank, 2);
    if (e.c.tp > 1) { // TP2's head halves: the whole vocabulary (its kernel's pitch); no host logits to rank
        e.draft_vocab = e.c.vocab;
        e.rank_log = false;
    }
    return s;
}

/// First prompt pays a one-time cost: every rank of a pair pays it here at the same step (GLM_NO_WARMUP: skip).
pub fn warmup(e: *Engine, cap: u32) !void {
    if (e.pr != null and cap >= 128 and platform.c.getenv("GLM_NO_WARMUP") == null) {
        var ids: [64]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
        var quiet: eng.Quiet = .{};
        _ = try e.generate(&ids, 2, &.{}, 0, .{ .ctx = &quiet, .prefilled = eng.Quiet.prefilled, .tokens = eng.Quiet.tokens, .cancelled = eng.Quiet.cancelled });
    }
}
