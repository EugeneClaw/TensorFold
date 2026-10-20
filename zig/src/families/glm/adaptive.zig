//! GLM-5.3-Flash's B3 adaptive-depth controller: the rolling accept rate that moves a reply's depth floor-to-ceiling.
const std = @import("std");
const platform = @import("sys");

/// B3: one depth switch in the rounds loop; the engine records each event for the harness to print.
pub const SwitchEvent = struct {
    round: u64,
    old_depth: u32,
    new_depth: u32,
    rolling_rate: f64,
    direction: []const u8, // "drop" or "restore"
};

/// The controller's tuning, read once at load (the engine's adaptive mode is its own on/off field).
pub const Config = struct {
    threshold: f64 = 0.65, // below this rolling accept rate, drop d3→d2
    high: f64 = 0.75, // above this rolling accept rate, restore d2→d3
    window: u32 = 10, // rolling window size in non-copy rounds
    min: u32 = 2, // floor depth (the d2 target)
    max: u32 = 3, // ceiling depth (the d3 starting point)
    /// Minimum rounds between switch decisions; the guard keeps the rate from flapping at the threshold.
    separation: u64 = 4,

    /// GLM_ADAPTIVE_THRESHOLD / HIGH / WINDOW / MIN / MAX; an unparseable value falls back to the default.
    pub fn fromEnv() Config {
        var c: Config = .{};
        if (platform.c.getenv("GLM_ADAPTIVE_THRESHOLD")) |v| c.threshold = std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.65;
        if (platform.c.getenv("GLM_ADAPTIVE_MAX")) |v| c.max = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 3;
        if (platform.c.getenv("GLM_ADAPTIVE_MIN")) |v| c.min = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2;
        if (platform.c.getenv("GLM_ADAPTIVE_HIGH")) |v| c.high = std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.75;
        if (platform.c.getenv("GLM_ADAPTIVE_WINDOW")) |v| c.window = std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch 10, 2, 64);
        return c;
    }

    /// The depth a reply opens at: the caller's depth bracketed by the floor and the ceiling.
    pub fn open(c: Config, on: bool, d: u32) u32 {
        if (!on or d == 0) return d; // depth 0 is the caller's no-drafting choice and stays 0
        var v = d;
        if (v < c.min) v = c.min;
        if (v > c.max) v = c.max;
        return v;
    }

    /// The depth a reply's cap check reserves: a mid-stream switch to the ceiling cannot trip ContextFull.
    pub fn cap(c: Config, on: bool, d: u32) u32 {
        return if (on) @max(d, c.max) else d;
    }
};

/// One reply's window: the ring buffer of recent non-copy-round accept rates, the accept tally and the switch log.
pub const Controller = struct {
    cfg: Config,
    buf: [64]f64 = @splat(0), // sized to the maximum window (64)
    idx: u32 = 0,
    len: u32 = 0,
    sum: f64 = 0,
    accepted: u64 = 0,
    drafted: u64 = 0,
    last_switch: u64 = 0,
    switches: u64 = 0,
    log: [32]SwitchEvent = @splat(.{ .round = 0, .old_depth = 0, .new_depth = 0, .rolling_rate = 0, .direction = "" }),
    log_len: u32 = 0,

    /// A fresh window for one reply; nothing carries over from the last one.
    pub fn init(cfg: Config) Controller {
        return .{ .cfg = cfg };
    }

    /// Record one non-copy round's accept rate at `round` and maybe switch depth; `d` is in force, updated in place.
    pub fn record(c: *Controller, on: bool, round: u64, keep: u32, d: *u32) void {
        if (!on or d.* == 0) return; // a copy round never reached here; depth 0 drafted nothing to measure
        c.accepted += keep - 1;
        c.drafted += d.*;
        const rate: f64 = @as(f64, @floatFromInt(keep - 1)) / @as(f64, @floatFromInt(d.*));
        const win = @as(u32, c.cfg.window);
        if (c.len < win) {
            c.buf[c.len] = rate;
            c.len += 1;
            c.sum += rate;
        } else {
            const old = c.buf[c.idx];
            c.buf[c.idx] = rate;
            c.idx = @intCast((c.idx + 1) % win);
            c.sum += rate - old;
        }
        // Decide only on a full window; partial windows read the warm-up transient and mis-price content.
        const decide_at: u32 = win;
        if (c.len < decide_at or round - c.last_switch < c.cfg.separation) return;
        const rolling: f64 = c.sum / @as(f64, @floatFromInt(c.len));
        if (d.* == c.cfg.max and rolling < c.cfg.threshold) {
            c.note(round, d.*, c.cfg.min, rolling, "drop");
            d.* = c.cfg.min;
            std.log.info("glm: B3 adaptive drop d{d}->d{d} at round {d}, rolling accept {d:.3} < {d:.3}", .{ c.cfg.max, c.cfg.min, round, rolling, c.cfg.threshold });
        } else if (d.* == c.cfg.min and rolling > c.cfg.high) {
            c.note(round, d.*, c.cfg.max, rolling, "restore");
            d.* = c.cfg.max;
            std.log.info("glm: B3 adaptive restore d{d}->d{d} at round {d}, rolling accept {d:.3} > {d:.3}", .{ c.cfg.min, c.cfg.max, round, rolling, c.cfg.high });
        }
    }

    /// One switch into the log (the first 32 are kept) and onto the rate limiter.
    fn note(c: *Controller, round: u64, old: u32, new: u32, rolling: f64, direction: []const u8) void {
        if (c.log_len < c.log.len) {
            c.log[c.log_len] = .{ .round = round, .old_depth = old, .new_depth = new, .rolling_rate = rolling, .direction = direction };
            c.log_len += 1;
        }
        c.switches += 1;
        c.last_switch = round;
    }
};

/// The reply's rollup: the switches observed, the depth in force at the end and the accept ratio over drafted tokens.
pub const Summary = struct {
    switches: u64 = 0,
    final_depth: u32 = 0,
    avg_accept: f64 = 0,
    log: ?[]u8 = null,
};

/// The window's rollup; a formatted switch log is allocated on `gpa` (null when no switch fired).
pub fn finish(c: *const Controller, gpa: std.mem.Allocator, d: u32) !Summary {
    var s: Summary = .{ .switches = c.switches, .final_depth = d };
    if (c.drafted > 0) s.avg_accept = @as(f64, @floatFromInt(c.accepted)) / @as(f64, @floatFromInt(c.drafted));
    if (c.log_len > 0) {
        var buf: std.ArrayList(u8) = .empty;
        for (c.log[0..c.log_len]) |sw| {
            try buf.print(gpa, "  B3 switch round {d}: d{d}->d{d} rolling accept {d:.3} ({s})\n", .{ sw.round, sw.old_depth, sw.new_depth, sw.rolling_rate, sw.direction });
        }
        s.log = try buf.toOwnedSlice(gpa);
    }
    return s;
}
