//! 12 launch classes for the profile gates + a whole-round sentinel; bit positions match forward.zig's Class.
const std = @import("std");
const mtl = @import("metal");
const counters_mod = @import("../../core/gpu_counters.zig");
const RecorderT = counters_mod.RecorderT;
const SpanT = counters_mod.SpanT;

/// Per-class GPU span tags; `_round` (bit 12) is a whole-round envelope that never gates dispatch.
pub const Kind = enum(u8) {
    hc = 0,
    kda = 1,
    mla = 2,
    dense = 3,
    route = 4,
    routed = 5,
    shared = 6,
    exchange = 7,
    combine = 8,
    head = 9,
    mtp = 10,
    ends = 11,
    _round = 12,

    pub const count = @typeInfo(Kind).@"enum".field_names.len; // 13

    /// The bit mask in forward.zig's Class for this kind (bit i = 1 << i, declaration order).
    pub fn bit(self: Kind) u32 {
        const idx = @backingInt(self);
        const shrunk: u5 = @intCast(idx);
        return @as(u32, 1) << @as(u5, @intCast(shrunk));
    }

    pub const names = [_][]const u8{
        "hc",     "kda",      "mla",     "dense", "route", "routed",
        "shared", "exchange", "combine", "head",  "mtp",   "ends",
        "_round",
    };

    pub fn name(self: Kind) []const u8 {
        return names[@backingInt(self)];
    }
};

/// A family-scoped Recorder alias (Kind-tagged): this is what the engine owns and forward.zig threads into Ctx.
pub const Recorder = RecorderT(Kind);
pub const Span = SpanT(Kind);

test "kind bit parity with forward.zig Class declaration order" {
    // Sanity: every named Kind maps to a unique bit and stays inside the 12+1 range we promise.
    inline for (@typeInfo(Kind).@"enum".field_names, 0..) |f, i| {
        try std.testing.expectEqualStrings(f, Kind.names[i]);
        try std.testing.expectEqual(@as(u8, i), @backingInt(@as(Kind, @fromBackingInt(@intCast(i)))));
    }
}
