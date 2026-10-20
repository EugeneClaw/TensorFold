//! GLM_COUNTER_SPANS: the per-class GPU timestamp recorder the engine owns, and the per-rep rollup each commit appends.
const std = @import("std");
const mtl = @import("metal");
const counters_class = @import("counters_class.zig");

/// One rep's measured rollup: per-class GPU seconds + wall clock + encoding time. BUBBLES derives per-rep from these.
pub const Rep = struct {
    wall_seconds: f64 = 0,
    encode_seconds: f64 = 0,
    gpu_seconds_whole: f64 = 0, // the round's GPU start..end (sums span classes to a wall, but separate from wall_seconds)
    per_class_seconds: [counters_class.Kind.count]f64 = @splat(0),
    per_class_calls: [counters_class.Kind.count]u64 = @splat(0),
    span_count: usize = 0, // total spans recorded for this rep
    prompt_tokens: u32 = 0,
    picked_rows: u32 = 0, // 1-row or 4-row windows
};

/// The per-rep accumulator; the engine appends one Rep per round or prompt chunk and tf-glm-run prints the table.
pub const SpanTable = std.ArrayList(Rep);

/// The recorder and its table as load allocates them: two heap roots freed as one pair.
pub const Owned = struct {
    rec: *counters_class.Recorder,
    table: *SpanTable,
};

/// GLM_COUNTER_SPANS=1 allocates the recorder and span table (last at load); null when the env is off or not 1.
pub fn attach(gpa: std.mem.Allocator, device: mtl.Device) !?Owned {
    const cs_env = std.c.getenv("GLM_COUNTER_SPANS") orelse return null;
    if (cs_env[0] != '1') return null;
    // MTLCounterSampleBuffer: 32768 B cap = 8 B × 4096 samples × 2 samples/span = 2048 spans.
    const max_spans: usize = if (std.c.getenv("GLM_COUNTER_MAX_SPANS")) |vv| std.fmt.parseInt(usize, std.mem.span(vv), 10) catch 2048 else 2048;
    const rec_ptr = if (gpa.create(counters_class.Recorder)) |p| p else |_| return error.OutOfMemory;
    rec_ptr.* = try counters_class.Recorder.init(gpa, device, .{ .enabled = true, .max_spans = max_spans });
    rec_ptr.sites = if (std.c.getenv("GLM_COUNTER_SPAN_SITES")) |v| v[0] != '0' else true;
    errdefer {
        rec_ptr.deinit() catch {};
        gpa.destroy(rec_ptr);
    }
    // Span table on the heap so the pointer outlives this scope.
    const table_ptr = try gpa.create(SpanTable);
    table_ptr.* = try SpanTable.initCapacity(gpa, 64);
    errdefer {
        table_ptr.deinit();
        gpa.destroy(table_ptr);
    }
    std.log.info("glm: GLM_COUNTER_SPANS=1, max_spans={d}, class_count={d}", .{ max_spans, counters_class.Kind.count });
    return .{ .rec = rec_ptr, .table = table_ptr };
}

/// The finished buffer's spans, aggregated per class into the table as the next rep (no recorder: nothing to add).
pub fn rollup(gpa: std.mem.Allocator, rec: ?*counters_class.Recorder, table: ?*SpanTable, gpu_seconds_whole: f64) void {
    const r = rec orelse return;
    // Wait already completed; read spans and aggregate per-class totals for this cb.
    const spans = r.readAfterCompletion() catch return;
    const t = table orelse return;
    var rep: Rep = .{ .gpu_seconds_whole = gpu_seconds_whole };
    for (spans) |s| {
        rep.span_count += 1;
        const idx = @backingInt(s.kind);
        rep.per_class_seconds[idx] += s.gpu_seconds;
        rep.per_class_calls[idx] += 1;
    }
    t.append(gpa, rep) catch {};
}

/// The table first, then the recorder that filled it (the deinit order, and attach's errdefer).
pub fn detach(gpa: std.mem.Allocator, rec: ?*counters_class.Recorder, table: ?*SpanTable) void {
    if (table) |t| {
        t.deinit(gpa);
        gpa.destroy(t);
    }
    if (rec) |r| {
        r.deinit() catch {};
        gpa.destroy(r);
    }
}
