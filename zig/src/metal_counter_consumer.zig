//! Model-free dependent FP32 operators with untracked resources and one externally completed command buffer.
const std = @import("std");
const mtl = @import("metal");
pub const counters = @import("core/gpu_counters.zig");
const source = @import("kernel_sources").counter_probe;
pub const element_count: u32 = 4096;
const classes = [_]counters.Class{ .dense, .activation, .copy };

pub const Report = struct {
    spans: [3]counters.Span,
    aggregates: [counters.class_count]counters.Aggregate,
    calibration: counters.Calibration,
    in_flight_rejected: bool,
};

pub const Consumer = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    pipeline: mtl.Pipeline,
    a: mtl.Buffer,
    b: mtl.Buffer,
    indirect: mtl.Buffer,
    gate: mtl.SharedEvent,
    next_gate: u64 = 1,

    pub fn init(device: mtl.Device) !Consumer {
        const queue = try device.queue();
        errdefer queue.deinit();
        const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer library.deinit();
        const pipeline = try mtl.Pipeline.init(device, library, "tf_counter_affine", false);
        errdefer pipeline.deinit();
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const a = try device.buffer(element_count * @sizeOf(f32), options);
        errdefer a.deinit();
        const b = try device.buffer(element_count * @sizeOf(f32), options);
        errdefer b.deinit();
        const indirect = try device.buffer(3 * @sizeOf(u32), options);
        errdefer indirect.deinit();
        @memcpy(indirect.slice(u32, 3), &[_]u32{ element_count / 256, 1, 1 });
        const gate = try device.sharedEvent();
        return .{
            .device = .{ .id = mtl.objc.retain(device.id) },
            .queue = queue,
            .pipeline = pipeline,
            .a = a,
            .b = b,
            .indirect = indirect,
            .gate = gate,
        };
    }

    pub fn deinit(self: Consumer) void {
        self.gate.deinit();
        self.indirect.deinit();
        self.b.deinit();
        self.a.deinit();
        self.pipeline.deinit();
        self.queue.deinit();
        self.device.deinit();
    }

    /// The only CPU wait is the consumer's original final completion wait, never a span wait.
    pub fn execute(self: *Consumer, recorder: *counters.Recorder) !void {
        for (self.a.slice(f32, element_count), 0..) |*value, i| value.* = initial(i);
        @memset(self.b.slice(f32, element_count), std.math.nan(f32));
        const cb = self.queue.commandBuffer();
        const gate_value = self.next_gate;
        self.next_gate += 1;
        cb.waitFor(self.gate, gate_value);
        var encoder = cb.compute(.serial); // the engine's order: the encoder is open when the recorder binds
        try recorder.beginCommand(cb);
        var submitted = false;
        errdefer {
            if (submitted) {
                self.gate.set(gate_value);
                cb.wait();
                recorder.discardAfterCompletion() catch {};
            } else recorder.cancelEncoding(&encoder) catch {};
        }
        const scales = [_]f32{ 2, 1, 0.5 };
        const biases = [_]f32{ 1, -3, 2 };
        for (classes, 0..) |kind, i| {
            try recorder.beginSpan(&encoder, kind);
            const input = if (i % 2 == 0) self.a else self.b;
            const output = if (i % 2 == 0) self.b else self.a;
            encoder.setPipeline(self.pipeline);
            encoder.setBuffer(input, 0, 0);
            encoder.setBuffer(output, 0, 1);
            encoder.setValue(element_count, 2);
            encoder.setValue(scales[i], 3);
            encoder.setValue(biases[i], 4);
            encoder.useResource(input, mtl.ResourceUsage.read);
            encoder.useResource(output, mtl.ResourceUsage.write);
            encoder.useResource(self.indirect, mtl.ResourceUsage.read);
            if (i == 1) {
                encoder.dispatchIndirect(self.indirect, 0, mtl.Size.of(256, 1, 1));
            } else encoder.dispatchThreads(mtl.Size.of(element_count, 1, 1), mtl.Size.of(256, 1, 1));
            encoder.barrier();
            try recorder.endSpan(&encoder);
        }
        try recorder.finishEncoding(&encoder);
        cb.commit();
        submitted = true;
        if (recorder.enabled()) {
            try expectFailure(error.CounterCommandNotCompleted, recorder.readAfterCompletion());
            try expectFailure(error.CounterRecordingInFlight, recorder.beginCommand(cb));
            try expectFailure(error.CounterCommandNotCompleted, recorder.discardAfterCompletion());
            try expectFailure(error.CounterCommandNotCompleted, recorder.deinit());
        }
        self.gate.set(gate_value);
        cb.wait();
        if (cb.failure() != null) return error.CounterConsumerGpuFailure;
        _ = try recorder.readAfterCompletion();
        for (self.b.slice(f32, element_count), 0..) |value, i| {
            const expected = initial((i + 3) % element_count) + 1;
            if (@as(u32, @bitCast(value)) != @as(u32, @bitCast(expected))) return error.CounterConsumerOutputMismatch;
        }
    }

    pub fn probe(self: *Consumer, allocator: std.mem.Allocator) !Report {
        var plain = try counters.Recorder.init(allocator, self.device, .{});
        defer plain.deinit() catch unreachable;
        try self.execute(&plain);
        var expected_bits: [element_count]u32 = undefined;
        for (self.b.slice(f32, element_count), &expected_bits) |value, *bits| bits.* = @bitCast(value);
        // GLM_COUNTER_MAX_SPANS: exercise a different buffer capacity (the engine's census uses 2048).
        const max_spans: usize = if (std.c.getenv("GLM_COUNTER_MAX_SPANS")) |v|
            std.fmt.parseInt(usize, std.mem.span(v), 10) catch 3
        else
            3;
        var instrumented = try counters.Recorder.init(allocator, self.device, .{ .enabled = true, .max_spans = max_spans });
        defer instrumented.deinit() catch unreachable;
        try self.execute(&instrumented);
        for (self.b.slice(f32, element_count), expected_bits) |value, bits| {
            if (@as(u32, @bitCast(value)) != bits) return error.CounterInstrumentationChangedBits;
        }
        const spans = try instrumented.readAfterCompletion();
        if (spans.len != 3) return error.CounterConsumerSpanCountMismatch;
        var report: Report = .{
            .spans = undefined,
            .aggregates = instrumented.aggregates(),
            .calibration = instrumented.calibration orelse return error.MissingClockCalibration,
            .in_flight_rejected = true,
        };
        @memcpy(&report.spans, spans);
        var seconds: f64 = 0;
        for (report.spans, classes) |span, kind| {
            if (span.kind != kind) return error.CounterConsumerClassMismatch;
            seconds += span.gpu_seconds;
        }
        if (seconds <= 0 or !std.math.isFinite(seconds)) return error.InvalidCounterSample;
        return report;
    }
};

fn initial(index: usize) f32 {
    const value: i32 = @as(i32, @intCast(index % 127)) - 63;
    return @floatFromInt(value);
}

fn expectFailure(expected: anyerror, result: anytype) !void {
    if (result) |_| {
        return error.CounterExpectedFailureMissing;
    } else |err| if (err != expected) return err;
}

fn supportedDevice() !mtl.Device {
    const device = mtl.Device.init() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (mtl.counters.query(device).mode() == .unsupported) {
        device.deinit();
        return error.SkipZigTest;
    }
    return device;
}

// GLM_COUNTER_SPANS crash repro: cases a-d vary pipeline vs counter alloc order; 1st setPipeline SIGSEGVs in AGX.
test "dispatch counter sample around the generated embed pipeline binds or crashes reproducibly" {
    const case = blk: {
        const v = std.c.getenv("GLM_REPRO_CASE") orelse break :blk 'a';
        break :blk v[0];
    };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    const sources = @import("kernel_sources");
    // Cases e/f: e = ~24 libs on workers (kernels.load Jobs); f = 100s MTLBuffers (wts.load shape); pre-recorder.
    if (case == 'e') {
        const Worker = struct {
            device: mtl.Device,
            entries: []const sources.glm.Kernel,
            fn run(self: *@This()) void {
                const wpool = mtl.objc.Pool.push();
                defer wpool.pop();
                for (self.entries) |entry| {
                    const lib = mtl.Library.fromSource(self.device, entry.source, mtl.CompileOptions.mlx()) catch return;
                    defer lib.deinit();
                    for (entry.functions) |fname| {
                        const p = mtl.Pipeline.init(self.device, lib, fname, false) catch return;
                        p.deinit();
                    }
                }
            }
        };
        var workers: [8]Worker = undefined;
        for (&workers, 0..) |*w, wi| w.* = .{ .device = .{ .id = mtl.objc.retain(device.id) }, .entries = sources.glm.all[wi * 4 ..][0..4] };
        var threads: [8]std.Thread = undefined;
        for (&threads, &workers) |*t, *w| t.* = std.Thread.spawn(.{}, Worker.run, .{w}) catch unreachable;
        for (&threads) |*t| t.join();
        for (&workers) |*w| mtl.objc.release(w.device.id);
        std.debug.print("case e: 8 worker threads compiled {} libraries and their pipelines\n", .{workers.len * 4});
    }
    if (case == 'f') {
        var big: [64]mtl.Buffer = undefined;
        for (&big) |*b| b.* = try device.buffer(8 << 20, mtl.ResourceOptions.shared);
        defer for (&big) |*b| b.deinit();
        std.debug.print("case f: 64 x 8 MB MTLBuffers allocated\n", .{});
    }
    if (case == 'g') {
        // Engine keeps every pipeline alive (~1000s); case e destroys workers, keep realistic pop across buffer.
        var alive = try std.ArrayList(mtl.Pipeline).initCapacity(std.testing.allocator, 512);
        defer {
            for (alive.items) |p| p.deinit();
            alive.deinit(std.testing.allocator);
        }
        for (0..2) |pass| {
            for (sources.glm.all) |entry| {
                const lib = mtl.Library.fromSource(device, entry.source, mtl.CompileOptions.mlx()) catch continue;
                defer lib.deinit();
                for (entry.functions) |fname| {
                    const p = mtl.Pipeline.init(device, lib, fname, false) catch continue;
                    try alive.append(std.testing.allocator, p);
                }
            }
            _ = pass;
        }
        std.debug.print("case g: {d} pipelines kept alive\n", .{alive.items.len});
    }
    // Case d mirrors the engine: the recorder's sample buffer is allocated before the wide pipelines.
    var late_recorder: ?counters.Recorder = null;
    defer if (late_recorder) |*r| r.deinit() catch unreachable;
    if (case == 'd' or case == 'e' or case == 'f' or case == 'g') {
        late_recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 2048 });
        std.debug.print("case {c}: 4096-sample buffer allocated\n", .{case});
    }
    const library = try mtl.Library.fromSource(device, sources.ops_embed_norm, mtl.CompileOptions.mlx());
    defer library.deinit();
    const embed = try mtl.Pipeline.init(device, library, "tf_embed_b8_g64", false);
    defer embed.deinit();
    std.debug.print("repro case {c}: embed pipeline created\n", .{case});
    const queue = try device.queue();
    defer queue.deinit();
    var recorder = if (late_recorder) |*r| r.* else try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 1 });
    defer if (late_recorder == null) recorder.deinit() catch unreachable;
    const cb = queue.commandBuffer();
    try recorder.beginCommand(cb);
    var encoder = cb.compute(.serial);
    std.debug.print("repro case {c}: beginCommand ok\n", .{case});
    switch (case) {
        'd', 'e', 'f', 'g' => {
            std.debug.print("case {c}: no span; about to setPipeline with the sample buffer present\n", .{case});
        },
        'b' => {
            encoder.setPipeline(embed);
            std.debug.print("case b: embed pipeline bound before sample\n", .{});
            try recorder.beginSpan(&encoder, .dense);
        },
        'c' => {
            const tiny_library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
            defer tiny_library.deinit();
            const tiny = try mtl.Pipeline.init(device, tiny_library, "tf_counter_affine", false);
            defer tiny.deinit();
            encoder.setPipeline(tiny);
            std.debug.print("case c: tiny pipeline bound before sample\n", .{});
            try recorder.beginSpan(&encoder, .dense);
        },
        else => {
            try recorder.beginSpan(&encoder, .dense);
            std.debug.print("case a: sample taken with no pipeline bound\n", .{});
        },
    }
    std.debug.print("repro case {c}: about to setPipeline(embed)\n", .{case});
    encoder.setPipeline(embed);
    std.debug.print("repro case {c}: setPipeline(embed) SURVIVED\n", .{case});
    const spanless = case == 'd' or case == 'e' or case == 'f' or case == 'g';
    if (!spanless) try recorder.endSpan(&encoder);
    try recorder.finishEncoding(&encoder);
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.CounterConsumerGpuFailure;
    if (!spanless) {
        const spans = try recorder.readAfterCompletion();
        if (spans.len != 1) return error.CounterConsumerSpanCountMismatch;
    }
}

test "sampled dependent direct and indirect operators preserve FP32 bits and repeat without stale samples" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    const first = try consumer.probe(std.testing.allocator);
    const second = try consumer.probe(std.testing.allocator);
    for (first.spans, second.spans, classes) |a, b, kind| {
        try std.testing.expectEqual(kind, a.kind);
        try std.testing.expectEqual(kind, b.kind);
        try std.testing.expect(b.begin_timestamp >= a.end_timestamp);
        try std.testing.expectEqual(@as(u64, 1), second.aggregates[@backingInt(kind)].calls);
    }
    try std.testing.expect(first.in_flight_rejected and second.in_flight_rejected);
}

test "bounded recorder rejects nesting, overflow, open spans and premature reuse then cancels and reuses" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 1 });
    defer recorder.deinit() catch unreachable;
    const cb = queue.commandBuffer();
    try recorder.beginCommand(cb);
    var encoder = cb.compute(.serial);
    errdefer recorder.cancelEncoding(&encoder) catch {};
    try std.testing.expectError(error.CounterEncoderStillOpen, recorder.deinit());
    try std.testing.expectError(error.CounterEncoderStillOpen, recorder.discardAfterCompletion());
    try std.testing.expectError(error.CounterRecordingInFlight, recorder.beginCommand(cb));
    try std.testing.expectError(error.NoActiveCounterSpan, recorder.endSpan(&encoder));
    try recorder.beginSpan(&encoder, .dense);
    try std.testing.expectError(error.NestedCounterSpan, recorder.beginSpan(&encoder, .copy));
    try std.testing.expectError(error.UnclosedCounterSpan, recorder.finishEncoding(&encoder));
    try recorder.endSpan(&encoder);
    try std.testing.expectError(error.CounterSpanCapacityExceeded, recorder.beginSpan(&encoder, .copy));
    try recorder.finishEncoding(&encoder);
    try std.testing.expectError(error.CounterCommandNotCompleted, recorder.readAfterCompletion());
    try recorder.cancelEncoding(&encoder);
    const next = queue.commandBuffer();
    try recorder.beginCommand(next);
    encoder = next.compute(.serial);
    try recorder.finishEncoding(&encoder);
    next.commit();
    next.wait();
    if (next.failure() != null) return error.CounterConsumerGpuFailure;
    try std.testing.expectEqual(@as(usize, 0), (try recorder.readAfterCompletion()).len);
    for (recorder.aggregates()) |aggregate| try std.testing.expectEqual(@as(u64, 0), aggregate.calls);
    try recorder.discardAfterCompletion();
}

test "completed recorder reuses bounded samples and read results do not double count" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    var recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 3 });
    defer recorder.deinit() catch unreachable;
    try consumer.execute(&recorder);
    const first = try recorder.readAfterCompletion();
    const prior_end = first[2].end_timestamp;
    _ = try recorder.readAfterCompletion();
    for (classes) |kind| try std.testing.expectEqual(@as(u64, 1), recorder.aggregates()[@backingInt(kind)].calls);
    try consumer.execute(&recorder);
    const second = try recorder.readAfterCompletion();
    try std.testing.expectEqual(@as(usize, 3), second.len);
    try std.testing.expect(second[0].begin_timestamp >= prior_end);
    for (classes) |kind| try std.testing.expectEqual(@as(u64, 1), recorder.aggregates()[@backingInt(kind)].calls);
}
