//! glm_grid against a host u16 reference of its exact RNE rule, at every k: catches a kernel that
//! compiles but silently does not write (the no-op failure mode the first window exposed downstream).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// The kernel's rule, on u16 bit patterns: RNE on the low 7-k mantissa bits, sign and exponent
/// carried bit-for-bit (carries fold naturally in the u16 field), NaN/Inf (exponent 0xff) pass through.
fn gridRef(u: u16, k: u5) u16 {
    const drop: u32 = 7 - @as(u32, k);
    const mask: u16 = @intCast((@as(u32, 1) << @intCast(drop)) - 1);
    const tie: u16 = @intCast(@as(u32, 1) << @intCast(drop - 1));
    if (((u >> 7) & 0xff) == 0xff) return u; // NaN and Inf pass through
    const rem = u & mask;
    var q = u & ~mask;
    if (rem > tie or (rem == tie and (q & (tie << 1)) != 0)) q += (tie << 1);
    return q;
}

fn bf16of(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

test "glm_grid: m_x grid equals the host RNE reference at every k, and is not a no-op" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, sources.glm_glue, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "glm_grid", false);
    defer pipe.deinit();

    const N = 8192;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const master = try std.testing.allocator.alloc(u16, N);
    defer std.testing.allocator.free(master);
    // the boundary cases first, then a spread that exercises ties, carries and subnormals
    const heads = [_]u16{ 0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x0001, 0x00ff, 0x7f7f, 0x0081, 0x0100 };
    for (master, 0..) |*v, i| v.* = if (i < heads.len) heads[i] else bf16of((rnd.float(f32) * 4 - 2) * std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)));

    const buf = try device.buffer(N * 2, mtl.ResourceOptions.shared);
    defer buf.deinit();

    for (1..7) |kk| {
        const k: u5 = @intCast(kk);
        @memcpy(buf.slice(u16, N), master); // fresh master values per k: the kernel writes in place
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipe);
        enc.setBuffer(buf, 0, 0);
        enc.setValue([2]u32{ N, k }, 1);
        enc.dispatchThreads(mtl.Size.of(N, 1, 1), mtl.Size.of(256, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        var changed: usize = 0;
        for (buf.slice(u16, N), master) |got, want_in| {
            try std.testing.expectEqual(gridRef(want_in, k), got);
            if (got != want_in) changed += 1;
        }
        // k = 6 rounds about a quarter of a uniform mantissa spread; k = 3 flushes nearly everything
        const floor: usize = if (k == 6) N / 8 else if (k == 3) N / 2 else 0;
        try std.testing.expect(changed >= floor);
    }
}
