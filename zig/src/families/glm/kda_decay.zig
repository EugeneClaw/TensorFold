//! The KDA decay rates A = exp(A_log) with MLX's Exp, on the GPU: one pass over the loaded layers' attention.
const std = @import("std");
const mtl = @import("metal");
const Engine = @import("engine.zig").Engine;

/// A = exp(A_log) for every KDA layer's attention (MLA layers have none); called once, after the weights land.
pub fn prepare(e: *Engine) !void {
    const cb = e.queue.commandBuffer();
    const enc = cb.compute(.serial);
    for (0..e.c.run) |li| switch (e.w.layers[li].attn) {
        .kda => |*a| {
            enc.setPipeline(e.k.exp_f32);
            enc.setBuffer(a.a_log.buf, a.a_log.off, 0);
            enc.setBuffer(a.a.buf, a.a.off, 1);
            enc.setValue(e.c.kda_heads, 2);
            enc.dispatchThreads(mtl.Size.of(e.c.kda_heads, 1, 1), mtl.Size.of(64, 1, 1));
        },
        .mla => {},
    };
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("glm: prepare failed: {s}", .{msg});
        return error.GpuFailed;
    }
}
