//! Engram rows from the release's own safetensors shards, for hosts that cannot hold the tables resident.
//!
//! The release packs one Engram layer to a shard: `model-00047-of-00048.safetensors` carries layer 1 and
//! `model-00048-of-00048.safetensors` layer 14, each as `layers.<L>.engram.embed.weight` (F8_E4M3, rows x 256) beside
//! `layers.<L>.engram.embed.scale` (F8_E8M0, rows x 8). The two tensors sit far apart in the file, so a row is two
//! positional reads, and a bounded LRU keeps the window's rows. No derived file is written: the bytes served are the
//! release's own, decoded by the same e4m3/ue8m0 tables the resident path uses.
const std = @import("std");
const f = @import("engram_rows.zig");
const eh = @import("engram_host.zig");
const st = @import("core").safetensors;

/// One row's record: 256 E4M3 values and 8 UE8M0 scales, the portable reader's own layout.
const row_bytes: usize = @intCast(f.row_bytes);
const values: usize = @intCast(f.values);
const scales: usize = values / 32;

pub const Stats = struct { hits: u64 = 0, misses: u64 = 0, issued: u64 = 0, reads: u64 = 0, bytes: u64 = 0 };

const Entry = struct { id: u64 = std.math.maxInt(u64), age: u64 = 0, bytes: [row_bytes]u8 = undefined };

const Table = struct {
    file: std.Io.File,
    /// Row 0's first weight byte and first scale byte, absolute in the shard.
    weight_at: u64,
    scale_at: u64,
    rows: u64,
    entries: []Entry,
};

/// The release packs one Engram layer to a shard; this is the pair load.zig plans for (same names, same layers).
const Known = struct { layer: u32, file: []const u8 };
const known = [_]Known{
    .{ .layer = 1, .file = "model-00047-of-00048.safetensors" },
    .{ .layer = 14, .file = "model-00048-of-00048.safetensors" },
};

/// The shard's two Engram tensors: dtype, shape and where row 0 starts.
const Extent = struct { weight_at: u64, scale_at: u64, rows: u64 };

pub const Rows = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    tables: []Table,
    clock: u64 = 0,
    stats: Stats = .{},
    /// TF_DSV41_ENGRAM_TRACE: one read line a window, for the read-path measurements.
    trace: bool = false,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, world: u32, layers: []const u32, cap: u32) !*Rows {
        if (world == 0 or rank >= world or cap == 0 or layers.len == 0) return error.BadEngramShard;
        const r = try a.create(Rows);
        errdefer a.destroy(r);
        const tables = try a.alloc(Table, layers.len);
        errdefer a.free(tables);
        var made: usize = 0;
        errdefer for (tables[0..made]) |t| {
            t.file.close(io);
            a.free(t.entries);
        };
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        for (layers, tables) |layer, *t| {
            const shard_name = for (known) |k| {
                if (k.layer == layer) break k.file;
            } else return error.UnknownEngramLayer;
            const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, shard_name });
            defer a.free(path);
            const file = try std.Io.Dir.cwd().openFile(io, path, .{});
            errdefer file.close(io);
            const size = try file.length(io);
            const ext = try extent(a, arena.allocator(), file, io, size, layer);
            const entries = try a.alloc(Entry, cap);
            @memset(entries, .{});
            t.* = .{ .file = file, .weight_at = ext.weight_at, .scale_at = ext.scale_at, .rows = ext.rows, .entries = entries };
            made += 1;
        }
        r.* = .{ .gpa = a, .io = io, .tables = tables, .trace = std.c.getenv("TF_DSV41_ENGRAM_TRACE") != null };
        return r;
    }

    fn extent(a: std.mem.Allocator, arena: std.mem.Allocator, file: std.Io.File, io: std.Io, size: u64, layer: u32) !Extent {
        if (size < 8) return error.BadSafetensors;
        var head: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &head, 0) != 8) return error.ShortRead;
        const header_bytes: usize = @intCast(std.mem.readInt(u64, &head, .little));
        if (header_bytes > @min(@as(usize, @intCast(size - 8)), 64 << 20)) return error.BadSafetensors;
        const json = try a.alloc(u8, header_bytes);
        defer a.free(json);
        if (try file.readPositionalAll(io, json, 8) != header_bytes) return error.ShortRead;
        const data_at: u64 = 8 + header_bytes;
        const names = try st.parseHeaderPrefix(arena, json, @intCast(size - data_at), null);
        var buf: [128]u8 = undefined;
        const w = names.get(try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.weight", .{layer})) orelse return error.MissingTensor;
        const s = names.get(try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.scale", .{layer})) orelse return error.MissingTensor;
        if (w.dtype != .f8_e4m3 or w.rank != 2 or w.shape[1] != values) return error.UnexpectedEngramTensor;
        if (s.dtype != .f8_e8m0 or s.rank != 2 or s.shape[1] != scales) return error.UnexpectedEngramTensor;
        if (s.shape[0] != w.shape[0]) return error.UnexpectedEngramTensor;
        if (w.end - w.begin < w.shape[0] * values or s.end - s.begin < s.shape[0] * scales) return error.UnexpectedEngramTensor;
        return .{ .weight_at = data_at + w.begin, .scale_at = data_at + s.begin, .rows = w.shape[0] };
    }

    pub fn close(r: *Rows) void {
        for (r.tables) |t| {
            t.file.close(r.io);
            r.gpa.free(t.entries);
        }
        r.gpa.free(r.tables);
        r.gpa.destroy(r);
    }

    fn validate(r: *Rows, li: usize, ids: []const i64) !void {
        if (li >= r.tables.len) return error.BadEngramShard;
        const total = r.tables[li].rows;
        for (ids) |id| if (id < 0 or @as(u64, @intCast(id)) >= total) return error.EngramRowOutsideShard;
    }

    fn record(r: *Rows, li: usize, id: u64) !*[row_bytes]u8 {
        const table = &r.tables[li];
        r.clock +%= 1;
        var slot = &table.entries[0];
        for (table.entries) |*entry| {
            if (entry.id == id) {
                entry.age = r.clock;
                r.stats.hits += 1;
                return &entry.bytes;
            }
            if (entry.age < slot.age) slot = entry;
        }
        // Publish the cache identity only after both reads succeed: a torn row is never served.
        var bytes: [row_bytes]u8 = undefined;
        if (try table.file.readPositionalAll(r.io, bytes[0..values], table.weight_at + id * values) != values) return error.ShortRead;
        if (try table.file.readPositionalAll(r.io, bytes[values..][0..scales], table.scale_at + id * scales) != scales) return error.ShortRead;
        r.stats.reads += 2;
        slot.* = .{ .id = id, .age = r.clock, .bytes = bytes };
        r.stats.misses += 1;
        r.stats.bytes += row_bytes;
        return &slot.bytes;
    }

    /// Synchronous prefetch: the Source contract permits completion before return.
    pub fn issue(r: *Rows, li: usize, ids: []const i64) !void {
        try r.validate(li, ids);
        for (ids) |id| {
            _ = try r.record(li, @intCast(id));
            r.stats.issued += 1;
        }
    }

    /// The GPU's own u32 row ids straight in: no widening copy a window.
    pub fn rowsU32(r: *Rows, li: usize, ids: []const u32, dim: usize, out: []align(1) u16) !void {
        if (li >= r.tables.len) return error.BadEngramShard;
        if (dim != values or out.len != try std.math.mul(usize, ids.len, values)) return error.BadEngramShard;
        const total = r.tables[li].rows;
        for (ids, 0..) |id, i| {
            if (id >= total) return error.EngramRowOutsideShard;
            const b = try r.record(li, id);
            for (0..values) |j| out[i * values + j] = f.bf16(f.e4m3(b[j]) * f.scale(b[values + j / 32]));
        }
    }

    pub fn rows(r: *Rows, li: usize, ids: []const i64, dim: usize, out: []u16) !void {
        try r.validate(li, ids);
        if (dim != values or out.len != try std.math.mul(usize, ids.len, values)) return error.BadEngramShard;
        for (ids, 0..) |id, i| {
            const b = try r.record(li, @intCast(id));
            for (0..values) |j| out[i * values + j] = f.bf16(f.e4m3(b[j]) * f.scale(b[values + j / 32]));
        }
    }

    /// One line a window when TF_DSV41_ENGRAM_TRACE is set: what this window read and what it hit.
    pub fn recordWindow(r: *Rows, window: u32) void {
        if (!r.trace) return;
        std.log.scoped(.dsv41).info("engram stream: rows={d} hits={d} misses={d} reads={d} bytes={d}", .{ window, r.stats.hits, r.stats.misses, r.stats.reads, r.stats.bytes });
    }

    pub fn source(r: *Rows) eh.Source {
        return .{ .ptr = r, .vtable = &.{ .issue = issueFn, .rows = rowsFn } };
    }
    fn issueFn(p: *anyopaque, li: usize, ids: []const i64) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.issue(li, ids);
    }
    fn rowsFn(p: *anyopaque, li: usize, ids: []const i64, dim: usize, out: []u16) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.rows(li, ids, dim, out);
    }
};
