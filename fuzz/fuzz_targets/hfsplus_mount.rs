// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Fuzzes mounting an HFS+ or HFSX volume and walking it.
//!
//! Directory hard links can make the catalog cyclic, so the walk is bounded
//! in depth and in total entries. Errors are fine; a panic is the bug.

#![no_main]

use std::ops::ControlFlow;
use std::sync::Arc;

use atlas_fs_core::fs::{
    DirBatch, DirProgress, FileSystem, MAX_ALLOCATION_BUCKETS, NodeId, Stream,
};
use atlas_fs_core::io::block::{BlockSource, MemSource};
use atlas_hfsplus::{HfsPlus, mount};
use libfuzzer_sys::fuzz_target;

/// Entries per `next_batch` call.
const BATCH: usize = 64;
/// Deepest folder the walk descends into.
const MAX_DEPTH: usize = 8;
/// Most entries the walk visits in total.
const MAX_ENTRIES: usize = 2000;

fuzz_target!(|data: &[u8]| {
    let source: Arc<dyn BlockSource> = Arc::new(MemSource::new(data));
    let Ok(volume) = mount(source) else {
        return;
    };
    exercise(&volume);
});

/// Everything the target does to a mounted volume. Add new calls here.
fn exercise(volume: &HfsPlus) {
    let _ = volume.info();
    let _ = volume.journal();
    let root = volume.root();
    let _ = volume.metadata(root);
    let _ = volume.allocation_map(MAX_ALLOCATION_BUCKETS);
    let mut budget = MAX_ENTRIES;
    walk(volume, root, 0, &mut budget);
    let _ = volume.identity();
    // The whole-volume scan, under the same entry budget as the walk.
    let mut seen = 0usize;
    let _ = volume.scan(&mut |entry| {
        let _ = (entry.node, entry.parent, entry.name.len());
        seen += 1;
        if seen >= MAX_ENTRIES {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
}

/// Lists `dir`, then looks up and describes every entry, descending into
/// folders until the depth or entry budget runs out.
fn walk(volume: &HfsPlus, dir: NodeId, depth: usize, budget: &mut usize) {
    let Ok(mut stream) = volume.read_dir(dir) else {
        return;
    };
    // Folders found in this listing, visited after it is fully read.
    let mut subdirs: Vec<NodeId> = Vec::new();
    let mut batch = DirBatch::new();
    loop {
        batch.clear();
        let progress = stream.next_batch(&mut batch, BATCH);
        for entry in batch.iter() {
            if *budget == 0 {
                return;
            }
            *budget -= 1;
            let _ = volume.lookup(dir, entry.name);
            let _ = volume.metadata(entry.node);
            visit_entry(volume, entry.node);
            if entry.is_browsable() {
                subdirs.push(entry.node);
            }
        }
        match progress {
            Ok(DirProgress::More) => {}
            Ok(DirProgress::Done) | Err(_) => break,
        }
    }
    if depth >= MAX_DEPTH {
        return;
    }
    for sub in subdirs {
        if *budget == 0 {
            return;
        }
        walk(volume, sub, depth + 1, budget);
    }
}

/// Most file content read per entry.
const READ_LIMIT: u64 = 256 * 1024;
/// Bytes per content read; one more than a decmpfs chunk, so reads straddle
/// chunk boundaries.
const STEP: usize = 64 * 1024 + 1;

fn visit_entry(volume: &HfsPlus, node: NodeId) {
    if let Ok(list) = volume.xattrs(node) {
        for attr in list.iter().take(8) {
            let _ = volume.read_xattr(node, &attr.name, 64 * 1024);
        }
    }
    let _ = volume.open(node, Stream::Resource);
    let _ = volume.extents(node, Stream::Data);
    let _ = volume.extents(node, Stream::Resource);
    let Ok(content) = volume.open(node, Stream::Data) else {
        return;
    };
    let end = content.len().min(READ_LIMIT);
    let mut buf = vec![0u8; STEP];
    let mut pos = 0;
    while pos < end {
        let n = usize::try_from(end - pos).map_or(STEP, |left| left.min(STEP));
        if content.read_at(pos, &mut buf[..n]).is_err() {
            break;
        }
        pos += n as u64;
    }
}
