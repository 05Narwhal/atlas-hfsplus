// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Fuzzes `VolumeHeader::parse` and the wrapper/header locator.

#![no_main]

use atlas_fs_core::io::block::MemSource;
use atlas_hfsplus::volume_header::{VolumeHeader, locate, read_header};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(header) = VolumeHeader::parse(data) {
        let _ = header.is_hfsx();
        let _ = header.is_journaled();
        let _ = header.unmounted_cleanly();
        let _ = header.volume_bytes();
    }
    // The same bytes as a whole device: wrapper detection and the header at
    // offset 1024.
    let source = MemSource::new(data);
    let _ = locate(&source);
    let _ = read_header(&source);
});
