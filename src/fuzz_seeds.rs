// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Writes seed inputs for the fuzz targets in `fuzz/`.

use std::fs;
use std::path::Path;

use super::compression::fixtures::{
    content, header, lzfse, lzvn, lzvn_literals, offsets_fork, pieces, zlib, zlib_fork,
};
use super::links::{DIR_PRIVATE_FOLDER, FILE_PRIVATE_FOLDER};
use super::test_volume::{
    FileSpec, MIN_JOURNAL, ROOT, TestJournal, TestList, VolumeBuilder, XattrValue, changed_blocks,
    wrap,
};

/// Offset of the volume header in a volume image.
const HEADER_OFFSET: usize = 1024;
/// Size of the volume header.
const HEADER_LEN: usize = 512;

/// Volumes covering what the builder supports: plain HFS+, HFSX in both key
/// orders, nested folders, hard links, a symlink, hidden files, a dirty
/// journal, a missing root thread and a classic HFS wrapper.
fn volumes() -> Vec<(&'static str, Vec<u8>)> {
    let mut plain = VolumeBuilder::new("Plain");
    plain
        .file(ROOT, "readme.txt", 16, FileSpec::regular(12))
        .file(
            ROOT,
            "data.bin",
            17,
            FileSpec::regular(0).modified(super::test_volume::MODIFIED + 99),
        )
        .file(
            ROOT,
            "content.txt",
            18,
            FileSpec {
                content: Some(b"hello, disk atlas".to_vec()),
                ..FileSpec::regular(0)
            },
        );

    let mut hfsx_fold = VolumeBuilder::new("Fold");
    hfsx_fold
        .hfsx(0xCF)
        .file(ROOT, "Readme", 16, FileSpec::regular(1))
        .file(ROOT, "readme2", 17, FileSpec::regular(2));

    let mut hfsx_binary = VolumeBuilder::new("Binary");
    hfsx_binary
        .hfsx(0xBC)
        .file(ROOT, "Readme", 16, FileSpec::regular(1))
        .file(ROOT, "readme", 17, FileSpec::regular(2));

    let mut nested = VolumeBuilder::new("Nested");
    nested
        .folder(ROOT, "Docs", 16)
        .folder(16, "Sub", 17)
        .folder(17, "Deeper", 18)
        .file(18, "deep.txt", 19, FileSpec::regular(3))
        .file(17, "mid.txt", 20, FileSpec::regular(4))
        .file(16, "top.txt", 21, FileSpec::regular(5))
        .folder(ROOT, "Empty", 22);

    // A classic Mac file with a type, a creator, a label and a custom icon.
    let mut classic = VolumeBuilder::new("Classic");
    classic
        .folder(ROOT, "Letters", 16)
        .file(
            16,
            "Read Me",
            17,
            FileSpec {
                finder_flags: 0x0404,
                content: Some(b"hello from 1998".to_vec()),
                ..FileSpec::regular(0)
            },
        )
        .file(16, "notes.txt", 18, FileSpec::regular(2048));

    let mut links = VolumeBuilder::new("Links");
    links
        .file(ROOT, "plain", 16, FileSpec::regular(7))
        .file(ROOT, ".dotfile", 17, FileSpec::regular(1))
        .file(ROOT, "Invisible", 18, FileSpec::regular(7).invisible())
        .file(ROOT, "symlink", 19, FileSpec::symlink(b"plain"))
        .folder(ROOT, "Docs", 20)
        .file(20, "a.txt", 21, FileSpec::regular(1))
        .folder_units(ROOT, &FILE_PRIVATE_FOLDER, 30, 0x5000)
        .file(30, "iNode100", 31, FileSpec::regular(555))
        .folder(30, "iNode101", 32)
        .file(30, "iNode102", 33, FileSpec::file_link(100))
        .file(30, "iNode103", 34, FileSpec::symlink(b"../target"))
        .folder_units(ROOT, &DIR_PRIVATE_FOLDER, 40, 0x5000)
        .folder(40, "dir_200", 41)
        .file(41, "inside.txt", 42, FileSpec::regular(9))
        .file(ROOT, "hard.txt", 50, FileSpec::file_link(100))
        .file(ROOT, "hard dir", 51, FileSpec::dir_link(200))
        .file(ROOT, "missing", 52, FileSpec::file_link(999))
        .file(ROOT, "to folder", 53, FileSpec::file_link(101))
        .file(ROOT, "chain", 54, FileSpec::file_link(102))
        .file(ROOT, "link to symlink", 55, FileSpec::file_link(103))
        .file(
            ROOT,
            "invisible link",
            56,
            FileSpec::file_link(100).invisible(),
        )
        .file(20, "hard in docs", 57, FileSpec::file_link(100));

    let mut journal = VolumeBuilder::new("Journal");
    journal
        .journal_dirty()
        .file(ROOT, "file", 16, FileSpec::regular(1));

    let mut no_thread = VolumeBuilder::new("NoThread");
    no_thread
        .without_root_thread()
        .file(ROOT, "file", 16, FileSpec::regular(1));

    let mut compressed = VolumeBuilder::new("Compressed");
    for (id, name, decmpfs, fork) in compressed_files() {
        compressed.compressed_file(ROOT, name, id, &decmpfs, fork.as_deref());
    }
    compressed
        .file(ROOT, "tagged", 90, FileSpec::regular(3))
        .xattr(90, "user.a", XattrValue::Inline(b"abc".to_vec()))
        .xattr(90, "big", XattrValue::Fork(content(9 * 4096 + 123)));

    let wrapped = wrap(&plain.build());
    let mut all = vec![
        ("compressed", compressed.build()),
        ("plain", plain.build()),
        ("hfsx_fold", hfsx_fold.build()),
        ("hfsx_binary", hfsx_binary.build()),
        ("nested", nested.build()),
        ("classic", classic.build()),
        ("links", links.build()),
        ("journal_dirty", journal.build()),
        ("no_root_thread", no_thread.build()),
        ("wrapped", wrapped),
    ];
    all.extend(journaled_volumes());
    all
}

/// Volumes with a dirty journal that replays: the blocks where a newer
/// volume differs from the image, journaled big-endian, little-endian, and
/// with the ring wrapping inside a block list.
fn journaled_volumes() -> Vec<(&'static str, Vec<u8>)> {
    let mut old = VolumeBuilder::new("Journaled");
    old.journaled(128, MIN_JOURNAL)
        .file(ROOT, "old.txt", 16, FileSpec::regular(5))
        .folder(ROOT, "Docs", 17);
    let mut new = old.clone();
    new.file(
        17,
        "new.txt",
        18,
        FileSpec {
            content: Some(b"replayed".to_vec()),
            ..FileSpec::regular(0)
        },
    );
    let (old_image, new_image) = (old.build(), new.build());
    let blocks = changed_blocks(&old_image, &new_image, 0, 512);
    let base = usize::try_from(old.journal_offset()).expect("journal offset fits usize");
    let journaled = |little_endian: bool, start: i64| {
        let journal = TestJournal {
            little_endian,
            start,
            lists: vec![
                TestList::first(7, blocks[..1].to_vec()),
                TestList::first(8, blocks[1..].to_vec()),
            ],
            ..TestJournal::new(MIN_JOURNAL)
        };
        let mut image = old_image.clone();
        journal.write(&mut image, base);
        image
    };
    let near_end = i64::try_from(MIN_JOURNAL).expect("journal size fits i64") - 2048;
    vec![
        ("journal_replay", journaled(false, 512)),
        ("journal_little_endian", journaled(true, 512)),
        ("journal_wrapped", journaled(true, near_end)),
    ]
}

/// (file id, name, decmpfs attribute, resource fork).
type CompressedSeed = (u32, &'static str, Vec<u8>, Option<Vec<u8>>);

/// One compressed file of each supported kind.
fn compressed_files() -> Vec<CompressedSeed> {
    let small = content(3000);
    // Two chunks, so the table and a chunk boundary are exercised.
    let two = content(70 * 1024);
    let inline = |ty: u32, data: &[u8], payload: &[u8]| {
        let mut x = header(ty, data.len() as u64);
        x.extend_from_slice(payload);
        x
    };
    let mut raw = vec![0xCC];
    raw.extend_from_slice(&small);
    let zlib_chunks: Vec<_> = pieces(&two).into_iter().map(zlib).collect();
    let lzvn_chunks: Vec<_> = pieces(&two).into_iter().map(lzvn_literals).collect();
    let lzfse_chunks: Vec<_> = pieces(&two).into_iter().map(lzfse).collect();
    let raw_chunks: Vec<_> = pieces(&two)
        .into_iter()
        .map(|p| [&[0xCC][..], p].concat())
        .collect();
    let rsrc = |ty: u32| header(ty, two.len() as u64);
    vec![
        (80, "type1", inline(1, &small, &small), None),
        (81, "type3", inline(3, &small, &zlib(&small)), None),
        (82, "type7", inline(7, &small, &lzvn(&small)), None),
        (83, "type9", inline(9, &small, &raw), None),
        (84, "type11", inline(11, &small, &lzfse(&small)), None),
        (85, "type4", rsrc(4), Some(zlib_fork(&zlib_chunks))),
        (86, "type8", rsrc(8), Some(offsets_fork(&lzvn_chunks))),
        (87, "type10", rsrc(10), Some(offsets_fork(&raw_chunks))),
        (88, "type12", rsrc(12), Some(offsets_fork(&lzfse_chunks))),
    ]
}

/// A `decmpfs` target input: the attribute's length (u16 LE), the
/// attribute, then the resource fork.
fn decmpfs_input(decmpfs: &[u8], fork: Option<&[u8]>) -> Vec<u8> {
    let len = u16::try_from(decmpfs.len()).expect("seed attribute fits u16");
    let mut input = len.to_le_bytes().to_vec();
    input.extend_from_slice(decmpfs);
    input.extend_from_slice(fork.unwrap_or(&[]));
    input
}

/// Writes seed files for the targets in `fuzz/` into the directory named by
/// `ATLAS_FUZZ_SEEDS`, one subfolder per target. Does nothing when the
/// variable is unset. Run it with
/// `cargo test --lib write_fuzz_seeds -- --ignored`.
#[test]
#[ignore = "writes files; run by the fuzz CI job"]
fn write_fuzz_seeds() {
    let Some(root) = std::env::var_os("ATLAS_FUZZ_SEEDS") else {
        return;
    };
    let root = Path::new(&root);
    let mount_dir = root.join("hfsplus_mount");
    let header_dir = root.join("hfsplus_volume_header");
    let decmpfs_dir = root.join("decmpfs");
    let journal_dir = root.join("hfsplus_journal");
    for dir in [&mount_dir, &header_dir, &decmpfs_dir, &journal_dir] {
        fs::create_dir_all(dir).expect("create seed directory");
    }
    for (name, image) in journaled_volumes() {
        fs::write(journal_dir.join(name), &image).expect("write journal seed");
    }
    for (_, name, decmpfs, fork) in compressed_files() {
        fs::write(
            decmpfs_dir.join(name),
            decmpfs_input(&decmpfs, fork.as_deref()),
        )
        .expect("write decmpfs seed");
    }
    for (name, image) in volumes() {
        fs::write(mount_dir.join(name), &image).expect("write mount seed");
        // A header slice for the plain parse; wrappers have none at 1024.
        if let Some(header) = image.get(HEADER_OFFSET..HEADER_OFFSET + HEADER_LEN) {
            fs::write(header_dir.join(name), header).expect("write header seed");
        }
        // The whole image too: the header target also locates wrappers.
        fs::write(header_dir.join(format!("{name}.img")), &image).expect("write image seed");
    }
}

/// Every journaled seed replays both transactions, so the journal target
/// starts on the valid path.
#[test]
fn journaled_seeds_replay() {
    use super::journal::JournalState;
    use super::test_volume::source;

    for (name, image) in journaled_volumes() {
        let fs = super::mount(source(image)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            fs.journal(),
            &JournalState::Replayed {
                transactions: 2,
                stopped_early: false
            },
            "{name}"
        );
    }
}

/// Every compressed seed decodes to its original content, so the seeds
/// start the fuzzer on the valid path of each kind.
#[test]
fn compressed_seeds_decode() {
    use std::sync::Arc;

    use atlas_fs_core::io::block::{BlockSource, MemSource};

    for (_, name, decmpfs, fork) in compressed_files() {
        let fork: Arc<dyn BlockSource> = Arc::new(MemSource::new(fork.unwrap_or_default()));
        let source = super::compression::open(&decmpfs, 1, &mut || Ok(Arc::clone(&fork)))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let expected = if source.len() == 3000 {
            content(3000)
        } else {
            content(70 * 1024)
        };
        let mut got = vec![0u8; expected.len()];
        source
            .read_at(0, &mut got)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got, expected, "{name}");
    }
}
