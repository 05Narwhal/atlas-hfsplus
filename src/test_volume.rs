// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! In-memory HFS+ and HFSX volumes for tests, built byte by byte.
//!
//! Layout, in 4096-byte allocation blocks: block 0 holds the volume header
//! at byte 1024; block 1 the extents overflow B-tree (a header node only);
//! then the data forks of files given content; then the catalog B-tree
//! (header node, one index node, the leaves); then one spare block, whose
//! last 1024 bytes hold the alternate header. Catalog records are sorted
//! with the volume's own name order, so lookups find them.
//!
//! Resource forks and fork-based attribute values take blocks after the
//! data forks. When any extended attribute is given, the attributes B-tree
//! follows the catalog, laid out like it. A fork-based attribute value gets
//! one extent per block, so a value of more than eight blocks continues in
//! extents records.
//!
//! The allocation file follows the catalog (and the attributes B-tree), just
//! before the spare block: a bitmap marking every block up to its own end
//! used, and, on a journaled volume, the journal info block, the journal and
//! the last block too. Its set bits agree with the header's free count.

use std::sync::Arc;

use super::attributes::fixtures::{attr_key, extents_data, fork_data, inline_data};
use super::btree::fixtures::{INDEX, LEAF, header, header_node, keyed, node, var_index};
use super::unicode::NameCompare;
use atlas_fs_core::io::block::{BlockSource, MemSource};

/// Allocation block size, and the catalog's node size.
pub const BLOCK: usize = 4096;

/// The root folder's ID.
pub const ROOT: u32 = 2;

const FOLDER_RECORD: u16 = 1;
const FILE_RECORD: u16 = 2;
const FOLDER_THREAD: u16 = 3;
const FILE_THREAD: u16 = 4;

/// `kIsInvisible`.
pub const FINDER_INVISIBLE: u16 = 0x4000;

/// Default dates, as raw HFS seconds: 1970-01-01 00:16:40 UTC and around it.
pub const CREATED: u32 = 2_082_844_800 + 500;
pub const MODIFIED: u32 = 2_082_844_800 + 1_000;
pub const ACCESSED: u32 = 2_082_844_800 + 1_500;

/// BSD modes.
pub const MODE_FILE: u16 = 0o100_644;
pub const MODE_DIR: u16 = 0o040_755;
pub const MODE_SYMLINK: u16 = 0o120_755;

/// `kHFSVolumeUnmountedBit`, `kHFSVolumeJournaledBit`.
const ATTR_UNMOUNTED: u32 = 1 << 8;
const ATTR_JOURNALED: u32 = 1 << 13;

/// `HFSJ`, the last-mounted version of a journaling mount.
pub const HFSJ: u32 = 0x4846_534A;
/// `kJIJournalInFSMask`, `kJIJournalOnOtherDeviceMask`, `kJIJournalNeedInitMask`.
pub const JIB_IN_FS: u32 = 1;
pub const JIB_ON_OTHER_DEVICE: u32 = 2;
pub const JIB_NEED_INIT: u32 = 4;
/// Journal header magics.
pub const JNL_MAGIC: u32 = 0x4A4E_4C78;
pub const JNL_OLD_MAGIC: u32 = 0x4A48_4452;
/// Block list header flags.
pub const BLHDR_CHECK: i32 = 1;
pub const BLHDR_FIRST: i32 = 2;
/// Smallest journal xnu accepts.
pub const MIN_JOURNAL: u64 = 256 * 1024;

/// The journal a [`VolumeBuilder`] reserves with [`VolumeBuilder::journaled`].
#[derive(Debug, Clone, Copy)]
pub struct JournalSetup {
    /// Total allocation blocks of the volume, fixed so two volumes with
    /// different content share one layout.
    pub total_blocks: u32,
    /// Journal size in bytes.
    pub size: u64,
    pub jib_flags: u32,
    pub last_mounted: u32,
    /// `kHFSVolumeUnmountedBit`.
    pub unmounted: bool,
}

impl JournalSetup {
    fn journal_blocks(&self) -> u32 {
        self.size.div_ceil(BLOCK as u64) as u32
    }

    /// First allocation block of the journal: the blocks before the spare
    /// last block.
    fn journal_block(&self) -> u32 {
        self.total_blocks - 1 - self.journal_blocks()
    }

    /// The journal info block, just before the journal.
    fn jib_block(&self) -> u32 {
        self.journal_block() - 1
    }
}

/// One block a [`TestList`] journals.
#[derive(Debug, Clone)]
pub struct TestBlock {
    /// Device block number in `jhdr_size` units; -1 is killed.
    pub bnum: i64,
    pub data: Vec<u8>,
    /// Recorded size; `data.len()` when `None`.
    pub bsize: Option<i32>,
    /// Recorded checksum; computed when `None` and the list checks
    /// checksums, else 0.
    pub cksum: Option<u32>,
}

impl TestBlock {
    pub fn new(bnum: i64, data: Vec<u8>) -> Self {
        Self {
            bnum,
            data,
            bsize: None,
            cksum: None,
        }
    }

    /// A killed block of `len` bytes: its data is in the journal but skipped.
    pub fn killed(len: usize) -> Self {
        Self {
            bnum: -1,
            data: vec![0xEE; len],
            bsize: None,
            cksum: Some(0),
        }
    }
}

/// One block list header and its blocks.
#[derive(Debug, Clone)]
pub struct TestList {
    pub flags: i32,
    pub seq: u32,
    pub blocks: Vec<TestBlock>,
    /// Overrides of the recorded fields.
    pub max_blocks: Option<u16>,
    pub num_blocks: Option<u16>,
    pub bytes_used: Option<i32>,
    /// Stores a wrong header checksum.
    pub bad_checksum: bool,
}

impl TestList {
    /// The first block list of a transaction, checking checksums.
    pub fn first(seq: u32, blocks: Vec<TestBlock>) -> Self {
        Self {
            flags: BLHDR_CHECK | BLHDR_FIRST,
            seq,
            blocks,
            max_blocks: None,
            num_blocks: None,
            bytes_used: None,
            bad_checksum: false,
        }
    }

    /// A later block list of the same transaction.
    pub fn more(seq: u32, blocks: Vec<TestBlock>) -> Self {
        Self {
            flags: BLHDR_CHECK,
            ..Self::first(seq, blocks)
        }
    }

    pub fn bad(self) -> Self {
        Self {
            bad_checksum: true,
            ..self
        }
    }
}

/// A journal to write into an image: header and ring of block lists.
#[derive(Debug, Clone)]
pub struct TestJournal {
    pub little_endian: bool,
    pub magic: u32,
    pub jhdr_size: i32,
    pub blhdr_size: i32,
    pub size: i64,
    /// Header `start`, and where the first list is written.
    pub start: i64,
    /// Header `end`; by default the position after the lists in range.
    pub end: Option<i64>,
    /// How many lists lie between `start` and `end`; the rest are written
    /// after `end` (transactions the header does not know about yet).
    pub in_range: Option<usize>,
    /// Stored header checksum; computed when `None`.
    pub header_checksum: Option<u32>,
    pub lists: Vec<TestList>,
}

impl TestJournal {
    /// An empty big-endian journal of `size` bytes with 512-byte headers
    /// and 4096-byte block list headers.
    pub fn new(size: u64) -> Self {
        Self {
            little_endian: false,
            magic: JNL_MAGIC,
            jhdr_size: 512,
            blhdr_size: 4096,
            size: size as i64,
            start: 512,
            end: None,
            in_range: None,
            header_checksum: None,
            lists: Vec::new(),
        }
    }

    fn put_u16(&self, b: &mut [u8], at: usize, v: u16) {
        let bytes = if self.little_endian {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        };
        b[at..at + 2].copy_from_slice(&bytes);
    }

    fn put_u32(&self, b: &mut [u8], at: usize, v: u32) {
        let bytes = if self.little_endian {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        };
        b[at..at + 4].copy_from_slice(&bytes);
    }

    fn put_u64(&self, b: &mut [u8], at: usize, v: u64) {
        let bytes = if self.little_endian {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        };
        b[at..at + 8].copy_from_slice(&bytes);
    }

    /// Writes the journal into `image` with its header at byte `base`.
    /// Returns the ring position of each list.
    pub fn write(&self, image: &mut [u8], base: usize) -> Vec<u64> {
        let jhdr = self.jhdr_size as u64;
        let size = self.size as u64;
        let mut positions = Vec::new();
        let mut pos = self.start as u64;
        if pos == size {
            pos = jhdr;
        }
        let mut end = pos;
        let put = |image: &mut [u8], mut pos: u64, bytes: &[u8]| -> u64 {
            for &b in bytes {
                image[base + pos as usize] = b;
                pos += 1;
                if pos == size {
                    pos = jhdr;
                }
            }
            pos
        };
        for (i, list) in self.lists.iter().enumerate() {
            positions.push(pos);
            let blhdr = self.blhdr_size as usize;
            let mut h = vec![0x5Au8; blhdr];
            h[..32].fill(0);
            let data_len: usize = list.blocks.iter().map(|b| b.data.len()).sum();
            let used = list.bytes_used.unwrap_or((blhdr + data_len) as i32);
            self.put_u16(
                &mut h,
                0,
                list.max_blocks.unwrap_or((blhdr / 16 - 1) as u16),
            );
            self.put_u16(
                &mut h,
                2,
                list.num_blocks.unwrap_or(list.blocks.len() as u16 + 1),
            );
            self.put_u32(&mut h, 4, used as u32);
            self.put_u32(&mut h, 12, list.flags as u32);
            // binfo[0].bnum holds a kernel pointer on disk: junk.
            self.put_u64(&mut h, 16, 0xFFFF_FF80_1234_5678);
            self.put_u32(&mut h, 28, list.seq);
            for (j, block) in list.blocks.iter().enumerate() {
                let at = 32 + 16 * j;
                assert!(at + 16 <= blhdr, "block list header too small");
                self.put_u64(&mut h, at, block.bnum as u64);
                let bsize = block.bsize.unwrap_or(block.data.len() as i32);
                self.put_u32(&mut h, at + 8, bsize as u32);
                let cksum = block.cksum.unwrap_or(if list.flags & BLHDR_CHECK != 0 {
                    super::journal::calc_checksum(&block.data)
                } else {
                    0
                });
                self.put_u32(&mut h, at + 12, cksum);
            }
            let mut sum = super::journal::calc_checksum(&h[..32]);
            if list.bad_checksum {
                sum ^= 1;
            }
            self.put_u32(&mut h, 8, sum);
            let start = pos;
            let mut p = put(image, pos, &h);
            for block in &list.blocks {
                p = put(image, p, &block.data);
            }
            let advance = if used > 0 {
                used as u64
            } else {
                (blhdr + data_len) as u64
            };
            pos = start + advance;
            while pos >= size {
                pos = pos - size + jhdr;
            }
            if self.in_range.is_none_or(|n| i < n) {
                end = pos;
            }
        }
        let end = self.end.unwrap_or(end as i64);

        let mut h = vec![0u8; 48];
        self.put_u32(&mut h, 0, self.magic);
        self.put_u32(&mut h, 4, 0x1234_5678);
        self.put_u64(&mut h, 8, self.start as u64);
        self.put_u64(&mut h, 16, end as u64);
        self.put_u64(&mut h, 24, self.size as u64);
        self.put_u32(&mut h, 32, self.blhdr_size as u32);
        self.put_u32(&mut h, 40, self.jhdr_size as u32);
        let seq = self.lists.last().map_or(1, |l| l.seq + 1);
        self.put_u32(&mut h, 44, seq);
        let sum = self
            .header_checksum
            .unwrap_or_else(|| super::journal::calc_checksum(&h[..44]));
        self.put_u32(&mut h, 36, sum);
        image[base..base + 48].copy_from_slice(&h);
        positions
    }
}

/// The `BLOCK`-byte blocks where `new` differs from `old`, as journal
/// blocks for images that start at `device_offset` in the partition.
pub fn changed_blocks(
    old: &[u8],
    new: &[u8],
    device_offset: u64,
    jhdr_size: u64,
) -> Vec<TestBlock> {
    assert_eq!(old.len(), new.len());
    old.chunks(BLOCK)
        .zip(new.chunks(BLOCK))
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (_, b))| {
            let device = device_offset + (i * BLOCK) as u64;
            assert_eq!(device % jhdr_size, 0);
            TestBlock::new((device / jhdr_size) as i64, b.to_vec())
        })
        .collect()
}

/// `name` as UTF-16 code units, stored as given (not decomposed).
pub fn units(name: &str) -> Vec<u16> {
    name.encode_utf16().collect()
}

/// A volume image as a block source.
pub fn source(bytes: Vec<u8>) -> Arc<dyn BlockSource> {
    Arc::new(MemSource::new(bytes))
}

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_be_bytes());
}

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_be_bytes());
}

/// Writes an `HFSPlusForkData` with one extent (or none when `blocks` is 0).
fn put_fork(b: &mut [u8], at: usize, logical: u64, start: u32, blocks: u32) {
    b[at..at + 8].copy_from_slice(&logical.to_be_bytes());
    put32(b, at + 12, blocks);
    if blocks > 0 {
        put32(b, at + 16, start);
        put32(b, at + 20, blocks);
    }
}

/// A file record's contents.
#[derive(Debug, Clone)]
pub struct FileSpec {
    pub mode: u16,
    /// Data fork logical size when `content` is `None`; no extents then.
    pub size: u64,
    /// Data fork bytes, written to blocks of their own.
    pub content: Option<Vec<u8>>,
    pub finder_flags: u16,
    /// `bsd.group_id`.
    pub group: u32,
    pub file_type: u32,
    pub creator: u32,
    /// `bsd.special`: the link reference of a hard link.
    pub special: u32,
    pub created: u32,
    pub modified: u32,
    pub accessed: u32,
}

impl FileSpec {
    /// A regular file of `size` bytes with no extents.
    pub fn regular(size: u64) -> Self {
        Self {
            mode: MODE_FILE,
            size,
            content: None,
            finder_flags: 0,
            group: 20,
            file_type: u32::from_be_bytes(*b"TEXT"),
            creator: u32::from_be_bytes(*b"ttxt"),
            special: 0,
            created: CREATED,
            modified: MODIFIED,
            accessed: ACCESSED,
        }
    }

    /// A symbolic link whose data fork is `target`.
    pub fn symlink(target: &[u8]) -> Self {
        Self {
            mode: MODE_SYMLINK,
            content: Some(target.to_vec()),
            file_type: u32::from_be_bytes(*b"slnk"),
            creator: u32::from_be_bytes(*b"rhap"),
            ..Self::regular(0)
        }
    }

    /// A file hard link to `iNode<reference>`.
    pub fn file_link(reference: u32) -> Self {
        Self {
            mode: 0o100_444,
            file_type: u32::from_be_bytes(*b"hlnk"),
            creator: u32::from_be_bytes(*b"hfs+"),
            special: reference,
            modified: MODIFIED + 7,
            ..Self::regular(0)
        }
    }

    /// A directory hard link to `dir_<reference>`.
    pub fn dir_link(reference: u32) -> Self {
        Self {
            file_type: u32::from_be_bytes(*b"fdrp"),
            creator: u32::from_be_bytes(*b"MACS"),
            ..Self::file_link(reference)
        }
    }

    /// With `kIsInvisible` set.
    pub fn invisible(self) -> Self {
        Self {
            finder_flags: FINDER_INVISIBLE,
            ..self
        }
    }

    /// With the content-modified date set.
    pub fn modified(self, modified: u32) -> Self {
        Self { modified, ..self }
    }
}

#[derive(Debug, Clone)]
enum Body {
    Folder {
        id: u32,
        finder_flags: u16,
    },
    File {
        id: u32,
        spec: FileSpec,
    },
    Thread {
        ty: u16,
        parent: u32,
        name: Vec<u16>,
    },
}

/// The value of an extended attribute given to [`VolumeBuilder::xattr`].
#[derive(Debug, Clone)]
pub enum XattrValue {
    /// Stored in the leaf record (`0x10`).
    Inline(Vec<u8>),
    /// Stored in a fork of its own (`0x20`), one extent per block, with
    /// extents records (`0x30`) past the eighth block.
    Fork(Vec<u8>),
}

/// Builds a volume. Every folder and file gets its thread record.
#[derive(Debug, Clone)]
pub struct VolumeBuilder {
    name: Vec<u16>,
    hfsx: Option<u8>,
    root_thread: bool,
    journal_dirty: bool,
    journal: Option<JournalSetup>,
    records: Vec<(u32, Vec<u16>, Body)>,
    /// BSD owner flags by file ID.
    owner_flags: Vec<(u32, u8)>,
    /// Resource fork bytes by file ID.
    resource_forks: Vec<(u32, Vec<u8>)>,
    /// Extended attributes: (node ID, name, value).
    xattrs: Vec<(u32, Vec<u16>, XattrValue)>,
}

impl VolumeBuilder {
    /// An HFS+ volume called `name` with an empty root.
    pub fn new(name: &str) -> Self {
        Self {
            name: units(name),
            hfsx: None,
            root_thread: true,
            journal_dirty: false,
            journal: None,
            records: Vec::new(),
            owner_flags: Vec::new(),
            resource_forks: Vec::new(),
            xattrs: Vec::new(),
        }
    }

    /// Sets the BSD owner flags of file `id` (`UF_COMPRESSED` is 0x20).
    pub fn owner_flags(&mut self, id: u32, flags: u8) -> &mut Self {
        self.owner_flags.push((id, flags));
        self
    }

    /// Gives file `id` a resource fork holding `bytes`.
    pub fn resource_fork(&mut self, id: u32, bytes: &[u8]) -> &mut Self {
        self.resource_forks.push((id, bytes.to_vec()));
        self
    }

    /// Gives node `id` the extended attribute `name`.
    pub fn xattr(&mut self, id: u32, name: &str, value: XattrValue) -> &mut Self {
        self.xattrs.push((id, units(name), value));
        self
    }

    /// A compressed file: `UF_COMPRESSED`, an empty data fork, `decmpfs`
    /// as its `com.apple.decmpfs` attribute (inline) and, when given, the
    /// resource fork holding the chunks.
    pub fn compressed_file(
        &mut self,
        parent: u32,
        name: &str,
        id: u32,
        decmpfs: &[u8],
        resource_fork: Option<&[u8]>,
    ) -> &mut Self {
        self.file(parent, name, id, FileSpec::regular(0))
            .owner_flags(id, 0x20)
            .xattr(
                id,
                "com.apple.decmpfs",
                XattrValue::Inline(decmpfs.to_vec()),
            );
        if let Some(fork) = resource_fork {
            self.resource_fork(id, fork);
        }
        self
    }

    /// Makes it HFSX with this catalog `keyCompareType` (0xCF or 0xBC).
    pub fn hfsx(&mut self, key_compare_type: u8) -> &mut Self {
        self.hfsx = Some(key_compare_type);
        self
    }

    /// Leaves out the root folder's thread record.
    pub fn without_root_thread(&mut self) -> &mut Self {
        self.root_thread = false;
        self
    }

    /// Marks the volume journaled and not cleanly unmounted, with no
    /// journal and no journaling last mount: macOS would not replay it.
    pub fn journal_dirty(&mut self) -> &mut Self {
        self.journal_dirty = true;
        self
    }

    /// Makes the volume `total_blocks` blocks long with an empty journal of
    /// `size` bytes in its last blocks (before the spare one) and the
    /// journal info block before it; journaled, last mounted by `HFSJ`, not
    /// unmounted cleanly. Two builders with the same arguments share the
    /// journal's place, so one's changed blocks can be journaled into the
    /// other's image.
    pub fn journaled(&mut self, total_blocks: u32, size: u64) -> &mut Self {
        self.journal = Some(JournalSetup {
            total_blocks,
            size,
            jib_flags: JIB_IN_FS,
            last_mounted: HFSJ,
            unmounted: false,
        });
        self
    }

    /// The journal setup, to change its flags. Panics unless `journaled`.
    pub fn journal_setup(&mut self) -> &mut JournalSetup {
        self.journal.as_mut().expect("journaled volume")
    }

    /// Volume byte of the journal (its header). Panics unless `journaled`.
    pub fn journal_offset(&self) -> u64 {
        let j = self.journal.expect("journaled volume");
        u64::from(j.journal_block()) * BLOCK as u64
    }

    /// A folder `id` called `name` (code units as stored) in `parent`.
    pub fn folder_units(
        &mut self,
        parent: u32,
        name: &[u16],
        id: u32,
        finder_flags: u16,
    ) -> &mut Self {
        self.records
            .push((parent, name.to_vec(), Body::Folder { id, finder_flags }));
        self.thread(FOLDER_THREAD, id, parent, name)
    }

    /// A folder `id` called `name` in `parent`.
    pub fn folder(&mut self, parent: u32, name: &str, id: u32) -> &mut Self {
        self.folder_units(parent, &units(name), id, 0)
    }

    /// A file `id` called `name` (code units as stored) in `parent`.
    pub fn file_units(&mut self, parent: u32, name: &[u16], id: u32, spec: FileSpec) -> &mut Self {
        self.records
            .push((parent, name.to_vec(), Body::File { id, spec }));
        self.thread(FILE_THREAD, id, parent, name)
    }

    /// A file `id` called `name` in `parent`.
    pub fn file(&mut self, parent: u32, name: &str, id: u32, spec: FileSpec) -> &mut Self {
        self.file_units(parent, &units(name), id, spec)
    }

    fn thread(&mut self, ty: u16, id: u32, parent: u32, name: &[u16]) -> &mut Self {
        self.records.push((
            id,
            Vec::new(),
            Body::Thread {
                ty,
                parent,
                name: name.to_vec(),
            },
        ));
        self
    }

    fn compare(&self) -> NameCompare {
        match self.hfsx {
            Some(0xBC) => NameCompare::Binary,
            _ => NameCompare::CaseFolding,
        }
    }

    /// The volume image.
    pub fn build(&self) -> Vec<u8> {
        let mut records = self.records.clone();
        records.push((
            1,
            self.name.clone(),
            Body::Folder {
                id: ROOT,
                finder_flags: 0,
            },
        ));
        if self.root_thread {
            records.push((
                ROOT,
                Vec::new(),
                Body::Thread {
                    ty: FOLDER_THREAD,
                    parent: 1,
                    name: self.name.clone(),
                },
            ));
        }
        let compare = self.compare();
        records.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| compare.compare(&a.1, &b.1)));

        // Data forks first: blocks 2 onwards.
        let mut content_blocks: Vec<u8> = Vec::new();
        let mut next_block: u32 = 2;
        let mut place = |bytes: &[u8]| -> (u32, u32) {
            let blocks = bytes.len().div_ceil(BLOCK);
            let start = next_block;
            next_block += blocks as u32;
            let mut padded = bytes.to_vec();
            padded.resize(blocks * BLOCK, 0);
            content_blocks.extend_from_slice(&padded);
            (start, blocks as u32)
        };
        let mut leaf_records: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for (parent, name, body) in &records {
            let data = match body {
                Body::Folder { id, finder_flags } => folder_data(*id, *finder_flags),
                Body::Thread { ty, parent, name } => thread_data(*ty, *parent, name),
                Body::File { id, spec } => {
                    let (logical, start, blocks) = match &spec.content {
                        Some(bytes) => {
                            let (start, blocks) = place(bytes);
                            (bytes.len() as u64, start, blocks)
                        }
                        None => (spec.size, 0, 0),
                    };
                    let mut d = file_data(*id, spec, logical, start, blocks);
                    for (_, flags) in self.owner_flags.iter().filter(|(i, _)| i == id) {
                        d[41] = *flags;
                    }
                    for (_, bytes) in self.resource_forks.iter().filter(|(i, _)| i == id) {
                        let (start, blocks) = place(bytes);
                        put_fork(&mut d, 168, bytes.len() as u64, start, blocks);
                    }
                    d
                }
            };
            leaf_records.push((key(*parent, name), data));
        }

        // Attribute records, with fork values placed after the forks.
        let mut xattrs = self.xattrs.clone();
        xattrs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut attr_records: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for (id, name, value) in &xattrs {
            match value {
                XattrValue::Inline(bytes) => {
                    attr_records.push((attr_key(*id, name, 0), inline_data(bytes)));
                }
                XattrValue::Fork(bytes) => {
                    let extents: Vec<(u32, u32)> = bytes
                        .chunks(BLOCK)
                        .map(|block| (place(block).0, 1))
                        .collect();
                    let mut groups = extents.chunks(8);
                    let first = groups.next().unwrap_or(&[]);
                    attr_records.push((
                        attr_key(*id, name, 0),
                        fork_data(bytes.len() as u64, extents.len() as u32, first),
                    ));
                    let mut mapped = first.len() as u32;
                    for group in groups {
                        attr_records.push((attr_key(*id, name, mapped), extents_data(group)));
                        mapped += group.len() as u32;
                    }
                }
            }
        }

        let catalog = catalog_nodes(&leaf_records, self.hfsx.unwrap_or(0));
        let catalog_start = next_block;
        let catalog_blocks = (catalog.len() / BLOCK) as u32;
        let attr_tree = if attr_records.is_empty() {
            Vec::new()
        } else {
            btree_nodes(&attr_records, 266, 0)
        };
        let attributes_start = catalog_start + catalog_blocks;
        let attributes_blocks = (attr_tree.len() / BLOCK) as u32;
        // The allocation file comes last, sized to the volume it describes.
        let allocation_start = attributes_start + attributes_blocks;
        let mut allocation_blocks = 1;
        let total_for = |allocation_blocks: u32| match &self.journal {
            Some(j) => j.total_blocks,
            None => allocation_start + allocation_blocks + 1,
        };
        loop {
            let need = (total_for(allocation_blocks) as usize)
                .div_ceil(8)
                .div_ceil(BLOCK) as u32;
            if need <= allocation_blocks {
                break;
            }
            allocation_blocks = need;
        }
        let used_blocks = allocation_start + allocation_blocks;
        let total_blocks = total_for(allocation_blocks);
        if let Some(j) = &self.journal {
            assert!(
                used_blocks <= j.jib_block(),
                "volume too small for its journal"
            );
        }

        let mut extents_header = header(0, 0, 0, 0, 1);
        extents_header.node_size = BLOCK as u16;
        extents_header.max_key_length = 10;
        let extents = header_node(&extents_header);

        let mut vh = vec![0u8; 512];
        let (signature, version) = if self.hfsx.is_some() {
            (*b"HX", 5)
        } else {
            (*b"H+", 4)
        };
        vh[0..2].copy_from_slice(&signature);
        put16(&mut vh, 2, version);
        let mut free_blocks = 1;
        let attributes = if let Some(j) = &self.journal {
            put32(&mut vh, 8, j.last_mounted);
            put32(&mut vh, 12, j.jib_block());
            free_blocks = j.jib_block() - used_blocks;
            ATTR_JOURNALED | if j.unmounted { ATTR_UNMOUNTED } else { 0 }
        } else if self.journal_dirty {
            put32(&mut vh, 12, 1); // journalInfoBlock
            ATTR_JOURNALED
        } else {
            ATTR_UNMOUNTED
        };
        put32(&mut vh, 4, attributes);
        put32(&mut vh, 40, BLOCK as u32);
        put32(&mut vh, 44, total_blocks);
        put32(&mut vh, 48, free_blocks);
        put32(&mut vh, 64, 1000); // next catalog ID
        put_fork(
            &mut vh,
            112,
            u64::from(allocation_blocks) * BLOCK as u64,
            allocation_start,
            allocation_blocks,
        );
        put_fork(&mut vh, 192, BLOCK as u64, 1, 1);
        put_fork(
            &mut vh,
            272,
            catalog.len() as u64,
            catalog_start,
            catalog_blocks,
        );
        if attributes_blocks > 0 {
            put_fork(
                &mut vh,
                352,
                attr_tree.len() as u64,
                attributes_start,
                attributes_blocks,
            );
        }

        let mut image = vec![0u8; BLOCK];
        image.extend_from_slice(&extents);
        image.extend_from_slice(&content_blocks);
        image.extend_from_slice(&catalog);
        image.extend_from_slice(&attr_tree);
        image.resize(total_blocks as usize * BLOCK, 0);
        // The bitmap, most significant bit first: everything up to the
        // allocation file's end is used, and so are the journal info block
        // and the journal (which run to the last block). The spare last
        // block of an unjournaled volume is the one free block, as
        // `free_blocks` says.
        let used = |block: u32| {
            block < used_blocks
                || self
                    .journal
                    .as_ref()
                    .is_some_and(|j| block >= j.jib_block())
        };
        let bitmap = allocation_start as usize * BLOCK;
        for block in (0..total_blocks).filter(|&b| used(b)) {
            image[bitmap + block as usize / 8] |= 0x80 >> (block % 8);
        }
        image[1024..1536].copy_from_slice(&vh);
        let alternate = image.len() - 1024;
        image[alternate..alternate + 512].copy_from_slice(&vh);
        if let Some(j) = &self.journal {
            let jib = j.jib_block() as usize * BLOCK;
            put32(&mut image, jib, j.jib_flags);
            let offset = u64::from(j.journal_block()) * BLOCK as u64;
            image[jib + 36..jib + 44].copy_from_slice(&offset.to_be_bytes());
            image[jib + 44..jib + 52].copy_from_slice(&j.size.to_be_bytes());
            TestJournal::new(j.size).write(&mut image, offset as usize);
        }
        image
    }
}

/// Catalog key bytes (after the key length).
pub fn key(parent: u32, name: &[u16]) -> Vec<u8> {
    let mut k = parent.to_be_bytes().to_vec();
    k.extend_from_slice(&(name.len() as u16).to_be_bytes());
    for u in name {
        k.extend_from_slice(&u.to_be_bytes());
    }
    k
}

fn dates(d: &mut [u8], created: u32, modified: u32, accessed: u32) {
    put32(d, 12, created);
    put32(d, 16, modified);
    put32(d, 20, modified); // attributeModDate
    put32(d, 24, accessed);
}

/// An `HFSPlusCatalogFolder`.
pub fn folder_data(id: u32, finder_flags: u16) -> Vec<u8> {
    let mut d = vec![0u8; 88];
    put16(&mut d, 0, FOLDER_RECORD);
    put32(&mut d, 8, id);
    dates(&mut d, CREATED, MODIFIED, ACCESSED);
    put32(&mut d, 32, 501);
    put16(&mut d, 42, MODE_DIR);
    put16(&mut d, 56, finder_flags);
    d
}

/// An `HFSPlusCatalogFile` whose data fork is one extent (none if `blocks` is 0).
fn file_data(id: u32, spec: &FileSpec, logical: u64, start: u32, blocks: u32) -> Vec<u8> {
    let mut d = vec![0u8; 248];
    put16(&mut d, 0, FILE_RECORD);
    put32(&mut d, 8, id);
    dates(&mut d, spec.created, spec.modified, spec.accessed);
    put32(&mut d, 32, 501);
    put32(&mut d, 36, spec.group);
    put16(&mut d, 42, spec.mode);
    put32(&mut d, 44, spec.special);
    put32(&mut d, 48, spec.file_type);
    put32(&mut d, 52, spec.creator);
    put16(&mut d, 56, spec.finder_flags);
    put_fork(&mut d, 88, logical, start, blocks);
    d
}

/// An `HFSPlusCatalogThread`.
fn thread_data(ty: u16, parent: u32, name: &[u16]) -> Vec<u8> {
    let mut d = vec![0u8; 10];
    put16(&mut d, 0, ty);
    put32(&mut d, 4, parent);
    put16(&mut d, 8, name.len() as u16);
    for u in name {
        d.extend_from_slice(&u.to_be_bytes());
    }
    d
}

/// The catalog B-tree: header node, an index node over the leaves, then the
/// leaves, each `BLOCK` bytes. `records` are (key, data) in key order.
fn catalog_nodes(records: &[(Vec<u8>, Vec<u8>)], key_compare_type: u8) -> Vec<u8> {
    btree_nodes(records, 516, key_compare_type)
}

/// A B-tree laid out as [`catalog_nodes`] lays out the catalog: header
/// node, one index node, the leaves. `records` are (key, data) in key order.
fn btree_nodes(
    records: &[(Vec<u8>, Vec<u8>)],
    max_key_length: u16,
    key_compare_type: u8,
) -> Vec<u8> {
    let mut leaves: Vec<Vec<Vec<u8>>> = vec![Vec::new()];
    let mut used = 14 + 2;
    for (k, d) in records {
        let rec = keyed(k, d);
        let cost = rec.len() + 2;
        if used + cost > BLOCK {
            leaves.push(Vec::new());
            used = 14 + 2;
        }
        used += cost;
        leaves.last_mut().unwrap().push(rec);
    }
    let count = leaves.len() as u32;
    let mut h = header(2, 1, 2, 1 + count, 2 + count);
    h.node_size = BLOCK as u16;
    h.max_key_length = max_key_length;
    h.leaf_records = records.len() as u32;
    h.key_compare_type = key_compare_type;
    let mut index = Vec::new();
    let mut nodes = Vec::new();
    for (i, leaf) in leaves.iter().enumerate() {
        let number = 2 + i as u32;
        // The first key of the leaf: skip the key length field.
        let first = &leaf[0];
        let key_len = usize::from(u16::from_be_bytes([first[0], first[1]]));
        index.push(var_index(&first[2..2 + key_len], number));
        let forward = if i + 1 < leaves.len() { number + 1 } else { 0 };
        let backward = if i == 0 { 0 } else { number - 1 };
        nodes.push(node(BLOCK, LEAF, 1, forward, backward, leaf));
    }
    let mut out = header_node(&h);
    out.extend_from_slice(&node(BLOCK, INDEX, 2, 0, 0, &index));
    for n in nodes {
        out.extend_from_slice(&n);
    }
    out
}

/// `volume` inside a classic HFS wrapper: the master directory block's
/// embedded extent starts at block 1 of 4096-byte blocks, after 8 sectors.
pub fn wrap(volume: &[u8]) -> Vec<u8> {
    assert_eq!(volume.len() % BLOCK, 0);
    let offset = 8 * 512 + BLOCK;
    let mut out = vec![0u8; offset + volume.len()];
    let mdb = 1024;
    out[mdb..mdb + 2].copy_from_slice(b"BD");
    put32(&mut out, mdb + 0x14, BLOCK as u32);
    put16(&mut out, mdb + 0x1C, 8);
    out[mdb + 0x7C..mdb + 0x7E].copy_from_slice(b"H+");
    put16(&mut out, mdb + 0x7E, 1);
    put16(&mut out, mdb + 0x80, (volume.len() / BLOCK) as u16);
    out[offset..].copy_from_slice(volume);
    out
}

// Finder flags as TN1150 gives them (`FInfo.fdFlags`, `DInfo.frFlags`),
// written out here rather than taken from the code under test.
pub const HAS_BUNDLE: u16 = 0x2000;
pub const IS_ALIAS: u16 = 0x8000;
/// `kColor` holds the label in bits 1 to 3.
pub const fn color(label: u16) -> u16 {
    label << 1
}

pub fn code(c: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*c)
}

/// Files and folders with Finder flags and type codes, a file hard link,
/// and a housekeeping folder with a file three levels inside it.
pub fn finder_volume() -> VolumeBuilder {
    let mut v = VolumeBuilder::new("Classic");
    v.folder_units(
        ROOT,
        &"Bundle Folder".encode_utf16().collect::<Vec<_>>(),
        16,
        HAS_BUNDLE | color(2),
    )
    .file(
        ROOT,
        "ReadMe",
        17,
        FileSpec {
            finder_flags: color(7),
            ..FileSpec::regular(10)
        },
    )
    .file(
        ROOT,
        "My Alias",
        18,
        FileSpec {
            finder_flags: IS_ALIAS | FINDER_INVISIBLE,
            file_type: 0,
            creator: 0,
            ..FileSpec::regular(0)
        },
    )
    .file(
        ROOT,
        "SimpleText",
        19,
        FileSpec {
            finder_flags: HAS_BUNDLE,
            file_type: code(b"APPL"),
            ..FileSpec::regular(100)
        },
    )
    .folder_units(ROOT, &super::links::FILE_PRIVATE_FOLDER, 30, 0x5000)
    .file(
        30,
        "iNode7",
        31,
        FileSpec {
            finder_flags: color(3),
            file_type: code(b"PICT"),
            ..FileSpec::regular(77)
        },
    )
    .file(
        ROOT,
        "Picture link",
        50,
        FileSpec {
            finder_flags: color(5),
            ..FileSpec::file_link(7)
        },
    )
    .folder(ROOT, ".Spotlight-V100", 60)
    .folder(60, "Store-V2", 61)
    .folder(61, "a", 62)
    .file(62, "deep", 63, FileSpec::regular(3))
    .folder(ROOT, "Sibling", 64)
    .file(64, "x", 65, FileSpec::regular(4));
    v
}
