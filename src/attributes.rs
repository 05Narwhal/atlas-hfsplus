// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Attributes B-tree: extended attributes and inline data.
//!
//! The attributes file (CNID 8) is a B-tree keyed by (file ID, attribute
//! name, start block). A value is stored inline in its leaf record (`0x10`),
//! or in a fork of its own whose descriptor is the record (`0x20`); a fork
//! with more than eight extents continues in extents records (`0x30`) keyed
//! by the same file and name, with the start block set to the blocks mapped
//! by the records before it (TN1150 "Attributes File"; xnu `hfs_xattr.c`).
//!
//! Key order is xnu's `hfs_attrkeycompare`: file ID, then the name compared
//! as raw UTF-16 code units (shorter first on a common prefix, no case
//! folding even on case-insensitive volumes), then start block.

use std::cmp::Ordering;
use std::sync::Arc;

use super::btree::{BTree, LeafCursor, Record};
use super::bytes::{be_u16, be_u32, bytes};
use super::extents::{ExtentRecord, ForkData, ForkSource, ForkType, Geometry, fork_map};
use super::extents::{FORK_DATA_LEN, parse_extent_record};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::io::block::BlockSource;

/// CNID of the attributes file.
pub const ATTRIBUTES_FILE_ID: u32 = 8;

/// Longest attribute name, in UTF-16 code units (`kHFSMaxAttrNameLen`).
pub const MAX_ATTR_NAME_UNITS: usize = 127;

const STRUCTURE: &str = "attributes B-tree";

/// Record types.
const INLINE_DATA: u32 = 0x10;
const FORK_DATA: u32 = 0x20;
const EXTENTS: u32 = 0x30;

/// Key bytes after `keyLength` before the name: pad, fileID, startBlock,
/// nameLength.
const KEY_HEADER_LEN: usize = 12;
/// Bytes of an inline record before its data: type, two reserved words, size.
const INLINE_HEADER_LEN: usize = 16;
/// Bytes of a fork or extents record before its descriptor: type, reserved.
const FORK_HEADER_LEN: usize = 8;
/// Size of an `HFSPlusExtentRecord`.
const EXTENT_RECORD_LEN: usize = 64;

/// One extended attribute of a file: its name and the size of its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xattr {
    /// The name, decoded from UTF-16; unpaired surrogates become U+FFFD.
    pub name: String,
    /// Size of the value in bytes.
    pub size: u64,
}

fn corrupt(location: Location, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(STRUCTURE, location, kind))
}

fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// The location of the record `cursor` handed out last.
fn record_location(cursor: &LeafCursor) -> Location {
    cursor
        .position()
        .map_or(Location::Unspecified, |(node, record)| Location::Record {
            node,
            record,
        })
}

/// An attribute key read in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AttrKey<'a> {
    pub(super) file_id: u32,
    pub(super) start_block: u32,
    /// The name as stored: big-endian UTF-16.
    pub(super) name_be: &'a [u8],
}

/// Parses an attribute key (the bytes after `keyLength`).
pub(super) fn parse_key(key: &[u8]) -> Result<AttrKey<'_>, CorruptKind> {
    let truncated = |needed: usize| CorruptKind::Truncated {
        needed: len64(needed),
        available: len64(key.len()),
    };
    let file_id = be_u32(key, 2).ok_or_else(|| truncated(KEY_HEADER_LEN))?;
    let start_block = be_u32(key, 6).ok_or_else(|| truncated(KEY_HEADER_LEN))?;
    let units = be_u16(key, 10).ok_or_else(|| truncated(KEY_HEADER_LEN))?;
    if usize::from(units) > MAX_ATTR_NAME_UNITS {
        return Err(CorruptKind::OutOfRange {
            field: "attribute name length",
            value: u64::from(units),
            limit_name: "longest attribute name",
            limit: len64(MAX_ATTR_NAME_UNITS),
        });
    }
    let len = usize::from(units).saturating_mul(2);
    let name_be = bytes(key, KEY_HEADER_LEN, len)
        .ok_or_else(|| truncated(KEY_HEADER_LEN.saturating_add(len)))?;
    Ok(AttrKey {
        file_id,
        start_block,
        name_be,
    })
}

/// The code units of a stored name.
fn units_be(name_be: &[u8]) -> impl Iterator<Item = u16> + '_ {
    name_be
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u16::from_be_bytes(*p))
}

impl AttrKey<'_> {
    /// `hfs_attrkeycompare`: file ID, name as binary code units, start block.
    fn order(&self, file_id: u32, name: &[u16], start_block: u32) -> Ordering {
        self.file_id
            .cmp(&file_id)
            .then_with(|| units_be(self.name_be).cmp(name.iter().copied()))
            .then_with(|| self.start_block.cmp(&start_block))
    }

    fn is(&self, file_id: u32, name: &[u16], start_block: u32) -> bool {
        self.order(file_id, name, start_block) == Ordering::Equal
    }
}

/// An attribute name as UTF-16 code units, on the stack.
struct AttrName {
    units: [u16; MAX_ATTR_NAME_UNITS],
    len: usize,
}

impl AttrName {
    /// `None` when `name` is longer than an attribute name can be.
    fn new(name: &str) -> Option<Self> {
        let mut units = [0u16; MAX_ATTR_NAME_UNITS];
        let mut len = 0usize;
        for u in name.encode_utf16() {
            *units.get_mut(len)? = u;
            len = len.checked_add(1)?;
        }
        Some(Self { units, len })
    }

    fn units(&self) -> &[u16] {
        self.units.get(..self.len).unwrap_or(&[])
    }
}

/// The type of an attribute record.
fn record_type(data: &[u8]) -> Result<u32, CorruptKind> {
    be_u32(data, 0).ok_or(CorruptKind::Truncated {
        needed: 4,
        available: len64(data.len()),
    })
}

/// An attribute's value as its first record describes it.
#[derive(Debug, Clone, Copy)]
pub(super) enum Value<'a> {
    /// The value bytes, in the leaf record.
    Inline(&'a [u8]),
    /// The value's fork.
    Fork(ForkData),
}

impl Value<'_> {
    fn size(&self) -> u64 {
        match self {
            Self::Inline(data) => len64(data.len()),
            Self::Fork(fork) => fork.logical_size,
        }
    }
}

/// Parses the first record of an attribute. Extents records (`0x30`) are
/// continuations and not values; they give `Ok(None)`.
pub(super) fn parse_value(data: &[u8]) -> Result<Option<Value<'_>>, CorruptKind> {
    match record_type(data)? {
        INLINE_DATA => {
            let size = be_u32(data, 12).ok_or(CorruptKind::Truncated {
                needed: len64(INLINE_HEADER_LEN),
                available: len64(data.len()),
            })?;
            let size = usize::try_from(size).unwrap_or(usize::MAX);
            let value = bytes(data, INLINE_HEADER_LEN, size).ok_or(CorruptKind::OutOfRange {
                field: "inline attribute size",
                value: len64(size),
                limit_name: "record length",
                limit: len64(data.len().saturating_sub(INLINE_HEADER_LEN)),
            })?;
            Ok(Some(Value::Inline(value)))
        }
        FORK_DATA => {
            let fork = data
                .get(FORK_HEADER_LEN..)
                .and_then(ForkData::parse)
                .ok_or(CorruptKind::Truncated {
                    needed: len64(FORK_HEADER_LEN.saturating_add(FORK_DATA_LEN)),
                    available: len64(data.len()),
                })?;
            Ok(Some(Value::Fork(fork)))
        }
        EXTENTS => Ok(None),
        other => Err(CorruptKind::InvalidValue {
            field: "attribute record type",
            value: u64::from(other),
        }),
    }
}

/// Parses an extents (`0x30`) record.
fn parse_extents(data: &[u8]) -> Result<ExtentRecord, CorruptKind> {
    let ty = record_type(data)?;
    if ty != EXTENTS {
        return Err(CorruptKind::InvalidValue {
            field: "attribute extents record type",
            value: u64::from(ty),
        });
    }
    data.get(FORK_HEADER_LEN..)
        .and_then(parse_extent_record)
        .ok_or(CorruptKind::Truncated {
            needed: len64(FORK_HEADER_LEN.saturating_add(EXTENT_RECORD_LEN)),
            available: len64(data.len()),
        })
}

/// Blocks in `record` up to its first empty extent.
fn record_blocks(record: &ExtentRecord) -> Option<u32> {
    record
        .iter()
        .take_while(|e| e.block_count != 0)
        .try_fold(0u32, |sum, e| sum.checked_add(e.block_count))
}

/// The attributes B-tree of a volume.
///
/// Cheap to clone (a [`BTree`] clone and an `Arc` to the volume), `Send +
/// Sync`, and every method takes `&self`.
#[derive(Clone)]
pub struct Attributes {
    pub(super) tree: BTree,
    volume: Arc<dyn BlockSource>,
    geometry: Geometry,
}

impl Attributes {
    /// Opens the attributes B-tree described by the volume header's
    /// attributes fork, mapped through the extents overflow tree.
    ///
    /// A fork of logical size 0 means the volume has no attributes file:
    /// `Ok(None)`. A fork that is present but fails to map or whose header
    /// node is damaged is an error.
    pub fn open(
        volume: Arc<dyn BlockSource>,
        geometry: Geometry,
        fork: &ForkData,
        extents: &BTree,
    ) -> Result<Option<Self>> {
        if fork.logical_size == 0 {
            return Ok(None);
        }
        let map = fork_map(
            geometry,
            ATTRIBUTES_FILE_ID,
            ForkType::Data,
            fork,
            Some(extents),
        )?;
        let tree = BTree::open(
            Arc::new(ForkSource::new(Arc::clone(&volume), Arc::new(map))),
            STRUCTURE,
        )?;
        Ok(Some(Self {
            tree,
            volume,
            geometry,
        }))
    }

    /// The value of attribute `name` of `file_id`, read into memory.
    ///
    /// `Ok(None)` when the file has no such attribute (or the name is longer
    /// than an attribute name can be). A value larger than `cap` bytes fails
    /// with `Corrupt` / `LimitExceeded` before anything is allocated.
    pub fn get(&self, file_id: u32, name: &str, cap: usize) -> Result<Option<Vec<u8>>> {
        let Some(name) = AttrName::new(name) else {
            return Ok(None);
        };
        let mut out = Vec::new();
        Ok(self
            .read_value(file_id, name.units(), cap, true, &mut out)?
            .map(|_| out))
    }

    /// Reads the first `min(len, size)` bytes of attribute `name` of
    /// `file_id` into `out` (cleared first, its allocation reused) and
    /// returns the value's full size, or `Ok(None)` when there is no such
    /// attribute.
    pub fn read_head(
        &self,
        file_id: u32,
        name: &str,
        len: usize,
        out: &mut Vec<u8>,
    ) -> Result<Option<u64>> {
        let Some(name) = AttrName::new(name) else {
            out.clear();
            return Ok(None);
        };
        self.read_value(file_id, name.units(), len, false, out)
    }

    /// Finds attribute `name` of `file_id` and reads its value into `out`:
    /// all of it (`whole`, failing when it is over `limit`), or its first
    /// `limit` bytes. Returns the value's size.
    fn read_value(
        &self,
        file_id: u32,
        name: &[u16],
        limit: usize,
        whole: bool,
        out: &mut Vec<u8>,
    ) -> Result<Option<u64>> {
        out.clear();
        let mut cursor = self.tree.seek(&mut |key: &[u8]| {
            let key = parse_key(key).map_err(|kind| corrupt(Location::Unspecified, kind))?;
            Ok(key.order(file_id, name, 0))
        })?;
        let fork = {
            let Some(rec) = cursor.next_record()? else {
                return Ok(None);
            };
            let found = first_record(rec, file_id, name);
            let value = match found {
                Ok(Some(value)) => value,
                Ok(None) => return Ok(None),
                Err(kind) => return Err(corrupt(record_location(&cursor), kind)),
            };
            let size = value.size();
            let target = if whole {
                if size > len64(limit) {
                    return Err(corrupt(
                        record_location(&cursor),
                        CorruptKind::LimitExceeded {
                            what: "attribute value size",
                            limit: len64(limit),
                        },
                    ));
                }
                size
            } else {
                size.min(len64(limit))
            };
            // `target` is at most `limit`, a usize.
            let target = usize::try_from(target).unwrap_or(limit);
            match value {
                Value::Inline(data) => {
                    out.extend_from_slice(data.get(..target).unwrap_or(data));
                    return Ok(Some(size));
                }
                Value::Fork(fork) => {
                    out.resize(target, 0);
                    fork
                }
            }
        };
        self.read_fork(file_id, name, &fork, &mut cursor, out)?;
        Ok(Some(fork.logical_size))
    }

    /// Fills `out` from the start of an attribute's fork: its first eight
    /// extents, then the extents records that follow the fork record.
    fn read_fork(
        &self,
        file_id: u32,
        name: &[u16],
        fork: &ForkData,
        cursor: &mut LeafCursor,
        out: &mut [u8],
    ) -> Result<()> {
        let at = Location::Object(u64::from(file_id));
        let block_size = u64::from(self.geometry.block_size);
        let covered = u64::from(fork.total_blocks).saturating_mul(block_size);
        if fork.logical_size > covered {
            return Err(corrupt(
                at,
                CorruptKind::OutOfRange {
                    field: "attribute fork logical size",
                    value: fork.logical_size,
                    limit_name: "bytes in its blocks",
                    limit: covered,
                },
            ));
        }
        if out.is_empty() {
            return Ok(());
        }
        let mut extents = fork.extents;
        let mut mapped: u32 = 0;
        let mut pos: usize = 0;
        let limit = self.tree.header().leaf_records;
        let mut continuations: u32 = 0;
        // Bounded: every pass maps at least one more block (or fails), the
        // mapped blocks never pass `total_blocks`, and the extents records
        // read are capped at the tree's record count.
        loop {
            let blocks = record_blocks(&extents).filter(|&b| b > 0).ok_or_else(|| {
                corrupt(
                    at,
                    CorruptKind::Other("attribute extents record maps no blocks"),
                )
            })?;
            let total = mapped
                .checked_add(blocks)
                .filter(|&t| t <= fork.total_blocks)
                .ok_or_else(|| {
                    corrupt(
                        at,
                        CorruptKind::OutOfRange {
                            field: "attribute fork mapped blocks",
                            value: u64::from(mapped).saturating_add(u64::from(blocks)),
                            limit_name: "fork blocks",
                            limit: u64::from(fork.total_blocks),
                        },
                    )
                })?;
            let piece_bytes = u64::from(blocks).saturating_mul(block_size);
            let rest = out.get_mut(pos..).unwrap_or(&mut []);
            let n = usize::try_from(piece_bytes)
                .unwrap_or(usize::MAX)
                .min(rest.len());
            if let Some(dst) = rest.get_mut(..n) {
                let piece = ForkData {
                    logical_size: len64(n),
                    clump_size: 0,
                    total_blocks: blocks,
                    extents,
                };
                let map = fork_map(self.geometry, file_id, ForkType::Data, &piece, None)?;
                ForkSource::new(Arc::clone(&self.volume), Arc::new(map)).read_at(0, dst)?;
            }
            pos = pos.saturating_add(n);
            mapped = total;
            if pos >= out.len() {
                return Ok(());
            }
            if continuations >= limit {
                return Err(corrupt(
                    at,
                    CorruptKind::LimitExceeded {
                        what: "attribute extents records",
                        limit: u64::from(limit),
                    },
                ));
            }
            continuations = continuations.saturating_add(1);
            let next = {
                let missing =
                    || corrupt(at, CorruptKind::Other("attribute extents record missing"));
                let rec = cursor.next_record()?.ok_or_else(missing)?;
                let key = parse_key(rec.key);
                let key = match key {
                    Ok(key) => key,
                    Err(kind) => return Err(corrupt(record_location(cursor), kind)),
                };
                if !key.is(file_id, name, mapped) {
                    return Err(missing());
                }
                parse_extents(rec.data)
            };
            extents = next.map_err(|kind| corrupt(record_location(cursor), kind))?;
        }
    }

    /// The attributes of `file_id`: names and value sizes, in key order.
    ///
    /// Bounded by the tree's leaf record count. A malformed record fails
    /// the call.
    pub fn list(&self, file_id: u32) -> Result<Vec<Xattr>> {
        let mut cursor = self.tree.seek(&mut |key: &[u8]| {
            let key = parse_key(key).map_err(|kind| corrupt(Location::Unspecified, kind))?;
            // Before every record of the file: (file_id, "", 0) is the
            // least key the file can have.
            Ok(key.order(file_id, &[], 0))
        })?;
        let limit = self.tree.header().leaf_records;
        let mut seen: u32 = 0;
        let mut out = Vec::new();
        loop {
            let Some(rec) = cursor.next_record()? else {
                return Ok(out);
            };
            let step = list_step(rec, file_id);
            match step.map_err(|kind| corrupt(record_location(&cursor), kind))? {
                ListStep::End => return Ok(out),
                ListStep::Skip => {}
                ListStep::Entry(x) => out.push(x),
            }
            seen = seen.saturating_add(1);
            if seen > limit {
                return Err(corrupt(
                    record_location(&cursor),
                    CorruptKind::LimitExceeded {
                        what: "attribute records of one file",
                        limit: u64::from(limit),
                    },
                ));
            }
        }
    }
}

/// The value an attribute's first record holds, when `rec` is the record
/// (`file_id`, `name`, start block 0).
fn first_record<'a>(
    rec: Record<'a>,
    file_id: u32,
    name: &[u16],
) -> Result<Option<Value<'a>>, CorruptKind> {
    let key = parse_key(rec.key)?;
    if !key.is(file_id, name, 0) {
        return Ok(None);
    }
    match parse_value(rec.data)? {
        Some(value) => Ok(Some(value)),
        None => Err(CorruptKind::Other(
            "attribute starts with an extents record",
        )),
    }
}

enum ListStep {
    Entry(Xattr),
    Skip,
    End,
}

fn list_step(rec: Record<'_>, file_id: u32) -> Result<ListStep, CorruptKind> {
    let key = parse_key(rec.key)?;
    if key.file_id != file_id {
        return Ok(ListStep::End);
    }
    let Some(value) = parse_value(rec.data)? else {
        return Ok(ListStep::Skip);
    };
    let name = char::decode_utf16(units_be(key.name_be))
        .map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER))
        .collect();
    Ok(ListStep::Entry(Xattr {
        name,
        size: value.size(),
    }))
}

#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod fixtures {
    //! Byte builders for attribute records, shared with the test volume.

    /// Key bytes (after `keyLength`) of an attribute record.
    pub fn attr_key(file_id: u32, name: &[u16], start_block: u32) -> Vec<u8> {
        let mut k = vec![0u8, 0];
        k.extend_from_slice(&file_id.to_be_bytes());
        k.extend_from_slice(&start_block.to_be_bytes());
        k.extend_from_slice(&(name.len() as u16).to_be_bytes());
        for u in name {
            k.extend_from_slice(&u.to_be_bytes());
        }
        k
    }

    /// An inline data record (`0x10`).
    pub fn inline_data(value: &[u8]) -> Vec<u8> {
        let mut d = vec![0u8; 16];
        d[0..4].copy_from_slice(&0x10u32.to_be_bytes());
        d[12..16].copy_from_slice(&(value.len() as u32).to_be_bytes());
        d.extend_from_slice(value);
        if d.len() % 2 == 1 {
            d.push(0);
        }
        d
    }

    /// Eight extent descriptors from up to eight (start, count) pairs.
    pub fn extent_bytes(extents: &[(u32, u32)]) -> Vec<u8> {
        assert!(extents.len() <= 8);
        let mut b = Vec::new();
        for i in 0..8 {
            let (s, c) = extents.get(i).copied().unwrap_or((0, 0));
            b.extend_from_slice(&s.to_be_bytes());
            b.extend_from_slice(&c.to_be_bytes());
        }
        b
    }

    /// A fork data record (`0x20`).
    pub fn fork_data(logical: u64, total_blocks: u32, extents: &[(u32, u32)]) -> Vec<u8> {
        let mut d = 0x20u32.to_be_bytes().to_vec();
        d.extend_from_slice(&[0; 4]);
        d.extend_from_slice(&logical.to_be_bytes());
        d.extend_from_slice(&[0; 4]); // clump size
        d.extend_from_slice(&total_blocks.to_be_bytes());
        d.extend_from_slice(&extent_bytes(extents));
        d
    }

    /// An extents record (`0x30`).
    pub fn extents_data(extents: &[(u32, u32)]) -> Vec<u8> {
        let mut d = 0x30u32.to_be_bytes().to_vec();
        d.extend_from_slice(&[0; 4]);
        d.extend_from_slice(&extent_bytes(extents));
        d
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::btree::fixtures::{INDEX, LEAF, header, header_node, image, keyed, node, var_index};
    use crate::test_volume::units;
    use atlas_fs_core::io::block::MemSource;

    const NODE: usize = 512;
    const BS: usize = 512;
    const FILE: u32 = 20;

    /// A 64-block volume where every byte of block `n` is `n`.
    fn volume() -> Arc<dyn BlockSource> {
        let bytes: Vec<u8> = (0..64u8).flat_map(|b| vec![b; BS]).collect();
        Arc::new(MemSource::new(bytes))
    }

    fn geometry() -> Geometry {
        Geometry {
            block_size: BS as u32,
            total_blocks: 64,
        }
    }

    fn rec(file: u32, name: &str, start: u32, data: &[u8]) -> Vec<u8> {
        keyed(&attr_key(file, &units(name), start), data)
    }

    /// Opens a tree of one leaf, or of an index over two leaves when
    /// `split` is given (records before `split` in the first leaf).
    fn attrs_from(records: &[Vec<u8>], split: Option<usize>) -> Attributes {
        let nodes = match split {
            None => {
                let mut h = header(1, 1, 1, 1, 2);
                h.leaf_records = records.len() as u32;
                h.max_key_length = 266;
                vec![header_node(&h), node(NODE, LEAF, 1, 0, 0, records)]
            }
            Some(at) => {
                let (a, b) = records.split_at(at);
                let mut h = header(2, 1, 2, 3, 4);
                h.leaf_records = records.len() as u32;
                h.max_key_length = 266;
                let first_key = |r: &Vec<u8>| {
                    let n = usize::from(u16::from_be_bytes([r[0], r[1]]));
                    r[2..2 + n].to_vec()
                };
                let index = vec![
                    var_index(&first_key(&a[0]), 2),
                    var_index(&first_key(&b[0]), 3),
                ];
                vec![
                    header_node(&h),
                    node(NODE, INDEX, 2, 0, 0, &index),
                    node(NODE, LEAF, 1, 3, 0, a),
                    node(NODE, LEAF, 1, 0, 2, b),
                ]
            }
        };
        Attributes {
            tree: BTree::open(image(&nodes), STRUCTURE).unwrap(),
            volume: volume(),
            geometry: geometry(),
        }
    }

    fn attrs(records: &[Vec<u8>]) -> Attributes {
        attrs_from(records, None)
    }

    fn corruption<T: std::fmt::Debug>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    fn sample() -> Vec<Vec<u8>> {
        vec![
            rec(FILE - 1, "z", 0, &inline_data(b"other file")),
            rec(FILE, "a", 0, &inline_data(b"")),
            rec(FILE, "com.apple.FinderInfo", 0, &inline_data(&[7u8; 32])),
            // 700 bytes in blocks 10 and 11.
            rec(FILE, "fork", 0, &fork_data(700, 2, &[(10, 2)])),
            rec(FILE, "user.x", 0, &inline_data(b"hello")),
            rec(FILE + 1, "a", 0, &inline_data(b"next file")),
        ]
    }

    #[test]
    fn inline_value() {
        let a = attrs(&sample());
        assert_eq!(a.get(FILE, "user.x", 100).unwrap().unwrap(), b"hello");
        assert_eq!(a.get(FILE, "a", 100).unwrap().unwrap(), b"");
        assert_eq!(
            a.get(FILE, "com.apple.FinderInfo", 32).unwrap().unwrap(),
            [7u8; 32]
        );
        assert_eq!(a.get(FILE - 1, "z", 100).unwrap().unwrap(), b"other file");
        assert_eq!(a.get(FILE + 1, "a", 100).unwrap().unwrap(), b"next file");
        // Head reads.
        let mut buf = Vec::with_capacity(4);
        assert_eq!(a.read_head(FILE, "user.x", 3, &mut buf).unwrap(), Some(5));
        assert_eq!(buf, b"hel");
        assert_eq!(a.read_head(FILE, "user.x", 30, &mut buf).unwrap(), Some(5));
        assert_eq!(buf, b"hello");
    }

    #[test]
    fn fork_value() {
        let a = attrs(&sample());
        let v = a.get(FILE, "fork", 1000).unwrap().unwrap();
        assert_eq!(v.len(), 700);
        assert!(v[..512].iter().all(|&b| b == 10));
        assert!(v[512..].iter().all(|&b| b == 11));
        let mut buf = Vec::new();
        assert_eq!(a.read_head(FILE, "fork", 16, &mut buf).unwrap(), Some(700));
        assert_eq!(buf, [10u8; 16]);
    }

    #[test]
    fn fork_value_continued_by_extents_records() {
        // Ten one-block extents: eight in the fork record, then two records
        // of one extent each. Physical blocks 20..30.
        let first: Vec<(u32, u32)> = (0..8).map(|i| (20 + i, 1)).collect();
        let size = 9 * BS as u64 + 100;
        let records = vec![
            rec(FILE, "big", 0, &fork_data(size, 10, &first)),
            rec(FILE, "big", 8, &extents_data(&[(28, 1)])),
            rec(FILE, "big", 9, &extents_data(&[(29, 1)])),
            rec(FILE, "c", 0, &inline_data(b"after")),
        ];
        for split in [None, Some(1), Some(2), Some(3)] {
            let a = attrs_from(&records, split);
            let v = a.get(FILE, "big", 1 << 20).unwrap().unwrap();
            assert_eq!(v.len() as u64, size);
            for (i, &b) in v.iter().enumerate() {
                assert_eq!(b as usize, 20 + i / BS, "byte {i} split {split:?}");
            }
            assert_eq!(a.get(FILE, "c", 10).unwrap().unwrap(), b"after");
            // The listing skips the continuation records.
            let names: Vec<_> = a.list(FILE).unwrap().into_iter().map(|x| x.name).collect();
            assert_eq!(names, ["big", "c"]);
        }
    }

    #[test]
    fn missing_or_misplaced_continuation_fails() {
        let first: Vec<(u32, u32)> = (0..8).map(|i| (20 + i, 1)).collect();
        let size = 9 * BS as u64;
        // No continuation at all.
        let a = attrs(&[rec(FILE, "big", 0, &fork_data(size, 9, &first))]);
        let c = corruption(a.get(FILE, "big", 1 << 20));
        assert_eq!(
            c.kind,
            CorruptKind::Other("attribute extents record missing")
        );
        // Wrong start block.
        let a = attrs(&[
            rec(FILE, "big", 0, &fork_data(size, 9, &first)),
            rec(FILE, "big", 7, &extents_data(&[(28, 1)])),
        ]);
        assert!(a.get(FILE, "big", 1 << 20).is_err());
        // A continuation that maps nothing.
        let a = attrs(&[
            rec(FILE, "big", 0, &fork_data(size, 9, &first)),
            rec(FILE, "big", 8, &extents_data(&[])),
        ]);
        let c = corruption(a.get(FILE, "big", 1 << 20));
        assert_eq!(
            c.kind,
            CorruptKind::Other("attribute extents record maps no blocks")
        );
        // A continuation record of the wrong type.
        let a = attrs(&[
            rec(FILE, "big", 0, &fork_data(size, 9, &first)),
            rec(FILE, "big", 8, &inline_data(b"x")),
        ]);
        assert!(matches!(
            corruption(a.get(FILE, "big", 1 << 20)).kind,
            CorruptKind::InvalidValue { .. }
        ));
        // A continuation mapping past total_blocks.
        let a = attrs(&[
            rec(FILE, "big", 0, &fork_data(size, 9, &first)),
            rec(FILE, "big", 8, &extents_data(&[(28, 2)])),
        ]);
        assert!(matches!(
            corruption(a.get(FILE, "big", 1 << 20)).kind,
            CorruptKind::OutOfRange { .. }
        ));
    }

    #[test]
    fn fork_value_bad_sizes() {
        // Logical size past the blocks.
        let a = attrs(&[rec(FILE, "f", 0, &fork_data(2000, 2, &[(10, 2)]))]);
        assert!(matches!(
            corruption(a.get(FILE, "f", 1 << 20)).kind,
            CorruptKind::OutOfRange { .. }
        ));
        // Extent past the volume.
        let a = attrs(&[rec(FILE, "f", 0, &fork_data(100, 1, &[(64, 1)]))]);
        assert!(a.get(FILE, "f", 1 << 20).is_err());
        // No extents at all for a non-empty value.
        let a = attrs(&[rec(FILE, "f", 0, &fork_data(100, 1, &[]))]);
        assert!(a.get(FILE, "f", 1 << 20).is_err());
        // An empty fork value reads as empty.
        let a = attrs(&[rec(FILE, "f", 0, &fork_data(0, 0, &[]))]);
        assert_eq!(a.get(FILE, "f", 10).unwrap().unwrap(), b"");
    }

    #[test]
    fn listing() {
        let a = attrs(&sample());
        assert_eq!(
            a.list(FILE).unwrap(),
            vec![
                Xattr {
                    name: "a".into(),
                    size: 0
                },
                Xattr {
                    name: "com.apple.FinderInfo".into(),
                    size: 32
                },
                Xattr {
                    name: "fork".into(),
                    size: 700
                },
                Xattr {
                    name: "user.x".into(),
                    size: 5
                },
            ]
        );
        assert!(a.list(999).unwrap().is_empty());
        assert!(a.list(0).unwrap().is_empty());
        assert_eq!(a.list(FILE + 1).unwrap().len(), 1);
        // Across a leaf boundary.
        let a = attrs_from(&sample(), Some(3));
        assert_eq!(a.list(FILE).unwrap().len(), 4);
        assert_eq!(a.get(FILE, "fork", 1000).unwrap().unwrap().len(), 700);
    }

    #[test]
    fn listing_bad_utf16_is_lossy() {
        let key = attr_key(FILE, &[0x61, 0xD800, 0x62], 0);
        let a = attrs(&[keyed(&key, &inline_data(b"v"))]);
        assert_eq!(a.list(FILE).unwrap()[0].name, "a\u{FFFD}b");
    }

    #[test]
    fn missing_attribute() {
        let a = attrs(&sample());
        assert_eq!(a.get(FILE, "nope", 100).unwrap(), None);
        assert_eq!(a.get(FILE, "USER.X", 100).unwrap(), None);
        assert_eq!(a.get(FILE, "user.", 100).unwrap(), None);
        assert_eq!(a.get(FILE, "user.xy", 100).unwrap(), None);
        assert_eq!(a.get(FILE + 2, "a", 100).unwrap(), None);
        assert_eq!(a.get(0, "a", 100).unwrap(), None);
        let long = "x".repeat(MAX_ATTR_NAME_UNITS + 1);
        assert_eq!(a.get(FILE, &long, 100).unwrap(), None);
        let mut buf = vec![1, 2, 3];
        assert_eq!(a.read_head(FILE, "nope", 16, &mut buf).unwrap(), None);
        assert!(buf.is_empty());
        // An empty tree.
        let a = attrs(&[]);
        assert_eq!(a.get(FILE, "a", 100).unwrap(), None);
        assert!(a.list(FILE).unwrap().is_empty());
    }

    #[test]
    fn missing_tree() {
        let h = header(0, 0, 0, 0, 1);
        let ext = BTree::open(image(&[header_node(&h)]), "extents").unwrap();
        let empty = ForkData::default();
        assert!(
            Attributes::open(volume(), geometry(), &empty, &ext)
                .unwrap()
                .is_none()
        );
        // Present but unreadable: an error.
        let bad = ForkData {
            logical_size: 512,
            clump_size: 0,
            total_blocks: 1,
            extents: [Default::default(); 8],
        };
        assert!(Attributes::open(volume(), geometry(), &bad, &ext).is_err());
    }

    #[test]
    fn value_over_the_cap() {
        let a = attrs(&sample());
        let c = corruption(a.get(FILE, "user.x", 4));
        assert!(matches!(
            c.kind,
            CorruptKind::LimitExceeded { limit: 4, .. }
        ));
        assert!(a.get(FILE, "user.x", 5).is_ok());
        let c = corruption(a.get(FILE, "fork", 699));
        assert!(matches!(c.kind, CorruptKind::LimitExceeded { .. }));
        // A huge claimed fork size fails before any allocation.
        let a = attrs(&[rec(FILE, "f", 0, &fork_data(u64::MAX, 2, &[(10, 2)]))]);
        assert!(a.get(FILE, "f", 1 << 20).is_err());
    }

    #[test]
    fn malformed_keys_and_records() {
        // Name length over 127.
        let mut key = attr_key(FILE, &units("a"), 0);
        key[10..12].copy_from_slice(&128u16.to_be_bytes());
        let a = attrs(&[keyed(&key, &inline_data(b"v"))]);
        assert!(matches!(
            corruption(a.get(FILE, "a", 10)).kind,
            CorruptKind::OutOfRange { .. }
        ));
        assert!(a.list(FILE).is_err());
        // Name length past the key.
        let mut key = attr_key(FILE, &units("a"), 0);
        key[10..12].copy_from_slice(&5u16.to_be_bytes());
        let a = attrs(&[keyed(&key, &inline_data(b"v"))]);
        assert!(matches!(
            corruption(a.get(FILE, "a", 10)).kind,
            CorruptKind::Truncated { .. }
        ));
        // A key too short for its header.
        let a = attrs(&[keyed(&[0, 0, 0, 0, 0, FILE as u8], &inline_data(b"v"))]);
        assert!(a.get(FILE, "a", 10).is_err());
        // A record too short for its type.
        let a = attrs(&[rec(FILE, "a", 0, &[0, 0])]);
        assert!(matches!(
            corruption(a.get(FILE, "a", 10)).kind,
            CorruptKind::Truncated { .. }
        ));
        // An inline record whose size runs past it.
        let mut d = inline_data(b"abcd");
        d[12..16].copy_from_slice(&100u32.to_be_bytes());
        let a = attrs(&[rec(FILE, "a", 0, &d)]);
        assert!(matches!(
            corruption(a.get(FILE, "a", 1000)).kind,
            CorruptKind::OutOfRange { .. }
        ));
        assert!(a.list(FILE).is_err());
        // An inline record cut before its size.
        let a = attrs(&[rec(FILE, "a", 0, &[0, 0, 0, 0x10, 0, 0, 0, 0])]);
        assert!(a.get(FILE, "a", 10).is_err());
        // A fork record cut short.
        let a = attrs(&[rec(FILE, "a", 0, &fork_data(1, 1, &[(10, 1)])[..40])]);
        assert!(matches!(
            corruption(a.get(FILE, "a", 10)).kind,
            CorruptKind::Truncated { .. }
        ));
        // Unknown record type.
        let mut d = inline_data(b"v");
        d[3] = 0x40;
        let a = attrs(&[rec(FILE, "a", 0, &d)]);
        assert!(matches!(
            corruption(a.get(FILE, "a", 10)).kind,
            CorruptKind::InvalidValue { value: 0x40, .. }
        ));
        assert!(a.list(FILE).is_err());
        // An attribute that starts with an extents record.
        let a = attrs(&[rec(FILE, "a", 0, &extents_data(&[(1, 1)]))]);
        assert!(a.get(FILE, "a", 10).is_err());
    }

    #[test]
    fn key_order_is_binary() {
        let k = attr_key(5, &units("Ab"), 3);
        let key = parse_key(&k).unwrap();
        assert_eq!(key.order(5, &units("Ab"), 3), Ordering::Equal);
        assert_eq!(key.order(5, &units("Ab"), 4), Ordering::Less);
        assert_eq!(key.order(5, &units("Ab"), 0), Ordering::Greater);
        // Binary: 'A' (0x41) < 'a' (0x61); no case folding.
        assert_eq!(key.order(5, &units("ab"), 0), Ordering::Less);
        // Shorter name first on a common prefix.
        assert_eq!(key.order(5, &units("A"), 9), Ordering::Greater);
        assert_eq!(key.order(5, &units("Abc"), 0), Ordering::Less);
        // File ID first.
        assert_eq!(key.order(6, &units(""), 0), Ordering::Less);
        assert_eq!(key.order(4, &units("zzz"), 99), Ordering::Greater);
    }

    #[test]
    fn attributes_is_send_sync_clone() {
        fn assert_traits<T: Send + Sync + Clone + 'static>() {}
        assert_traits::<Attributes>();
    }
}
