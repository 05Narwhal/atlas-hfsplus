// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Fork reader and the extents overflow file.
//!
//! A fork is a run of allocation blocks described by extents: the first
//! eight in its [`ForkData`], any further ones in the extents overflow
//! B-tree.

use std::sync::Arc;

use super::btree::BTree;
use super::bytes::{be_u32, be_u64, u8_at};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::io::block::{BlockSource, check_range};

/// Size of an on-disk `HFSPlusForkData` in bytes.
pub const FORK_DATA_LEN: usize = 80;

/// Size of an on-disk `HFSPlusExtentDescriptor` in bytes.
pub const EXTENT_LEN: usize = 8;

/// One contiguous run of allocation blocks (`HFSPlusExtentDescriptor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExtentDescriptor {
    /// First allocation block of the run.
    pub start_block: u32,
    /// Number of allocation blocks in the run. Zero ends an extent list.
    pub block_count: u32,
}

impl ExtentDescriptor {
    /// Parses one descriptor from the start of `b`. `None` if `b` is shorter
    /// than [`EXTENT_LEN`].
    pub fn parse(b: &[u8]) -> Option<Self> {
        Some(Self {
            start_block: be_u32(b, 0)?,
            block_count: be_u32(b, 4)?,
        })
    }
}

/// Eight extent descriptors (`HFSPlusExtentRecord`), as stored in a fork
/// descriptor and in each extents overflow record.
pub type ExtentRecord = [ExtentDescriptor; 8];

/// Parses an `HFSPlusExtentRecord` (64 bytes) from the start of `b`.
pub fn parse_extent_record(b: &[u8]) -> Option<ExtentRecord> {
    let mut out = [ExtentDescriptor::default(); 8];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = ExtentDescriptor::parse(b.get(i.checked_mul(EXTENT_LEN)?..)?)?;
    }
    Some(out)
}

/// A fork descriptor (`HFSPlusForkData`): the fork's size and its first
/// eight extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ForkData {
    /// Size of the fork's content in bytes.
    pub logical_size: u64,
    /// Allocation hint; not needed for reading.
    pub clump_size: u32,
    /// Allocation blocks the fork occupies, over all its extents.
    pub total_blocks: u32,
    /// The first eight extents.
    pub extents: ExtentRecord,
}

impl ForkData {
    /// Parses a fork descriptor from the start of `b`. `None` if `b` is
    /// shorter than [`FORK_DATA_LEN`]. The values are not validated here;
    /// [`resolve`](self) checks them against the volume.
    pub fn parse(b: &[u8]) -> Option<Self> {
        Some(Self {
            logical_size: be_u64(b, 0)?,
            clump_size: be_u32(b, 8)?,
            total_blocks: be_u32(b, 12)?,
            extents: parse_extent_record(b.get(16..)?)?,
        })
    }
}

/// Which fork of a file. The values are the `forkType` byte of an extents
/// overflow key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForkType {
    /// The data fork.
    Data = 0x00,
    /// The resource fork.
    Resource = 0xFF,
}

/// The `forkType` byte of a fork type.
fn fork_type_byte(t: ForkType) -> u8 {
    match t {
        ForkType::Data => 0x00,
        ForkType::Resource => 0xFF,
    }
}

/// Structure name for errors in a fork's mapping or reads.
const FORK: &str = "fork";

/// Structure name for errors in extents overflow records.
const OVERFLOW: &str = "extents overflow B-tree";

/// The volume's allocation geometry, from the volume header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Allocation block size in bytes.
    pub block_size: u32,
    /// Allocation blocks in the volume.
    pub total_blocks: u32,
}

/// One run of a fork: where it starts in the fork, in allocation blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    logical_block: u32,
    extent: ExtentDescriptor,
}

/// A fork's extents as sorted runs: logical block -> physical block.
///
/// Runs are contiguous in the fork from block 0 to the mapped end; there are
/// no holes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForkMap {
    file_id: u32,
    block_size: u32,
    logical_size: u64,
    mapped_blocks: u32,
    runs: Vec<Run>,
}

impl ForkMap {
    /// Size of the fork's content in bytes, from its fork descriptor.
    pub fn logical_size(&self) -> u64 {
        self.logical_size
    }

    /// Bytes covered by the extents. May be less than `logical_size` on a
    /// damaged volume.
    fn mapped_bytes(&self) -> u64 {
        u64::from(self.mapped_blocks).saturating_mul(u64::from(self.block_size))
    }

    /// The runs in fork order: (logical byte offset, extent).
    pub fn runs(&self) -> impl Iterator<Item = (u64, ExtentDescriptor)> + '_ {
        let block_size = u64::from(self.block_size);
        self.runs.iter().map(move |r| {
            (
                u64::from(r.logical_block).saturating_mul(block_size),
                r.extent,
            )
        })
    }

    /// Maps fork byte `pos` to (volume byte offset, bytes left in its run).
    /// `None` when `pos` is past the mapped extents.
    fn locate(&self, pos: u64) -> Option<(u64, u64)> {
        let bs = u64::from(self.block_size);
        let after = self
            .runs
            .partition_point(|r| u64::from(r.logical_block).saturating_mul(bs) <= pos);
        let run = self.runs.get(after.checked_sub(1)?)?;
        let run_start = u64::from(run.logical_block).checked_mul(bs)?;
        let run_len = u64::from(run.extent.block_count).checked_mul(bs)?;
        let within = pos.checked_sub(run_start)?;
        let avail = run_len.checked_sub(within).filter(|&n| n > 0)?;
        let physical = u64::from(run.extent.start_block)
            .checked_mul(bs)?
            .checked_add(within)?;
        Some((physical, avail))
    }

    /// A fork from runs the caller collected itself (classic HFS), in fork
    /// order. Every run must be non-empty and end inside
    /// `geometry.total_blocks`; the runs' sum must fit in `u32`. The runs
    /// may cover more or fewer blocks than `logical_size` needs: reads past
    /// the logical size are refused by [`ForkSource`], reads past the mapped
    /// runs fail as unmapped.
    pub fn from_runs(
        file_id: u32,
        logical_size: u64,
        runs: Vec<ExtentDescriptor>,
        geometry: Geometry,
    ) -> Result<ForkMap> {
        let error = |kind| {
            Error::Corrupt(Corruption::new(
                FORK,
                Location::Object(u64::from(file_id)),
                kind,
            ))
        };
        let mut mapped: u32 = 0;
        let mut out = Vec::with_capacity(runs.len());
        for extent in runs {
            if extent.block_count == 0 {
                return Err(error(CorruptKind::InvalidValue {
                    field: "extent block count",
                    value: 0,
                }));
            }
            let end = extent
                .start_block
                .checked_add(extent.block_count)
                .ok_or_else(|| {
                    error(CorruptKind::Overflow {
                        field: "extent end",
                    })
                })?;
            if end > geometry.total_blocks {
                return Err(error(CorruptKind::OutOfRange {
                    field: "extent end",
                    value: u64::from(end),
                    limit_name: "volume blocks",
                    limit: u64::from(geometry.total_blocks),
                }));
            }
            let next = mapped.checked_add(extent.block_count).ok_or_else(|| {
                error(CorruptKind::Overflow {
                    field: "mapped blocks",
                })
            })?;
            out.push(Run {
                logical_block: mapped,
                extent,
            });
            mapped = next;
        }
        Ok(ForkMap {
            file_id,
            block_size: geometry.block_size,
            logical_size,
            mapped_blocks: mapped,
            runs: out,
        })
    }

    fn unmapped(&self, value: u64) -> Error {
        Error::Corrupt(Corruption::new(
            FORK,
            Location::Object(u64::from(self.file_id)),
            CorruptKind::OutOfRange {
                field: "read end",
                value,
                limit_name: "end of mapped extents",
                limit: self.mapped_bytes(),
            },
        ))
    }
}

/// A fork as a BlockSource: len() is the logical size, offsets are fork-relative, reads split across runs.
/// read_bulk forwards to the volume's read_bulk. sector_size is the volume's.
#[derive(Clone)]
pub struct ForkSource {
    volume: Arc<dyn BlockSource>,
    map: Arc<ForkMap>,
}

impl ForkSource {
    /// The fork described by `map`, read from `volume`.
    pub fn new(volume: Arc<dyn BlockSource>, map: Arc<ForkMap>) -> Self {
        Self { volume, map }
    }

    fn read(&self, offset: u64, buf: &mut [u8], bulk: bool) -> Result<()> {
        let map = &*self.map;
        check_range(offset, buf.len(), map.logical_size)?;
        // check_range guarantees the end fits in u64.
        let end = offset.saturating_add(u64::try_from(buf.len()).unwrap_or(u64::MAX));
        if end > map.mapped_bytes() {
            return Err(map.unmapped(end));
        }
        let mut pos = offset;
        let mut rest = buf;
        // Bounded: every pass consumes at least one byte, and at most one
        // pass per run is needed.
        while !rest.is_empty() {
            let (physical, avail) = map.locate(pos).ok_or_else(|| map.unmapped(end))?;
            let n = usize::try_from(avail).unwrap_or(usize::MAX).min(rest.len());
            let (head, tail) = rest
                .split_at_mut_checked(n)
                .ok_or_else(|| map.unmapped(end))?;
            if bulk {
                self.volume.read_bulk(physical, head)?;
            } else {
                self.volume.read_at(physical, head)?;
            }
            pos = pos.saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
            rest = tail;
        }
        Ok(())
    }
}

impl BlockSource for ForkSource {
    fn len(&self) -> u64 {
        self.map.logical_size
    }

    fn sector_size(&self) -> u32 {
        self.volume.sector_size()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read(offset, buf, false)
    }

    fn read_bulk(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read(offset, buf, true)
    }
}

/// Accumulates runs while a fork is mapped.
struct Mapper {
    geometry: Geometry,
    file_id: u32,
    fork_blocks: u32,
    mapped: u32,
    runs: Vec<Run>,
}

impl Mapper {
    fn error(&self, kind: CorruptKind) -> Error {
        Error::Corrupt(Corruption::new(
            FORK,
            Location::Object(u64::from(self.file_id)),
            kind,
        ))
    }

    /// Appends the extents of `record` up to the first empty one. Returns
    /// the number of blocks added.
    fn add(&mut self, record: &ExtentRecord) -> Result<u32> {
        let before = self.mapped;
        for extent in record.iter().take_while(|e| e.block_count != 0) {
            let end = extent
                .start_block
                .checked_add(extent.block_count)
                .ok_or_else(|| {
                    self.error(CorruptKind::Overflow {
                        field: "extent end",
                    })
                })?;
            if end > self.geometry.total_blocks {
                return Err(self.error(CorruptKind::OutOfRange {
                    field: "extent end",
                    value: u64::from(end),
                    limit_name: "volume blocks",
                    limit: u64::from(self.geometry.total_blocks),
                }));
            }
            let mapped = self
                .mapped
                .checked_add(extent.block_count)
                .filter(|&m| m <= self.fork_blocks)
                .ok_or_else(|| {
                    self.error(CorruptKind::OutOfRange {
                        field: "mapped blocks",
                        value: u64::from(self.mapped).saturating_add(u64::from(extent.block_count)),
                        limit_name: "fork blocks",
                        limit: u64::from(self.fork_blocks),
                    })
                })?;
            self.runs.push(Run {
                logical_block: self.mapped,
                extent: *extent,
            });
            self.mapped = mapped;
        }
        Ok(self.mapped.saturating_sub(before))
    }

    fn missing(&self) -> Error {
        self.error(CorruptKind::Other("extents overflow record missing"))
    }
}

/// An extents overflow key (`HFSPlusExtentKey`) without its length field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExtentKey {
    fork_type: u8,
    file_id: u32,
    start_block: u32,
}

impl ExtentKey {
    /// Parses the key bytes after `keyLength`: forkType, pad, fileID, startBlock.
    fn parse(key: &[u8], file_id: u32) -> Result<Self> {
        let parsed = (|| {
            Some(Self {
                fork_type: u8_at(key, 0)?,
                file_id: be_u32(key, 2)?,
                start_block: be_u32(key, 6)?,
            })
        })();
        parsed.ok_or_else(|| {
            Error::Corrupt(Corruption::new(
                OVERFLOW,
                Location::Object(u64::from(file_id)),
                CorruptKind::Truncated {
                    needed: 10,
                    available: u64::try_from(key.len()).unwrap_or(u64::MAX),
                },
            ))
        })
    }

    /// TN1150 order: fileID, then forkType, then startBlock.
    fn order(&self) -> (u32, u8, u32) {
        (self.file_id, self.fork_type, self.start_block)
    }
}

/// Maps `fork` of file `file_id`. `overflow` is the extents overflow B-tree; None means "the first eight extents must cover the fork" (used for the extents file itself).
pub fn fork_map(
    geometry: Geometry,
    file_id: u32,
    fork_type: ForkType,
    fork: &ForkData,
    overflow: Option<&BTree>,
) -> Result<ForkMap> {
    let mut m = Mapper {
        geometry,
        file_id,
        fork_blocks: fork.total_blocks,
        mapped: 0,
        runs: Vec::new(),
    };
    m.add(&fork.extents)?;

    if m.mapped < m.fork_blocks {
        let tree = overflow.ok_or_else(|| m.missing())?;
        let ft = fork_type_byte(fork_type);
        let target = (file_id, ft, m.mapped);
        let mut cursor =
            tree.seek(&mut |key: &[u8]| Ok(ExtentKey::parse(key, file_id)?.order().cmp(&target)))?;
        let limit = tree.header().leaf_records;
        let mut used: u32 = 0;
        // Bounded by the tree's leaf record count, and every record must add
        // blocks while mapped stays at or below fork_blocks.
        while m.mapped < m.fork_blocks {
            if used >= limit {
                return Err(Error::Corrupt(Corruption::new(
                    OVERFLOW,
                    Location::Object(u64::from(file_id)),
                    CorruptKind::LimitExceeded {
                        what: "extents overflow records",
                        limit: u64::from(limit),
                    },
                )));
            }
            used = used.saturating_add(1);
            let record = {
                let rec = cursor.next_record()?.ok_or_else(|| m.missing())?;
                let key = ExtentKey::parse(rec.key, file_id)?;
                let want = ExtentKey {
                    fork_type: ft,
                    file_id,
                    start_block: m.mapped,
                };
                if key != want {
                    return Err(m.missing());
                }
                parse_extent_record(rec.data).ok_or_else(|| {
                    Error::Corrupt(Corruption::new(
                        OVERFLOW,
                        Location::Object(u64::from(file_id)),
                        CorruptKind::Truncated {
                            needed: 64,
                            available: u64::try_from(rec.data.len()).unwrap_or(u64::MAX),
                        },
                    ))
                })?
            };
            if m.add(&record)? == 0 {
                return Err(m.error(CorruptKind::Other("extents overflow record maps no blocks")));
            }
        }
    }

    Ok(ForkMap {
        file_id,
        block_size: geometry.block_size,
        logical_size: fork.logical_size,
        mapped_blocks: m.mapped,
        runs: m.runs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::btree::fixtures::{INDEX, LEAF, header, header_node, keyed, node, tree, var_index};
    use atlas_fs_core::io::block::MemSource;

    const BS: usize = 512;
    const FILE: u32 = 7;
    const DATA: u8 = 0x00;
    const RSRC: u8 = 0xFF;

    fn geometry() -> Geometry {
        Geometry {
            block_size: 512,
            total_blocks: 64,
        }
    }

    /// A 64-block volume where every byte of block `n` is `n`.
    fn volume() -> Arc<dyn BlockSource> {
        let bytes: Vec<u8> = (0..64u8).flat_map(|b| vec![b; BS]).collect();
        Arc::new(MemSource::new(bytes))
    }

    fn ext(start_block: u32, block_count: u32) -> ExtentDescriptor {
        ExtentDescriptor {
            start_block,
            block_count,
        }
    }

    /// An extent record from up to eight extents, the rest empty.
    fn record(extents: &[ExtentDescriptor]) -> ExtentRecord {
        assert!(extents.len() <= 8);
        let mut r = [ExtentDescriptor::default(); 8];
        r[..extents.len()].copy_from_slice(extents);
        r
    }

    /// Eight one-block extents at physical blocks 1, 3, 5, ... 15.
    fn eight_singles() -> Vec<ExtentDescriptor> {
        (0..8).map(|i| ext(2 * i + 1, 1)).collect()
    }

    fn fork(logical_size: u64, total_blocks: u32, extents: &[ExtentDescriptor]) -> ForkData {
        ForkData {
            logical_size,
            clump_size: 0,
            total_blocks,
            extents: record(extents),
        }
    }

    fn map_fork(fork_type: ForkType, fork: &ForkData, overflow: Option<&BTree>) -> Result<ForkMap> {
        fork_map(geometry(), FILE, fork_type, fork, overflow)
    }

    fn map_data(fork: &ForkData, overflow: Option<&BTree>) -> Result<ForkMap> {
        map_fork(ForkType::Data, fork, overflow)
    }

    fn source(map: ForkMap) -> ForkSource {
        ForkSource::new(volume(), Arc::new(map))
    }

    fn corrupt<T: std::fmt::Debug>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            Err(e) => panic!("expected Corrupt, got {e:?}"),
            Ok(v) => panic!("expected an error, got {v:?}"),
        }
    }

    fn is_missing(c: &Corruption) -> bool {
        matches!(c.kind, CorruptKind::Other(s) if s.contains("record missing"))
    }

    /// Key bytes of an extents overflow record, after the key length.
    fn okey(fork_type: u8, file_id: u32, start_block: u32) -> Vec<u8> {
        let mut k = vec![fork_type, 0];
        k.extend_from_slice(&file_id.to_be_bytes());
        k.extend_from_slice(&start_block.to_be_bytes());
        k
    }

    fn extent_bytes(extents: &[ExtentDescriptor]) -> Vec<u8> {
        let mut b = Vec::new();
        for e in record(extents) {
            b.extend_from_slice(&e.start_block.to_be_bytes());
            b.extend_from_slice(&e.block_count.to_be_bytes());
        }
        b
    }

    /// A leaf record of the extents overflow tree.
    fn orec(fork_type: u8, file_id: u32, start: u32, extents: &[ExtentDescriptor]) -> Vec<u8> {
        keyed(&okey(fork_type, file_id, start), &extent_bytes(extents))
    }

    fn leaf(forward: u32, backward: u32, records: &[Vec<u8>]) -> Vec<u8> {
        node(512, LEAF, 1, forward, backward, records)
    }

    /// A one-leaf overflow tree whose header claims `leaf_records` records.
    fn one_leaf_tree(records: &[Vec<u8>], leaf_records: u32) -> BTree {
        let mut h = header(1, 1, 1, 1, 2);
        h.leaf_records = leaf_records;
        tree(&[header_node(&h), leaf(0, 0, records)]).unwrap()
    }

    /// Reads the whole fork and checks every byte against `phys`, the
    /// physical block of each logical block.
    fn assert_content(src: &ForkSource, phys: &[u8]) {
        let len = usize::try_from(src.len()).unwrap();
        assert!(len <= phys.len() * BS);
        let mut buf = vec![0u8; len];
        src.read_at(0, &mut buf).unwrap();
        for (pos, &b) in buf.iter().enumerate() {
            assert_eq!(b, phys[pos / BS], "byte {pos}");
        }
    }

    #[test]
    fn maps_first_eight_extents_without_overflow() {
        let f = fork(2500, 5, &[ext(10, 2), ext(20, 3)]);
        let m = map_data(&f, None).unwrap();
        assert_eq!(m.logical_size(), 2500);
        let runs: Vec<_> = m.runs().collect();
        assert_eq!(runs, [(0, ext(10, 2)), (1024, ext(20, 3))]);
    }

    #[test]
    fn maps_all_eight_extents_in_order() {
        let f = fork(8 * 512, 8, &eight_singles());
        let m = map_data(&f, None).unwrap();
        let runs: Vec<_> = m.runs().collect();
        assert_eq!(runs.len(), 8);
        for (i, (offset, e)) in runs.into_iter().enumerate() {
            assert_eq!(offset, i as u64 * 512);
            assert_eq!(e, ext(2 * i as u32 + 1, 1));
        }
    }

    #[test]
    fn extents_after_the_first_empty_one_are_ignored() {
        let mut f = fork(1024, 2, &[ext(10, 2)]);
        f.extents[2] = ext(20, 3);
        let m = map_data(&f, None).unwrap();
        assert_eq!(m.runs().collect::<Vec<_>>(), [(0, ext(10, 2))]);
    }

    #[test]
    fn empty_fork_maps_and_reads_nothing() {
        let m = map_data(&fork(0, 0, &[]), None).unwrap();
        assert_eq!(m.runs().count(), 0);
        let src = source(m);
        assert_eq!(src.len(), 0);
        assert!(src.is_empty());
        src.read_at(0, &mut []).unwrap();
        assert!(matches!(
            src.read_at(0, &mut [0u8; 1]),
            Err(Error::OutOfBounds { .. })
        ));
    }

    #[test]
    fn fork_source_len_and_sector_size() {
        let src = source(map_data(&fork(2500, 5, &[ext(10, 2), ext(20, 3)]), None).unwrap());
        assert_eq!(src.len(), 2500);
        assert_eq!(src.sector_size(), 512);
    }

    #[test]
    fn read_inside_one_run() {
        let src = source(map_data(&fork(2500, 5, &[ext(10, 2), ext(20, 3)]), None).unwrap());
        let mut buf = [0u8; 50];
        src.read_at(100, &mut buf).unwrap();
        assert_eq!(buf, [10u8; 50]);
        // Across a block boundary but inside the run.
        let mut buf = [0u8; 200];
        src.read_at(400, &mut buf).unwrap();
        assert_eq!(buf[..112], [10u8; 112]);
        assert_eq!(buf[112..], [11u8; 88]);
    }

    #[test]
    fn read_spanning_two_non_adjacent_runs() {
        let src = source(map_data(&fork(2500, 5, &[ext(10, 2), ext(20, 3)]), None).unwrap());
        let mut buf = [0u8; 48];
        src.read_at(1000, &mut buf).unwrap();
        // Logical block 1 is physical block 11; logical block 2 is physical 20.
        assert_eq!(buf[..24], [11u8; 24]);
        assert_eq!(buf[24..], [20u8; 24]);
    }

    #[test]
    fn read_whole_fork_and_up_to_logical_size() {
        let src = source(map_data(&fork(2500, 5, &[ext(10, 2), ext(20, 3)]), None).unwrap());
        assert_content(&src, &[10, 11, 20, 21, 22]);
        // Ends exactly at the logical size, inside the last block.
        let mut buf = [0u8; 100];
        src.read_at(2400, &mut buf).unwrap();
        assert_eq!(buf, [22u8; 100]);
        // An empty read at the end is fine.
        src.read_at(2500, &mut []).unwrap();
    }

    #[test]
    fn read_past_logical_size_fails() {
        let src = source(map_data(&fork(2500, 5, &[ext(10, 2), ext(20, 3)]), None).unwrap());
        let mut buf = [0u8; 101];
        assert!(matches!(
            src.read_at(2400, &mut buf),
            Err(Error::OutOfBounds { .. })
        ));
        assert!(matches!(
            src.read_bulk(2400, &mut buf),
            Err(Error::OutOfBounds { .. })
        ));
        assert!(matches!(
            src.read_at(2501, &mut []),
            Err(Error::OutOfBounds { .. })
        ));
        assert!(matches!(
            src.read_at(u64::MAX, &mut [0u8; 4]),
            Err(Error::OutOfBounds { .. })
        ));
    }

    #[test]
    fn read_bulk_matches_read_at() {
        let src = source(map_data(&fork(2500, 5, &[ext(10, 2), ext(20, 3)]), None).unwrap());
        for (offset, len) in [(0, 2500), (1000, 48), (511, 2), (1023, 1025), (2499, 1)] {
            let mut a = vec![0u8; len];
            let mut b = vec![0u8; len];
            src.read_at(offset, &mut a).unwrap();
            src.read_bulk(offset, &mut b).unwrap();
            assert_eq!(a, b, "offset {offset} len {len}");
        }
    }

    #[test]
    fn logical_size_beyond_mapped_extents() {
        // 2000 bytes claimed, but the extents cover only two blocks.
        let src = source(map_data(&fork(2000, 2, &[ext(10, 2)]), None).unwrap());
        assert_eq!(src.len(), 2000);
        let mut buf = vec![0u8; 1024];
        src.read_at(0, &mut buf).unwrap();
        assert_eq!(buf[..512], [10u8; 512]);
        assert_eq!(buf[512..], [11u8; 512]);

        let mut buf = [0u8; 100];
        let c = corrupt(src.read_at(1000, &mut buf));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                limit_name: "end of mapped extents",
                limit: 1024,
                value: 1100,
                ..
            }
        ));
        let c = corrupt(src.read_bulk(1024, &mut [0u8; 1]));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                limit_name: "end of mapped extents",
                ..
            }
        ));
    }

    #[test]
    fn extent_end_overflowing_u32() {
        let f = fork(512 * 32, 32, &[ext(0xFFFF_FFF0, 0x20)]);
        let c = corrupt(map_data(&f, None));
        assert!(matches!(
            c.kind,
            CorruptKind::Overflow {
                field: "extent end"
            }
        ));
    }

    #[test]
    fn extent_past_volume_end() {
        let f = fork(512 * 10, 10, &[ext(60, 10)]);
        let c = corrupt(map_data(&f, None));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                limit_name: "volume blocks",
                value: 70,
                limit: 64,
                ..
            }
        ));
        // Ending exactly on the last block is fine.
        assert!(map_data(&fork(512 * 4, 4, &[ext(60, 4)]), None).is_ok());
    }

    #[test]
    fn extents_summing_past_fork_blocks() {
        let f = fork(512 * 4, 4, &[ext(10, 3), ext(20, 3)]);
        let c = corrupt(map_data(&f, None));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                limit_name: "fork blocks",
                value: 6,
                limit: 4,
                ..
            }
        ));
    }

    #[test]
    fn fork_needing_overflow_without_a_tree() {
        let f = fork(512 * 10, 10, &[ext(10, 2)]);
        assert!(is_missing(&corrupt(map_data(&f, None))));
    }

    #[test]
    fn overflow_record_completes_the_fork() {
        let f = fork(10 * 512, 10, &eight_singles());
        let t = one_leaf_tree(&[orec(DATA, FILE, 8, &[ext(40, 2)])], 1);
        let m = map_data(&f, Some(&t)).unwrap();
        let runs: Vec<_> = m.runs().collect();
        assert_eq!(runs.len(), 9);
        assert_eq!(runs[8], (8 * 512, ext(40, 2)));
        let src = source(m);
        assert_content(&src, &[1, 3, 5, 7, 9, 11, 13, 15, 40, 41]);
    }

    #[test]
    fn overflow_tree_is_not_consulted_when_extents_suffice() {
        let f = fork(1024, 2, &[ext(10, 2)]);
        // A tree that would fail on any use.
        let t = one_leaf_tree(&[], 0);
        let m = map_data(&f, Some(&t)).unwrap();
        assert_eq!(m.runs().count(), 1);
    }

    #[test]
    fn overflow_records_in_two_leaves() {
        // Fork of 15 blocks: eight single blocks, then record A at 8 (4
        // blocks) and record B at 12 (3 blocks). A and B sit in different
        // leaves under an index node.
        let a = orec(DATA, FILE, 8, &[ext(30, 2), ext(34, 2)]);
        let b = orec(DATA, FILE, 12, &[ext(40, 3)]);
        let mut h = header(2, 1, 2, 3, 4);
        h.leaf_records = 4;
        let index = node(
            512,
            INDEX,
            2,
            0,
            0,
            &[
                var_index(&okey(DATA, 5, 8), 2),
                var_index(&okey(DATA, FILE, 12), 3),
            ],
        );
        let t = tree(&[
            header_node(&h),
            index,
            leaf(3, 0, &[orec(DATA, 5, 8, &[ext(60, 2)]), a]),
            leaf(0, 2, &[b, orec(DATA, 9, 8, &[ext(5, 1)])]),
        ])
        .unwrap();

        let f = fork(15 * 512, 15, &eight_singles());
        let m = map_data(&f, Some(&t)).unwrap();
        let runs: Vec<_> = m.runs().collect();
        assert_eq!(runs.len(), 11);
        assert_eq!(runs[8], (8 * 512, ext(30, 2)));
        assert_eq!(runs[9], (10 * 512, ext(34, 2)));
        assert_eq!(runs[10], (12 * 512, ext(40, 3)));
        let src = source(m);
        assert_content(
            &src,
            &[1, 3, 5, 7, 9, 11, 13, 15, 30, 31, 34, 35, 40, 41, 42],
        );
    }

    /// Records for other files and for the resource fork, in key order.
    fn interleaved_tree() -> BTree {
        one_leaf_tree(
            &[
                orec(DATA, 3, 8, &[ext(60, 2)]),
                orec(DATA, FILE, 8, &[ext(30, 2), ext(34, 2)]),
                orec(DATA, FILE, 12, &[ext(40, 3)]),
                orec(RSRC, FILE, 8, &[ext(50, 2)]),
                orec(DATA, 9, 8, &[ext(5, 1)]),
            ],
            5,
        )
    }

    #[test]
    fn overflow_ignores_other_files_and_the_resource_fork() {
        let t = interleaved_tree();
        let f = fork(15 * 512, 15, &eight_singles());
        let m = map_data(&f, Some(&t)).unwrap();
        let src = source(m);
        assert_content(
            &src,
            &[1, 3, 5, 7, 9, 11, 13, 15, 30, 31, 34, 35, 40, 41, 42],
        );
    }

    #[test]
    fn resource_fork_picks_its_own_overflow_records() {
        let t = interleaved_tree();
        let f = fork(10 * 512, 10, &eight_singles());
        let m = map_fork(ForkType::Resource, &f, Some(&t)).unwrap();
        let runs: Vec<_> = m.runs().collect();
        assert_eq!(runs.len(), 9);
        assert_eq!(runs[8], (8 * 512, ext(50, 2)));
        let src = source(m);
        assert_content(&src, &[1, 3, 5, 7, 9, 11, 13, 15, 50, 51]);
        // A data fork of 12 blocks takes the data record instead.
        let f = fork(12 * 512, 12, &eight_singles());
        let m = map_data(&f, Some(&t)).unwrap();
        assert_eq!(m.runs().nth(8).unwrap(), (8 * 512, ext(30, 2)));
    }

    #[test]
    fn overflow_record_with_wrong_start_block() {
        // The record that follows the seek target starts at 9, not 8.
        let t = one_leaf_tree(&[orec(DATA, FILE, 9, &[ext(30, 2)])], 1);
        let f = fork(10 * 512, 10, &eight_singles());
        assert!(is_missing(&corrupt(map_data(&f, Some(&t)))));
    }

    #[test]
    fn second_overflow_record_with_wrong_start_block() {
        let t = one_leaf_tree(
            &[
                orec(DATA, FILE, 8, &[ext(30, 2)]),
                orec(DATA, FILE, 11, &[ext(40, 3)]),
            ],
            2,
        );
        let f = fork(13 * 512, 13, &eight_singles());
        assert!(is_missing(&corrupt(map_data(&f, Some(&t)))));
    }

    #[test]
    fn overflow_record_for_another_file_is_not_taken() {
        let t = one_leaf_tree(&[orec(DATA, 9, 8, &[ext(30, 2)])], 1);
        let f = fork(10 * 512, 10, &eight_singles());
        assert!(is_missing(&corrupt(map_data(&f, Some(&t)))));
    }

    #[test]
    fn overflow_tree_runs_out_of_records() {
        // The chain ends after one record but the fork wants more.
        let t = one_leaf_tree(&[orec(DATA, FILE, 8, &[ext(30, 2)])], 1);
        let f = fork(13 * 512, 13, &eight_singles());
        // With the header claiming one record, the limit is reached first.
        let c = corrupt(map_data(&f, Some(&t)));
        assert!(matches!(c.kind, CorruptKind::LimitExceeded { .. }));
        // With an honest count the missing record is reported.
        let t = one_leaf_tree(&[orec(DATA, FILE, 8, &[ext(30, 2)])], 9);
        assert!(is_missing(&corrupt(map_data(&f, Some(&t)))));
    }

    #[test]
    fn empty_overflow_tree_means_missing() {
        let f = fork(10 * 512, 10, &eight_singles());
        let mut h = header(0, 0, 0, 0, 1);
        let t = tree(&[header_node(&h)]).unwrap();
        // An empty tree has no leaf records, so the bound is hit at once.
        assert!(matches!(
            corrupt(map_data(&f, Some(&t))).kind,
            CorruptKind::LimitExceeded { limit: 0, .. }
        ));
        // With an inflated count the missing record is reported.
        h.leaf_records = 1;
        let t = tree(&[header_node(&h)]).unwrap();
        assert!(is_missing(&corrupt(map_data(&f, Some(&t)))));
    }

    #[test]
    fn overflow_record_mapping_no_blocks() {
        let t = one_leaf_tree(&[orec(DATA, FILE, 8, &[])], 1);
        let f = fork(10 * 512, 10, &eight_singles());
        let c = corrupt(map_data(&f, Some(&t)));
        assert!(matches!(c.kind, CorruptKind::Other(s) if s.contains("maps no blocks")));
    }

    #[test]
    fn overflow_record_with_first_extent_empty_and_later_extent_set() {
        // Extents after an empty one do not count.
        let mut data = extent_bytes(&[]);
        data[16..20].copy_from_slice(&30u32.to_be_bytes());
        data[20..24].copy_from_slice(&2u32.to_be_bytes());
        let t = one_leaf_tree(&[keyed(&okey(DATA, FILE, 8), &data)], 1);
        let f = fork(10 * 512, 10, &eight_singles());
        let c = corrupt(map_data(&f, Some(&t)));
        assert!(matches!(c.kind, CorruptKind::Other(s) if s.contains("maps no blocks")));
    }

    #[test]
    fn overflow_leaf_record_limit() {
        let a = orec(DATA, FILE, 8, &[ext(30, 2)]);
        let b = orec(DATA, FILE, 10, &[ext(40, 3)]);
        let f = fork(13 * 512, 13, &eight_singles());
        // One record allowed, two needed.
        let t = one_leaf_tree(&[a.clone(), b.clone()], 1);
        let c = corrupt(map_data(&f, Some(&t)));
        assert!(matches!(
            c.kind,
            CorruptKind::LimitExceeded {
                what: "extents overflow records",
                limit: 1
            }
        ));
        // A header that claims no records at all.
        let t = one_leaf_tree(&[a.clone(), b.clone()], 0);
        assert!(matches!(
            corrupt(map_data(&f, Some(&t))).kind,
            CorruptKind::LimitExceeded { limit: 0, .. }
        ));
        // Two allowed is enough.
        let t = one_leaf_tree(&[a, b], 2);
        assert!(map_data(&f, Some(&t)).is_ok());
    }

    #[test]
    fn overflow_record_data_too_short() {
        let t = one_leaf_tree(&[keyed(&okey(DATA, FILE, 8), &[0u8; 63])], 1);
        let f = fork(10 * 512, 10, &eight_singles());
        let c = corrupt(map_data(&f, Some(&t)));
        assert!(matches!(
            c.kind,
            CorruptKind::Truncated {
                needed: 64,
                available: 63
            }
        ));
    }

    #[test]
    fn overflow_key_too_short() {
        let t = one_leaf_tree(&[keyed(&okey(DATA, FILE, 8)[..9], &[0u8; 64])], 1);
        let f = fork(10 * 512, 10, &eight_singles());
        let c = corrupt(map_data(&f, Some(&t)));
        assert!(matches!(
            c.kind,
            CorruptKind::Truncated {
                needed: 10,
                available: 9
            }
        ));
    }

    #[test]
    fn overflow_record_past_volume_or_fork() {
        let f = fork(10 * 512, 10, &eight_singles());
        let t = one_leaf_tree(&[orec(DATA, FILE, 8, &[ext(60, 2), ext(63, 2)])], 1);
        assert!(matches!(
            corrupt(map_data(&f, Some(&t))).kind,
            CorruptKind::OutOfRange {
                limit_name: "volume blocks",
                ..
            }
        ));
        let t = one_leaf_tree(&[orec(DATA, FILE, 8, &[ext(30, 3)])], 1);
        assert!(matches!(
            corrupt(map_data(&f, Some(&t))).kind,
            CorruptKind::OutOfRange {
                limit_name: "fork blocks",
                ..
            }
        ));
    }

    #[test]
    fn from_runs_maps_and_reads_in_order() {
        let runs = vec![ext(10, 2), ext(20, 1), ext(5, 2)];
        let m = ForkMap::from_runs(FILE, 4 * 512 + 10, runs, geometry()).unwrap();
        assert_eq!(
            m.runs().collect::<Vec<_>>(),
            [(0, ext(10, 2)), (1024, ext(20, 1)), (1536, ext(5, 2))]
        );
        assert_content(&source(m), &[10, 11, 20, 5, 6]);
        // Runs ending exactly on the last volume block are fine.
        assert!(ForkMap::from_runs(FILE, 512, vec![ext(63, 1)], geometry()).is_ok());
    }

    #[test]
    fn from_runs_physical_larger_and_smaller_than_logical() {
        // Physical larger: reads stop at the logical size.
        let m = ForkMap::from_runs(FILE, 100, vec![ext(10, 4)], geometry()).unwrap();
        let src = source(m);
        assert_content(&src, &[10]);
        assert!(matches!(
            src.read_at(0, &mut [0u8; 101]),
            Err(Error::OutOfBounds { .. })
        ));
        // Physical smaller: the unmapped tail fails.
        let src = source(ForkMap::from_runs(FILE, 2048, vec![ext(10, 2)], geometry()).unwrap());
        src.read_at(0, &mut [0u8; 1024]).unwrap();
        let c = corrupt(src.read_at(1000, &mut [0u8; 100]));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                limit_name: "end of mapped extents",
                limit: 1024,
                ..
            }
        ));
    }

    #[test]
    fn from_runs_empty() {
        let m = ForkMap::from_runs(FILE, 0, Vec::new(), geometry()).unwrap();
        assert_eq!(m.runs().count(), 0);
        let src = source(m);
        assert_eq!(src.len(), 0);
        src.read_at(0, &mut []).unwrap();
    }

    #[test]
    fn from_runs_rejects_bad_runs() {
        let c = corrupt(ForkMap::from_runs(
            FILE,
            512 * 10,
            vec![ext(10, 2), ext(60, 8)],
            geometry(),
        ));
        assert_eq!(c.location, Location::Object(u64::from(FILE)));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "extent end",
                value: 68,
                limit_name: "volume blocks",
                limit: 64,
            }
        ));
        let c = corrupt(ForkMap::from_runs(
            FILE,
            512,
            vec![ext(0xFFFF_FFF0, 0x20)],
            geometry(),
        ));
        assert!(matches!(
            c.kind,
            CorruptKind::Overflow {
                field: "extent end"
            }
        ));
        let c = corrupt(ForkMap::from_runs(
            FILE,
            512,
            vec![ext(10, 1), ext(20, 0)],
            geometry(),
        ));
        assert!(matches!(
            c.kind,
            CorruptKind::InvalidValue {
                field: "extent block count",
                value: 0
            }
        ));
        // Overlapping runs whose sum passes u32 on a huge volume.
        let huge = Geometry {
            block_size: 512,
            total_blocks: u32::MAX,
        };
        let c = corrupt(ForkMap::from_runs(
            FILE,
            0,
            vec![ext(0, u32::MAX), ext(0, 1)],
            huge,
        ));
        assert!(matches!(
            c.kind,
            CorruptKind::Overflow {
                field: "mapped blocks"
            }
        ));
    }

    #[test]
    fn fork_source_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ForkSource>();
        assert_send_sync::<ForkMap>();
    }

    #[test]
    fn parses_fork_data() {
        let mut b = vec![0u8; FORK_DATA_LEN];
        b[0..8].copy_from_slice(&0x1_0000_0000u64.to_be_bytes());
        b[8..12].copy_from_slice(&7u32.to_be_bytes());
        b[12..16].copy_from_slice(&30u32.to_be_bytes());
        b[16..20].copy_from_slice(&100u32.to_be_bytes());
        b[20..24].copy_from_slice(&10u32.to_be_bytes());
        b[72..76].copy_from_slice(&500u32.to_be_bytes());
        b[76..80].copy_from_slice(&20u32.to_be_bytes());
        let f = ForkData::parse(&b).unwrap();
        assert_eq!(f.logical_size, 0x1_0000_0000);
        assert_eq!(f.clump_size, 7);
        assert_eq!(f.total_blocks, 30);
        assert_eq!(
            f.extents[0],
            ExtentDescriptor {
                start_block: 100,
                block_count: 10
            }
        );
        assert_eq!(f.extents[1], ExtentDescriptor::default());
        assert_eq!(
            f.extents[7],
            ExtentDescriptor {
                start_block: 500,
                block_count: 20
            }
        );
        assert_eq!(ForkData::parse(&b[..79]), None);
    }
}
