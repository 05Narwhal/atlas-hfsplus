// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Volume header and HFS wrapper detection.
//!
//! A mount calls [`locate`] to find the HFS+ volume inside its source (the
//! whole source, or the volume embedded in a classic HFS wrapper), cuts a
//! slice at the result, and calls [`read_header`] on the slice.

use super::bytes::{be_u16, be_u32, be_u64};
use super::extents::ForkData;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::Timestamp;
use atlas_fs_core::io::block::BlockSource;

/// `H+`, the signature of an HFS+ volume header (version 4).
pub const SIGNATURE_HFS_PLUS: u16 = 0x482B;
/// `HX`, the signature of an HFSX volume header (version 5).
pub const SIGNATURE_HFSX: u16 = 0x4858;
/// Byte offset of the volume header from the start of the volume.
pub const VOLUME_HEADER_OFFSET: u64 = 1024;
/// Size of the volume header in bytes.
pub const VOLUME_HEADER_LEN: usize = 512;
/// `HFSJ` (`kHFSJMountVersion`): the `lastMountedVersion` of a volume last
/// mounted by journaling Mac OS X. Only then does macOS replay the journal.
pub const JOURNALED_MOUNT_VERSION: u32 = 0x4846_534A;

/// `BD`, the signature of a classic HFS master directory block.
const SIGNATURE_HFS: u16 = 0x4244;
/// The alternate volume header lies this many bytes before the volume's end.
const ALTERNATE_HEADER_FROM_END: u64 = 1024;
/// Smallest source `locate` accepts: the header's offset plus its length.
const MIN_LOCATE_LEN: u64 = 1536;
/// Seconds from 1904-01-01 to 1970-01-01.
const HFS_EPOCH_TO_UNIX: i64 = 2_082_844_800;
/// Bytes per sector in the wrapper's `drAlBlSt`.
const SECTOR_LEN: u64 = 512;

/// Smallest allocation block size.
const MIN_BLOCK_SIZE: u32 = 512;
// Real volumes use 512 bytes to 64 KiB. The cap only keeps the buffers the
// engine sizes from the block size sane when a header is damaged.
/// Largest allocation block size accepted.
const MAX_BLOCK_SIZE: u32 = 1 << 20;

/// `kHFSVolumeUnmountedBit`.
const ATTR_UNMOUNTED: u32 = 1 << 8;
/// `kHFSVolumeJournaledBit`.
const ATTR_JOURNALED: u32 = 1 << 13;

const STRUCTURE: &str = "volume header";
const WRAPPER: &str = "HFS wrapper";

/// The fields of an HFS+ or HFSX volume header (`HFSPlusVolumeHeader`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VolumeHeader {
    /// `H+` or `HX`.
    pub signature: u16,
    /// 4 for HFS+, 5 for HFSX.
    pub version: u16,
    /// Volume attribute bits.
    pub attributes: u32,
    /// Which implementation last mounted the volume (`H+Lx` for journaled).
    pub last_mounted_version: u32,
    /// Allocation block of the journal info block; 0 when there is no journal.
    pub journal_info_block: u32,
    /// Creation date (local time, unlike the others).
    pub create_date: u32,
    /// Last modification date.
    pub modify_date: u32,
    /// Last backup date.
    pub backup_date: u32,
    /// Date of the last consistency check.
    pub checked_date: u32,
    /// Number of files on the volume.
    pub file_count: u32,
    /// Number of folders on the volume.
    pub folder_count: u32,
    /// Allocation block size in bytes.
    pub block_size: u32,
    /// Allocation blocks on the volume.
    pub total_blocks: u32,
    /// Unused allocation blocks.
    pub free_blocks: u32,
    /// Where the allocator searches next.
    pub next_allocation: u32,
    /// Default clump size for resource forks.
    pub rsrc_clump_size: u32,
    /// Default clump size for data forks.
    pub data_clump_size: u32,
    /// The next catalog node id to assign.
    pub next_catalog_id: u32,
    /// Times the volume has been mounted writable.
    pub write_count: u32,
    /// Text encodings in use, one bit each.
    pub encodings_bitmap: u64,
    /// Eight words for the Finder; word 5 is the Mac OS X boot folder id.
    pub finder_info: [u32; 8],
    /// The allocation bitmap file.
    pub allocation_file: ForkData,
    /// The extents overflow file.
    pub extents_file: ForkData,
    /// The catalog file.
    pub catalog_file: ForkData,
    /// The attributes file.
    pub attributes_file: ForkData,
    /// The startup file.
    pub startup_file: ForkData,
}

/// A buffer shorter than a structure needs.
fn truncated(needed: u64, available: usize) -> Error {
    Error::Corrupt(Corruption::new(
        STRUCTURE,
        Location::Unspecified,
        CorruptKind::Truncated {
            needed,
            available: u64::try_from(available).unwrap_or(u64::MAX),
        },
    ))
}

/// Builds a "volume header" corruption at a field's byte offset.
fn bad(at: usize, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(
        STRUCTURE,
        Location::Byte(u64::try_from(at).unwrap_or(u64::MAX)),
        kind,
    ))
}

/// Field reads over the header buffer; a field that does not fit is `Truncated`.
struct Fields<'a>(&'a [u8]);

impl Fields<'_> {
    fn short(&self) -> Error {
        truncated(VOLUME_HEADER_LEN as u64, self.0.len())
    }

    fn u16(&self, at: usize) -> Result<u16> {
        be_u16(self.0, at).ok_or_else(|| self.short())
    }

    fn u32(&self, at: usize) -> Result<u32> {
        be_u32(self.0, at).ok_or_else(|| self.short())
    }

    fn u64(&self, at: usize) -> Result<u64> {
        be_u64(self.0, at).ok_or_else(|| self.short())
    }

    fn fork(&self, at: usize) -> Result<ForkData> {
        self.0
            .get(at..)
            .and_then(ForkData::parse)
            .ok_or_else(|| self.short())
    }
}

impl VolumeHeader {
    /// Parses and validates a 512-byte header.
    ///
    /// The offsets are those of TN1150 "Volume Header". Beyond the signature
    /// and version, the block size, the block counts and every special
    /// file's extents are checked against the volume's size, so later code
    /// can trust them.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < VOLUME_HEADER_LEN {
            return Err(truncated(VOLUME_HEADER_LEN as u64, bytes.len()));
        }
        let f = Fields(bytes);
        let signature = f.u16(0)?;
        let version = f.u16(2)?;
        match (signature, version) {
            (SIGNATURE_HFS_PLUS, 4) | (SIGNATURE_HFSX, 5) => {}
            (SIGNATURE_HFS_PLUS | SIGNATURE_HFSX, _) => {
                return Err(bad(
                    2,
                    CorruptKind::InvalidValue {
                        field: "version",
                        value: u64::from(version),
                    },
                ));
            }
            _ => {
                return Err(bad(
                    0,
                    CorruptKind::BadSignature {
                        expected: u64::from(SIGNATURE_HFS_PLUS),
                        found: u64::from(signature),
                    },
                ));
            }
        }

        let mut finder_info = [0u32; 8];
        for (i, word) in finder_info.iter_mut().enumerate() {
            // 80 + 4 * i, with i < 8.
            let at = i.checked_mul(4).and_then(|n| n.checked_add(80));
            *word = f.u32(at.ok_or_else(|| f.short())?)?;
        }

        let header = Self {
            signature,
            version,
            attributes: f.u32(4)?,
            last_mounted_version: f.u32(8)?,
            journal_info_block: f.u32(12)?,
            create_date: f.u32(16)?,
            modify_date: f.u32(20)?,
            backup_date: f.u32(24)?,
            checked_date: f.u32(28)?,
            file_count: f.u32(32)?,
            folder_count: f.u32(36)?,
            block_size: f.u32(40)?,
            total_blocks: f.u32(44)?,
            free_blocks: f.u32(48)?,
            next_allocation: f.u32(52)?,
            rsrc_clump_size: f.u32(56)?,
            data_clump_size: f.u32(60)?,
            next_catalog_id: f.u32(64)?,
            write_count: f.u32(68)?,
            encodings_bitmap: f.u64(72)?,
            finder_info,
            allocation_file: f.fork(112)?,
            extents_file: f.fork(192)?,
            catalog_file: f.fork(272)?,
            attributes_file: f.fork(352)?,
            startup_file: f.fork(432)?,
        };
        header.validate()?;
        Ok(header)
    }

    fn validate(&self) -> Result<()> {
        let bs = self.block_size;
        if !bs.is_power_of_two() || bs < MIN_BLOCK_SIZE {
            return Err(bad(
                40,
                CorruptKind::InvalidValue {
                    field: "block size",
                    value: u64::from(bs),
                },
            ));
        }
        if bs > MAX_BLOCK_SIZE {
            return Err(bad(
                40,
                CorruptKind::OutOfRange {
                    field: "block size",
                    value: u64::from(bs),
                    limit_name: "maximum block size",
                    limit: u64::from(MAX_BLOCK_SIZE),
                },
            ));
        }
        if self.total_blocks == 0 {
            return Err(bad(
                44,
                CorruptKind::InvalidValue {
                    field: "total blocks",
                    value: 0,
                },
            ));
        }
        if self.free_blocks > self.total_blocks {
            return Err(bad(
                48,
                CorruptKind::OutOfRange {
                    field: "free blocks",
                    value: u64::from(self.free_blocks),
                    limit_name: "total blocks",
                    limit: u64::from(self.total_blocks),
                },
            ));
        }
        self.validate_fork(112, &self.allocation_file, false)?;
        // The extents file never uses overflow records (TN1150), so its eight
        // extents must hold all of it.
        self.validate_fork(192, &self.extents_file, true)?;
        // The catalog may continue in the extents overflow file, so its
        // first eight extents need not cover it.
        self.validate_fork(272, &self.catalog_file, false)?;
        self.validate_fork(352, &self.attributes_file, false)?;
        self.validate_fork(432, &self.startup_file, false)?;
        Ok(())
    }

    /// Checks one special file's descriptor at header offset `at` against
    /// the volume. `must_cover` requires the eight extents to hold every
    /// block the fork claims.
    fn validate_fork(&self, at: usize, fork: &ForkData, must_cover: bool) -> Result<()> {
        let mut in_extents: u64 = 0;
        for extent in &fork.extents {
            // A zero-length extent ends the list; its start is meaningless.
            if extent.block_count == 0 {
                continue;
            }
            let end = extent
                .start_block
                .checked_add(extent.block_count)
                .ok_or_else(|| {
                    bad(
                        at,
                        CorruptKind::Overflow {
                            field: "extent end",
                        },
                    )
                })?;
            if end > self.total_blocks {
                return Err(bad(
                    at,
                    CorruptKind::OutOfRange {
                        field: "extent end",
                        value: u64::from(end),
                        limit_name: "total blocks",
                        limit: u64::from(self.total_blocks),
                    },
                ));
            }
            in_extents = in_extents
                .checked_add(u64::from(extent.block_count))
                .ok_or_else(|| {
                    bad(
                        at,
                        CorruptKind::Overflow {
                            field: "extent block count sum",
                        },
                    )
                })?;
        }
        // `fork.total_blocks` counts the extents beyond the eighth too, so
        // the first eight can never hold more than it, and it can never
        // exceed the volume.
        if in_extents > u64::from(fork.total_blocks) {
            return Err(bad(
                at,
                CorruptKind::OutOfRange {
                    field: "blocks in the first eight extents",
                    value: in_extents,
                    limit_name: "fork total blocks",
                    limit: u64::from(fork.total_blocks),
                },
            ));
        }
        if fork.total_blocks > self.total_blocks {
            return Err(bad(
                at,
                CorruptKind::OutOfRange {
                    field: "fork total blocks",
                    value: u64::from(fork.total_blocks),
                    limit_name: "total blocks",
                    limit: u64::from(self.total_blocks),
                },
            ));
        }
        if must_cover && in_extents < u64::from(fork.total_blocks) {
            return Err(bad(
                at,
                CorruptKind::Other("extents file is not covered by its eight extents"),
            ));
        }
        let capacity = u64::from(fork.total_blocks)
            .checked_mul(u64::from(self.block_size))
            .ok_or_else(|| {
                bad(
                    at,
                    CorruptKind::Overflow {
                        field: "fork capacity",
                    },
                )
            })?;
        if fork.logical_size > capacity {
            return Err(bad(
                at,
                CorruptKind::OutOfRange {
                    field: "fork logical size",
                    value: fork.logical_size,
                    limit_name: "fork capacity",
                    limit: capacity,
                },
            ));
        }
        Ok(())
    }

    /// True for an HFSX volume.
    pub fn is_hfsx(&self) -> bool {
        self.signature == SIGNATURE_HFSX
    }

    /// `kHFSVolumeJournaledBit` (bit 13) is set and `journal_info_block` is
    /// nonzero.
    pub fn is_journaled(&self) -> bool {
        self.attributes & ATTR_JOURNALED != 0 && self.journal_info_block != 0
    }

    /// True when the last mount journaled (`lastMountedVersion` is
    /// [`JOURNALED_MOUNT_VERSION`]), so a journal on the volume is current.
    pub fn last_mounted_journaled(&self) -> bool {
        self.last_mounted_version == JOURNALED_MOUNT_VERSION
    }

    /// `kHFSVolumeUnmountedBit` (bit 8): the volume was unmounted cleanly.
    pub fn unmounted_cleanly(&self) -> bool {
        self.attributes & ATTR_UNMOUNTED != 0
    }

    /// `block_size * total_blocks`, or `None` on overflow.
    pub fn volume_bytes(&self) -> Option<u64> {
        u64::from(self.block_size).checked_mul(u64::from(self.total_blocks))
    }
}

/// Where the HFS+ volume lies inside a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Located {
    /// Byte offset of the volume's first byte in the source.
    pub offset: u64,
    /// Length of the volume in bytes.
    pub len: u64,
    /// True when the volume is embedded in a classic HFS wrapper.
    pub wrapped: bool,
}

fn wrapper_error(at: u64, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(WRAPPER, Location::Byte(at), kind))
}

/// Byte position in the source of the field at `field` in the master
/// directory block.
fn mdb_at(field: u64) -> u64 {
    VOLUME_HEADER_OFFSET.saturating_add(field)
}

/// Finds the HFS+ volume in `source` by reading the 512 bytes at byte 1024.
///
/// `H+` or `HX` there means the volume is the whole source. An HFS master
/// directory block (`BD`) whose `drEmbedSigWord` is `H+` means the volume is
/// the embedded extent, wherever the wrapper puts it. Anything else is
/// `BadSignature`. Only the signature is checked here; [`read_header`]
/// validates the header itself.
pub fn locate(source: &dyn BlockSource) -> Result<Located> {
    let size = source.len();
    if size < MIN_LOCATE_LEN {
        return Err(Error::Corrupt(Corruption::new(
            STRUCTURE,
            Location::Unspecified,
            CorruptKind::Truncated {
                needed: MIN_LOCATE_LEN,
                available: size,
            },
        )));
    }
    let mut buf = [0u8; VOLUME_HEADER_LEN];
    source.read_at(VOLUME_HEADER_OFFSET, &mut buf)?;
    let f = Fields(&buf);
    let signature = f.u16(0)?;
    if signature == SIGNATURE_HFS_PLUS || signature == SIGNATURE_HFSX {
        return Ok(Located {
            offset: 0,
            len: size,
            wrapped: false,
        });
    }

    // From here the block is the HFS master directory block. Its offsets
    // (TN1150 "HFS Wrapper") are relative to the block at byte 1024.
    let embed_sig = f.u16(0x7C)?; // drEmbedSigWord
    if signature != SIGNATURE_HFS || embed_sig != SIGNATURE_HFS_PLUS {
        let (found, at) = if signature == SIGNATURE_HFS {
            (embed_sig, mdb_at(0x7C))
        } else {
            (signature, VOLUME_HEADER_OFFSET)
        };
        return Err(Error::Corrupt(Corruption::new(
            STRUCTURE,
            Location::Byte(at),
            CorruptKind::BadSignature {
                expected: u64::from(SIGNATURE_HFS_PLUS),
                found: u64::from(found),
            },
        )));
    }
    let block_size = u64::from(f.u32(0x14)?); // drAlBlkSiz
    let first_block = u64::from(f.u16(0x1C)?); // drAlBlSt, in 512-byte sectors
    let start = u64::from(f.u16(0x7E)?); // drEmbedExtent.startBlock
    let count = u64::from(f.u16(0x80)?); // drEmbedExtent.blockCount
    if block_size == 0 || block_size % SECTOR_LEN != 0 {
        return Err(wrapper_error(
            mdb_at(0x14),
            CorruptKind::InvalidValue {
                field: "allocation block size",
                value: block_size,
            },
        ));
    }
    let overflow = |field| wrapper_error(mdb_at(0x7E), CorruptKind::Overflow { field });
    let offset = first_block
        .checked_mul(SECTOR_LEN)
        .and_then(|base| base.checked_add(start.checked_mul(block_size)?))
        .ok_or_else(|| overflow("embedded volume offset"))?;
    let len = count
        .checked_mul(block_size)
        .ok_or_else(|| overflow("embedded volume length"))?;
    if len == 0 {
        return Err(wrapper_error(
            mdb_at(0x80),
            CorruptKind::InvalidValue {
                field: "embedded block count",
                value: 0,
            },
        ));
    }
    let end = offset
        .checked_add(len)
        .ok_or_else(|| overflow("embedded volume end"))?;
    if end > size {
        return Err(wrapper_error(
            mdb_at(0x7E),
            CorruptKind::OutOfRange {
                field: "embedded volume end",
                value: end,
                limit_name: "source size",
                limit: size,
            },
        ));
    }
    Ok(Located {
        offset,
        len,
        wrapped: true,
    })
}

/// Reads and parses the header at `offset` of `volume`.
fn read_header_at(volume: &dyn BlockSource, offset: u64) -> Result<VolumeHeader> {
    let mut buf = [0u8; VOLUME_HEADER_LEN];
    volume.read_at(offset, &mut buf)?;
    VolumeHeader::parse(&buf)
}

/// Reads the volume header of `volume`, which is the HFS+ volume itself (a
/// slice at [`Located::offset`], for a wrapped one).
///
/// The header at byte 1024 is tried first. If it is unreadable or fails
/// validation, the alternate header 1024 bytes before the end of `volume`
/// is tried and returned if it is good. When both fail, the primary's error
/// is returned.
pub fn read_header(volume: &dyn BlockSource) -> Result<VolumeHeader> {
    let primary = match read_header_at(volume, VOLUME_HEADER_OFFSET) {
        Ok(header) => return Ok(header),
        Err(e @ (Error::Corrupt(_) | Error::OutOfBounds { .. })) => e,
        // A device that fails to read says nothing about the alternate.
        Err(e) => return Err(e),
    };
    if let Some(alternate) = volume.len().checked_sub(ALTERNATE_HEADER_FROM_END)
        && alternate != VOLUME_HEADER_OFFSET
        && let Ok(header) = read_header_at(volume, alternate)
    {
        return Ok(header);
    }
    Err(primary)
}

/// An HFS+ date (unsigned seconds since 1904-01-01 00:00 GMT) as a
/// `Timestamp`. Zero means the field was never set and gives `None`.
pub fn hfs_time(seconds: u32) -> Option<Timestamp> {
    if seconds == 0 {
        return None;
    }
    let unix = i64::from(seconds).checked_sub(HFS_EPOCH_TO_UNIX)?;
    Some(Timestamp::from_unix(unix, 0))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use atlas_fs_core::io::block::{MemSource, Slice};

    const BS: u32 = 4096;
    const TOTAL: u32 = 1000;

    fn put16(b: &mut [u8], at: usize, v: u16) {
        b[at..at + 2].copy_from_slice(&v.to_be_bytes());
    }

    fn put32(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_be_bytes());
    }

    fn put_fork(b: &mut [u8], at: usize, logical: u64, blocks: u32, extents: &[(u32, u32)]) {
        b[at..at + 8].copy_from_slice(&logical.to_be_bytes());
        put32(b, at + 8, 0x10000); // clump size
        put32(b, at + 12, blocks);
        for (i, (start, count)) in extents.iter().enumerate() {
            put32(b, at + 16 + i * 8, *start);
            put32(b, at + 20 + i * 8, *count);
        }
    }

    /// A valid 512-byte header.
    fn header(signature: u16, version: u16) -> Vec<u8> {
        let mut b = vec![0u8; VOLUME_HEADER_LEN];
        put16(&mut b, 0, signature);
        put16(&mut b, 2, version);
        put32(&mut b, 4, ATTR_UNMOUNTED);
        put32(&mut b, 8, 0x3130_2E30); // lastMountedVersion
        put32(&mut b, 16, 3_000_000_000); // create
        put32(&mut b, 20, 3_100_000_000); // modify
        put32(&mut b, 24, 0); // backup unset
        put32(&mut b, 28, 3_200_000_000); // checked
        put32(&mut b, 32, 77); // files
        put32(&mut b, 36, 12); // folders
        put32(&mut b, 40, BS);
        put32(&mut b, 44, TOTAL);
        put32(&mut b, 48, 500);
        put32(&mut b, 52, 9);
        put32(&mut b, 56, 0x20000);
        put32(&mut b, 60, 0x30000);
        put32(&mut b, 64, 4242);
        put32(&mut b, 68, 31337); // writeCount
        b[72..80].copy_from_slice(&0x8000_0000_0000_0101u64.to_be_bytes());
        put32(&mut b, 80 + 5 * 4, 0xCAFE); // finderInfo[5]
        put_fork(&mut b, 112, 4096, 1, &[(0, 1)]);
        put_fork(&mut b, 192, 4096, 1, &[(1, 1)]);
        put_fork(&mut b, 272, 8192, 2, &[(2, 2)]);
        put_fork(&mut b, 352, 4096, 1, &[(4, 1)]);
        put_fork(&mut b, 432, 4096, 1, &[(5, 1)]);
        b
    }

    fn hfs_plus() -> Vec<u8> {
        header(SIGNATURE_HFS_PLUS, 4)
    }

    fn corrupt<T: std::fmt::Debug>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    fn first(f: &ForkData) -> (u32, u32) {
        (f.extents[0].start_block, f.extents[0].block_count)
    }

    #[test]
    fn parses_hfs_plus() {
        let h = VolumeHeader::parse(&hfs_plus()).unwrap();
        assert_eq!((h.signature, h.version), (SIGNATURE_HFS_PLUS, 4));
        assert_eq!(h.last_mounted_version, 0x3130_2E30);
        assert_eq!(h.create_date, 3_000_000_000);
        assert_eq!(h.modify_date, 3_100_000_000);
        assert_eq!(h.backup_date, 0);
        assert_eq!(h.checked_date, 3_200_000_000);
        assert_eq!((h.file_count, h.folder_count), (77, 12));
        assert_eq!(
            (h.block_size, h.total_blocks, h.free_blocks),
            (BS, TOTAL, 500)
        );
        assert_eq!(h.next_allocation, 9);
        assert_eq!((h.rsrc_clump_size, h.data_clump_size), (0x20000, 0x30000));
        assert_eq!(h.next_catalog_id, 4242);
        assert_eq!(h.write_count, 31337);
        assert_eq!(h.encodings_bitmap, 0x8000_0000_0000_0101);
        assert_eq!(h.finder_info[5], 0xCAFE);
        assert_eq!(h.finder_info[4], 0);
        assert_eq!(first(&h.allocation_file), (0, 1));
        assert_eq!(first(&h.extents_file), (1, 1));
        assert_eq!(first(&h.catalog_file), (2, 2));
        assert_eq!(h.catalog_file.logical_size, 8192);
        assert_eq!(first(&h.attributes_file), (4, 1));
        assert_eq!(first(&h.startup_file), (5, 1));
        assert_eq!(h.volume_bytes(), Some(u64::from(BS) * u64::from(TOTAL)));
        assert!(!h.is_hfsx());
    }

    #[test]
    fn parses_hfsx_and_ignores_trailing_bytes() {
        let mut b = header(SIGNATURE_HFSX, 5);
        b.extend_from_slice(&[0xFF; 100]);
        let h = VolumeHeader::parse(&b).unwrap();
        assert!(h.is_hfsx());
        assert_eq!(h.version, 5);
        assert_eq!(h.write_count, 31337);
        assert_eq!(first(&h.catalog_file), (2, 2));
    }

    #[test]
    fn flags() {
        let mut b = hfs_plus();
        let h = VolumeHeader::parse(&b).unwrap();
        assert!(h.unmounted_cleanly());
        assert!(!h.is_journaled());

        // Journaled bit alone is not enough: the info block must be set.
        put32(&mut b, 4, ATTR_JOURNALED);
        let h = VolumeHeader::parse(&b).unwrap();
        assert!(!h.unmounted_cleanly());
        assert!(!h.is_journaled());

        put32(&mut b, 12, 6);
        assert!(VolumeHeader::parse(&b).unwrap().is_journaled());

        // An info block without the bit is not journaled.
        put32(&mut b, 4, ATTR_UNMOUNTED);
        let h = VolumeHeader::parse(&b).unwrap();
        assert!(!h.is_journaled());
        assert!(h.unmounted_cleanly());
    }

    #[test]
    fn volume_bytes_is_checked() {
        let mut h = VolumeHeader::parse(&hfs_plus()).unwrap();
        h.block_size = u32::MAX;
        h.total_blocks = u32::MAX;
        assert_eq!(
            h.volume_bytes(),
            Some(u64::from(u32::MAX) * u64::from(u32::MAX))
        );
    }

    #[test]
    fn rejects_wrong_signature() {
        let c = corrupt(VolumeHeader::parse(&header(0x4244, 4)));
        assert_eq!(c.structure, "volume header");
        assert!(matches!(
            c.kind,
            CorruptKind::BadSignature { found: 0x4244, .. }
        ));
    }

    #[test]
    fn rejects_mismatched_version() {
        for (sig, version) in [
            (SIGNATURE_HFS_PLUS, 5),
            (SIGNATURE_HFSX, 4),
            (SIGNATURE_HFS_PLUS, 0),
        ] {
            let c = corrupt(VolumeHeader::parse(&header(sig, version)));
            assert!(
                matches!(
                    c.kind,
                    CorruptKind::InvalidValue {
                        field: "version",
                        ..
                    }
                ),
                "{sig:#x} v{version}: {c}"
            );
        }
    }

    #[test]
    fn rejects_bad_block_sizes() {
        for bs in [1000, 256, 0, 1024 + 512] {
            let mut b = hfs_plus();
            put32(&mut b, 40, bs);
            let c = corrupt(VolumeHeader::parse(&b));
            assert!(
                matches!(
                    c.kind,
                    CorruptKind::InvalidValue {
                        field: "block size",
                        ..
                    }
                ),
                "{bs}: {c}"
            );
        }
        let mut b = hfs_plus();
        put32(&mut b, 40, 1 << 21);
        assert!(matches!(
            corrupt(VolumeHeader::parse(&b)).kind,
            CorruptKind::OutOfRange {
                field: "block size",
                ..
            }
        ));
        for bs in [512, 65536, 1 << 20] {
            let mut b = hfs_plus();
            put32(&mut b, 40, bs);
            // Keep the fixture forks inside their capacity at the small size.
            for at in [112, 192, 352, 432] {
                put_fork(&mut b, at, 512, 1, &[(at as u32, 1)]);
            }
            put_fork(&mut b, 272, 1024, 2, &[(2, 2)]);
            VolumeHeader::parse(&b).unwrap();
        }
    }

    #[test]
    fn rejects_bad_block_counts() {
        let mut b = hfs_plus();
        put32(&mut b, 44, 0);
        assert!(matches!(
            corrupt(VolumeHeader::parse(&b)).kind,
            CorruptKind::InvalidValue {
                field: "total blocks",
                ..
            }
        ));

        let mut b = hfs_plus();
        put32(&mut b, 48, TOTAL + 1);
        assert!(matches!(
            corrupt(VolumeHeader::parse(&b)).kind,
            CorruptKind::OutOfRange {
                field: "free blocks",
                ..
            }
        ));
        put32(&mut b, 48, TOTAL);
        VolumeHeader::parse(&b).unwrap();
    }

    #[test]
    fn rejects_extents_past_the_volume() {
        let mut b = hfs_plus();
        put_fork(&mut b, 272, 8192, 2, &[(TOTAL - 1, 2)]);
        let c = corrupt(VolumeHeader::parse(&b));
        assert_eq!(c.location, Location::Byte(272));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "extent end",
                ..
            }
        ));

        // Exactly to the end is fine.
        put_fork(&mut b, 272, 8192, 2, &[(TOTAL - 2, 2)]);
        VolumeHeader::parse(&b).unwrap();

        // start + count overflowing u32 is an error, not a wrap.
        put_fork(&mut b, 352, 4096, 1, &[(u32::MAX, 2)]);
        let c = corrupt(VolumeHeader::parse(&b));
        assert_eq!(c.location, Location::Byte(352));
        assert!(matches!(c.kind, CorruptKind::Overflow { .. }));
    }

    #[test]
    fn extents_file_must_be_covered_by_its_extents() {
        let mut b = hfs_plus();
        put_fork(&mut b, 192, 3 * 4096, 3, &[(1, 1)]);
        let c = corrupt(VolumeHeader::parse(&b));
        assert_eq!(c.location, Location::Byte(192));
        assert!(matches!(c.kind, CorruptKind::Other(_)));

        // Extents that do cover it are fine.
        put_fork(&mut b, 192, 3 * 4096, 3, &[(1, 1), (10, 2)]);
        VolumeHeader::parse(&b).unwrap();
    }

    #[test]
    fn catalog_may_continue_in_overflow() {
        let mut b = hfs_plus();
        put_fork(&mut b, 272, 50 * 4096, 50, &[(2, 2)]);
        VolumeHeader::parse(&b).unwrap();
    }

    #[test]
    fn rejects_inconsistent_fork_counts() {
        // More blocks in the extents than the fork claims.
        let mut b = hfs_plus();
        put_fork(&mut b, 272, 4096, 1, &[(2, 2)]);
        assert!(matches!(
            corrupt(VolumeHeader::parse(&b)).kind,
            CorruptKind::OutOfRange {
                limit_name: "fork total blocks",
                ..
            }
        ));
        // A fork claiming more blocks than the volume has.
        let mut b = hfs_plus();
        put_fork(&mut b, 272, 4096, TOTAL + 1, &[(2, 2)]);
        assert!(matches!(
            corrupt(VolumeHeader::parse(&b)).kind,
            CorruptKind::OutOfRange {
                field: "fork total blocks",
                ..
            }
        ));
    }

    #[test]
    fn rejects_logical_size_past_capacity() {
        let mut b = hfs_plus();
        put_fork(&mut b, 272, 2 * 4096 + 1, 2, &[(2, 2)]);
        let c = corrupt(VolumeHeader::parse(&b));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "fork logical size",
                ..
            }
        ));
        put_fork(&mut b, 272, u64::MAX, 2, &[(2, 2)]);
        assert!(VolumeHeader::parse(&b).is_err());
    }

    #[test]
    fn rejects_short_buffers() {
        let b = hfs_plus();
        for len in [0, 1, 3, 111, 431, 511] {
            let c = corrupt(VolumeHeader::parse(&b[..len]));
            assert!(
                matches!(c.kind, CorruptKind::Truncated { needed: 512, .. }),
                "{len}: {c}"
            );
        }
    }

    /// A source of `total` bytes with an HFS+ header at byte 1024.
    fn plain_source(total: usize) -> MemSource {
        let mut bytes = vec![0u8; total];
        bytes[1024..1536].copy_from_slice(&hfs_plus());
        MemSource::new(bytes)
    }

    #[test]
    fn locates_bare_volume() {
        let src = plain_source(8192);
        assert_eq!(
            locate(&src).unwrap(),
            Located {
                offset: 0,
                len: 8192,
                wrapped: false
            }
        );
        let mut bytes = vec![0u8; 2048];
        bytes[1024..1536].copy_from_slice(&header(SIGNATURE_HFSX, 5));
        assert!(!locate(&MemSource::new(bytes)).unwrap().wrapped);
        // Exactly the minimum length.
        assert!(locate(&plain_source(1536)).is_ok());
    }

    const WRAPPER_BLOCK: u32 = 4096;
    const WRAPPER_SECTORS: u16 = 8;
    const EMBED_START: u16 = 3;
    const EMBED_COUNT: u16 = 5;
    const EMBED_OFFSET: u64 = 8 * 512 + 3 * 4096;
    const EMBED_LEN: u64 = 5 * 4096;

    /// An HFS wrapper over `total` bytes with an HFS+ header inside.
    fn wrapped_bytes(total: usize, embed_sig: u16) -> Vec<u8> {
        let mut bytes = vec![0u8; total];
        let mdb = 1024;
        put16(&mut bytes, mdb, 0x4244);
        put32(&mut bytes, mdb + 0x14, WRAPPER_BLOCK);
        put16(&mut bytes, mdb + 0x1C, WRAPPER_SECTORS);
        put16(&mut bytes, mdb + 0x7C, embed_sig);
        put16(&mut bytes, mdb + 0x7E, EMBED_START);
        put16(&mut bytes, mdb + 0x80, EMBED_COUNT);
        let inner = usize::try_from(EMBED_OFFSET + 1024).unwrap();
        if inner + 512 <= total {
            bytes[inner..inner + 512].copy_from_slice(&hfs_plus());
        }
        bytes
    }

    fn full_wrapped_len() -> usize {
        usize::try_from(EMBED_OFFSET + EMBED_LEN).unwrap()
    }

    #[test]
    fn locates_wrapped_volume() {
        let bytes = wrapped_bytes(full_wrapped_len(), SIGNATURE_HFS_PLUS);
        let src: Arc<dyn BlockSource> = Arc::new(MemSource::new(bytes));
        let located = locate(src.as_ref()).unwrap();
        assert_eq!(
            located,
            Located {
                offset: EMBED_OFFSET,
                len: EMBED_LEN,
                wrapped: true
            }
        );
        let volume = Slice::new(Arc::clone(&src), located.offset, located.len).unwrap();
        let h = read_header(&volume).unwrap();
        assert_eq!(h.write_count, 31337);
    }

    #[test]
    fn rejects_wrapper_past_the_source() {
        let total = full_wrapped_len() - 1;
        let c = corrupt(locate(&MemSource::new(wrapped_bytes(
            total,
            SIGNATURE_HFS_PLUS,
        ))));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "embedded volume end",
                ..
            }
        ));
    }

    #[test]
    fn rejects_bad_wrapper_fields() {
        let total = full_wrapped_len();
        for block_size in [0u32, 1000, 4097] {
            let mut bytes = wrapped_bytes(total, SIGNATURE_HFS_PLUS);
            put32(&mut bytes, 1024 + 0x14, block_size);
            let c = corrupt(locate(&MemSource::new(bytes)));
            assert!(
                matches!(
                    c.kind,
                    CorruptKind::InvalidValue {
                        field: "allocation block size",
                        ..
                    }
                ),
                "{block_size}: {c}"
            );
        }
        let mut bytes = wrapped_bytes(total, SIGNATURE_HFS_PLUS);
        put16(&mut bytes, 1024 + 0x80, 0);
        assert!(matches!(
            corrupt(locate(&MemSource::new(bytes))).kind,
            CorruptKind::InvalidValue {
                field: "embedded block count",
                ..
            }
        ));
        // Huge values stay inside checked arithmetic and fail.
        let mut bytes = wrapped_bytes(total, SIGNATURE_HFS_PLUS);
        put32(&mut bytes, 1024 + 0x14, 0xFFFF_FE00);
        put16(&mut bytes, 1024 + 0x7E, 0xFFFF);
        put16(&mut bytes, 1024 + 0x80, 0xFFFF);
        assert!(locate(&MemSource::new(bytes)).is_err());
    }

    #[test]
    fn plain_hfs_is_not_hfs_plus() {
        let c = corrupt(locate(&MemSource::new(wrapped_bytes(
            full_wrapped_len(),
            0,
        ))));
        assert!(matches!(c.kind, CorruptKind::BadSignature { found: 0, .. }));
        let c = corrupt(locate(&MemSource::new(vec![0u8; 4096])));
        assert!(matches!(c.kind, CorruptKind::BadSignature { found: 0, .. }));
    }

    #[test]
    fn locate_rejects_short_sources() {
        for len in [0, 512, 1024, 1535] {
            let c = corrupt(locate(&MemSource::new(vec![0u8; len])));
            assert!(
                matches!(c.kind, CorruptKind::Truncated { needed: 1536, .. }),
                "{len}: {c}"
            );
        }
    }

    /// A 16 KiB volume with the primary and alternate headers as given.
    fn volume_with(primary: &[u8], alternate: &[u8]) -> MemSource {
        let mut bytes = vec![0u8; 16384];
        bytes[1024..1536].copy_from_slice(primary);
        let alt = bytes.len() - 1024;
        bytes[alt..alt + 512].copy_from_slice(alternate);
        MemSource::new(bytes)
    }

    #[test]
    fn read_header_prefers_primary() {
        let mut other = hfs_plus();
        put32(&mut other, 68, 1);
        let h = read_header(&volume_with(&hfs_plus(), &other)).unwrap();
        assert_eq!(h.write_count, 31337);
    }

    #[test]
    fn read_header_falls_back_to_alternate() {
        let mut alternate = hfs_plus();
        put32(&mut alternate, 68, 99);
        let mut damaged = hfs_plus();
        put16(&mut damaged, 0, 0);
        let h = read_header(&volume_with(&damaged, &alternate)).unwrap();
        assert_eq!(h.write_count, 99);

        // A primary that has the signature but fails validation too.
        let mut damaged = hfs_plus();
        put32(&mut damaged, 40, 1000);
        let h = read_header(&volume_with(&damaged, &alternate)).unwrap();
        assert_eq!(h.write_count, 99);
    }

    #[test]
    fn read_header_returns_primary_error_when_both_fail() {
        let mut primary = hfs_plus();
        put32(&mut primary, 40, 1000); // bad block size
        let mut alternate = hfs_plus();
        put16(&mut alternate, 0, 0); // bad signature
        let c = corrupt(read_header(&volume_with(&primary, &alternate)));
        assert!(matches!(
            c.kind,
            CorruptKind::InvalidValue {
                field: "block size",
                ..
            }
        ));
    }

    #[test]
    fn read_header_on_tiny_volumes() {
        // Too short for a header at 1024: an error, no panic.
        assert!(read_header(&MemSource::new(vec![0u8; 100])).is_err());
        assert!(read_header(&MemSource::new(vec![0u8; 1024])).is_err());
        assert!(read_header(&MemSource::new(Vec::new())).is_err());
    }

    #[test]
    fn hfs_time_conversion() {
        assert_eq!(hfs_time(0), None);
        assert_eq!(hfs_time(2_082_844_800), Some(Timestamp::from_unix(0, 0)));
        assert_eq!(hfs_time(2_082_844_801), Some(Timestamp::from_unix(1, 0)));
        // Before 1970 is negative, not wrapped.
        assert_eq!(hfs_time(1), Some(Timestamp::from_unix(-2_082_844_799, 0)));
        // The far end of the range converts without panicking.
        assert!(hfs_time(u32::MAX).is_some());
    }
}
