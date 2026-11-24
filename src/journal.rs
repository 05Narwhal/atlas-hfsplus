// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Journal replay as a read-only overlay.
//!
//! A journaled HFS+ volume that was not unmounted cleanly can hold metadata
//! in its journal that never reached its home blocks. macOS replays the
//! journal at mount by writing those blocks; [`open`] does the same walk
//! (xnu `replay_journal`) and keeps the result in memory as a
//! [`JournalOverlay`]: a [`BlockSource`] over the partition that returns the
//! journal's copy of every replayed byte. Nothing is written to the disk.
//!
//! On-disk layout (xnu `hfs_journal.h`, `hfs_format.h`): the volume header's
//! `journalInfoBlock` names a big-endian `JournalInfoBlock` whose `offset`
//! (relative to the HFS+ volume) and `size` give the journal. The journal
//! begins with a `journal_header` and is otherwise a ring of transactions,
//! each one or more block lists: a `block_list_header` of `blhdr_size` bytes
//! followed by the blocks' data. Journal structures are in the byte order of
//! the Mac that wrote them; the magic number says which. Block numbers count
//! `jhdr_size` bytes from the start of the partition (the HFS wrapper's
//! start for a wrapped volume).
//!
//! Every value read from the journal is checked before use, beyond what xnu
//! checks: block sizes are positive, a block list's data fits in it and in
//! one lap of the ring, every block lies on the device, the walk covers at
//! most one lap and at most [`MAX_BLOCKS`] blocks.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use super::bytes::array;
use super::volume_header::VolumeHeader;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::io::block::{BlockSource, check_range};

/// `kJIJournalInFSMask`: the journal lives inside this volume.
const JIB_IN_FS: u32 = 1;
/// `kJIJournalOnOtherDeviceMask`: the journal lives on another device.
const JIB_ON_OTHER_DEVICE: u32 = 2;
/// `kJIJournalNeedInitMask`: the journal is to be initialized, not replayed.
const JIB_NEED_INIT: u32 = 4;
/// Bytes of the `JournalInfoBlock` read: flags, device signature, offset, size.
const JIB_LEN: usize = 52;

/// `JNLx`, `JOURNAL_HEADER_MAGIC`.
const MAGIC: u32 = 0x4A4E_4C78;
/// `JHDR`, `OLD_JOURNAL_HEADER_MAGIC`.
const OLD_MAGIC: u32 = 0x4A48_4452;
/// Bytes of the `journal_header` read.
const HEADER_LEN: usize = 48;
/// `JOURNAL_HEADER_CKSUM_SIZE`: the header bytes its checksum covers.
const HEADER_CHECKSUM_LEN: usize = 44;
/// Byte range of the header's checksum field.
const HEADER_CHECKSUM_FIELD: Range<usize> = 36..40;

/// `BLHDR_CHECKSUM_SIZE`: the block list header bytes its checksum covers
/// (the 16-byte header and `binfo[0]`).
const BLHDR_LEN: usize = 32;
/// Byte range of the block list header's checksum field.
const BLHDR_CHECKSUM_FIELD: Range<usize> = 8..12;
/// Size of one `block_info`.
const BLOCK_INFO_LEN: u64 = 16;
/// `BLHDR_CHECK_CHECKSUMS`.
const CHECK_CHECKSUMS: i32 = 1;
/// `BLHDR_FIRST_HEADER`: this block list starts a transaction.
const FIRST_HEADER: i32 = 2;
/// The block number of a killed block, whose data is skipped.
const KILLED: i64 = -1;

/// Smallest journal xnu accepts.
const MIN_JOURNAL_SIZE: u64 = 256 * 1024;
/// Largest journal, and largest start and end offset, xnu accepts.
const MAX_JOURNAL_SIZE: u64 = 1 << 30;
/// `start` and `end` are multiples of this.
const JOURNAL_ALIGN: u64 = 512;
/// Smallest `jhdr_size`.
const MIN_JHDR_SIZE: u64 = 512;
/// Largest `jhdr_size` (the largest sector size).
const MAX_JHDR_SIZE: u64 = 65536;
/// Most blocks (not counting killed ones) one replay accepts.
pub const MAX_BLOCKS: u64 = 1 << 20;

/// Text of [`JournalState::NotReplayed`] for an external journal that may
/// be dirty.
const EXTERNAL_JOURNAL: &str = "journal is on another device";

const JIB: &str = "journal info block";
const HEADER: &str = "journal header";
const BLOCK_LIST: &str = "journal block list";

pub use atlas_fs_core::fs::JournalState;

/// The device to read the volume through, and what was done to get it.
pub struct Opened {
    /// The device itself, or a [`JournalOverlay`] over it.
    pub device: Arc<dyn BlockSource>,
    /// What happened to the journal.
    pub state: JournalState,
}

/// Opens the journal of the volume at `volume_offset` in `device` (the
/// partition, before any wrapper is cut away) and replays it in memory.
///
/// `header` is the volume header as it is on disk. Fails only when the
/// device cannot be read (`Error::Io`, `Error::AccessDenied`); a journal
/// that is damaged or cannot be replayed is reported in
/// [`Opened::state`] with the device unchanged.
pub fn open(
    device: Arc<dyn BlockSource>,
    volume_offset: u64,
    header: &VolumeHeader,
) -> Result<Opened> {
    let plan = plan(&*device, volume_offset, header);
    finish(device, plan)
}

/// Turns a replay plan into the device and state `open` returns.
fn finish(device: Arc<dyn BlockSource>, plan: Result<Plan>) -> Result<Opened> {
    match plan {
        Ok(Plan::Clean) => Ok(Opened {
            device,
            state: JournalState::Clean,
        }),
        Ok(Plan::External) => Ok(Opened {
            device,
            state: JournalState::NotReplayed {
                reason: EXTERNAL_JOURNAL.to_string(),
            },
        }),
        Ok(Plan::Replay {
            patches,
            transactions,
            stopped_early,
        }) => {
            let device: Arc<dyn BlockSource> = if patches.is_empty() {
                device
            } else {
                Arc::new(JournalOverlay {
                    inner: device,
                    patches,
                })
            };
            Ok(Opened {
                device,
                state: JournalState::Replayed {
                    transactions,
                    stopped_early,
                },
            })
        }
        Err(e @ (Error::Io { .. } | Error::AccessDenied { .. })) => Err(e),
        Err(e) => Ok(Opened {
            device,
            state: JournalState::NotReplayed {
                reason: e.to_string(),
            },
        }),
    }
}

/// What to do with the device.
enum Plan {
    Clean,
    /// The journal is elsewhere and the volume was not unmounted cleanly.
    External,
    Replay {
        patches: BTreeMap<u64, Patch>,
        transactions: u32,
        stopped_early: bool,
    },
}

fn corrupt(structure: &'static str, at: u64, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(structure, Location::Byte(at), kind))
}

fn out_of_range(
    structure: &'static str,
    at: u64,
    field: &'static str,
    value: u64,
    limit_name: &'static str,
    limit: u64,
) -> Error {
    corrupt(
        structure,
        at,
        CorruptKind::OutOfRange {
            field,
            value,
            limit_name,
            limit,
        },
    )
}

fn invalid(structure: &'static str, at: u64, field: &'static str, value: u64) -> Error {
    corrupt(structure, at, CorruptKind::InvalidValue { field, value })
}

fn overflow(structure: &'static str, at: u64, field: &'static str) -> Error {
    corrupt(structure, at, CorruptKind::Overflow { field })
}

/// A signed on-disk value as `u64` for an error message (two's complement).
fn raw_i64(v: i64) -> u64 {
    u64::from_ne_bytes(v.to_ne_bytes())
}

/// Finds the journal through the journal info block and replays it.
fn plan(device: &dyn BlockSource, volume_offset: u64, header: &VolumeHeader) -> Result<Plan> {
    // xnu hfs_mountfs: the journal is only opened (and replayed) when the
    // last mount journaled; otherwise it is stale.
    if !header.is_journaled() || !header.last_mounted_journaled() {
        return Ok(Plan::Clean);
    }
    let device_len = device.len();
    let mut volume_len = device_len
        .checked_sub(volume_offset)
        .ok_or(Error::OutOfBounds {
            offset: volume_offset,
            len: 0,
            size: device_len,
        })?;
    if let Some(bytes) = header.volume_bytes() {
        volume_len = volume_len.min(bytes);
    }

    let jib_at = u64::from(header.journal_info_block)
        .checked_mul(u64::from(header.block_size))
        .ok_or_else(|| overflow(JIB, 0, "journal info block offset"))?;
    let jib_end = jib_at
        .checked_add(JIB_LEN as u64)
        .ok_or_else(|| overflow(JIB, jib_at, "journal info block end"))?;
    if jib_end > volume_len {
        return Err(out_of_range(
            JIB,
            jib_at,
            "journal info block end",
            jib_end,
            "volume size",
            volume_len,
        ));
    }
    let jib_device = volume_offset
        .checked_add(jib_at)
        .ok_or_else(|| overflow(JIB, jib_at, "journal info block offset"))?;
    let mut jib = [0u8; JIB_LEN];
    device.read_at(jib_device, &mut jib)?;
    let short = || {
        corrupt(
            JIB,
            jib_at,
            CorruptKind::Truncated {
                needed: JIB_LEN as u64,
                available: 0,
            },
        )
    };
    let flags = array::<4>(&jib, 0)
        .map(u32::from_be_bytes)
        .ok_or_else(short)?;
    let offset = array::<8>(&jib, 36)
        .map(u64::from_be_bytes)
        .ok_or_else(short)?;
    let size = array::<8>(&jib, 44)
        .map(u64::from_be_bytes)
        .ok_or_else(short)?;

    if flags & JIB_NEED_INIT != 0 {
        return Ok(Plan::Clean);
    }
    if flags & JIB_IN_FS == 0 || flags & JIB_ON_OTHER_DEVICE != 0 {
        return Ok(if header.unmounted_cleanly() {
            Plan::Clean
        } else {
            Plan::External
        });
    }
    let end = offset
        .checked_add(size)
        .ok_or_else(|| overflow(JIB, jib_at, "journal end"))?;
    if end > volume_len {
        return Err(out_of_range(
            JIB,
            jib_at,
            "journal end",
            end,
            "volume size",
            volume_len,
        ));
    }
    let base = volume_offset
        .checked_add(offset)
        .ok_or_else(|| overflow(JIB, jib_at, "journal offset"))?;
    open_at(device, base, size)
}

/// Replays the journal of `region_len` bytes at device byte `base`.
fn open_at(device: &dyn BlockSource, base: u64, region_len: u64) -> Result<Plan> {
    let device_len = device.len();
    match base.checked_add(region_len) {
        Some(end) if end <= device_len => {}
        _ => {
            return Err(out_of_range(
                HEADER,
                0,
                "journal end",
                base.saturating_add(region_len),
                "device size",
                device_len,
            ));
        }
    }
    if region_len < HEADER_LEN as u64 {
        return Err(corrupt(
            HEADER,
            0,
            CorruptKind::Truncated {
                needed: HEADER_LEN as u64,
                available: region_len,
            },
        ));
    }
    let mut raw = [0u8; HEADER_LEN];
    device.read_at(base, &mut raw)?;
    let header = Header::parse(&raw, region_len)?;
    if header.start == header.end {
        return Ok(Plan::Clean);
    }
    let ring = Ring {
        device,
        base,
        first: header.jhdr_size,
        size: header.size,
        // Cannot underflow: parse checks size > jhdr_size.
        lap: header.size.saturating_sub(header.jhdr_size),
    };
    let walk = walk(&ring, &header, device_len)?;
    let patches = patches(&ring, &walk.lists);
    Ok(Plan::Replay {
        patches,
        transactions: walk.transactions(),
        stopped_early: walk.stopped_early,
    })
}

/// `calc_checksum` from xnu: a shift-xor over `bytes`, inverted. The walk
/// sums blocks in pieces with [`fold`]; this whole-slice form builds test
/// journals.
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
pub fn calc_checksum(bytes: &[u8]) -> u32 {
    checksum(bytes.iter().copied())
}

fn checksum(bytes: impl Iterator<Item = u8>) -> u32 {
    !fold(0, bytes)
}

/// Folds `bytes` into the running state `c` of [`calc_checksum`] (before
/// its final inversion), so a long block can be summed in pieces.
fn fold(mut c: u32, bytes: impl Iterator<Item = u8>) -> u32 {
    for b in bytes {
        c = c.wrapping_shl(8) ^ c.wrapping_add(u32::from(b));
    }
    c
}

/// Bytes of block data read at a time to check a block's checksum.
const CHECKSUM_CHUNK: u64 = 64 * 1024;

/// The checksum of `bytes` with the checksum field `skip` read as zero.
fn checksum_without(bytes: &[u8], skip: Range<usize>) -> u32 {
    checksum(
        bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| if skip.contains(&i) { 0 } else { b }),
    )
}

/// The journal's byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Big,
    Little,
}

impl Order {
    fn u16(self, b: &[u8], at: usize) -> Option<u16> {
        let a = array::<2>(b, at)?;
        Some(match self {
            Self::Big => u16::from_be_bytes(a),
            Self::Little => u16::from_le_bytes(a),
        })
    }

    fn u32(self, b: &[u8], at: usize) -> Option<u32> {
        let a = array::<4>(b, at)?;
        Some(match self {
            Self::Big => u32::from_be_bytes(a),
            Self::Little => u32::from_le_bytes(a),
        })
    }

    fn i32(self, b: &[u8], at: usize) -> Option<i32> {
        let a = array::<4>(b, at)?;
        Some(match self {
            Self::Big => i32::from_be_bytes(a),
            Self::Little => i32::from_le_bytes(a),
        })
    }

    fn i64(self, b: &[u8], at: usize) -> Option<i64> {
        let a = array::<8>(b, at)?;
        Some(match self {
            Self::Big => i64::from_be_bytes(a),
            Self::Little => i64::from_le_bytes(a),
        })
    }
}

/// A validated `journal_header`. Offsets are journal bytes; `start` and
/// `end` are already wrapped (`size` becomes `jhdr_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    order: Order,
    /// The header used the old `JHDR` magic.
    old_magic: bool,
    /// The stored header checksum matched. xnu only logs a mismatch, and so
    /// does nothing here; tests read it.
    checksum_ok: bool,
    start: u64,
    end: u64,
    size: u64,
    blhdr_size: u64,
    jhdr_size: u64,
}

impl Header {
    /// Parses the first [`HEADER_LEN`] bytes of a journal whose region (from
    /// the journal info block) is `region_len` bytes.
    fn parse(raw: &[u8; HEADER_LEN], region_len: u64) -> Result<Self> {
        let short = || {
            corrupt(
                HEADER,
                0,
                CorruptKind::Truncated {
                    needed: HEADER_LEN as u64,
                    available: 0,
                },
            )
        };
        let be = Order::Big.u32(raw, 0).ok_or_else(short)?;
        let le = Order::Little.u32(raw, 0).ok_or_else(short)?;
        let (order, magic) = if be == MAGIC || be == OLD_MAGIC {
            (Order::Big, be)
        } else if le == MAGIC || le == OLD_MAGIC {
            (Order::Little, le)
        } else {
            return Err(corrupt(
                HEADER,
                0,
                CorruptKind::BadSignature {
                    expected: u64::from(MAGIC),
                    found: u64::from(be),
                },
            ));
        };
        let i64_at = |at| order.i64(raw, at).ok_or_else(short);
        let i32_at = |at| order.i32(raw, at).ok_or_else(short);
        let start = i64_at(8)?;
        let end = i64_at(16)?;
        let size = i64_at(24)?;
        let blhdr_size = i32_at(32)?;
        let stored = order.u32(raw, 36).ok_or_else(short)?;
        let jhdr_size = i32_at(40)?;
        let checksum_ok = raw
            .get(..HEADER_CHECKSUM_LEN)
            .is_some_and(|b| checksum_without(b, HEADER_CHECKSUM_FIELD) == stored);

        let jhdr_size = u64::try_from(jhdr_size)
            .ok()
            .filter(|s| (MIN_JHDR_SIZE..=MAX_JHDR_SIZE).contains(s) && s.is_power_of_two())
            .ok_or_else(|| invalid(HEADER, 40, "jhdr_size", raw_i64(i64::from(jhdr_size))))?;
        let size = u64::try_from(size)
            .ok()
            .filter(|s| (MIN_JOURNAL_SIZE..=MAX_JOURNAL_SIZE).contains(s))
            .ok_or_else(|| invalid(HEADER, 24, "journal size", raw_i64(size)))?;
        if size > region_len {
            return Err(out_of_range(
                HEADER,
                24,
                "journal size",
                size,
                "journal info block size",
                region_len,
            ));
        }
        if size <= jhdr_size {
            return Err(out_of_range(
                HEADER,
                40,
                "jhdr_size",
                jhdr_size,
                "journal size",
                size,
            ));
        }
        let lap = size.saturating_sub(jhdr_size);
        let pointer = |value: i64, at: u64, field: &'static str| -> Result<u64> {
            let v = u64::try_from(value)
                .ok()
                .filter(|v| *v > 0 && *v <= size && *v <= MAX_JOURNAL_SIZE)
                .ok_or_else(|| invalid(HEADER, at, field, raw_i64(value)))?;
            if v % JOURNAL_ALIGN != 0 {
                return Err(invalid(HEADER, at, field, v));
            }
            let v = if v == size { jhdr_size } else { v };
            if v < jhdr_size {
                return Err(out_of_range(HEADER, at, field, v, "jhdr_size", jhdr_size));
            }
            Ok(v)
        };
        let start = pointer(start, 8, "start")?;
        let end = pointer(end, 16, "end")?;
        let blhdr_size = u64::try_from(blhdr_size)
            .ok()
            .filter(|s| *s >= BLHDR_LEN as u64)
            .ok_or_else(|| invalid(HEADER, 32, "blhdr_size", raw_i64(i64::from(blhdr_size))))?;
        if blhdr_size > lap {
            return Err(out_of_range(
                HEADER,
                32,
                "blhdr_size",
                blhdr_size,
                "journal data size",
                lap,
            ));
        }
        // xnu journal_open checks this after replay.
        let max_blocks = (blhdr_size / BLOCK_INFO_LEN).saturating_sub(1);
        let limit = size.checked_div(jhdr_size).unwrap_or(0);
        if max_blocks > limit {
            return Err(out_of_range(
                HEADER,
                32,
                "block list capacity",
                max_blocks,
                "journal blocks",
                limit,
            ));
        }
        Ok(Self {
            order,
            old_magic: magic == OLD_MAGIC,
            checksum_ok,
            start,
            end,
            size,
            blhdr_size,
            jhdr_size,
        })
    }
}

/// The journal's ring: journal bytes `[first, size)` of the region at
/// device byte `base`. Positions are always inside the ring.
struct Ring<'a> {
    device: &'a dyn BlockSource,
    base: u64,
    /// `jhdr_size`: the ring starts after the header block.
    first: u64,
    size: u64,
    /// `size - first`, the ring's length.
    lap: u64,
}

impl Ring<'_> {
    /// The position `n` bytes after `pos`; `n` is at most one lap.
    fn advance(&self, pos: u64, n: u64) -> u64 {
        // pos < size <= 1 GiB and n <= lap: no overflow, one wrap at most.
        let next = pos.saturating_add(n);
        if next >= self.size {
            next.saturating_sub(self.lap)
        } else {
            next
        }
    }

    /// The device ranges holding `len` journal bytes at `pos`, split where
    /// they cross the ring's end: (device byte, length) pairs, the second
    /// one empty when there is no wrap. `len` is at most one lap.
    fn segments(&self, pos: u64, len: u64) -> [(u64, u64); 2] {
        let first = len.min(self.size.saturating_sub(pos));
        let rest = len.saturating_sub(first);
        [
            (self.base.saturating_add(pos), first),
            (self.base.saturating_add(self.first), rest),
        ]
    }

    /// Reads `buf.len()` journal bytes at `pos`, wrapping at the ring's end
    /// (xnu `do_journal_io`).
    fn read(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        let [(at, first), (wrapped_at, _)] = self.segments(pos, len);
        let split = usize::try_from(first).unwrap_or(usize::MAX).min(buf.len());
        let (head, tail) = buf.split_at_mut(split);
        self.device.read_at(at, head)?;
        if !tail.is_empty() {
            self.device.read_at(wrapped_at, tail)?;
        }
        Ok(())
    }
}

/// One block a block list replays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    /// Device byte the block belongs at.
    device: u64,
    len: u64,
    /// Ring position of the block's data.
    journal: u64,
}

/// An accepted block list.
#[derive(Debug, Clone, PartialEq, Eq)]
struct List {
    /// Its live (not killed) blocks, in order.
    blocks: Vec<Block>,
    first_header: bool,
}

/// The block lists a replay applies, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Walk {
    lists: Vec<List>,
    stopped_early: bool,
}

impl Walk {
    /// The transactions applied: the block lists that start one, or every
    /// block list when none is marked (journals that predate the flag).
    fn transactions(&self) -> u32 {
        let marked = self.lists.iter().filter(|l| l.first_header).count();
        let count = if marked > 0 { marked } else { self.lists.len() };
        u32::try_from(count).unwrap_or(u32::MAX)
    }
}

/// Walks the journal from `start` the way xnu `replay_journal` does and
/// returns the block lists to apply.
///
/// Fails when the journal cannot be replayed at all: the first block list
/// is damaged, or a block list is damaged before any transaction start was
/// seen. A damaged block list after a transaction start keeps only the block
/// lists before the most recent one (xnu restarts the replay with `end` set
/// there).
fn walk(ring: &Ring<'_>, header: &Header, device_len: u64) -> Result<Walk> {
    let order = header.order;
    let orig_start = header.start;
    let mut pos = header.start;
    let end = header.end;
    // xnu keeps walking past `end` while block lists still check out
    // (`check_past_jnl_end`): transactions written after the header was.
    // Every place xnu clears that flag ends its walk at once, so here they
    // all leave the loop instead. xnu's second pass (after a bad
    // transaction, with the flag clear) is the truncation at the bottom.
    // xnu `in_uncharted_territory`: the walk has reached `end`.
    let mut uncharted = false;
    let mut last_seq: u32 = 0;
    // Bytes walked so far; the walk stops at one lap.
    let mut advanced: u64 = 0;
    let mut live_blocks: u64 = 0;
    let mut lists: Vec<List> = Vec::new();
    // Index in `lists` of the block list that started the newest
    // transaction (xnu `txn_start_offset`).
    let mut txn_start: Option<usize> = None;
    let mut raw = [0u8; BLHDR_LEN];
    let mut infos: Vec<u8> = Vec::new();
    let mut data: Vec<u8> = Vec::new();

    // Bounded: every pass either leaves the loop or adds bytes_used (at
    // least BLHDR_LEN) to `advanced`, which stops the walk at one lap.
    'walk: loop {
        if advanced >= ring.lap {
            break;
        }
        let failure: Error = 'list: {
            ring.read(pos, &mut raw)?;
            let short = || {
                corrupt(
                    BLOCK_LIST,
                    pos,
                    CorruptKind::Truncated {
                        needed: BLHDR_LEN as u64,
                        available: 0,
                    },
                )
            };
            let stored = order.u32(&raw, 8).ok_or_else(short)?;
            let computed = checksum_without(&raw, BLHDR_CHECKSUM_FIELD);
            if computed != stored {
                if uncharted {
                    // Past `end`: the first bad block list ends the extras.
                    break 'walk;
                }
                let e = corrupt(
                    BLOCK_LIST,
                    pos,
                    CorruptKind::BadSignature {
                        expected: u64::from(computed),
                        found: u64::from(stored),
                    },
                );
                if pos == orig_start {
                    return Err(e);
                }
                break 'list e;
            }

            let seq = order.u32(&raw, 28).ok_or_else(short)?;
            if last_seq != 0 && seq != 0 && seq != last_seq && seq != last_seq.wrapping_add(1) {
                // A transaction from an earlier lap: the journal ends here.
                break 'walk;
            }
            last_seq = seq;
            if pos == end && last_seq == 0 {
                // A journal without sequence numbers cannot tell extra
                // transactions from stale ones: stop at `end`.
                break 'walk;
            }

            let max_blocks = order.u16(&raw, 0).ok_or_else(short)?;
            let num_blocks = order.u16(&raw, 2).ok_or_else(short)?;
            let bytes_used = order.i32(&raw, 4).ok_or_else(short)?;
            let flags = order.i32(&raw, 12).ok_or_else(short)?;
            let max_allowed = header.size.checked_div(header.jhdr_size).unwrap_or(0);
            if max_blocks == 0 || u64::from(max_blocks) > max_allowed {
                break 'list out_of_range(
                    BLOCK_LIST,
                    pos,
                    "max_blocks",
                    u64::from(max_blocks),
                    "journal blocks",
                    max_allowed,
                );
            }
            if num_blocks == 0 || num_blocks > max_blocks {
                break 'list out_of_range(
                    BLOCK_LIST,
                    pos,
                    "num_blocks",
                    u64::from(num_blocks),
                    "max_blocks",
                    u64::from(max_blocks),
                );
            }
            // Ours: the block infos fit in the block list header.
            let info_bytes = u64::from(num_blocks).saturating_mul(BLOCK_INFO_LEN);
            if info_bytes.saturating_add(BLOCK_INFO_LEN) > header.blhdr_size {
                break 'list out_of_range(
                    BLOCK_LIST,
                    pos,
                    "num_blocks",
                    u64::from(num_blocks),
                    "block list header size",
                    header.blhdr_size,
                );
            }
            // binfo[1..num_blocks]; binfo[0] was in `raw`.
            let rest = info_bytes.saturating_sub(BLOCK_INFO_LEN);
            infos.clear();
            infos.resize(usize::try_from(rest).unwrap_or(0), 0);
            ring.read(ring.advance(pos, BLHDR_LEN as u64), &mut infos)?;
            let infos_short = || {
                corrupt(
                    BLOCK_LIST,
                    pos,
                    CorruptKind::Truncated {
                        needed: rest,
                        available: 0,
                    },
                )
            };
            let mut bnum_error = None;
            for (i, info) in infos.as_chunks::<16>().0.iter().enumerate() {
                let bnum = order.i64(info, 0).ok_or_else(infos_short)?;
                if bnum < 0 && bnum != KILLED {
                    bnum_error = Some((i, bnum));
                    break;
                }
            }
            if let Some((i, bnum)) = bnum_error {
                let at = info_at(pos, i);
                break 'list invalid(BLOCK_LIST, at, "block number", raw_i64(bnum));
            }

            let first_header = flags & FIRST_HEADER != 0;
            if first_header {
                txn_start = Some(lists.len());
            }

            // Sizes, ranges and the data's extent, before reading any data.
            let mut blocks = Vec::new();
            let mut consumed = header.blhdr_size;
            let mut largest: u64 = 0;
            let mut data_pos = ring.advance(pos, header.blhdr_size);
            let mut checks: Vec<(u64, u64, u32)> = Vec::new();
            let check_sums = flags & CHECK_CHECKSUMS != 0;
            let mut block_error = None;
            for (i, info) in infos.as_chunks::<16>().0.iter().enumerate() {
                let at = info_at(pos, i);
                let bnum = order.i64(info, 0).ok_or_else(infos_short)?;
                let bsize = order.i32(info, 8).ok_or_else(infos_short)?;
                let cksum = order.u32(info, 12).ok_or_else(infos_short)?;
                let Some(len) = u64::try_from(bsize).ok().filter(|l| *l > 0) else {
                    block_error = Some(invalid(
                        BLOCK_LIST,
                        at,
                        "block size",
                        raw_i64(i64::from(bsize)),
                    ));
                    break;
                };
                consumed = consumed.saturating_add(len);
                if consumed > ring.lap {
                    block_error = Some(out_of_range(
                        BLOCK_LIST,
                        at,
                        "block list length",
                        consumed,
                        "journal data size",
                        ring.lap,
                    ));
                    break;
                }
                if bnum != KILLED {
                    let device = u64::try_from(bnum)
                        .ok()
                        .and_then(|b| b.checked_mul(header.jhdr_size));
                    let block_end = device.and_then(|d| d.checked_add(len));
                    let (Some(device), Some(block_end)) = (device, block_end) else {
                        block_error = Some(overflow(BLOCK_LIST, at, "block end"));
                        break;
                    };
                    if block_end > device_len {
                        block_error = Some(out_of_range(
                            BLOCK_LIST,
                            at,
                            "block end",
                            block_end,
                            "device size",
                            device_len,
                        ));
                        break;
                    }
                    live_blocks = live_blocks.saturating_add(1);
                    if live_blocks > MAX_BLOCKS {
                        block_error = Some(corrupt(
                            BLOCK_LIST,
                            at,
                            CorruptKind::LimitExceeded {
                                what: "replayed blocks",
                                limit: MAX_BLOCKS,
                            },
                        ));
                        break;
                    }
                    if check_sums && cksum != 0 {
                        largest = largest.max(len);
                        checks.push((data_pos, len, cksum));
                    }
                    blocks.push(Block {
                        device,
                        len,
                        journal: data_pos,
                    });
                }
                data_pos = ring.advance(data_pos, len);
            }
            if let Some(e) = block_error {
                break 'list e;
            }
            let used = u64::try_from(bytes_used)
                .ok()
                .filter(|u| *u >= consumed && *u <= ring.lap);
            let Some(used) = used else {
                break 'list out_of_range(
                    BLOCK_LIST,
                    pos.saturating_add(4),
                    "bytes_used",
                    raw_i64(i64::from(bytes_used)),
                    "journal data size",
                    ring.lap,
                );
            };

            // Block checksums, read through one buffer of at most
            // CHECKSUM_CHUNK bytes: a block's size comes from the disk and
            // can be most of a 1 GiB ring.
            if !checks.is_empty() {
                data.clear();
                data.resize(usize::try_from(largest.min(CHECKSUM_CHUNK)).unwrap_or(0), 0);
            }
            let mut sum_error = None;
            'sums: for &(at, len, cksum) in &checks {
                let mut c: u32 = 0;
                let mut done: u64 = 0;
                // Bounded: each pass reads at least one byte of `len`.
                while done < len {
                    let n = len.saturating_sub(done).min(CHECKSUM_CHUNK);
                    let Some(buf) = usize::try_from(n).ok().and_then(|n| data.get_mut(..n)) else {
                        sum_error = Some(overflow(BLOCK_LIST, pos, "block size"));
                        break 'sums;
                    };
                    // done < len <= one lap: one wrap at most.
                    ring.read(ring.advance(at, done), buf)?;
                    c = fold(c, buf.iter().copied());
                    done = done.saturating_add(n);
                }
                if !c != cksum {
                    sum_error = Some(corrupt(
                        BLOCK_LIST,
                        pos,
                        CorruptKind::Other("block checksum mismatch"),
                    ));
                    break;
                }
            }
            if let Some(e) = sum_error {
                break 'list e;
            }

            // Ours: never more than one lap in total.
            let total = advanced.saturating_add(used);
            if total > ring.lap {
                break 'walk;
            }
            lists.push(List {
                blocks,
                first_header,
            });
            advanced = total;
            pos = ring.advance(pos, used);
            if pos == end {
                uncharted = true;
            }
            continue 'walk;
        };

        // Bad transaction (xnu `bad_txn_handling`): keep the block lists
        // before the newest transaction start, if there is one.
        let Some(keep) = txn_start else {
            return Err(failure);
        };
        lists.truncate(keep);
        return Ok(Walk {
            lists,
            stopped_early: true,
        });
    }
    Ok(Walk {
        lists,
        stopped_early: false,
    })
}

/// Journal byte of `binfo[i + 1]` in the block list at `pos` (for errors).
fn info_at(pos: u64, i: usize) -> u64 {
    let index = u64::try_from(i).unwrap_or(u64::MAX).saturating_add(1);
    pos.saturating_add(index.saturating_mul(BLOCK_INFO_LEN))
}

/// Bytes of the device replaced by journal bytes: `len` bytes from device
/// byte `from` (inside the journal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Patch {
    len: u64,
    from: u64,
}

/// The patches `lists` make, applied in order: a later block replaces the
/// bytes of earlier ones it overlaps.
fn patches(ring: &Ring<'_>, lists: &[List]) -> BTreeMap<u64, Patch> {
    let mut map = BTreeMap::new();
    for block in lists.iter().flat_map(|l| &l.blocks) {
        let [(from, first), (wrapped_from, rest)] = ring.segments(block.journal, block.len);
        insert(&mut map, block.device, first, from);
        insert(
            &mut map,
            block.device.saturating_add(first),
            rest,
            wrapped_from,
        );
    }
    map
}

/// Puts `len` bytes from `from` at `start`, trimming or splitting the
/// patches it overlaps. Ranges were checked against the device, so nothing
/// here overflows.
fn insert(map: &mut BTreeMap<u64, Patch>, start: u64, len: u64, from: u64) {
    if len == 0 {
        return;
    }
    let end = start.saturating_add(len);
    // Patches do not overlap, so their ends rise with their starts: walk
    // back from `end` until one ends at or before `start`.
    let overlapped: Vec<u64> = map
        .range(..end)
        .rev()
        .take_while(|(k, p)| k.saturating_add(p.len) > start)
        .map(|(k, _)| *k)
        .collect();
    for k in overlapped {
        let Some(p) = map.remove(&k) else { continue };
        let p_end = k.saturating_add(p.len);
        if k < start {
            map.insert(
                k,
                Patch {
                    len: start.saturating_sub(k),
                    from: p.from,
                },
            );
        }
        if p_end > end {
            map.insert(
                end,
                Patch {
                    len: p_end.saturating_sub(end),
                    from: p.from.saturating_add(end.saturating_sub(k)),
                },
            );
        }
    }
    map.insert(start, Patch { len, from });
}

/// A device as it reads after journal replay.
///
/// Reads come from the device, with every replayed byte taken from the
/// journal's copy on the same device. Built by [`open`]; offsets are device
/// offsets.
pub struct JournalOverlay {
    inner: Arc<dyn BlockSource>,
    /// Device start → patch; patches do not overlap.
    patches: BTreeMap<u64, Patch>,
}

impl JournalOverlay {
    /// Copies the journal's bytes over the replayed parts of the `buf.len()`
    /// bytes at `offset` (already read from the device).
    fn apply(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        // check_range passed: no overflow.
        let end = offset.saturating_add(len);
        for (&k, p) in self.patches.range(..end).rev() {
            let p_end = k.saturating_add(p.len);
            if p_end <= offset {
                break;
            }
            let s = k.max(offset);
            let e = p_end.min(end);
            let from = p.from.saturating_add(s.saturating_sub(k));
            let a = usize::try_from(s.saturating_sub(offset)).unwrap_or(usize::MAX);
            let b = usize::try_from(e.saturating_sub(offset)).unwrap_or(usize::MAX);
            let dst = buf.get_mut(a..b).ok_or(Error::OutOfBounds {
                offset,
                len,
                size: self.inner.len(),
            })?;
            self.inner.read_at(from, dst)?;
        }
        Ok(())
    }
}

impl BlockSource for JournalOverlay {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn sector_size(&self) -> u32 {
        self.inner.sector_size()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        check_range(offset, buf.len(), self.len())?;
        self.inner.read_at(offset, buf)?;
        self.apply(offset, buf)
    }

    fn read_bulk(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        check_range(offset, buf.len(), self.len())?;
        self.inner.read_bulk(offset, buf)?;
        self.apply(offset, buf)
    }
}

// `HfsPlus` holds one; this names the failure here if a field breaks it.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<JournalOverlay>();
    assert_send_sync::<Opened>();
};

#[cfg(test)]
mod tests {
    use super::super::test_volume::{
        BLHDR_FIRST, FileSpec, JIB_IN_FS, JIB_NEED_INIT, JIB_ON_OTHER_DEVICE, JNL_OLD_MAGIC,
        JournalSetup, MIN_JOURNAL, ROOT, TestBlock, TestJournal, TestList, VolumeBuilder, source,
    };
    use super::super::volume_header::read_header;
    use super::*;
    use atlas_fs_core::io::block::MemSource;

    /// The synthetic device: 2 MiB, the journal at 1 MiB.
    const DEV: usize = 2 << 20;
    const BASE: usize = 1 << 20;
    const JSIZE: u64 = MIN_JOURNAL;

    fn device() -> Vec<u8> {
        (0..DEV).map(|i| (i % 251) as u8).collect()
    }

    struct Run {
        image: Vec<u8>,
        opened: Opened,
        positions: Vec<u64>,
    }

    /// Writes `j` into the synthetic device at `BASE` and opens it.
    fn run(j: &TestJournal) -> Run {
        let mut image = device();
        let positions = j.write(&mut image, BASE);
        let src: Arc<dyn BlockSource> = Arc::new(MemSource::new(image.clone()));
        let plan = open_at(&*src, BASE as u64, JSIZE);
        let opened = finish(src, plan).unwrap();
        Run {
            image,
            opened,
            positions,
        }
    }

    fn read_all(d: &dyn BlockSource) -> Vec<u8> {
        let mut b = vec![0u8; d.len() as usize];
        d.read_at(0, &mut b).unwrap();
        b
    }

    /// `image` with the live blocks of `lists` written at home, in order.
    fn applied(image: &[u8], lists: &[TestList]) -> Vec<u8> {
        let mut out = image.to_vec();
        for block in lists.iter().flat_map(|l| &l.blocks) {
            if block.bnum >= 0 {
                let at = block.bnum as usize * 512;
                out[at..at + block.data.len()].copy_from_slice(&block.data);
            }
        }
        out
    }

    /// Asserts the device reads as `image` with exactly `lists` applied.
    fn assert_applied(r: &Run, lists: &[TestList]) {
        assert!(
            read_all(&*r.opened.device) == applied(&r.image, lists),
            "device does not read as the replayed image"
        );
    }

    fn assert_unchanged(r: &Run) {
        assert!(
            read_all(&*r.opened.device) == r.image,
            "device does not read as on disk"
        );
    }

    fn blk(bnum: i64, len: usize, v: u8) -> TestBlock {
        TestBlock::new(bnum, vec![v; len])
    }

    fn first(seq: u32, blocks: Vec<TestBlock>) -> TestList {
        TestList::first(seq, blocks)
    }

    fn journal(lists: Vec<TestList>) -> TestJournal {
        TestJournal {
            lists,
            ..TestJournal::new(JSIZE)
        }
    }

    fn replayed(r: &Run) -> (u32, bool) {
        match r.opened.state {
            JournalState::Replayed {
                transactions,
                stopped_early,
            } => (transactions, stopped_early),
            ref s => panic!("expected Replayed, got {s:?}"),
        }
    }

    fn not_replayed(state: &JournalState) -> &str {
        match state {
            JournalState::NotReplayed { reason } => reason,
            s => panic!("expected NotReplayed, got {s:?}"),
        }
    }

    fn header_of(j: &TestJournal) -> Result<Header> {
        let mut image = vec![0u8; JSIZE as usize];
        j.write(&mut image, 0);
        let raw: [u8; HEADER_LEN] = image[..HEADER_LEN].try_into().unwrap();
        Header::parse(&raw, JSIZE)
    }

    // ---- checksum and header ----

    #[test]
    fn checksum_matches_xnu() {
        assert_eq!(calc_checksum(&[]), 0xFFFF_FFFF);
        assert_eq!(calc_checksum(&[1]), !1);
        // (1 << 8) ^ (1 + 2) = 259
        assert_eq!(calc_checksum(&[1, 2]), !259);
        assert_eq!(calc_checksum(&[1, 2, 3]), !((259u32 << 8) ^ (259 + 3)));
        // Long inputs shift bits out and wrap, as C unsigned arithmetic does.
        let mut c: u32 = 0;
        for _ in 0..64 {
            c = c.wrapping_shl(8) ^ c.wrapping_add(0xFF);
        }
        assert_eq!(calc_checksum(&[0xFF; 64]), !c);
        assert_eq!(
            checksum_without(&[1, 2, 9, 9, 3], 2..4),
            calc_checksum(&[1, 2, 0, 0, 3])
        );
    }

    #[test]
    fn header_in_both_byte_orders() {
        for le in [false, true] {
            let j = TestJournal {
                little_endian: le,
                start: 4096,
                end: Some(8192),
                ..TestJournal::new(JSIZE)
            };
            let h = header_of(&j).unwrap();
            let order = if le { Order::Little } else { Order::Big };
            assert_eq!(h.order, order);
            assert_eq!(
                (h.start, h.end, h.size, h.blhdr_size, h.jhdr_size),
                (4096, 8192, JSIZE, 4096, 512)
            );
            assert!(h.checksum_ok);
            assert!(!h.old_magic);
        }
        // The old magic, in either order.
        for le in [false, true] {
            let j = TestJournal {
                little_endian: le,
                magic: JNL_OLD_MAGIC,
                ..TestJournal::new(JSIZE)
            };
            let h = header_of(&j).unwrap();
            assert!(h.old_magic);
            assert_eq!(h.order, if le { Order::Little } else { Order::Big });
        }
        // start and end equal to size wrap to jhdr_size.
        let j = TestJournal {
            jhdr_size: 4096,
            blhdr_size: 1024,
            start: JSIZE as i64,
            end: Some(JSIZE as i64),
            ..TestJournal::new(JSIZE)
        };
        let h = header_of(&j).unwrap();
        assert_eq!((h.start, h.end), (4096, 4096));
    }

    #[test]
    fn header_checksum_mismatch_is_tolerated() {
        let lists = vec![first(1, vec![blk(4, 512, 0xAB)])];
        let j = TestJournal {
            header_checksum: Some(0x1234),
            ..journal(lists.clone())
        };
        assert!(!header_of(&j).unwrap().checksum_ok);
        let r = run(&j);
        assert_eq!(replayed(&r), (1, false));
        assert_applied(&r, &lists);
    }

    #[test]
    fn header_bounds_are_rejected() {
        let base = || TestJournal {
            end: Some(4096),
            ..TestJournal::new(JSIZE)
        };
        let size = JSIZE as i64;
        let cases: Vec<(&str, TestJournal)> = vec![
            (
                "magic",
                TestJournal {
                    magic: 0x1234_5678,
                    ..base()
                },
            ),
            (
                "jhdr 0",
                TestJournal {
                    jhdr_size: 0,
                    ..base()
                },
            ),
            (
                "jhdr 256",
                TestJournal {
                    jhdr_size: 256,
                    ..base()
                },
            ),
            (
                "jhdr 768",
                TestJournal {
                    jhdr_size: 768,
                    ..base()
                },
            ),
            (
                "jhdr 128 KiB",
                TestJournal {
                    jhdr_size: 131_072,
                    ..base()
                },
            ),
            (
                "size below 256 KiB",
                TestJournal {
                    size: size - 512,
                    ..base()
                },
            ),
            (
                "size past the region",
                TestJournal {
                    size: size + 512,
                    ..base()
                },
            ),
            ("size negative", TestJournal { size: -1, ..base() }),
            ("start 0", TestJournal { start: 0, ..base() }),
            (
                "start negative",
                TestJournal {
                    start: -512,
                    ..base()
                },
            ),
            (
                "start past size",
                TestJournal {
                    start: size + 512,
                    ..base()
                },
            ),
            (
                "start unaligned",
                TestJournal {
                    start: 1000,
                    ..base()
                },
            ),
            (
                "start inside the header block",
                TestJournal {
                    jhdr_size: 4096,
                    start: 512,
                    end: Some(8192),
                    ..base()
                },
            ),
            (
                "end 0",
                TestJournal {
                    end: Some(0),
                    ..base()
                },
            ),
            (
                "end past size",
                TestJournal {
                    end: Some(size + 512),
                    ..base()
                },
            ),
            (
                "end unaligned",
                TestJournal {
                    end: Some(4100),
                    ..base()
                },
            ),
            (
                "blhdr_size 16",
                TestJournal {
                    blhdr_size: 16,
                    ..base()
                },
            ),
            (
                "blhdr_size negative",
                TestJournal {
                    blhdr_size: -4096,
                    ..base()
                },
            ),
            (
                "blhdr_size past the ring",
                TestJournal {
                    blhdr_size: size as i32,
                    ..base()
                },
            ),
            (
                "blhdr_size capacity",
                TestJournal {
                    jhdr_size: 4096,
                    start: 4096,
                    end: Some(8192),
                    blhdr_size: 16 * 66,
                    ..base()
                },
            ),
        ];
        for (name, j) in cases {
            let r = run(&j);
            let reason = not_replayed(&r.opened.state);
            assert!(reason.starts_with("journal header"), "{name}: {reason}");
            assert_unchanged(&r);
        }
        // The limits themselves pass: 64 block infos for a 64-block journal.
        let j = TestJournal {
            jhdr_size: 4096,
            start: 4096,
            end: Some(4096),
            blhdr_size: 16 * 65,
            ..base()
        };
        assert_eq!(run(&j).opened.state, JournalState::Clean);
    }

    #[test]
    fn empty_journal_is_clean() {
        let r = run(&TestJournal::new(JSIZE));
        assert_eq!(r.opened.state, JournalState::Clean);
        let r = run(&TestJournal {
            start: 65536,
            end: Some(65536),
            ..TestJournal::new(JSIZE)
        });
        assert_eq!(r.opened.state, JournalState::Clean);
        assert_unchanged(&r);
    }

    // ---- replay ----

    #[test]
    fn replays_in_both_byte_orders() {
        for (le, magic) in [
            (false, MAGIC),
            (true, MAGIC),
            (false, JNL_OLD_MAGIC),
            (true, JNL_OLD_MAGIC),
        ] {
            let lists = vec![
                first(10, vec![blk(0, 4096, 0x11), blk(16, 1024, 0x22)]),
                first(11, vec![blk(40, 512, 0x33)]),
            ];
            let j = TestJournal {
                little_endian: le,
                magic,
                ..journal(lists.clone())
            };
            let r = run(&j);
            assert_eq!(replayed(&r), (2, false), "le {le}");
            assert_applied(&r, &lists);
        }
    }

    #[test]
    fn bad_first_block_list_is_not_replayed() {
        let lists = vec![
            first(1, vec![blk(0, 512, 1)]).bad(),
            first(2, vec![blk(1, 512, 2)]),
        ];
        let r = run(&journal(lists));
        let reason = not_replayed(&r.opened.state);
        assert!(
            reason.starts_with("journal block list byte offset 512"),
            "{reason}"
        );
        assert_unchanged(&r);
    }

    #[test]
    fn damage_before_any_transaction_start_is_not_replayed() {
        // Block lists that never set FIRST_HEADER: nothing to fall back to.
        let lists = vec![
            TestList::more(1, vec![blk(0, 512, 1)]),
            TestList::more(1, vec![blk(1, 512, 2)]).bad(),
        ];
        let r = run(&journal(lists));
        not_replayed(&r.opened.state);
        assert_unchanged(&r);
    }

    #[test]
    fn journals_without_transaction_starts_count_block_lists() {
        let lists = vec![
            TestList::more(1, vec![blk(0, 512, 1)]),
            TestList::more(2, vec![blk(1, 512, 2)]),
            TestList::more(3, vec![blk(2, 512, 3)]),
        ];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (3, false));
        assert_applied(&r, &lists);
    }

    #[test]
    fn damaged_transaction_keeps_those_before_the_newest_start() {
        // xnu restarts the replay with `end` at the newest transaction start
        // it saw, so T2 (complete and sound) goes with the damaged T3.
        let lists = vec![
            first(1, vec![blk(0, 512, 1)]),
            first(2, vec![blk(1, 512, 2)]),
            first(3, vec![blk(2, 512, 3)]).bad(),
        ];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (1, true));
        assert_applied(&r, &lists[..1]);

        // Damage in the second transaction: nothing is left to replay.
        let r = run(&journal(vec![lists[0].clone(), lists[2].clone()]));
        assert_eq!(replayed(&r), (0, true));
        assert_unchanged(&r);
    }

    #[test]
    fn bad_second_block_list_drops_its_whole_transaction() {
        let lists = vec![
            first(1, vec![blk(0, 512, 1)]),
            first(2, vec![blk(1, 512, 2)]),
            TestList::more(2, vec![blk(2, 512, 3)]),
            first(3, vec![blk(3, 512, 4)]),
            TestList::more(3, vec![blk(4, 512, 5)]).bad(),
        ];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (2, true));
        assert_applied(&r, &lists[..3]);
    }

    #[test]
    fn sequence_break_ends_the_journal_quietly() {
        let lists = vec![
            first(5, vec![blk(0, 512, 1)]),
            first(6, vec![blk(1, 512, 2)]),
            first(9, vec![blk(2, 512, 3)]),
            first(10, vec![blk(3, 512, 4)]),
        ];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (2, false));
        assert_applied(&r, &lists[..2]);

        // Repeats, zero, and wrapping past u32::MAX are not breaks.
        let lists = vec![
            first(u32::MAX, vec![blk(0, 512, 1)]),
            first(u32::MAX, vec![blk(1, 512, 2)]),
            first(0, vec![blk(2, 512, 3)]),
            first(77, vec![blk(3, 512, 4)]),
        ];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (4, false));
        assert_applied(&r, &lists);
        let lists = vec![
            first(u32::MAX, vec![blk(0, 512, 1)]),
            first(0, vec![blk(1, 512, 2)]),
        ];
        assert_eq!(replayed(&run(&journal(lists))), (2, false));
    }

    #[test]
    fn transactions_past_end() {
        let a = || first(5, vec![blk(0, 512, 1)]);
        let past = |lists: Vec<TestList>| {
            run(&TestJournal {
                in_range: Some(1),
                ..journal(lists)
            })
        };
        // Written after the header was: replayed.
        let lists = vec![a(), first(6, vec![blk(1, 512, 2)])];
        let r = past(lists.clone());
        assert_eq!(replayed(&r), (2, false));
        assert_applied(&r, &lists);
        // A bad checksum out there ends the walk; it is no failure.
        let lists = vec![a(), first(6, vec![blk(1, 512, 2)]).bad()];
        let r = past(lists.clone());
        assert_eq!(replayed(&r), (1, false));
        assert_applied(&r, &lists[..1]);
        // A stale transaction from an earlier lap.
        let lists = vec![a(), first(3, vec![blk(1, 512, 2)])];
        let r = past(lists.clone());
        assert_eq!(replayed(&r), (1, false));
        assert_applied(&r, &lists[..1]);
        // A journal without sequence numbers never goes past `end`.
        let lists = vec![
            first(0, vec![blk(0, 512, 1)]),
            first(0, vec![blk(1, 512, 2)]),
        ];
        let r = past(lists.clone());
        assert_eq!(replayed(&r), (1, false));
        assert_applied(&r, &lists[..1]);
        // Damage behind a good checksum out there is a bad transaction.
        let mut bad = first(7, vec![blk(2, 512, 3)]);
        bad.blocks[0].bsize = Some(0);
        let lists = vec![a(), first(6, vec![blk(1, 512, 2)]), bad];
        let r = past(lists.clone());
        assert_eq!(replayed(&r), (2, true));
        assert_applied(&r, &lists[..2]);
    }

    #[test]
    fn block_list_header_wraps_the_ring() {
        let lists = vec![
            first(1, vec![blk(0, 4096, 0x41)]),
            first(2, vec![blk(8, 2048, 0x42)]),
        ];
        let r = run(&TestJournal {
            start: JSIZE as i64 - 1024,
            ..journal(lists.clone())
        });
        // The 4096-byte header wraps: 1024 bytes before the end, 3072 after.
        assert_eq!(r.positions, [JSIZE - 1024, 512 + 3072 + 4096]);
        assert_eq!(replayed(&r), (2, false));
        assert_applied(&r, &lists);
    }

    #[test]
    fn block_data_wraps_the_ring() {
        let lists = vec![
            first(1, vec![blk(0, 4096, 0x51), blk(20, 512, 0x52)]),
            first(2, vec![blk(8, 2048, 0x53)]),
        ];
        let r = run(&TestJournal {
            start: JSIZE as i64 - 4096 - 1024,
            ..journal(lists.clone())
        });
        // The first block's data: 1024 bytes before the end, 3072 after.
        assert_eq!(r.positions, [JSIZE - 5120, 512 + 3072 + 512]);
        assert_eq!(replayed(&r), (2, false));
        assert_applied(&r, &lists);
        // start == size is the ring's first byte.
        let r = run(&TestJournal {
            start: JSIZE as i64,
            ..journal(lists.clone())
        });
        assert_eq!(r.positions[0], 512);
        assert_applied(&r, &lists);
    }

    #[test]
    fn later_blocks_win_where_they_overlap() {
        let lists = vec![
            first(1, vec![blk(0, 4096, 0xA1), blk(16, 4096, 0xA2)]),
            // Inside the first block.
            first(2, vec![blk(2, 1024, 0xB1)]),
            // Across the first block's end, and over the B1 bytes' start.
            first(3, vec![blk(7, 1024, 0xC1), blk(1, 1024, 0xC2)]),
            // Across the second block's start; twice in one transaction.
            first(4, vec![blk(14, 4096, 0xD1), blk(15, 1024, 0xD2)]),
            TestList::more(4, vec![blk(16, 512, 0xD3)]),
        ];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (4, false));
        assert_applied(&r, &lists);
    }

    #[test]
    fn killed_blocks_are_skipped_but_keep_their_space() {
        let lists = vec![first(
            1,
            vec![
                blk(0, 1024, 0x11),
                TestBlock::killed(2048),
                blk(8, 512, 0x22),
            ],
        )];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (1, false));
        assert_applied(&r, &lists);
    }

    #[test]
    fn block_checksums() {
        // A zero stored checksum is not checked.
        let mut b = blk(0, 512, 1);
        b.cksum = Some(0);
        let lists = vec![first(1, vec![b])];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (1, false));
        assert_applied(&r, &lists);

        // A mismatch is a bad transaction. This one started a transaction,
        // so xnu replays up to its start: nothing.
        let mut b = blk(0, 512, 1);
        b.cksum = Some(0xDEAD);
        let r = run(&journal(vec![first(1, vec![b.clone()])]));
        assert_eq!(replayed(&r), (0, true));
        assert_unchanged(&r);

        // Without BLHDR_CHECK_CHECKSUMS the stored value does not matter.
        let lists = vec![TestList {
            flags: BLHDR_FIRST,
            ..first(1, vec![b])
        }];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (1, false));
        assert_applied(&r, &lists);
    }

    #[test]
    fn block_checksums_span_several_chunks() {
        // 70 KiB: two CHECKSUM_CHUNK reads, the second one partial.
        let len = 70 * 1024;
        let data: Vec<u8> = (0..len).map(|i| (i % 253) as u8).collect();
        let lists = vec![first(1, vec![TestBlock::new(100, data.clone())])];
        let r = run(&journal(lists.clone()));
        assert_eq!(replayed(&r), (1, false));
        assert_applied(&r, &lists);

        // A stored checksum that differs only by a byte past the first chunk.
        let mut other = data.clone();
        other[CHECKSUM_CHUNK as usize + 1] ^= 0xFF;
        let mut b = TestBlock::new(100, data);
        b.cksum = Some(calc_checksum(&other));
        let r = run(&journal(vec![first(1, vec![b])]));
        assert_eq!(replayed(&r), (0, true));
        assert_unchanged(&r);
    }

    #[test]
    fn damaged_block_lists() {
        type Edit = fn(&mut TestList);
        // Found before the block list's transaction-start flag is read: the
        // replay keeps what precedes the previous start (T1).
        let early: &[(&str, Edit)] = &[
            ("max_blocks 0", |l| l.max_blocks = Some(0)),
            ("max_blocks past the journal", |l| l.max_blocks = Some(513)),
            ("num_blocks 0", |l| l.num_blocks = Some(0)),
            ("num_blocks past max_blocks", |l| {
                l.max_blocks = Some(1);
                l.num_blocks = Some(2);
            }),
            ("num_blocks past the header", |l| {
                l.max_blocks = Some(300);
                l.num_blocks = Some(256);
            }),
            ("negative block number", |l| l.blocks[0].bnum = -2),
        ];
        // Found after it: T3 is the newest start, T1 and T2 stay.
        let late: &[(&str, Edit)] = &[
            ("zero block size", |l| l.blocks[0].bsize = Some(0)),
            ("negative block size", |l| l.blocks[0].bsize = Some(-512)),
            ("block past the device", |l| {
                l.blocks[0].bnum = (DEV / 512) as i64;
            }),
            ("block end past the device", |l| {
                l.blocks[0].bnum = (DEV / 512 - 1) as i64;
            }),
            ("block number overflow", |l| l.blocks[0].bnum = i64::MAX / 2),
            ("bytes_used short", |l| l.bytes_used = Some(4096)),
            ("bytes_used past the ring", |l| {
                l.bytes_used = Some(JSIZE as i32);
            }),
            ("bytes_used negative", |l| l.bytes_used = Some(-1)),
            ("block list longer than the ring", |l| {
                l.blocks[0].bsize = Some(JSIZE as i32);
            }),
            ("block checksum", |l| l.blocks[0].cksum = Some(1)),
        ];
        let make = |edit: Edit| {
            let mut t3 = first(3, vec![blk(2, 1024, 3), blk(9, 512, 4)]);
            edit(&mut t3);
            vec![
                first(1, vec![blk(0, 512, 1)]),
                first(2, vec![blk(1, 512, 2)]),
                t3,
            ]
        };
        for (name, edit) in early {
            let lists = make(*edit);
            let r = run(&journal(lists.clone()));
            assert_eq!(replayed(&r), (1, true), "{name}");
            assert_applied(&r, &lists[..1]);
        }
        for (name, edit) in late {
            let lists = make(*edit);
            let r = run(&journal(lists.clone()));
            assert_eq!(replayed(&r), (2, true), "{name}");
            assert_applied(&r, &lists[..2]);
        }
    }

    #[test]
    fn a_ring_that_never_ends_stops_after_one_lap() {
        // Every block list has the same sequence number and together they
        // fill the ring exactly, so a walk past `end` comes back to `start`
        // and finds it valid again: xnu would go round for ever.
        let mut lists: Vec<TestList> = (0..255)
            .map(|i| first(7, vec![blk(i, 512, i as u8)]))
            .collect();
        lists.push(first(7, Vec::new()));
        let j = TestJournal {
            blhdr_size: 512,
            in_range: Some(3),
            ..journal(lists.clone())
        };
        let r = run(&j);
        // The last list ends where the first begins.
        assert_eq!(r.positions[255], JSIZE - 512);
        assert_eq!(replayed(&r), (256, false));
        assert_applied(&r, &lists);
    }

    #[test]
    fn overlay_reads_across_patch_edges() {
        let lists = vec![
            first(1, vec![blk(1, 1024, 0x61), blk(10, 512, 0x62)]),
            first(2, vec![blk(3, 512, 0x63), blk(4000, 512, 0x64)]),
        ];
        let r = run(&journal(lists.clone()));
        let want = applied(&r.image, &lists);
        let d = &*r.opened.device;
        assert_eq!(d.len(), DEV as u64);
        assert_eq!(d.sector_size(), 512);
        let dev = DEV as u64;
        for (offset, len) in [
            (0u64, 1usize),
            (511, 2),
            (512, 1024),
            (1000, 5000),
            (1535, 2),
            (5119, 514),
            (5150, 10),
            (5630, 4),
            (4000 * 512 - 1, 514),
            (dev - 10, 10),
            (0, DEV),
        ] {
            let mut got = vec![0u8; len];
            d.read_at(offset, &mut got).unwrap();
            let want = &want[offset as usize..offset as usize + len];
            assert!(got == want, "read_at {offset} {len}");
            d.read_bulk(offset, &mut got).unwrap();
            assert!(got == want, "read_bulk {offset} {len}");
        }
        assert!(matches!(
            d.read_at(dev - 1, &mut [0u8; 2]),
            Err(Error::OutOfBounds { .. })
        ));
        assert!(matches!(
            d.read_bulk(u64::MAX, &mut [0u8; 1]),
            Err(Error::OutOfBounds { .. })
        ));
        assert!(d.read_at(dev, &mut []).is_ok());
    }

    #[test]
    fn patches_split_and_trim() {
        let list = |m: &BTreeMap<u64, Patch>| -> Vec<(u64, u64, u64)> {
            m.iter().map(|(&k, p)| (k, p.len, p.from)).collect()
        };
        let mut m = BTreeMap::new();
        insert(&mut m, 100, 100, 1000);
        insert(&mut m, 150, 10, 5000);
        assert_eq!(
            list(&m),
            [(100, 50, 1000), (150, 10, 5000), (160, 40, 1060)]
        );
        insert(&mut m, 90, 20, 7000);
        insert(&mut m, 155, 100, 8000);
        assert_eq!(
            list(&m),
            [
                (90, 20, 7000),
                (110, 40, 1010),
                (150, 5, 5000),
                (155, 100, 8000)
            ]
        );
        insert(&mut m, 300, 0, 1);
        assert_eq!(m.len(), 4);
        insert(&mut m, 0, 1000, 9);
        assert_eq!(list(&m), [(0, 1000, 9)]);
    }

    // ---- through the volume header and journal info block ----

    /// A volume with a one-transaction dirty journal, after `edit`.
    fn dirty_volume(edit: impl FnOnce(&mut JournalSetup)) -> Vec<u8> {
        let mut v = VolumeBuilder::new("J");
        v.journaled(128, MIN_JOURNAL)
            .file(ROOT, "a", 16, FileSpec::regular(1));
        edit(v.journal_setup());
        let mut image = v.build();
        journal(vec![first(1, vec![blk(4, 512, 0x77)])])
            .write(&mut image, v.journal_offset() as usize);
        image
    }

    fn open_volume(image: Vec<u8>) -> (Opened, u8) {
        let src = source(image);
        let header = read_header(&*src).unwrap();
        let opened = open(src, 0, &header).unwrap();
        let mut byte = [0u8];
        opened.device.read_at(4 * 512, &mut byte).unwrap();
        (opened, byte[0])
    }

    #[test]
    fn replays_through_the_journal_info_block() {
        let (opened, byte) = open_volume(dirty_volume(|_| {}));
        assert_eq!(
            opened.state,
            JournalState::Replayed {
                transactions: 1,
                stopped_early: false
            }
        );
        assert_eq!(byte, 0x77);
    }

    #[test]
    fn journals_macos_would_not_replay_are_clean() {
        type Edit = fn(&mut JournalSetup);
        let cases: [(&str, Edit); 3] = [
            ("non-journaling last mount", |j| {
                j.last_mounted = u32::from_be_bytes(*b"10.0");
            }),
            ("needs init", |j| j.jib_flags = JIB_IN_FS | JIB_NEED_INIT),
            ("external, unmounted cleanly", |j| {
                j.jib_flags = JIB_ON_OTHER_DEVICE;
                j.unmounted = true;
            }),
        ];
        for (name, edit) in cases {
            let (opened, byte) = open_volume(dirty_volume(edit));
            assert_eq!(opened.state, JournalState::Clean, "{name}");
            assert_eq!(byte, 0, "{name}");
        }
    }

    #[test]
    fn external_journal_on_a_dirty_volume_is_not_replayed() {
        for flags in [JIB_ON_OTHER_DEVICE, 0, JIB_IN_FS | JIB_ON_OTHER_DEVICE] {
            let (opened, byte) = open_volume(dirty_volume(|j| j.jib_flags = flags));
            assert_eq!(
                opened.state,
                JournalState::NotReplayed {
                    reason: "journal is on another device".to_string()
                }
            );
            assert_eq!(byte, 0);
        }
    }

    #[test]
    fn journal_info_block_bounds() {
        let image = dirty_volume(|_| {});
        let header = read_header(&MemSource::new(image.clone())).unwrap();
        let jib = (header.journal_info_block * header.block_size) as usize;
        let volume = image.len() as u64;
        for (offset, size) in [
            (volume, MIN_JOURNAL),
            (volume - MIN_JOURNAL + 512, MIN_JOURNAL),
            (u64::MAX - 10, MIN_JOURNAL),
            (4096, u64::MAX),
        ] {
            let mut image = image.clone();
            image[jib + 36..jib + 44].copy_from_slice(&offset.to_be_bytes());
            image[jib + 44..jib + 52].copy_from_slice(&size.to_be_bytes());
            let (opened, byte) = open_volume(image);
            let reason = not_replayed(&opened.state);
            assert!(reason.starts_with("journal info block"), "{reason}");
            assert_eq!(byte, 0);
        }
        // A journal info block past the volume.
        let mut header = header;
        header.journal_info_block = header.total_blocks;
        let src = source(image);
        let opened = open(Arc::clone(&src), 0, &header).unwrap();
        not_replayed(&opened.state);
        // One whose offset overflows, and a volume offset past the device.
        header.journal_info_block = u32::MAX;
        header.block_size = u32::MAX;
        not_replayed(&open(Arc::clone(&src), 0, &header).unwrap().state);
        not_replayed(&open(src, u64::MAX, &header).unwrap().state);
    }
}
