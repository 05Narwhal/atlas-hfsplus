// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! decmpfs decompression (the decmpfs header is little-endian).
//!
//! A file whose BSD owner flags carry `UF_COMPRESSED` has an empty data
//! fork and a `com.apple.decmpfs` attribute: a 16-byte header (magic `cmpf`
//! stored little-endian, so `fpmc` on disk; compression type; uncompressed
//! size), then, for the "xattr" kinds, the whole compressed payload. The
//! "resource fork" kinds keep the payload in the file's resource fork in
//! chunks of 64 KiB of output:
//!
//! - Type 4 (zlib): a classic resource fork. Big-endian data offset at 0
//!   (normally 0x100); at the data offset a big-endian resource length, then
//!   a little-endian chunk count and that many little-endian (offset, size)
//!   pairs, offsets counted from just after the resource length.
//! - Types 8, 10, 12 (LZVN, raw, LZFSE): `chunk_count + 1` little-endian
//!   offsets from the start of the fork; chunk `k` is `[off[k], off[k+1])`.
//!
//! Inside a chunk (or an xattr payload): zlib data whose first byte has a
//! low nibble of 0xF is stored raw after that byte; LZVN data whose first
//! byte is 0x06 (the end-of-stream opcode) is raw after it; LZFSE data whose
//! first byte is 0xFF is raw after it; the raw kinds (9, 10) are prefixed
//! with 0xCC. Type 1 is the uncompressed bytes right after the header.
//!
//! Layouts checked against apfs-fuse `ApfsLib/Decmpfs.cpp` (read, not
//! copied), the libfshfs format notes and xnu `bsd/sys/decmpfs.h`.

use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex, PoisonError};

use lzfse_rust::LzfseRingDecoder;

use super::attributes::Attributes;
use super::bytes::{be_u32, le_u32, le_u64};
use super::catalog::{FileRecord, kind_of_mode};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::NodeKind;
use atlas_fs_core::io::block::{BlockSource, MemSource, check_range};

/// Name of the attribute that holds the decmpfs header.
pub const DECMPFS_XATTR: &str = "com.apple.decmpfs";

/// `cmpf`, the header magic, as the little-endian `u32` it is stored as.
pub const DECMPFS_MAGIC: u32 = 0x636D_7066;

/// Size of the decmpfs header.
pub const HEADER_LEN: usize = 16;

/// Bytes of output per resource fork chunk.
pub const CHUNK_SIZE: usize = 0x1_0000;

/// `UF_COMPRESSED` in the BSD owner flags byte.
pub const UF_COMPRESSED: u8 = 0x20;

/// Largest decmpfs attribute read: a header and one chunk's worth of
/// payload. xnu itself refuses more than 3802 bytes on HFS+.
pub const MAX_DECMPFS_XATTR: usize = CHUNK_SIZE + HEADER_LEN;

/// Largest uncompressed size accepted for the kinds that keep their payload
/// in the attribute, which are decoded into memory in one piece.
pub const MAX_INLINE_SIZE: u64 = 16 << 20;

/// Largest compressed chunk accepted: a raw chunk is 64 KiB plus its
/// prefix byte; the rest is slack.
pub const MAX_COMPRESSED_CHUNK: u64 = 2 * CHUNK_SIZE as u64;

const STRUCTURE: &str = "decmpfs";

/// Table entries read per batch while validating a chunk table.
const TABLE_BATCH: usize = 1024;

fn corrupt(file_id: u32, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(
        STRUCTURE,
        Location::Object(u64::from(file_id)),
        kind,
    ))
}

fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// True when `file` is stored compressed: `UF_COMPRESSED` on a regular
/// file (xnu treats the flag on anything else as not compressed).
pub fn is_compressed(file: &FileRecord) -> bool {
    file.bsd.owner_flags & UF_COMPRESSED != 0 && kind_of_mode(file.bsd.file_mode) == NodeKind::File
}

/// The uncompressed size from `file_id`'s decmpfs header, or `None` when
/// the header cannot be read (no attribute, bad magic, read error). Reuses
/// `buf` for the 16 header bytes. Unsupported compression types still give
/// their size.
pub fn compressed_size(attributes: &Attributes, file_id: u32, buf: &mut Vec<u8>) -> Option<u64> {
    attributes
        .read_head(file_id, DECMPFS_XATTR, HEADER_LEN, buf)
        .ok()??;
    DecmpfsHeader::parse(buf, file_id)
        .ok()
        .map(|h| h.uncompressed_size)
}

/// The decmpfs header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecmpfsHeader {
    /// The compression type (1, 3, 4, 7 to 12 are read here).
    pub compression_type: u32,
    /// Size of the file's content once decompressed.
    pub uncompressed_size: u64,
}

impl DecmpfsHeader {
    /// Parses the header at the start of `b`. `file_id` names the file in
    /// errors. Fails as `Corrupt` when `b` is short or the magic is wrong.
    pub fn parse(b: &[u8], file_id: u32) -> Result<Self> {
        let truncated = || {
            corrupt(
                file_id,
                CorruptKind::Truncated {
                    needed: len64(HEADER_LEN),
                    available: len64(b.len()),
                },
            )
        };
        let magic = le_u32(b, 0).ok_or_else(truncated)?;
        if magic != DECMPFS_MAGIC {
            return Err(corrupt(
                file_id,
                CorruptKind::BadSignature {
                    expected: u64::from(DECMPFS_MAGIC),
                    found: u64::from(magic),
                },
            ));
        }
        Ok(Self {
            compression_type: le_u32(b, 4).ok_or_else(truncated)?,
            uncompressed_size: le_u64(b, 8).ok_or_else(truncated)?,
        })
    }
}

/// How a chunk or payload is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    /// Type 1: the bytes as they are, no prefix.
    Plain,
    Zlib,
    Lzvn,
    /// Types 9 and 10: `0xCC`, then the bytes.
    Raw,
    Lzfse,
}

/// Where the payload is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage {
    Xattr,
    ResourceFork,
}

fn kind_of(compression_type: u32) -> Option<(Codec, Storage)> {
    Some(match compression_type {
        1 => (Codec::Plain, Storage::Xattr),
        3 => (Codec::Zlib, Storage::Xattr),
        4 => (Codec::Zlib, Storage::ResourceFork),
        7 => (Codec::Lzvn, Storage::Xattr),
        8 => (Codec::Lzvn, Storage::ResourceFork),
        9 => (Codec::Raw, Storage::Xattr),
        10 => (Codec::Raw, Storage::ResourceFork),
        11 => (Codec::Lzfse, Storage::Xattr),
        12 => (Codec::Lzfse, Storage::ResourceFork),
        _ => return None,
    })
}

/// The decompressed content of a compressed file.
///
/// `xattr` is the whole `com.apple.decmpfs` attribute; `resource_fork` is
/// called only for the resource fork kinds. The xattr kinds are decoded
/// here, once, into memory; the resource fork kinds give a source that
/// decodes chunks on demand. A compression type this reader does not know
/// (LZBITMAP 13 and 14, anything else) is `Error::Unsupported`.
pub fn open(
    xattr: &[u8],
    file_id: u32,
    resource_fork: &mut dyn FnMut() -> Result<Arc<dyn BlockSource>>,
) -> Result<Arc<dyn BlockSource>> {
    let header = DecmpfsHeader::parse(xattr, file_id)?;
    let Some((codec, storage)) = kind_of(header.compression_type) else {
        return Err(Error::unsupported(format!(
            "decmpfs compression type {}",
            header.compression_type
        )));
    };
    match storage {
        Storage::Xattr => {
            let size = header.uncompressed_size;
            if size > MAX_INLINE_SIZE {
                return Err(corrupt(
                    file_id,
                    CorruptKind::LimitExceeded {
                        what: "uncompressed size of an attribute payload",
                        limit: MAX_INLINE_SIZE,
                    },
                ));
            }
            let payload = xattr.get(HEADER_LEN..).unwrap_or(&[]);
            // At most MAX_INLINE_SIZE, so it fits a usize.
            let mut out = vec![0u8; usize::try_from(size).unwrap_or(0)];
            if !out.is_empty() {
                let mut lzfse = None;
                decode(codec, payload, &mut out, &mut lzfse)
                    .map_err(|kind| corrupt(file_id, kind))?;
            }
            Ok(Arc::new(MemSource::new(out)))
        }
        Storage::ResourceFork => {
            let fork = resource_fork()?;
            Ok(Arc::new(DecmpfsSource::new(
                fork,
                codec,
                header.uncompressed_size,
                file_id,
            )?))
        }
    }
}

/// Decodes `src` into exactly `out.len()` bytes.
fn decode(
    codec: Codec,
    src: &[u8],
    out: &mut [u8],
    lzfse: &mut Option<Box<LzfseRingDecoder>>,
) -> Result<(), CorruptKind> {
    let first = src.first().copied();
    let rest = src.get(1..).unwrap_or(&[]);
    match codec {
        Codec::Plain => stored(src, out),
        Codec::Raw => match first {
            Some(0xCC) => stored(rest, out),
            other => Err(CorruptKind::InvalidValue {
                field: "uncompressed chunk prefix",
                value: other.map_or(0, u64::from),
            }),
        },
        Codec::Zlib => {
            if first.is_some_and(|b| b & 0x0F == 0x0F) {
                return stored(rest, out);
            }
            match miniz_oxide::inflate::decompress_slice_iter_to_slice(
                out,
                std::iter::once(src),
                true,
                false,
            ) {
                Ok(n) => exact(n, out.len()),
                // Includes output that would run past `out`: miniz stops at
                // the end of the slice instead of growing anything.
                Err(_) => Err(CorruptKind::Other(
                    "zlib data is invalid or longer than its chunk",
                )),
            }
        }
        Codec::Lzvn => {
            if first == Some(0x06) {
                return stored(rest, out);
            }
            // lzfse_rust has no bare LZVN decoder: wrap the stream as an
            // LZFSE `bvxn` block (magic, raw size, payload size, all LE)
            // followed by the end-of-stream block `bvx$`.
            let raw = u32::try_from(out.len()).map_err(|_| CorruptKind::Overflow {
                field: "LZVN raw size",
            })?;
            let payload = u32::try_from(src.len()).map_err(|_| CorruptKind::Overflow {
                field: "LZVN payload size",
            })?;
            let mut header = [0u8; 12];
            header[..4].copy_from_slice(b"bvxn");
            header[4..8].copy_from_slice(&raw.to_le_bytes());
            header[8..].copy_from_slice(&payload.to_le_bytes());
            let mut reader = (&header[..]).chain(src).chain(&b"bvx$"[..]);
            lzfse_decode(lzfse, &mut reader, out)
        }
        Codec::Lzfse => {
            if first == Some(0xFF) {
                return stored(rest, out);
            }
            let mut reader = src;
            lzfse_decode(lzfse, &mut reader, out)
        }
    }
}

/// Copies stored bytes, which must be exactly the expected length.
fn stored(data: &[u8], out: &mut [u8]) -> Result<(), CorruptKind> {
    exact(data.len(), out.len())?;
    out.copy_from_slice(data);
    Ok(())
}

fn exact(got: usize, want: usize) -> Result<(), CorruptKind> {
    if got == want {
        Ok(())
    } else {
        Err(CorruptKind::InvalidValue {
            field: "decompressed chunk length",
            value: len64(got),
        })
    }
}

/// A writer into a fixed slice that refuses anything past its end, so a
/// decoder can never produce more than the expected output.
struct Capped<'a> {
    out: &'a mut [u8],
    pos: usize,
}

impl Write for Capped<'_> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let end = self.pos.checked_add(data.len());
        match end.and_then(|end| self.out.get_mut(self.pos..end)) {
            Some(dst) => {
                dst.copy_from_slice(data);
                self.pos = end.unwrap_or(self.pos);
                Ok(data.len())
            }
            None => Err(io::Error::other("decompressed data longer than its chunk")),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Runs the LZFSE decoder (kept in `decoder` for reuse: its ring buffers
/// are 640 KiB) from `reader` into exactly `out`.
pub fn lzfse_decode<R: Read>(
    decoder: &mut Option<Box<LzfseRingDecoder>>,
    reader: &mut R,
    out: &mut [u8],
) -> Result<(), CorruptKind> {
    let want = out.len();
    let mut writer = Capped { out, pos: 0 };
    let result = decoder
        .get_or_insert_with(Box::default)
        .decode(reader, &mut writer);
    match result {
        Ok(_) => exact(writer.pos, want),
        Err(_) => {
            // Do not reuse a decoder that stopped half way.
            *decoder = None;
            Err(CorruptKind::Other(
                "LZFSE or LZVN data is invalid or longer than its chunk",
            ))
        }
    }
}

/// Where a resource fork kind keeps its chunk table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Table {
    /// Type 4: count and (offset, size) pairs at `base`, offsets from `base`.
    Pairs { base: u64 },
    /// Types 8, 10, 12: `chunks + 1` offsets from the start of the fork.
    Offsets,
}

/// The last decoded chunk and the buffers reused to decode the next one.
#[derive(Default)]
struct Cache {
    chunk: Option<u64>,
    data: Vec<u8>,
    input: Vec<u8>,
    lzfse: Option<Box<LzfseRingDecoder>>,
}

/// The decompressed content of a resource fork kind: decodes 64 KiB chunks
/// on demand and keeps the last one.
pub struct DecmpfsSource {
    fork: Arc<dyn BlockSource>,
    codec: Codec,
    table: Table,
    size: u64,
    chunks: u64,
    file_id: u32,
    cache: Mutex<Cache>,
}

impl DecmpfsSource {
    /// Validates the chunk table of `fork` for `size` bytes of output:
    /// the count matches `ceil(size / 64 KiB)`, chunks are in order, do not
    /// overlap the table or each other, lie inside the fork and are at most
    /// [`MAX_COMPRESSED_CHUNK`] bytes. The table is read in batches, never
    /// held whole.
    fn new(fork: Arc<dyn BlockSource>, codec: Codec, size: u64, file_id: u32) -> Result<Self> {
        let chunks = size.div_ceil(len64(CHUNK_SIZE));
        let table = if codec == Codec::Zlib {
            let mut b = [0u8; 4];
            read(&*fork, 0, &mut b, file_id)?;
            let data_offset = u64::from(be_u32(&b, 0).unwrap_or(0));
            Table::Pairs {
                base: data_offset.checked_add(4).ok_or_else(|| {
                    corrupt(
                        file_id,
                        CorruptKind::Overflow {
                            field: "resource data offset",
                        },
                    )
                })?,
            }
        } else {
            Table::Offsets
        };
        let source = Self {
            fork,
            codec,
            table,
            size,
            chunks,
            file_id,
            cache: Mutex::new(Cache::default()),
        };
        source.validate()?;
        Ok(source)
    }

    fn corrupt(&self, kind: CorruptKind) -> Error {
        corrupt(self.file_id, kind)
    }

    /// Byte offset of the first table entry and the bytes per chunk entry.
    fn entries(&self) -> Result<(u64, u64)> {
        match self.table {
            Table::Pairs { base } => {
                let mut b = [0u8; 4];
                read(&*self.fork, base, &mut b, self.file_id)?;
                let count = u64::from(le_u32(&b, 0).unwrap_or(0));
                if count != self.chunks {
                    return Err(self.corrupt(CorruptKind::InvalidValue {
                        field: "chunk count",
                        value: count,
                    }));
                }
                Ok((base.saturating_add(4), 8))
            }
            Table::Offsets => Ok((0, 4)),
        }
    }

    /// Checks the whole table, reading it in batches.
    fn validate(&self) -> Result<()> {
        let fork_len = self.fork.len();
        let (start, per) = self.entries()?;
        // Offsets has one entry more than there are chunks.
        let entries = match self.table {
            Table::Pairs { .. } => self.chunks,
            Table::Offsets => self.chunks.saturating_add(1),
        };
        let table_end = entries
            .checked_mul(per)
            .and_then(|n| n.checked_add(start))
            .filter(|&end| end <= fork_len)
            .ok_or_else(|| {
                self.corrupt(CorruptKind::OutOfRange {
                    field: "chunk table end",
                    value: entries.saturating_mul(per).saturating_add(start),
                    limit_name: "resource fork length",
                    limit: fork_len,
                })
            })?;

        let mut buf = vec![0u8; TABLE_BATCH.saturating_mul(8)];
        // The end of the previous chunk; the first must start at or after
        // the table.
        let mut prev_end = table_end;
        let mut done: u64 = 0;
        // Bounded: `done` grows by at least one entry per pass up to `entries`.
        while done < entries {
            let batch = entries.saturating_sub(done).min(len64(TABLE_BATCH));
            let bytes = usize::try_from(batch.saturating_mul(per)).unwrap_or(0);
            let chunk = buf.get_mut(..bytes).unwrap_or(&mut []);
            let at = start.saturating_add(done.saturating_mul(per));
            read(&*self.fork, at, chunk, self.file_id)?;
            for (i, entry) in chunk
                .chunks_exact(usize::try_from(per).unwrap_or(4))
                .enumerate()
            {
                let index = done.saturating_add(len64(i));
                match self.table {
                    Table::Pairs { base } => {
                        let off = u64::from(le_u32(entry, 0).unwrap_or(0));
                        let len = u64::from(le_u32(entry, 4).unwrap_or(0));
                        let chunk_start = base.saturating_add(off);
                        self.check_chunk(index, prev_end, chunk_start, len)?;
                        prev_end = chunk_start.saturating_add(len);
                    }
                    Table::Offsets => {
                        let off = u64::from(le_u32(entry, 0).unwrap_or(0));
                        if index == 0 {
                            if off != table_end {
                                return Err(self.corrupt(CorruptKind::InvalidValue {
                                    field: "first chunk offset",
                                    value: off,
                                }));
                            }
                        } else {
                            let len = off.checked_sub(prev_end).ok_or_else(|| {
                                self.corrupt(CorruptKind::InvalidValue {
                                    field: "chunk offset out of order",
                                    value: off,
                                })
                            })?;
                            self.check_chunk(index.saturating_sub(1), prev_end, prev_end, len)?;
                        }
                        prev_end = off;
                    }
                }
            }
            done = done.saturating_add(batch);
        }
        Ok(())
    }

    /// Checks one chunk's range: starts at or after `min_start`, is 1 to
    /// [`MAX_COMPRESSED_CHUNK`] bytes and ends inside the fork.
    fn check_chunk(&self, index: u64, min_start: u64, start: u64, len: u64) -> Result<()> {
        if start < min_start {
            return Err(self.corrupt(CorruptKind::InvalidValue {
                field: "chunk offset out of order",
                value: index,
            }));
        }
        if len == 0 || len > MAX_COMPRESSED_CHUNK {
            return Err(self.corrupt(CorruptKind::OutOfRange {
                field: "compressed chunk size",
                value: len,
                limit_name: "largest compressed chunk",
                limit: MAX_COMPRESSED_CHUNK,
            }));
        }
        let fork_len = self.fork.len();
        match start.checked_add(len) {
            Some(end) if end <= fork_len => Ok(()),
            _ => Err(self.corrupt(CorruptKind::OutOfRange {
                field: "chunk end",
                value: start.saturating_add(len),
                limit_name: "resource fork length",
                limit: fork_len,
            })),
        }
    }

    /// (start, length) of compressed chunk `k` in the fork, from its table
    /// entry, checked again as it is read.
    fn chunk_range(&self, k: u64) -> Result<(u64, u64)> {
        let mut b = [0u8; 8];
        let (start, len) = match self.table {
            Table::Pairs { base } => {
                let at = k
                    .checked_mul(8)
                    .and_then(|n| n.checked_add(base))
                    .and_then(|n| n.checked_add(4))
                    .ok_or_else(|| {
                        self.corrupt(CorruptKind::Overflow {
                            field: "chunk entry",
                        })
                    })?;
                read(&*self.fork, at, &mut b, self.file_id)?;
                let off = u64::from(le_u32(&b, 0).unwrap_or(0));
                let len = u64::from(le_u32(&b, 4).unwrap_or(0));
                (base.saturating_add(off), len)
            }
            Table::Offsets => {
                let at = k.checked_mul(4).ok_or_else(|| {
                    self.corrupt(CorruptKind::Overflow {
                        field: "chunk entry",
                    })
                })?;
                read(&*self.fork, at, &mut b, self.file_id)?;
                let off = u64::from(le_u32(&b, 0).unwrap_or(0));
                let next = u64::from(le_u32(&b, 4).unwrap_or(0));
                let len = next.checked_sub(off).ok_or_else(|| {
                    self.corrupt(CorruptKind::InvalidValue {
                        field: "chunk offset out of order",
                        value: next,
                    })
                })?;
                (off, len)
            }
        };
        self.check_chunk(k, 0, start, len)?;
        Ok((start, len))
    }

    /// Output bytes of chunk `k`: 64 KiB, or the remainder for the last.
    fn chunk_len(&self, k: u64) -> u64 {
        let start = k.saturating_mul(len64(CHUNK_SIZE));
        self.size.saturating_sub(start).min(len64(CHUNK_SIZE))
    }

    /// Makes chunk `k` the cached one.
    fn load(&self, cache: &mut Cache, k: u64) -> Result<()> {
        if cache.chunk == Some(k) {
            return Ok(());
        }
        cache.chunk = None;
        let (start, len) = self.chunk_range(k)?;
        let Cache {
            data, input, lzfse, ..
        } = cache;
        // Both at most MAX_COMPRESSED_CHUNK and CHUNK_SIZE.
        input.resize(usize::try_from(len).unwrap_or(0), 0);
        self.fork.read_at(start, input)?;
        data.resize(usize::try_from(self.chunk_len(k)).unwrap_or(0), 0);
        decode(self.codec, input, data, lzfse).map_err(|kind| self.corrupt(kind))?;
        cache.chunk = Some(k);
        Ok(())
    }
}

/// `read_at` with a short source turned into `Corrupt`.
fn read(source: &dyn BlockSource, at: u64, buf: &mut [u8], file_id: u32) -> Result<()> {
    match source.read_at(at, buf) {
        Err(Error::OutOfBounds { offset, len, size }) => Err(corrupt(
            file_id,
            CorruptKind::OutOfRange {
                field: "resource fork read end",
                value: offset.saturating_add(len),
                limit_name: "resource fork length",
                limit: size,
            },
        )),
        other => other,
    }
}

impl BlockSource for DecmpfsSource {
    fn len(&self) -> u64 {
        self.size
    }

    fn sector_size(&self) -> u32 {
        self.fork.sector_size()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        check_range(offset, buf.len(), self.size)?;
        let chunk = len64(CHUNK_SIZE);
        let mut pos = offset;
        let mut rest = buf;
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        // Bounded: each pass copies at least one byte, since pos < size
        // leaves at least one byte in the chunk holding pos.
        while !rest.is_empty() {
            let k = pos.checked_div(chunk).unwrap_or(0);
            let within = usize::try_from(pos.checked_rem(chunk).unwrap_or(0)).unwrap_or(0);
            self.load(&mut cache, k)?;
            let src = cache.data.get(within..).unwrap_or(&[]);
            let n = src.len().min(rest.len());
            if n == 0 {
                return Err(self.corrupt(CorruptKind::Other("decoded chunk is empty")));
            }
            let (head, tail) = rest.split_at_mut(n);
            head.copy_from_slice(src.get(..n).unwrap_or(&[]));
            pos = pos.saturating_add(len64(n));
            rest = tail;
        }
        Ok(())
    }
}

#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod fixtures {
    //! Builders for compressed payloads and resource forks, shared with the
    //! test volume.

    use super::{CHUNK_SIZE, DECMPFS_MAGIC};

    /// A decmpfs header.
    pub fn header(compression_type: u32, size: u64) -> Vec<u8> {
        let mut h = DECMPFS_MAGIC.to_le_bytes().to_vec();
        h.extend_from_slice(&compression_type.to_le_bytes());
        h.extend_from_slice(&size.to_le_bytes());
        h
    }

    /// `data` as a zlib stream.
    pub fn zlib(data: &[u8]) -> Vec<u8> {
        miniz_oxide::deflate::compress_to_vec_zlib(data, 6)
    }

    /// `data` as an LZFSE stream.
    pub fn lzfse(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        lzfse_rust::encode_bytes(data, &mut out).unwrap();
        out
    }

    /// `data` as an LZVN stream from lzfse_rust's encoder, which writes a
    /// `bvxn` block for inputs of 20 to 4095 bytes.
    pub fn lzvn(data: &[u8]) -> Vec<u8> {
        assert!((20..4096).contains(&data.len()));
        let block = lzfse(data);
        assert_eq!(&block[..4], b"bvxn");
        assert_eq!(
            u32::from_le_bytes(block[4..8].try_into().unwrap()) as usize,
            data.len()
        );
        let n = u32::from_le_bytes(block[8..12].try_into().unwrap()) as usize;
        assert_eq!(&block[12 + n..], b"bvx$");
        block[12..12 + n].to_vec()
    }

    /// `data` as an LZVN stream of literals only (any length), ending with
    /// the end-of-stream opcode and its seven pad bytes.
    pub fn lzvn_literals(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for run in data.chunks(271) {
            if run.len() >= 16 {
                out.push(0xE0);
                out.push((run.len() - 16) as u8);
            } else {
                out.push(0xE0 | run.len() as u8);
            }
            out.extend_from_slice(run);
        }
        out.extend_from_slice(&[0x06, 0, 0, 0, 0, 0, 0, 0]);
        out
    }

    /// A type 8, 10 or 12 resource fork: the offset table, then the chunks.
    pub fn offsets_fork(chunks: &[Vec<u8>]) -> Vec<u8> {
        let mut at = 4 * (chunks.len() + 1);
        let mut out = Vec::new();
        out.extend_from_slice(&(at as u32).to_le_bytes());
        for c in chunks {
            at += c.len();
            out.extend_from_slice(&(at as u32).to_le_bytes());
        }
        for c in chunks {
            out.extend_from_slice(c);
        }
        out
    }

    /// A type 4 resource fork: resource header, resource length, count and
    /// (offset, size) pairs, the chunks.
    pub fn zlib_fork(chunks: &[Vec<u8>]) -> Vec<u8> {
        let mut out = vec![0u8; 0x100];
        out[0..4].copy_from_slice(&0x100u32.to_be_bytes());
        let table = 4 + 8 * chunks.len();
        let total: usize = table + chunks.iter().map(Vec::len).sum::<usize>();
        out[8..12].copy_from_slice(&(total as u32 + 4).to_be_bytes());
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        let mut at = table;
        for c in chunks {
            out.extend_from_slice(&(at as u32).to_le_bytes());
            out.extend_from_slice(&(c.len() as u32).to_le_bytes());
            at += c.len();
        }
        for c in chunks {
            out.extend_from_slice(c);
        }
        out
    }

    /// `data` split into 64 KiB pieces.
    pub fn pieces(data: &[u8]) -> Vec<&[u8]> {
        data.chunks(CHUNK_SIZE).collect()
    }

    /// Test content: varied but compressible.
    pub fn content(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i % 251) as u8 ^ ((i / 1000) as u8))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    const FILE: u32 = 77;

    fn mem(bytes: Vec<u8>) -> Arc<dyn BlockSource> {
        Arc::new(MemSource::new(bytes))
    }

    fn no_fork() -> Result<Arc<dyn BlockSource>> {
        panic!("the resource fork was not expected to be read")
    }

    /// Opens an xattr kind.
    fn open_xattr(ty: u32, size: u64, payload: &[u8]) -> Result<Arc<dyn BlockSource>> {
        let mut x = header(ty, size);
        x.extend_from_slice(payload);
        open(&x, FILE, &mut no_fork)
    }

    /// Opens a resource fork kind.
    fn open_fork(ty: u32, size: u64, fork: Vec<u8>) -> Result<Arc<dyn BlockSource>> {
        let x = header(ty, size);
        let mut fork = Some(fork);
        open(&x, FILE, &mut || Ok(mem(fork.take().unwrap())))
    }

    fn read_all(src: &dyn BlockSource) -> Vec<u8> {
        let mut buf = vec![0u8; src.len() as usize];
        src.read_at(0, &mut buf).unwrap();
        buf
    }

    fn corruption<T>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            Err(e) => panic!("expected Corrupt, got {e:?}"),
            Ok(_) => panic!("expected Corrupt, got Ok"),
        }
    }

    // ---- header ----

    #[test]
    fn header_parse() {
        let h = DecmpfsHeader::parse(&header(4, 123_456), FILE).unwrap();
        assert_eq!(
            h,
            DecmpfsHeader {
                compression_type: 4,
                uncompressed_size: 123_456
            }
        );
        // On disk the magic reads "fpmc".
        assert_eq!(&header(3, 0)[..4], b"fpmc");
        let mut bad = header(3, 1);
        bad[0] = b'x';
        assert!(matches!(
            corruption(DecmpfsHeader::parse(&bad, FILE)).kind,
            CorruptKind::BadSignature { .. }
        ));
        for len in [0, 3, 4, 8, 15] {
            assert!(matches!(
                corruption(DecmpfsHeader::parse(&header(3, 1)[..len], FILE)).kind,
                CorruptKind::Truncated { .. }
            ));
        }
        // Trailing bytes are the payload, not part of the header.
        let mut long = header(9, 2);
        long.extend_from_slice(&[0xCC, 1, 2]);
        assert_eq!(
            DecmpfsHeader::parse(&long, FILE).unwrap().uncompressed_size,
            2
        );
    }

    // ---- xattr kinds ----

    #[test]
    fn xattr_kinds_round_trip() {
        let data = content(3000);
        let size = data.len() as u64;
        let mut raw_prefixed = vec![0xCC];
        raw_prefixed.extend_from_slice(&data);
        let mut zlib_raw = vec![0xFF];
        zlib_raw.extend_from_slice(&data);
        let mut lzvn_raw = vec![0x06];
        lzvn_raw.extend_from_slice(&data);
        let mut lzfse_raw = vec![0xFF];
        lzfse_raw.extend_from_slice(&data);
        let cases: Vec<(u32, Vec<u8>)> = vec![
            (1, data.clone()),
            (3, zlib(&data)),
            (3, zlib_raw),
            (7, lzvn(&data)),
            (7, lzvn_literals(&data)),
            (7, lzvn_raw),
            (9, raw_prefixed),
            (11, lzfse(&data)),
            (11, lzfse_raw),
        ];
        for (ty, payload) in cases {
            let src = open_xattr(ty, size, &payload).unwrap();
            assert_eq!(src.len(), size, "type {ty}");
            assert_eq!(read_all(&*src), data, "type {ty}");
        }
    }

    #[test]
    fn xattr_kind_larger_than_a_chunk() {
        // One zlib stream for the whole file, as afsctool writes it.
        let data = content(200_000);
        let src = open_xattr(3, data.len() as u64, &zlib(&data)).unwrap();
        assert_eq!(read_all(&*src), data);
        let src = open_xattr(11, data.len() as u64, &lzfse(&data)).unwrap();
        assert_eq!(read_all(&*src), data);
    }

    #[test]
    fn xattr_kind_empty_file() {
        for ty in [1, 3, 7, 9, 11] {
            let src = open_xattr(ty, 0, &[]).unwrap();
            assert_eq!(src.len(), 0);
        }
    }

    #[test]
    fn xattr_kind_wrong_length() {
        let data = content(1000);
        // Claims more than the stream holds, and less.
        for size in [999, 1001] {
            assert!(open_xattr(3, size, &zlib(&data)).is_err());
            assert!(open_xattr(7, size, &lzvn(&data)).is_err());
            assert!(open_xattr(11, size, &lzfse(&data)).is_err());
            assert!(open_xattr(1, size, &data).is_err());
            let mut raw = vec![0xCC];
            raw.extend_from_slice(&data);
            assert!(open_xattr(9, size, &raw).is_err());
        }
        // Raw without its prefix.
        assert!(matches!(
            corruption(open_xattr(9, 1000, &data)).kind,
            CorruptKind::InvalidValue { .. }
        ));
        // Garbage.
        assert!(open_xattr(3, 10, &[0x78, 1, 2, 3]).is_err());
        assert!(open_xattr(7, 10, &[1, 2, 3]).is_err());
        assert!(open_xattr(11, 10, b"bvx2garbage").is_err());
        assert!(open_xattr(3, 10, &[]).is_err());
    }

    #[test]
    fn xattr_kind_size_limit() {
        let c = corruption(open_xattr(3, MAX_INLINE_SIZE + 1, &zlib(b"x")));
        assert!(matches!(c.kind, CorruptKind::LimitExceeded { .. }));
        let c = corruption(open_xattr(3, u64::MAX, &zlib(b"x")));
        assert!(matches!(c.kind, CorruptKind::LimitExceeded { .. }));
    }

    #[test]
    fn unsupported_types() {
        for ty in [0, 2, 5, 6, 13, 14, 15, 0x8000_0001] {
            assert!(
                matches!(open_xattr(ty, 10, &[0; 10]), Err(Error::Unsupported(_))),
                "type {ty}"
            );
        }
    }

    // ---- resource fork kinds ----

    #[test]
    fn resource_fork_kinds_round_trip() {
        // 150 KiB: chunks of 64, 64 and 22 KiB.
        let data = content(150 * 1024);
        let size = data.len() as u64;
        let parts = pieces(&data);
        assert_eq!(parts.len(), 3);
        let raw = |p: &[u8], prefix: u8| {
            let mut v = vec![prefix];
            v.extend_from_slice(p);
            v
        };
        let cases: Vec<(u32, Vec<u8>)> = vec![
            (
                4,
                zlib_fork(&parts.iter().map(|p| zlib(p)).collect::<Vec<_>>()),
            ),
            (
                4,
                zlib_fork(&parts.iter().map(|p| raw(p, 0xFF)).collect::<Vec<_>>()),
            ),
            (
                8,
                offsets_fork(&parts.iter().map(|p| lzvn_literals(p)).collect::<Vec<_>>()),
            ),
            (
                8,
                offsets_fork(&parts.iter().map(|p| raw(p, 0x06)).collect::<Vec<_>>()),
            ),
            (
                10,
                offsets_fork(&parts.iter().map(|p| raw(p, 0xCC)).collect::<Vec<_>>()),
            ),
            (
                12,
                offsets_fork(&parts.iter().map(|p| lzfse(p)).collect::<Vec<_>>()),
            ),
            (
                12,
                offsets_fork(&parts.iter().map(|p| raw(p, 0xFF)).collect::<Vec<_>>()),
            ),
        ];
        for (ty, fork) in cases {
            let src = open_fork(ty, size, fork).unwrap();
            assert_eq!(src.len(), size);
            assert_eq!(read_all(&*src), data, "type {ty}");
            // Reads that straddle chunk boundaries, out of order.
            for (at, len) in [
                (65_530u64, 20usize),
                (131_000, 1_000),
                (0, 1),
                (size - 1, 1),
                (65_536, 65_536),
                (10, 140_000),
                (65_535, 2),
            ] {
                let mut buf = vec![0u8; len];
                src.read_at(at, &mut buf).unwrap();
                assert_eq!(
                    buf,
                    data[at as usize..at as usize + len],
                    "type {ty} at {at}"
                );
            }
            assert!(matches!(
                src.read_at(size, &mut [0u8; 1]),
                Err(Error::OutOfBounds { .. })
            ));
            src.read_at(size, &mut []).unwrap();
        }
    }

    #[test]
    fn mixed_lzvn_chunks() {
        // A literal chunk and an encoder-made chunk with real matches.
        let data = content(CHUNK_SIZE + 2000);
        let parts = pieces(&data);
        let fork = offsets_fork(&[lzvn_literals(parts[0]), lzvn(parts[1])]);
        let src = open_fork(8, data.len() as u64, fork).unwrap();
        assert_eq!(read_all(&*src), data);
    }

    #[test]
    fn resource_fork_exact_multiple_and_empty() {
        let data = content(2 * CHUNK_SIZE);
        let parts = pieces(&data);
        let fork = zlib_fork(&parts.iter().map(|p| zlib(p)).collect::<Vec<_>>());
        let src = open_fork(4, data.len() as u64, fork).unwrap();
        assert_eq!(read_all(&*src), data);
        // An empty file: no chunks.
        let src = open_fork(4, 0, zlib_fork(&[])).unwrap();
        assert_eq!(src.len(), 0);
        let src = open_fork(12, 0, offsets_fork(&[])).unwrap();
        assert_eq!(src.len(), 0);
    }

    fn three_chunk_offsets_fork() -> (u64, Vec<u8>) {
        let data = content(150 * 1024);
        let parts = pieces(&data);
        let fork = offsets_fork(&parts.iter().map(|p| lzfse(p)).collect::<Vec<_>>());
        (data.len() as u64, fork)
    }

    fn set_le(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn le(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn corrupt_offset_table() {
        let (size, good) = three_chunk_offsets_fork();
        assert!(open_fork(12, size, good.clone()).is_ok());

        // Non-monotonic.
        let mut f = good.clone();
        let (o1, o2) = (le(&f, 4), le(&f, 8));
        set_le(&mut f, 4, o2);
        set_le(&mut f, 8, o1);
        assert!(open_fork(12, size, f).is_err());
        // Past the fork's end.
        let mut f = good.clone();
        let end = f.len() as u32;
        set_le(&mut f, 12, end + 1);
        assert!(matches!(
            corruption(open_fork(12, size, f)).kind,
            CorruptKind::OutOfRange { .. }
        ));
        // Wrong count: the size says four chunks, the table holds three.
        assert!(open_fork(12, size + CHUNK_SIZE as u64, good.clone()).is_err());
        // Wrong count: the size says two chunks.
        assert!(open_fork(12, size - 40 * 1024, good.clone()).is_err());
        // A zero-length chunk.
        let mut f = good.clone();
        let o1 = le(&f, 4);
        set_le(&mut f, 8, o1);
        assert!(open_fork(12, size, f).is_err());
        // First offset inside the table.
        let mut f = good.clone();
        set_le(&mut f, 0, 4);
        assert!(open_fork(12, size, f).is_err());
        // A chunk larger than allowed.
        let mut f = good.clone();
        f.resize(f.len() + 200_000, 0);
        let o2 = le(&f, 8);
        set_le(&mut f, 12, o2 + MAX_COMPRESSED_CHUNK as u32 + 1);
        assert!(open_fork(12, size, f).is_err());
        // A fork too short for its table.
        assert!(open_fork(12, size, good[..10].to_vec()).is_err());
        assert!(open_fork(12, size, Vec::new()).is_err());
        // A huge claimed size: the table cannot fit, no allocation.
        assert!(open_fork(12, u64::MAX, good).is_err());
    }

    #[test]
    fn corrupt_zlib_table() {
        let data = content(150 * 1024);
        let parts = pieces(&data);
        let good = zlib_fork(&parts.iter().map(|p| zlib(p)).collect::<Vec<_>>());
        let size = data.len() as u64;
        assert!(open_fork(4, size, good.clone()).is_ok());
        let pair = |i: usize| 0x108 + 8 * i;
        // Count does not match the size.
        let mut f = good.clone();
        set_le(&mut f, 0x104, 2);
        assert!(matches!(
            corruption(open_fork(4, size, f)).kind,
            CorruptKind::InvalidValue {
                field: "chunk count",
                ..
            }
        ));
        // Overlapping chunks.
        let mut f = good.clone();
        let first = le(&f, pair(0));
        set_le(&mut f, pair(1), first);
        assert!(open_fork(4, size, f).is_err());
        // A chunk overlapping the table.
        let mut f = good.clone();
        set_le(&mut f, pair(0), 0);
        assert!(open_fork(4, size, f).is_err());
        // A chunk past the end.
        let mut f = good.clone();
        set_le(&mut f, pair(2) + 4, 1 << 20);
        assert!(open_fork(4, size, f).is_err());
        // A data offset past the fork.
        let mut f = good.clone();
        f[0..4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(open_fork(4, size, f).is_err());
        assert!(open_fork(4, size, vec![0, 0]).is_err());
    }

    #[test]
    fn chunk_decoding_to_the_wrong_length() {
        // The middle chunk holds 1000 bytes too few.
        let data = content(150 * 1024);
        let parts = pieces(&data);
        let short = &parts[1][..CHUNK_SIZE - 1000];
        let build = |enc: &dyn Fn(&[u8]) -> Vec<u8>| -> Vec<Vec<u8>> {
            vec![enc(parts[0]), enc(short), enc(parts[2])]
        };
        let forks = [
            (4, zlib_fork(&build(&|p| zlib(p)))),
            (8, offsets_fork(&build(&|p| lzvn_literals(p)))),
            (12, offsets_fork(&build(&|p| lzfse(p)))),
            (
                10,
                offsets_fork(&build(&|p| {
                    let mut v = vec![0xCC];
                    v.extend_from_slice(p);
                    v
                })),
            ),
        ];
        for (ty, fork) in forks {
            let src = open_fork(ty, data.len() as u64, fork).unwrap();
            // The first chunk is fine; the second fails as Corrupt.
            let mut buf = vec![0u8; 100];
            src.read_at(0, &mut buf).unwrap();
            assert_eq!(buf, data[..100]);
            assert!(corruption(src.read_at(CHUNK_SIZE as u64, &mut buf)).structure == STRUCTURE);
            // And the failure does not poison later reads of good chunks.
            src.read_at(2 * CHUNK_SIZE as u64, &mut buf).unwrap();
            assert_eq!(buf, data[2 * CHUNK_SIZE..2 * CHUNK_SIZE + 100]);
        }
        // Too long: a full chunk where only the 22 KiB remainder belongs.
        let fork = offsets_fork(&[lzfse(parts[0]), lzfse(parts[1]), lzfse(parts[0])]);
        let src = open_fork(12, data.len() as u64, fork).unwrap();
        assert!(src.read_at(2 * CHUNK_SIZE as u64, &mut [0u8; 10]).is_err());
    }

    #[test]
    fn decompression_bombs_fail() {
        // 64 MiB of zeros compresses to a few KiB; a chunk may produce only
        // 64 KiB. The decoders write into the 64 KiB chunk buffer and stop
        // at its end, so nothing grows.
        let zeros = vec![0u8; 64 << 20];
        let z = zlib(&zeros);
        let l = lzfse(&zeros);
        assert!(z.len() < MAX_COMPRESSED_CHUNK as usize);
        assert!(l.len() < MAX_COMPRESSED_CHUNK as usize);
        drop(zeros);
        let src = open_fork(4, CHUNK_SIZE as u64, zlib_fork(std::slice::from_ref(&z))).unwrap();
        assert!(src.read_at(0, &mut [0u8; 16]).is_err());
        let src = open_fork(
            12,
            CHUNK_SIZE as u64,
            offsets_fork(std::slice::from_ref(&l)),
        )
        .unwrap();
        assert!(src.read_at(0, &mut [0u8; 16]).is_err());
        // The xattr kinds: the claimed size bounds the output.
        assert!(open_xattr(3, 100, &z).is_err());
        assert!(open_xattr(11, 100, &l).is_err());
        // An LZVN stream of a 100-byte claim that produces more.
        let lv = lzvn_literals(&[7u8; 5000]);
        assert!(open_xattr(7, 100, &lv).is_err());
    }

    #[test]
    fn source_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DecmpfsSource>();
    }
}
