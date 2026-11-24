// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The allocation file (TN1150 "Allocation File"): a bitmap with one bit per
//! allocation block, most significant bit first within each byte, set when
//! the block is in use. [`density`] reduces it to the bucket densities of an
//! [`AllocationMap`](atlas_fs_core::fs::AllocationMap).

use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::MAX_ALLOCATION_BUCKETS;
use atlas_fs_core::io::block::BlockSource;

/// CNID of the allocation file.
pub const ALLOCATION_FILE_ID: u32 = 6;

/// Structure name for errors in the allocation file.
const STRUCTURE: &str = "allocation file";

/// Largest bulk read of the bitmap, in bytes: the one buffer `density` uses.
const PIECE: usize = 1 << 20;

/// `HEAD[k]`: the bits of a byte from bit `k` (MSB first) to its end.
const HEAD: [u8; 8] = [0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01];
/// `TAIL[k]`: the bits of a byte from its start to bit `k` (MSB first), inclusive.
const TAIL: [u8; 8] = [0x80, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC, 0xFE, 0xFF];

fn corrupt(kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(
        STRUCTURE,
        Location::Object(u64::from(ALLOCATION_FILE_ID)),
        kind,
    ))
}

/// How full each of `buckets` slices of a volume of `total_blocks` blocks
/// is, from its allocation `bitmap` (the allocation file's content).
///
/// The bucket count is clamped to 1 ..= [`MAX_ALLOCATION_BUCKETS`] and to
/// `total_blocks`. Bucket `i` covers blocks `[i*T/n, (i+1)*T/n)`, so each
/// has at least one block; its value is `ceil(used * 255 / blocks)`: 0 only
/// when empty, 255 only when full. Bits past `total_blocks` are ignored.
///
/// Fails with `Corrupt` when `total_blocks` is 0 or the bitmap holds fewer
/// than `ceil(total_blocks / 8)` bytes, and with the bitmap's read errors.
/// Reads it in pieces of at most 1 MiB through one buffer.
pub fn density(bitmap: &dyn BlockSource, total_blocks: u32, buckets: usize) -> Result<Vec<u8>> {
    density_in_pieces(bitmap, total_blocks, buckets, PIECE)
}

/// [`density`] with reads of at most `piece` bytes (at least 1).
fn density_in_pieces(
    bitmap: &dyn BlockSource,
    total_blocks: u32,
    buckets: usize,
    piece: usize,
) -> Result<Vec<u8>> {
    if total_blocks == 0 {
        return Err(corrupt(CorruptKind::InvalidValue {
            field: "total blocks",
            value: 0,
        }));
    }
    let total = u64::from(total_blocks);
    let needed = total.div_ceil(8);
    let available = bitmap.len();
    if available < needed {
        return Err(corrupt(CorruptKind::Truncated { needed, available }));
    }
    // total_blocks fits in usize on every target this builds for (>= 32 bit).
    let n = buckets
        .clamp(1, MAX_ALLOCATION_BUCKETS)
        .min(usize::try_from(total_blocks).unwrap_or(usize::MAX));
    let n64 = u64::try_from(n).unwrap_or(u64::MAX);
    // i <= 4096 and T < 2^32, so the product cannot saturate.
    let bound = |i: u64| i.saturating_mul(total).checked_div(n64).unwrap_or(total);

    let piece_len = usize::try_from(needed).map_or(piece, |b| b.min(piece.max(1)));
    let mut buf = vec![0u8; piece_len];
    let mut density = Vec::with_capacity(n);

    // The bucket being counted: blocks [lo, hi), `used` of them so far.
    let mut bucket: u64 = 0;
    let mut lo: u64 = 0;
    let mut hi = bound(1);
    let mut used: u64 = 0;
    // Byte offset of the current piece. Bounded: each pass reads at least
    // one byte, and stops once every bucket is done.
    let mut at: u64 = 0;
    while at < needed && bucket < n64 {
        let left = needed.saturating_sub(at);
        let len = usize::try_from(left).map_or(piece_len, |l| l.min(piece_len));
        let chunk = buf.get_mut(..len).ok_or_else(|| ended_early(at))?;
        bitmap.read_bulk(at, chunk)?;
        let first_bit = at.saturating_mul(8);
        let len64 = u64::try_from(len).unwrap_or(u64::MAX);
        let end_bit = first_bit.saturating_add(len64.saturating_mul(8)).min(total);
        // Every bucket that ends in this piece, then the start of the one
        // that continues past it. Bounded: each pass finishes a bucket or
        // leaves the loop.
        while bucket < n64 {
            let from = to_local(lo.max(first_bit), first_bit)?;
            let to = to_local(hi.min(end_bit), first_bit)?;
            used = used.saturating_add(count_ones(chunk, from, to));
            if hi > end_bit {
                break;
            }
            density.push(scale(used, hi.saturating_sub(lo)));
            used = 0;
            bucket = bucket.saturating_add(1);
            lo = hi;
            hi = bound(bucket.saturating_add(1));
        }
        at = at.saturating_add(len64);
    }
    if density.len() != n {
        return Err(ended_early(at));
    }
    Ok(density)
}

/// A bitmap that ran out before the last bucket. `density` reads
/// `ceil(total / 8)` bytes, which hold every bucket, so this means a logic
/// error rather than damage; it is still an error, never a panic.
fn ended_early(at: u64) -> Error {
    Error::Corrupt(Corruption::new(
        STRUCTURE,
        Location::Byte(at),
        CorruptKind::Other("allocation bitmap ended before the last bucket"),
    ))
}

/// Bit `bit` of the volume as a bit index into the piece starting at bit
/// `first_bit` (`bit >= first_bit`, within the piece).
fn to_local(bit: u64, first_bit: u64) -> Result<usize> {
    bit.checked_sub(first_bit)
        .and_then(|b| usize::try_from(b).ok())
        .ok_or_else(|| ended_early(first_bit / 8))
}

/// `ceil(used * 255 / count)`, at most 255; 0 for an empty `count`.
fn scale(used: u64, count: u64) -> u8 {
    let value = used
        .saturating_mul(255)
        .saturating_add(count.saturating_sub(1))
        .checked_div(count)
        .unwrap_or(0);
    u8::try_from(value).unwrap_or(u8::MAX)
}

/// Set bits of `bytes` from bit `from` to bit `to` (exclusive), counting
/// bits most significant first. Bits outside `bytes` count as clear.
fn count_ones(bytes: &[u8], from: usize, to: usize) -> u64 {
    let Some(last_bit) = to.checked_sub(1).filter(|&l| l >= from) else {
        return 0;
    };
    let (first, last) = (from / 8, last_bit / 8);
    let head = HEAD.get(from & 7).copied().unwrap_or(0);
    let tail = TAIL.get(last_bit & 7).copied().unwrap_or(0);
    let byte = |i: usize| bytes.get(i).copied().unwrap_or(0);
    if first == last {
        return u64::from((byte(first) & head & tail).count_ones());
    }
    let edges = (byte(first) & head)
        .count_ones()
        .saturating_add((byte(last) & tail).count_ones());
    let middle = first
        .checked_add(1)
        .and_then(|start| bytes.get(start..last))
        .unwrap_or(&[]);
    u64::from(edges).saturating_add(count_bytes(middle))
}

/// Set bits in `bytes`, eight bytes at a time.
fn count_bytes(bytes: &[u8]) -> u64 {
    let (words, rest) = bytes.as_chunks::<8>();
    let mut total: u64 = 0;
    // Wrapping adds: a slice holds fewer than 2^61 bytes, so the count of
    // its bits cannot wrap, and plain adds keep the loop vectorizable.
    for word in words {
        total = total.wrapping_add(u64::from(u64::from_ne_bytes(*word).count_ones()));
    }
    for b in rest {
        total = total.wrapping_add(u64::from(b.count_ones()));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_fs_core::io::block::MemSource;

    /// The densities, one bit at a time.
    fn naive(bitmap: &[u8], total: u32, buckets: usize) -> Vec<u8> {
        let t = u64::from(total);
        let n = buckets.clamp(1, MAX_ALLOCATION_BUCKETS).min(total as usize) as u64;
        (0..n)
            .map(|i| {
                let (lo, hi) = (i * t / n, (i + 1) * t / n);
                let used = (lo..hi)
                    .filter(|&b| bitmap[(b / 8) as usize] & (0x80 >> (b % 8)) != 0)
                    .count() as u64;
                ((used * 255).div_ceil(hi - lo)) as u8
            })
            .collect()
    }

    /// A deterministic, uneven bit pattern.
    fn pattern(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                // Runs of full and empty bytes too, not only noise.
                match (i / 97) % 4 {
                    0 => 0xFF,
                    1 => 0x00,
                    _ => x as u8,
                }
            })
            .collect()
    }

    fn src(bytes: &[u8]) -> MemSource {
        MemSource::new(bytes.to_vec())
    }

    #[test]
    fn count_ones_masks_partial_bytes() {
        let b = [0b1010_1010, 0xFF, 0x0F, 0b1000_0001];
        assert_eq!(count_ones(&b, 0, 0), 0);
        assert_eq!(count_ones(&b, 3, 3), 0);
        assert_eq!(count_ones(&b, 5, 3), 0);
        assert_eq!(count_ones(&b, 0, 1), 1);
        assert_eq!(count_ones(&b, 1, 2), 0);
        assert_eq!(count_ones(&b, 0, 8), 4);
        assert_eq!(count_ones(&b, 1, 7), 3);
        assert_eq!(count_ones(&b, 7, 9), 1);
        assert_eq!(count_ones(&b, 0, 32), 4 + 8 + 4 + 2);
        assert_eq!(count_ones(&b, 20, 25), 4 + 1);
        assert_eq!(count_ones(&b, 31, 32), 1);
        // Past the slice counts as clear.
        assert_eq!(count_ones(&b, 30, 1000), 1);
        // Every range of a longer pattern, against a per-bit count.
        let p = pattern(40, 9);
        let bit = |i: usize| u64::from(p[i / 8] & (0x80 >> (i % 8)) != 0);
        for from in 0..p.len() * 8 {
            for to in from..=p.len() * 8 {
                let want: u64 = (from..to).map(bit).sum();
                assert_eq!(count_ones(&p, from, to), want, "{from}..{to}");
            }
        }
    }

    #[test]
    fn scale_rounds_up_and_saturates_only_when_full() {
        assert_eq!(scale(0, 10), 0);
        assert_eq!(scale(1, 10), 26);
        assert_eq!(scale(1, 1_000_000), 1);
        assert_eq!(scale(999_999, 1_000_000), 255);
        assert_eq!(scale(4_294_967_294, 4_294_967_295), 255);
        assert_eq!(scale(10, 10), 255);
        assert_eq!(scale(0, 0), 0);
    }

    #[test]
    fn matches_a_per_bit_count_at_any_piece_size() {
        let bitmap = pattern(300, 7);
        for total in [1, 7, 8, 9, 63, 64, 65, 1000, 2399, 2400] {
            let bytes = &bitmap[..(total as usize).div_ceil(8)];
            for buckets in [0, 1, 2, 3, 5, 8, 13, 64, 100, 999, 5000] {
                let want = naive(bytes, total, buckets);
                for piece in [1, 2, 3, 7, 8, 64, PIECE] {
                    let got = density_in_pieces(&src(bytes), total, buckets, piece).unwrap();
                    assert_eq!(got, want, "total {total} buckets {buckets} piece {piece}");
                }
            }
        }
    }

    #[test]
    fn bucket_counts_are_clamped() {
        let bitmap = pattern(1250, 3);
        let d = |total: u32, buckets: usize| density(&src(&bitmap), total, buckets).unwrap().len();
        assert_eq!(d(10_000, 0), 1);
        assert_eq!(d(10_000, 1), 1);
        assert_eq!(d(10_000, 4096), 4096);
        assert_eq!(d(10_000, 4097), 4096);
        assert_eq!(d(10_000, usize::MAX), 4096);
        assert_eq!(d(10, 11), 10);
        assert_eq!(d(10, 10), 10);
        assert_eq!(d(1, 4096), 1);
    }

    #[test]
    fn full_empty_and_nearly_empty_buckets() {
        // 64 blocks in 4 buckets of 16: full, empty, one block, all but one.
        let bitmap = [0xFF, 0xFF, 0, 0, 0, 0x01, 0xFF, 0xFE];
        assert_eq!(density(&src(&bitmap), 64, 4).unwrap(), [255, 0, 16, 240]);
        // One block used in a bucket of 4096 still shows.
        let mut big = vec![0u8; 512];
        big[300] = 0x10;
        assert_eq!(density(&src(&big), 4096, 1).unwrap(), [1]);
    }

    #[test]
    fn bits_past_the_last_block_are_ignored() {
        // 12 blocks: the last 4 bits of byte 1 are not blocks.
        let bitmap = [0xFF, 0xFF, 0xFF];
        assert_eq!(density(&src(&bitmap), 12, 12).unwrap(), [255; 12]);
        assert_eq!(density(&src(&[0x00, 0x0F]), 12, 3).unwrap(), [0, 0, 0]);
        assert_eq!(density(&src(&[0x00, 0x1F]), 12, 3).unwrap(), [0, 0, 64]);
    }

    #[test]
    fn buckets_straddle_one_mib_pieces() {
        // 1.5 MiB and a bit of bitmap, in buckets that are not whole bytes
        // and cross the piece boundary at bit 8 Mi.
        let len = PIECE + PIECE / 2 + 3;
        let bitmap = pattern(len, 11);
        let total = (len * 8 - 5) as u32;
        for buckets in [7, 4096] {
            let got = density(&src(&bitmap), total, buckets).unwrap();
            assert_eq!(got, naive(&bitmap, total, buckets), "buckets {buckets}");
        }
        // One bucket spanning the boundary, its bits only on either side.
        let mut sparse = vec![0u8; len];
        sparse[PIECE - 1] = 0x01;
        sparse[PIECE] = 0x80;
        let got = density(&src(&sparse), total, 4096).unwrap();
        assert_eq!(got, naive(&sparse, total, 4096));
        assert_eq!(got.iter().filter(|&&d| d > 0).count(), 1);
    }

    #[test]
    fn damaged_bitmaps_fail() {
        let short = density(&src(&[0xFF; 3]), 25, 4).unwrap_err();
        match short {
            Error::Corrupt(c) => {
                assert_eq!(c.structure, STRUCTURE);
                assert_eq!(
                    c.kind,
                    CorruptKind::Truncated {
                        needed: 4,
                        available: 3
                    }
                );
            }
            e => panic!("expected Corrupt, got {e:?}"),
        }
        assert!(matches!(
            density(&src(&[0xFF; 3]), 0, 4),
            Err(Error::Corrupt(c)) if c.structure == STRUCTURE
        ));
        assert!(matches!(density(&src(&[]), 1, 1), Err(Error::Corrupt(_))));
        // Exactly long enough, and longer, are fine.
        assert_eq!(density(&src(&[0xFF; 4]), 25, 1).unwrap(), [255]);
        assert_eq!(density(&src(&[0xFF; 9]), 25, 1).unwrap(), [255]);
    }

    /// Bitmap bytes per second through `density` from memory. Run with
    /// `cargo test -p disk-atlas-lib --release allocation_density_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing; run in release with --ignored --nocapture"]
    // Printing the measurement is this test's purpose.
    #[allow(clippy::print_stdout)]
    fn allocation_density_throughput() {
        const LEN: usize = 64 << 20;
        let bitmap = src(&pattern(LEN, 5));
        let total = u32::try_from(LEN * 8).unwrap();
        // Warm up, then take the best of a few runs.
        let mut sink = density(&bitmap, total, MAX_ALLOCATION_BUCKETS).unwrap();
        let mut best = std::time::Duration::MAX;
        for _ in 0..5 {
            let start = std::time::Instant::now();
            sink = density(&bitmap, total, MAX_ALLOCATION_BUCKETS).unwrap();
            best = best.min(start.elapsed());
        }
        assert_eq!(sink.len(), MAX_ALLOCATION_BUCKETS);
        let gbps = LEN as f64 / best.as_secs_f64() / 1e9;
        println!(
            "allocation density: 64 MiB bitmap in {best:?}, {gbps:.2} GB/s (with the read copy)"
        );
    }
}
