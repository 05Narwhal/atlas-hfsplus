// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Checked field reads for the HFS+ modules.
//!
//! Every helper returns `None` when the field does not fit in the buffer, so
//! a caller turns a short buffer into a `Corruption` with `ok_or`.

/// `len` bytes at `at`, or `None` when they do not fit.
pub fn bytes(buf: &[u8], at: usize, len: usize) -> Option<&[u8]> {
    buf.get(at..at.checked_add(len)?)
}

/// `N` bytes at `at` as an array, or `None` when they do not fit.
pub fn array<const N: usize>(buf: &[u8], at: usize) -> Option<[u8; N]> {
    bytes(buf, at, N)?.try_into().ok()
}

/// Big-endian `u8` at `at`.
pub fn u8_at(buf: &[u8], at: usize) -> Option<u8> {
    buf.get(at).copied()
}

/// Big-endian `u16` at `at`.
pub fn be_u16(buf: &[u8], at: usize) -> Option<u16> {
    array(buf, at).map(u16::from_be_bytes)
}

/// Big-endian `i16` at `at`.
pub fn be_i16(buf: &[u8], at: usize) -> Option<i16> {
    array(buf, at).map(i16::from_be_bytes)
}

/// Big-endian `u32` at `at`.
pub fn be_u32(buf: &[u8], at: usize) -> Option<u32> {
    array(buf, at).map(u32::from_be_bytes)
}

/// Big-endian `u64` at `at`.
pub fn be_u64(buf: &[u8], at: usize) -> Option<u64> {
    array(buf, at).map(u64::from_be_bytes)
}

/// Little-endian `u32` at `at` (decmpfs headers, chunk tables).
pub fn le_u32(buf: &[u8], at: usize) -> Option<u32> {
    array(buf, at).map(u32::from_le_bytes)
}

/// Little-endian `u64` at `at`.
pub fn le_u64(buf: &[u8], at: usize) -> Option<u64> {
    array(buf, at).map(u64::from_le_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_bounds() {
        let b = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        assert_eq!(u8_at(&b, 7), Some(0xF0));
        assert_eq!(u8_at(&b, 8), None);
        assert_eq!(be_u16(&b, 0), Some(0x1234));
        assert_eq!(be_i16(&b, 6), Some(i16::from_be_bytes([0xDE, 0xF0])));
        assert_eq!(be_u32(&b, 4), Some(0x9ABC_DEF0));
        assert_eq!(be_u32(&b, 5), None);
        assert_eq!(be_u64(&b, 0), Some(0x1234_5678_9ABC_DEF0));
        assert_eq!(le_u32(&b, 0), Some(0x7856_3412));
        assert_eq!(le_u64(&b, 1), None);
        assert_eq!(bytes(&b, usize::MAX, 2), None);
        assert_eq!(bytes(&b, 8, 0), Some(&[][..]));
    }
}
