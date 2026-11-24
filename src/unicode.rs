// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Name comparison tables and Unicode decomposition.
//!
//! HFS+ stores catalog names as big-endian UTF-16 in Apple's variant of
//! canonical decomposition (TN1150 "Canonical Decomposition"): Unicode 3.2
//! canonical decompositions applied fully, Hangul syllables split into
//! conjoining jamo, and U+2000-U+2FFF, U+F900-U+FAFF and everything outside
//! the BMP left alone (the macOS converter decomposes 16-bit code units only,
//! so surrogate pairs pass through). Catalog keys are ordered by
//! [`NameCompare`]: TN1150 `FastUnicodeCompare` on HFS+ and case-insensitive
//! HFSX, unsigned code unit order on case-sensitive HFSX.
//!
//! [`decompose`] turns a host name into the stored form for lookup.
//! [`compose`] and [`compose_stored`] turn a stored name into its display
//! form: NFC, except that nothing in the excluded ranges is changed or
//! produced. That makes the display form decompose back to the stored form,
//! so a name the user sees can always be looked up again.
//!
//! The tables come from `tools/hfs_unicode` (see its README).

use std::cmp::Ordering;

#[rustfmt::skip]
mod tables;

use tables::{COMBINING_CLASS, COMPOSE, DECOMP_DATA, DECOMP_INDEX, DECOMP_KEYS, LOWER_CASE};

/// How a volume orders catalog names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameCompare {
    /// HFS+, and HFSX with keyCompareType 0xCF: TN1150 `FastUnicodeCompare`.
    CaseFolding,
    /// HFSX with keyCompareType 0xBC: unsigned UTF-16 code unit order.
    Binary,
}

impl NameCompare {
    /// From an HFSX catalog header's keyCompareType: 0xCF -> CaseFolding,
    /// 0xBC -> Binary, anything else None.
    ///
    /// Plain HFS+ always uses CaseFolding; the catalog decides that, not this
    /// function.
    pub fn from_key_compare_type(b: u8) -> Option<Self> {
        match b {
            0xCF => Some(Self::CaseFolding),
            0xBC => Some(Self::Binary),
            _ => None,
        }
    }

    /// Orders two names given as UTF-16 code units.
    ///
    /// No allocation and no UTF-16 validation: unpaired surrogates compare as
    /// the code units they are, as they do on disk.
    pub fn compare(self, a: &[u16], b: &[u16]) -> Ordering {
        match self {
            Self::CaseFolding => fast_unicode_compare(a.iter().copied(), b.iter().copied()),
            Self::Binary => a.cmp(b),
        }
    }

    /// Orders a stored name (big-endian UTF-16 bytes, read in place from a
    /// catalog key) against a target in code units.
    ///
    /// A trailing odd byte is ignored (the catalog validates lengths before
    /// calling).
    pub fn compare_stored(self, stored_be: &[u8], target: &[u16]) -> Ordering {
        let stored = be_units(stored_be);
        match self {
            Self::CaseFolding => fast_unicode_compare(stored, target.iter().copied()),
            Self::Binary => stored.cmp(target.iter().copied()),
        }
    }
}

/// Appends `name` to `out` in the decomposed UTF-16 form HFS+ stores. Does
/// not clear `out`.
///
/// Unicode 3.2 canonical decomposition, applied fully, then canonical
/// reordering by combining class. Hangul syllables decompose
/// algorithmically. U+2000-U+2FFF and U+F900-U+FAFF (TN1150) and all
/// characters outside the BMP are not decomposed; the latter become
/// surrogate pairs as they are.
pub fn decompose(name: &str, out: &mut Vec<u16>) {
    let mut cps = Vec::with_capacity(name.len());
    for ch in name.chars() {
        push_decomposed(u32::from(ch), &mut cps);
    }
    canonical_order(&mut cps);
    for cp in cps {
        push_utf16(cp, out);
    }
}

/// Appends the display form of a stored name (big-endian UTF-16 bytes) to
/// `out`: NFC that leaves the excluded ranges alone (see the module docs). Unpaired surrogates become U+FFFD; a trailing odd byte is
/// ignored. Does not clear `out`.
pub fn compose_stored(stored_be: &[u8], out: &mut String) {
    compose_units(be_units(stored_be), out);
}

/// Same as [`compose_stored`], from code units.
pub fn compose(units: &[u16], out: &mut String) {
    compose_units(units.iter().copied(), out);
}

/// Big-endian code units of `bytes`; a trailing odd byte is dropped.
fn be_units(bytes: &[u8]) -> impl Iterator<Item = u16> + '_ {
    let (pairs, _odd) = bytes.as_chunks::<2>();
    pairs.iter().map(|&pair| u16::from_be_bytes(pair))
}

/// TN1150 case folding of one code unit; 0 means ignorable.
#[inline]
fn fold(c: u16) -> u16 {
    let [high, low] = c.to_be_bytes();
    match LOWER_CASE.get(usize::from(high)) {
        Some(&0) | None => c,
        // Page indices are multiples of 256 that leave room for a whole page
        // (checked by the generator and by a test), so `|` adds the low byte.
        Some(&page) => LOWER_CASE
            .get(usize::from(page) | usize::from(low))
            .copied()
            .unwrap_or(c),
    }
}

/// The next non-ignorable folded code unit, or 0 at the end of the string.
#[inline]
fn next_folded(units: &mut impl Iterator<Item = u16>) -> u16 {
    units.map(fold).find(|&c| c != 0).unwrap_or(0)
}

/// TN1150 `FastUnicodeCompare`. U+0000 folds to 0xFFFF in the table, so 0 is
/// free to mean "ignorable" and "end of string". A string that ends first
/// yields 0 and so orders before any remaining character.
fn fast_unicode_compare(
    mut a: impl Iterator<Item = u16>,
    mut b: impl Iterator<Item = u16>,
) -> Ordering {
    // Each turn consumes at least one unit from a non-empty side, or returns.
    loop {
        let c1 = next_folded(&mut a);
        let c2 = next_folded(&mut b);
        if c1 != c2 {
            return c1.cmp(&c2);
        }
        if c1 == 0 {
            return Ordering::Equal;
        }
    }
}

/// Apple's decomposition exclusions: TN1150's two ranges, plus everything
/// outside the BMP.
fn hfs_excluded(cp: u32) -> bool {
    matches!(cp, 0x2000..=0x2FFF | 0xF900..=0xFAFF | 0x10000..)
}

/// Appends Apple's full canonical decomposition of `cp` (not yet reordered).
fn push_decomposed(cp: u32, out: &mut Vec<u32>) {
    if push_hangul_jamo(cp, out) {
        return;
    }
    if hfs_excluded(cp) {
        out.push(cp);
        return;
    }
    match decomposition(cp) {
        Some(units) => out.extend(
            char::decode_utf16(units.iter().copied())
                .map(|r| u32::from(r.unwrap_or(char::REPLACEMENT_CHARACTER))),
        ),
        None => out.push(cp),
    }
}

/// The table decomposition of `cp` in UTF-16, if it has one. The generator
/// stores full decompositions, so no recursion is needed.
fn decomposition(cp: u32) -> Option<&'static [u16]> {
    let i = DECOMP_KEYS.binary_search(&cp).ok()?;
    let start = usize::from(*DECOMP_INDEX.get(i)?);
    let end = usize::from(*DECOMP_INDEX.get(i.checked_add(1)?)?);
    DECOMP_DATA.get(start..end)
}

/// Canonical combining class of `cp`.
fn combining_class(cp: u32) -> u8 {
    // Nothing below U+0300 has a non-zero class (checked by a test).
    if cp < 0x0300 {
        return 0;
    }
    COMBINING_CLASS
        .binary_search_by(|&(first, last, _)| {
            if last < cp {
                Ordering::Less
            } else if first > cp {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .ok()
        .and_then(|i| COMBINING_CLASS.get(i))
        .map_or(0, |&(_, _, class)| class)
}

/// Stable-sorts every run of non-starters by combining class.
fn canonical_order(cps: &mut [u32]) {
    for run in cps.split_mut(|&cp| combining_class(cp) == 0) {
        if run.len() > 1 {
            run.sort_by_key(|&cp| combining_class(cp));
        }
    }
}

fn push_utf16(cp: u32, out: &mut Vec<u16>) {
    let ch = char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER);
    let mut buf = [0u16; 2];
    out.extend_from_slice(ch.encode_utf16(&mut buf));
}

// Hangul syllable algorithm (Unicode 3.2, section 3.12).
const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

/// Appends the jamo of a Hangul syllable; false if `cp` is not one.
fn push_hangul_jamo(cp: u32, out: &mut Vec<u32>) -> bool {
    let Some(s) = cp.checked_sub(S_BASE).filter(|&s| s < S_COUNT) else {
        return false;
    };
    let l = L_BASE.checked_add(s / N_COUNT);
    let v = V_BASE.checked_add(s % N_COUNT / T_COUNT);
    let t = s % T_COUNT;
    let (Some(l), Some(v)) = (l, v) else {
        return false;
    };
    out.push(l);
    out.push(v);
    if t != 0
        && let Some(t) = T_BASE.checked_add(t)
    {
        out.push(t);
    }
    true
}

/// The primary composite of `first` followed by `second`, if any.
fn compose_pair(first: u32, second: u32) -> Option<u32> {
    // L + V -> LV
    if let (Some(l), Some(v)) = (first.checked_sub(L_BASE), second.checked_sub(V_BASE))
        && l < L_COUNT
        && v < V_COUNT
    {
        return l
            .checked_mul(V_COUNT)?
            .checked_add(v)?
            .checked_mul(T_COUNT)?
            .checked_add(S_BASE);
    }
    // LV + T -> LVT
    if let (Some(s), Some(t)) = (first.checked_sub(S_BASE), second.checked_sub(T_BASE))
        && s < S_COUNT
        && s % T_COUNT == 0
        && t > 0
        && t < T_COUNT
    {
        return first.checked_add(t);
    }
    let key = (u16::try_from(first).ok()?, u16::try_from(second).ok()?);
    let i = COMPOSE
        .binary_search_by(|&(a, b, _)| (a, b).cmp(&key))
        .ok()?;
    COMPOSE
        .get(i)
        .map(|&(_, _, composite)| u32::from(composite))
}

/// Display form of a UTF-16 unit sequence, appended to `out`: Unicode 3.2
/// NFC built on Apple's decomposition, never composing into an excluded
/// code point, so decomposing the result gives the stored form back.
fn compose_units(units: impl Iterator<Item = u16>, out: &mut String) {
    let mut cps = Vec::new();
    for r in char::decode_utf16(units) {
        let ch = r.unwrap_or(char::REPLACEMENT_CHARACTER);
        push_decomposed(u32::from(ch), &mut cps);
    }
    canonical_order(&mut cps);

    // Canonical composition (UAX #15): each character either joins the last
    // starter, if it is not blocked from it, or is kept.
    let mut composed: Vec<u32> = Vec::with_capacity(cps.len());
    let mut starter: Option<usize> = None;
    // Class of the last character kept since the starter, or 0 when the
    // starter is the last character kept; 256 marks "no usable starter".
    let mut last_class: u16 = 256;
    for cp in cps {
        let class = u16::from(combining_class(cp));
        if let Some(pos) = starter
            && (last_class == 0 || last_class < class)
            && let Some(slot) = composed.get_mut(pos)
            && let Some(c) = compose_pair(*slot, cp)
            && !hfs_excluded(c)
        {
            *slot = c;
            continue;
        }
        if class == 0 {
            starter = Some(composed.len());
        }
        last_class = class;
        composed.push(cp);
    }
    out.extend(composed.into_iter().filter_map(char::from_u32));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf16(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    fn be(units: &[u16]) -> Vec<u8> {
        units.iter().flat_map(|u| u.to_be_bytes()).collect()
    }

    fn dec(s: &str) -> Vec<u16> {
        let mut out = Vec::new();
        decompose(s, &mut out);
        out
    }

    fn comp(units: &[u16]) -> String {
        let mut out = String::new();
        compose(units, &mut out);
        out
    }

    fn cf(a: &str, b: &str) -> Ordering {
        NameCompare::CaseFolding.compare(&utf16(a), &utf16(b))
    }

    // --- tables -----------------------------------------------------------

    #[test]
    fn lower_case_table_shape() {
        assert_eq!(LOWER_CASE.len(), 11 * 256);
        for &page in &LOWER_CASE[..256] {
            assert!(page == 0 || (page % 256 == 0 && usize::from(page) + 256 <= LOWER_CASE.len()));
        }
        // Spot checks against TN1150's FastUnicodeCompare.c.
        assert_eq!(fold(0x0000), 0xFFFF);
        assert_eq!(fold(0x0041), 0x0061);
        assert_eq!(fold(0x00C6), 0x00E6);
        assert_eq!(fold(0x0110), 0x0111);
        assert_eq!(fold(0x2160), 0x2170);
        assert_eq!(fold(0xFF21), 0xFF41);
        assert_eq!(fold(0x200C), 0);
        assert_eq!(fold(0xFEFF), 0);
        assert_eq!(fold(0x00E9), 0x00E9);
        assert_eq!(fold(0x4E00), 0x4E00);
    }

    #[test]
    fn decomposition_tables_shape() {
        assert!(DECOMP_KEYS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(DECOMP_INDEX.len(), DECOMP_KEYS.len() + 1);
        assert_eq!(DECOMP_INDEX[0], 0);
        assert!(DECOMP_INDEX.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(
            usize::from(*DECOMP_INDEX.last().unwrap()),
            DECOMP_DATA.len()
        );
        assert!(COMBINING_CLASS.windows(2).all(|w| w[0].1 < w[1].0));
        assert!(COMBINING_CLASS.iter().all(|&(a, b, c)| a <= b && c != 0));
        assert!(COMBINING_CLASS[0].0 >= 0x0300);
        assert!(
            COMPOSE
                .windows(2)
                .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1))
        );
        // Every decomposition decodes as valid UTF-16.
        for (i, &key) in DECOMP_KEYS.iter().enumerate() {
            let data = &DECOMP_DATA[usize::from(DECOMP_INDEX[i])..usize::from(DECOMP_INDEX[i + 1])];
            assert!(
                char::decode_utf16(data.iter().copied()).all(|r| r.is_ok()),
                "{key:04X}"
            );
        }
    }

    // The tables are checked against their inputs by
    // `cargo run --manifest-path tools/hfs_unicode/Cargo.toml -- --check ...`
    // (see tools/hfs_unicode/README.md); the inputs are not in the repository.

    // --- CaseFolding ------------------------------------------------------

    #[test]
    fn case_folding_ignores_case() {
        assert_eq!(cf("ABC", "abc"), Ordering::Equal);
        assert_eq!(cf("a", "B"), Ordering::Less);
        assert_eq!(cf("B", "a"), Ordering::Greater);
        assert_eq!(cf("\u{C6}", "\u{E6}"), Ordering::Equal);
    }

    #[test]
    fn case_folding_skips_ignorables() {
        assert_eq!(cf("a\u{200C}b", "ab"), Ordering::Equal);
        assert_eq!(cf("\u{FEFF}ab\u{FEFF}", "AB"), Ordering::Equal);
        assert_eq!(cf("\u{200C}", ""), Ordering::Equal);
        assert_eq!(cf("a\u{200D}", "a\u{206F}c"), Ordering::Less);
    }

    #[test]
    fn case_folding_nul_sorts_last() {
        assert_eq!(cf("\u{0}", "z"), Ordering::Greater);
        assert_eq!(cf("\u{0}", "\u{FFFD}"), Ordering::Greater);
        assert_eq!(cf("a\u{0}", "a"), Ordering::Greater);
        assert_eq!(cf("\u{0}", "\u{0}"), Ordering::Equal);
    }

    #[test]
    fn case_folding_prefix_orders_first() {
        assert_eq!(cf("abc", "ABCD"), Ordering::Less);
        assert_eq!(cf("abcd", "ABC"), Ordering::Greater);
        assert_eq!(cf("", "a"), Ordering::Less);
        assert_eq!(cf("", ""), Ordering::Equal);
    }

    #[test]
    fn case_folding_does_not_normalize() {
        // Stored names are decomposed; a composed name is a different key.
        assert_ne!(cf("\u{E9}", "e\u{301}"), Ordering::Equal);
    }

    #[test]
    fn compare_stored_agrees_with_compare() {
        let names = [
            "",
            "a",
            "A",
            "ab",
            "aB\u{200C}",
            "B",
            "abc",
            "\u{0}",
            "e\u{301}",
            "\u{E9}",
            "Z\u{FEFF}",
            "\u{1F600}",
            "\u{FF21}",
            "\u{2160}x",
        ];
        for mode in [NameCompare::CaseFolding, NameCompare::Binary] {
            for a in names {
                for b in names {
                    let (a16, b16) = (utf16(a), utf16(b));
                    let expected = mode.compare(&a16, &b16);
                    assert_eq!(
                        mode.compare_stored(&be(&a16), &b16),
                        expected,
                        "{mode:?} {a:?} {b:?}"
                    );
                    assert_eq!(
                        mode.compare(&b16, &a16),
                        expected.reverse(),
                        "{mode:?} {a:?} {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn compare_stored_ignores_odd_byte() {
        for mode in [NameCompare::CaseFolding, NameCompare::Binary] {
            assert_eq!(
                mode.compare_stored(&[0x00, 0x61, 0x00], &[0x61]),
                Ordering::Equal
            );
            assert_eq!(mode.compare_stored(&[0x41], &[]), Ordering::Equal);
            assert_eq!(mode.compare_stored(&[], &[]), Ordering::Equal);
        }
    }

    #[test]
    fn compare_handles_lone_surrogates() {
        for mode in [NameCompare::CaseFolding, NameCompare::Binary] {
            assert_eq!(mode.compare(&[0xD800], &[0xDC00]), Ordering::Less);
            assert_eq!(
                mode.compare(&[0xDC00, 0x61], &[0xDC00, 0x41]).is_eq(),
                mode == NameCompare::CaseFolding
            );
        }
    }

    // --- Binary -----------------------------------------------------------

    #[test]
    fn binary_is_code_unit_order() {
        let b = NameCompare::Binary;
        assert_ne!(b.compare(&utf16("a"), &utf16("A")), Ordering::Equal);
        assert_eq!(b.compare(&utf16("B"), &utf16("a")), Ordering::Less);
        assert_eq!(b.compare(&utf16("ab"), &utf16("abc")), Ordering::Less);
        // Ignorables are significant on case-sensitive HFSX.
        assert_ne!(
            b.compare(&utf16("a\u{200C}b"), &utf16("ab")),
            Ordering::Equal
        );
        // Code unit order, not code point order: a surrogate sorts below U+FFFD.
        assert_eq!(
            b.compare(&utf16("\u{1F600}"), &utf16("\u{FFFD}")),
            Ordering::Less
        );
    }

    #[test]
    fn key_compare_type() {
        assert_eq!(
            NameCompare::from_key_compare_type(0xCF),
            Some(NameCompare::CaseFolding)
        );
        assert_eq!(
            NameCompare::from_key_compare_type(0xBC),
            Some(NameCompare::Binary)
        );
        assert_eq!(NameCompare::from_key_compare_type(0x00), None);
        assert_eq!(NameCompare::from_key_compare_type(0xFF), None);
    }

    // --- decompose --------------------------------------------------------

    #[test]
    fn decompose_basic() {
        assert_eq!(dec("\u{E9}"), [0x0065, 0x0301]);
        assert_eq!(dec("abc"), utf16("abc"));
        assert_eq!(dec(""), [] as [u16; 0]);
        // Recursive: U+01D5 -> U+00DC U+0304 -> U U+0308 U+0304.
        assert_eq!(dec("\u{1D5}"), [0x0055, 0x0308, 0x0304]);
        // Singleton outside the excluded ranges.
        assert_eq!(dec("\u{37E}"), [0x003B]);
    }

    #[test]
    fn decompose_appends() {
        let mut out = vec![0x0041];
        decompose("\u{E9}", &mut out);
        assert_eq!(out, [0x0041, 0x0065, 0x0301]);
    }

    #[test]
    fn decompose_reorders_marks() {
        // U+1E0B (d, dot above 230) then U+0323 (dot below 220).
        assert_eq!(dec("\u{1E0B}\u{323}"), [0x0064, 0x0323, 0x0307]);
        // U+1E69 decomposes to s + dot below + dot above, already ordered.
        assert_eq!(dec("\u{1E69}"), [0x0073, 0x0323, 0x0307]);
        // Equal classes keep their order.
        assert_eq!(dec("a\u{301}\u{300}"), [0x0061, 0x0301, 0x0300]);
        // A starter ends a run.
        assert_eq!(dec("a\u{301}b\u{323}"), [0x0061, 0x0301, 0x0062, 0x0323]);
    }

    #[test]
    fn decompose_hangul() {
        assert_eq!(dec("\u{AC00}"), [0x1100, 0x1161]);
        assert_eq!(dec("\u{AC01}"), [0x1100, 0x1161, 0x11A8]);
        assert_eq!(dec("\u{D7A3}"), [0x1112, 0x1175, 0x11C2]);
    }

    #[test]
    fn decompose_leaves_excluded_ranges() {
        assert_eq!(dec("\u{2126}"), [0x2126]); // OHM SIGN
        assert_eq!(dec("\u{212B}"), [0x212B]); // ANGSTROM SIGN
        assert_eq!(dec("\u{2000}"), [0x2000]);
        assert_eq!(dec("\u{F900}"), [0xF900]);
        assert_eq!(dec("\u{FAFF}"), [0xFAFF]);
        assert_eq!(dec("\u{2F800}"), [0xD87E, 0xDC00]);
    }

    #[test]
    fn decompose_non_bmp() {
        assert_eq!(dec("\u{1F600}"), [0xD83D, 0xDE00]);
        // U+1D15E MUSICAL SYMBOL HALF NOTE has a canonical decomposition, but
        // macOS only decomposes BMP code units, so it is stored as is.
        assert_eq!(dec("\u{1D15E}"), [0xD834, 0xDD5E]);
    }

    // --- compose ----------------------------------------------------------

    #[test]
    fn compose_basic() {
        assert_eq!(comp(&[0x0065, 0x0301]), "\u{E9}");
        assert_eq!(comp(&utf16("abc")), "abc");
        assert_eq!(comp(&[]), "");
        assert_eq!(comp(&[0x0055, 0x0308, 0x0304]), "\u{1D5}");
        // Composition exclusion: U+0958 stays decomposed.
        assert_eq!(comp(&[0x0915, 0x093C]), "\u{915}\u{93C}");
        // A blocked mark stays separate: dot below composes, dot above then
        // joins the result.
        assert_eq!(comp(&[0x0064, 0x0323, 0x0307]), "\u{1E0D}\u{307}");
        // A leading combining mark has no starter.
        assert_eq!(comp(&[0x0301, 0x0065]), "\u{301}e");
    }

    #[test]
    fn compose_appends() {
        let mut out = String::from("x");
        compose(&[0x0065, 0x0301], &mut out);
        assert_eq!(out, "x\u{E9}");
    }

    #[test]
    fn compose_hangul() {
        assert_eq!(comp(&[0x1100, 0x1161]), "\u{AC00}");
        assert_eq!(comp(&[0x1100, 0x1161, 0x11A8]), "\u{AC01}");
        assert_eq!(comp(&[0x1112, 0x1175, 0x11C2]), "\u{D7A3}");
        // A lone trailing consonant does not compose with a plain letter.
        assert_eq!(comp(&[0x0061, 0x11A8]), "a\u{11A8}");
    }

    #[test]
    fn compose_replaces_lone_surrogates() {
        assert_eq!(comp(&[0xD800]), "\u{FFFD}");
        assert_eq!(comp(&[0xDC00]), "\u{FFFD}");
        assert_eq!(comp(&[0x0061, 0xD800, 0x0062]), "a\u{FFFD}b");
        assert_eq!(comp(&[0xDC00, 0xD800]), "\u{FFFD}\u{FFFD}");
        assert_eq!(comp(&[0xD83D, 0xDE00]), "\u{1F600}");
    }

    #[test]
    fn compose_stored_ignores_odd_byte() {
        let mut out = String::new();
        compose_stored(&[0x00, 0x65, 0x03, 0x01, 0xFF], &mut out);
        assert_eq!(out, "\u{E9}");
        out.clear();
        compose_stored(&[0x41], &mut out);
        assert_eq!(out, "");
    }

    #[test]
    fn compose_of_decompose_is_display_form() {
        let cases = [
            ("Caf\u{E9}", "Caf\u{E9}"),
            ("Cafe\u{301}", "Caf\u{E9}"),
            ("\u{D55C}\u{AD6D}\u{C5B4}", "\u{D55C}\u{AD6D}\u{C5B4}"),
            ("\u{1E0B}\u{323}", "\u{1E0D}\u{307}"),
            // Excluded ranges and non-BMP characters are shown as stored.
            ("\u{212B}ngstr\u{F6}m", "\u{212B}ngstr\u{F6}m"),
            ("\u{2126} \u{F900}", "\u{2126} \u{F900}"),
            ("\u{1F600} \u{1D15E}", "\u{1F600} \u{1D15E}"),
            // UAX #15's example: U+1E9B U+0323 is already NFC.
            ("A\u{30A}\u{1E9B}\u{323}", "\u{C5}\u{1E9B}\u{323}"),
            ("\u{0}/:", "\u{0}/:"),
        ];
        for (s, nfc) in cases {
            assert_eq!(comp(&dec(s)), nfc, "{s:?}");
        }
    }

    #[test]
    fn compose_never_produces_excluded_composites() {
        // U+2260 NOT EQUAL TO is canonically U+003D U+0338 but lies in an
        // excluded range: a stored "=" + U+0338 must not be shown as U+2260,
        // which would not decompose back to what is stored.
        assert_eq!(comp(&[0x003D, 0x0338]), "=\u{338}");
    }

    #[test]
    fn display_form_decomposes_to_stored_form() {
        // For every stored form the volume can hold (decompose's output),
        // decompose(compose(stored)) == stored.
        let mut stored = Vec::new();
        let mut shown = String::new();
        let mut again = Vec::new();
        for ch in (0..=0x10FFFF).filter_map(char::from_u32) {
            for s in [ch.to_string(), format!("a{ch}\u{301}\u{323}")] {
                stored.clear();
                decompose(&s, &mut stored);
                shown.clear();
                compose(&stored, &mut shown);
                again.clear();
                decompose(&shown, &mut again);
                assert_eq!(again, stored, "{:?}", ch as u32);
            }
        }
    }

    #[test]
    fn never_panics_on_any_unit_pair() {
        // Every BMP code unit, alone and followed by a combining mark.
        let mut out = String::new();
        for u in 0..=u16::MAX {
            out.clear();
            compose(&[u, 0x0301], &mut out);
            let _ = NameCompare::CaseFolding.compare(&[u], &[0x0301, u]);
            let _ = NameCompare::CaseFolding.compare_stored(&u.to_be_bytes(), &[u]);
        }
        let mut units = Vec::new();
        for cp in (0..=0x10FFFF).filter_map(char::from_u32).step_by(7) {
            units.clear();
            decompose(cp.encode_utf8(&mut [0; 4]), &mut units);
        }
    }
}
