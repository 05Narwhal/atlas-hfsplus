// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Hard links, directory hard links and symbolic links.
//!
//! A file hard link is a small file record of type `hlnk`, creator `hfs+`,
//! whose link reference (`bsd.special`) names the real file `iNode<ref>` in
//! a hidden folder of the root. A directory hard link (Time Machine) is a
//! file record of type `fdrp`, creator `MACS`, naming the folder `dir_<ref>`
//! in a second hidden folder of the root (TN1150 "Hard Links"; directory
//! links are not in TN1150 and follow xnu).
//!
//! Symbolic links need nothing here: they are files whose BSD mode says
//! symlink, with the target path in the data fork. `mod.rs` reads it.

use super::catalog::{Catalog, CatalogRecord, FileRecord, ROOT_FOLDER_ID};
use atlas_fs_core::error::Result;

/// `hlnk`: the Finder type of a file hard link.
pub const FILE_LINK_TYPE: u32 = 0x686C_6E6B;
/// `hfs+`: the Finder creator of a file hard link.
pub const FILE_LINK_CREATOR: u32 = 0x6866_732B;
/// `fdrp`: the Finder type of a directory hard link.
pub const DIR_LINK_TYPE: u32 = 0x6664_7270;
/// `MACS`: the Finder creator of a directory hard link.
pub const DIR_LINK_CREATOR: u32 = 0x4D41_4353;

/// The folder that holds file link targets: four NULs, then
/// "HFS+ Private Data". A direct child of the root.
pub const FILE_PRIVATE_FOLDER: [u16; 21] = [
    0, 0, 0, 0, 0x48, 0x46, 0x53, 0x2B, 0x20, 0x50, 0x72, 0x69, 0x76, 0x61, 0x74, 0x65, 0x20, 0x44,
    0x61, 0x74, 0x61,
];

/// The folder that holds directory link targets: ".HFS+ Private Directory
/// Data" and a carriage return. A direct child of the root.
pub const DIR_PRIVATE_FOLDER: [u16; 29] = [
    0x2E, 0x48, 0x46, 0x53, 0x2B, 0x20, 0x50, 0x72, 0x69, 0x76, 0x61, 0x74, 0x65, 0x20, 0x44, 0x69,
    0x72, 0x65, 0x63, 0x74, 0x6F, 0x72, 0x79, 0x20, 0x44, 0x61, 0x74, 0x61, 0x0D,
];

/// Name prefix of a file link target.
pub(super) const FILE_TARGET_PREFIX: &[u8] = b"iNode";
/// Name prefix of a directory link target.
pub(super) const DIR_TARGET_PREFIX: &[u8] = b"dir_";

/// Which kind of hard link a file record is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    /// A file hard link (`hlnk`/`hfs+`).
    File,
    /// A directory hard link (`fdrp`/`MACS`).
    Directory,
}

/// A hard link: its kind and link reference number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Link {
    /// File or directory link.
    pub kind: LinkKind,
    /// The number in the target's name (`bsd.special`).
    pub reference: u32,
}

/// The hard link `file` is, or `None` for an ordinary file.
pub fn link_of(file: &FileRecord) -> Option<Link> {
    let kind = match (file.file_type, file.creator) {
        (FILE_LINK_TYPE, FILE_LINK_CREATOR) => LinkKind::File,
        (DIR_LINK_TYPE, DIR_LINK_CREATOR) => LinkKind::Directory,
        _ => return None,
    };
    Some(Link {
        kind,
        reference: file.bsd.special,
    })
}

/// Longest target name: "iNode" plus the ten digits of `u32::MAX`.
const TARGET_NAME_CAP: usize = 15;

/// A target name in UTF-16 code units, built on the stack.
pub(super) struct TargetName {
    units: [u16; TARGET_NAME_CAP],
    len: usize,
}

impl TargetName {
    /// `prefix` followed by `reference` in decimal. `None` only if the name
    /// would not fit, which the prefixes used here rule out.
    pub(super) fn new(prefix: &[u8], reference: u32) -> Option<Self> {
        let mut digits = [0u8; 10];
        let mut count = 0usize;
        let mut rest = reference;
        // Bounded: a u32 has at most ten decimal digits, and `get_mut`
        // fails on an eleventh.
        loop {
            let digit = u8::try_from(rest.checked_rem(10)?).ok()?;
            *digits.get_mut(count)? = b'0'.checked_add(digit)?;
            count = count.checked_add(1)?;
            rest = rest.checked_div(10)?;
            if rest == 0 {
                break;
            }
        }
        let mut units = [0u16; TARGET_NAME_CAP];
        let mut len = 0usize;
        let ascii = prefix
            .iter()
            .chain(digits.get(..count)?.iter().rev())
            .copied();
        for b in ascii {
            *units.get_mut(len)? = u16::from(b);
            len = len.checked_add(1)?;
        }
        Some(Self { units, len })
    }

    pub(super) fn units(&self) -> &[u16] {
        self.units.get(..self.len).unwrap_or(&[])
    }
}

/// Finds hard link targets. Holds a [`Catalog`] clone and the IDs of the
/// two private folders, so it is cheap to clone and a listing can own one.
#[derive(Clone)]
pub struct LinkResolver {
    catalog: Catalog,
    file_folder: Option<u32>,
    dir_folder: Option<u32>,
}

impl LinkResolver {
    /// Finds the private folders in the root. Either may be absent (most
    /// volumes have neither); a record of that name that is not a folder
    /// counts as absent. Fails only when the catalog cannot be read.
    pub fn new(catalog: Catalog) -> Result<Self> {
        let folder = |name: &[u16]| -> Result<Option<u32>> {
            Ok(match catalog.find(ROOT_FOLDER_ID, name)? {
                Some(CatalogRecord::Folder(f)) => Some(f.folder_id),
                Some(CatalogRecord::File(_)) | None => None,
            })
        };
        let file_folder = folder(&FILE_PRIVATE_FOLDER)?;
        let dir_folder = folder(&DIR_PRIVATE_FOLDER)?;
        Ok(Self {
            catalog,
            file_folder,
            dir_folder,
        })
    }

    /// ID of the folder holding file link targets, if the volume has one.
    pub fn file_folder(&self) -> Option<u32> {
        self.file_folder
    }

    /// ID of the folder holding directory link targets, if the volume has one.
    pub fn dir_folder(&self) -> Option<u32> {
        self.dir_folder
    }

    /// True when `id` is one of the private folders, which listings and
    /// lookups of the root leave out.
    pub fn is_private_folder(&self, id: u32) -> bool {
        Some(id) == self.file_folder || Some(id) == self.dir_folder
    }

    /// True when a listing of `parent` leaves out its child `id`: the
    /// private folders of the root. The one rule the listing, `lookup` and
    /// `scan` all apply; names hidden later (`design/m3-leftovers.md` §1.4)
    /// join here.
    pub fn hidden_in_listing(&self, parent: u32, id: u32) -> bool {
        parent == ROOT_FOLDER_ID && self.is_private_folder(id)
    }

    /// The target of `link`.
    ///
    /// A file link resolves to a file that is not itself a link; a directory
    /// link resolves to a folder. A missing private folder, a missing
    /// target, or a target of the wrong kind gives `Ok(None)`. `Err` means
    /// the catalog could not be read.
    pub fn resolve(&self, link: Link) -> Result<Option<CatalogRecord>> {
        let (folder, prefix) = match link.kind {
            LinkKind::File => (self.file_folder, FILE_TARGET_PREFIX),
            LinkKind::Directory => (self.dir_folder, DIR_TARGET_PREFIX),
        };
        let Some(folder) = folder else {
            return Ok(None);
        };
        let Some(name) = TargetName::new(prefix, link.reference) else {
            return Ok(None);
        };
        Ok(
            match (link.kind, self.catalog.find(folder, name.units())?) {
                (LinkKind::File, Some(CatalogRecord::File(f))) if link_of(&f).is_none() => {
                    Some(CatalogRecord::File(f))
                }
                (LinkKind::Directory, Some(CatalogRecord::Folder(f))) => {
                    Some(CatalogRecord::Folder(f))
                }
                _ => None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(units: &[u16]) -> String {
        String::from_utf16(units).unwrap()
    }

    #[test]
    fn target_names() {
        let name = |prefix, n| text(TargetName::new(prefix, n).unwrap().units());
        assert_eq!(name(FILE_TARGET_PREFIX, 0), "iNode0");
        assert_eq!(name(FILE_TARGET_PREFIX, 7), "iNode7");
        assert_eq!(name(FILE_TARGET_PREFIX, 1234), "iNode1234");
        assert_eq!(name(FILE_TARGET_PREFIX, u32::MAX), "iNode4294967295");
        assert_eq!(name(DIR_TARGET_PREFIX, 10), "dir_10");
        assert_eq!(name(DIR_TARGET_PREFIX, u32::MAX), "dir_4294967295");
        // A prefix that leaves no room fails instead of truncating.
        assert!(TargetName::new(b"0123456789ABCDEF", 1).is_none());
        assert!(TargetName::new(b"abcdef", u32::MAX).is_none());
    }

    #[test]
    fn private_folder_names() {
        assert_eq!(
            text(&FILE_PRIVATE_FOLDER),
            "\u{0}\u{0}\u{0}\u{0}HFS+ Private Data"
        );
        assert_eq!(text(&DIR_PRIVATE_FOLDER), ".HFS+ Private Directory Data\r");
    }

    #[test]
    fn link_codes_spell_their_names() {
        assert_eq!(FILE_LINK_TYPE.to_be_bytes(), *b"hlnk");
        assert_eq!(FILE_LINK_CREATOR.to_be_bytes(), *b"hfs+");
        assert_eq!(DIR_LINK_TYPE.to_be_bytes(), *b"fdrp");
        assert_eq!(DIR_LINK_CREATOR.to_be_bytes(), *b"MACS");
    }
}
