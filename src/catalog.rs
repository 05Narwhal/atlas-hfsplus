// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Catalog file: lookup, paged listing, ID to path.
//!
//! The catalog B-tree is keyed by (parent ID, name). Every folder and file
//! has a record keyed by its parent and its own name, and a thread record
//! keyed by (its own ID, empty name) that gives the parent and name back.
//! All children of a folder are contiguous in key order, right after the
//! folder's thread, so a listing is one descent plus a walk along the leaf
//! chain (TN1150 "Catalog File").

use std::cmp::Ordering;

use super::attributes::Attributes;
use super::btree::{BTree, LeafCursor, Record};
use super::bytes::{be_u16, be_u32, bytes, u8_at};
use super::compression::{compressed_size, is_compressed};
use super::extents::ForkData;
use super::links::{Link, LinkResolver, link_of};
use super::unicode::{NameCompare, compose, compose_stored, decompose};
use super::volume_header::hfs_time;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::{DirBatch, DirEntry, DirProgress, DirStream, EntryFlags, NodeId, NodeKind};

/// The parent ID of the root folder.
pub const ROOT_PARENT_ID: u32 = 1;

/// The root folder's ID.
pub const ROOT_FOLDER_ID: u32 = 2;

/// Longest name a catalog key can hold, in UTF-16 code units.
pub const MAX_NAME_UNITS: usize = 255;

const STRUCTURE: &str = "catalog B-tree";

/// `recordType` values.
const FOLDER_RECORD: u16 = 1;
const FILE_RECORD: u16 = 2;
const FOLDER_THREAD: u16 = 3;
const FILE_THREAD: u16 = 4;

/// Size of `HFSPlusCatalogFolder`.
const FOLDER_RECORD_LEN: usize = 88;
/// Size of `HFSPlusCatalogFile`.
const FILE_RECORD_LEN: usize = 248;
/// Bytes of `HFSPlusCatalogThread` before the name's code units.
const THREAD_HEADER_LEN: usize = 10;
/// Bytes of a catalog key before the name's code units.
const KEY_HEADER_LEN: usize = 6;

/// `kIsInvisible` in the Finder flags (`FInfo.fdFlags`, `DInfo.frFlags`).
const FINDER_INVISIBLE: u16 = 0x4000;
/// `kIsAlias` in the Finder flags.
const FINDER_ALIAS: u16 = 0x8000;
/// `kHasBundle` in the Finder flags.
const FINDER_HAS_BUNDLE: u16 = 0x2000;
/// `kColor` in the Finder flags: the label, bits 1 to 3.
const FINDER_COLOR: u16 = 0x000E;
/// Shift of `kColor`.
const FINDER_COLOR_SHIFT: u32 = 1;

/// File type bits of `fileMode`, and the values the listing tells apart.
const MODE_TYPE_MASK: u16 = 0o170_000;
const MODE_SYMLINK: u16 = 0o120_000;
const MODE_REGULAR: u16 = 0o100_000;

/// Hard cap on the folders walked by [`Catalog::path`].
const MAX_PATH_DEPTH: u32 = 4096;

fn corrupt(location: Location, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(STRUCTURE, location, kind))
}

/// Length as `u64` for error values.
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

/// The bytes of `units` UTF-16 code units at `at` in `buf`, or the
/// `Truncated` error for them.
fn name_bytes(buf: &[u8], at: usize, units: u16) -> Result<&[u8], CorruptKind> {
    let len = usize::from(units).saturating_mul(2);
    bytes(buf, at, len).ok_or(CorruptKind::Truncated {
        needed: len64(at.saturating_add(len)),
        available: len64(buf.len()),
    })
}

/// A name length checked against [`MAX_NAME_UNITS`].
fn check_name_len(field: &'static str, units: u16) -> Result<(), CorruptKind> {
    if usize::from(units) > MAX_NAME_UNITS {
        return Err(CorruptKind::OutOfRange {
            field,
            value: u64::from(units),
            limit_name: "longest name",
            limit: len64(MAX_NAME_UNITS),
        });
    }
    Ok(())
}

/// A catalog key read in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogKey<'a> {
    /// The parent folder's ID.
    pub parent_id: u32,
    /// The name as stored: big-endian UTF-16, decomposed.
    pub name_be: &'a [u8],
}

impl<'a> CatalogKey<'a> {
    /// Parses a key (the bytes after the key length field). `None` when the
    /// name is longer than [`MAX_NAME_UNITS`] or runs past the key. Bytes
    /// after the name are ignored.
    pub fn parse(key: &'a [u8]) -> Option<Self> {
        parse_key(key).ok()
    }
}

fn parse_key(key: &[u8]) -> Result<CatalogKey<'_>, CorruptKind> {
    let truncated = || CorruptKind::Truncated {
        needed: len64(KEY_HEADER_LEN),
        available: len64(key.len()),
    };
    let parent_id = be_u32(key, 0).ok_or_else(truncated)?;
    let units = be_u16(key, 4).ok_or_else(truncated)?;
    check_name_len("key name length", units)?;
    let name_be = name_bytes(key, KEY_HEADER_LEN, units)?;
    Ok(CatalogKey { parent_id, name_be })
}

/// The BSD ownership and mode of a folder or file (`HFSPlusBSDInfo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BsdInfo {
    /// Owner's user ID.
    pub owner_id: u32,
    /// Group ID.
    pub group_id: u32,
    /// Flags settable by the superuser.
    pub admin_flags: u8,
    /// Flags settable by the owner (`UF_COMPRESSED` lives here).
    pub owner_flags: u8,
    /// File type and permission bits, as `st_mode`.
    pub file_mode: u16,
    /// Link count of an indirect node, the device of a device file.
    pub special: u32,
}

/// The five dates of a folder or file, as raw HFS seconds (0 is unset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dates {
    /// `createDate`.
    pub created: u32,
    /// `contentModDate`.
    pub content_modified: u32,
    /// `attributeModDate`.
    pub attributes_modified: u32,
    /// `accessDate`.
    pub accessed: u32,
    /// `backupDate`.
    pub backed_up: u32,
}

/// A folder record (`HFSPlusCatalogFolder`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FolderRecord {
    /// `flags`.
    pub flags: u16,
    /// Number of direct children.
    pub valence: u32,
    /// The folder's ID.
    pub folder_id: u32,
    /// The folder's dates.
    pub dates: Dates,
    /// Ownership and mode.
    pub bsd: BsdInfo,
    /// Finder flags from the user info.
    pub finder_flags: u16,
    /// `userInfo` then `finderInfo` as stored (`DInfo` and `DXInfo`).
    pub finder_info: [u8; 32],
    /// Hint for converting the name to a legacy encoding.
    pub text_encoding: u32,
    /// Number of child folders, when the volume keeps it.
    pub folder_count: u32,
}

/// A file record (`HFSPlusCatalogFile`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileRecord {
    /// `flags`.
    pub flags: u16,
    /// The file's ID.
    pub file_id: u32,
    /// The file's dates.
    pub dates: Dates,
    /// Ownership and mode.
    pub bsd: BsdInfo,
    /// Finder file type (`hlnk` for a hard link).
    pub file_type: u32,
    /// Finder creator code.
    pub creator: u32,
    /// Finder flags from the user info.
    pub finder_flags: u16,
    /// `userInfo` then `finderInfo` as stored (`FInfo` and `FXInfo`).
    pub finder_info: [u8; 32],
    /// Hint for converting the name to a legacy encoding.
    pub text_encoding: u32,
    /// The data fork.
    pub data_fork: ForkData,
    /// The resource fork.
    pub resource_fork: ForkData,
}

/// A folder or file record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogRecord {
    /// A folder.
    Folder(FolderRecord),
    /// A file.
    File(FileRecord),
}

impl CatalogRecord {
    /// The folder's or file's ID.
    pub fn id(&self) -> u32 {
        match self {
            Self::Folder(f) => f.folder_id,
            Self::File(f) => f.file_id,
        }
    }

    /// The Finder flags from the user info.
    pub fn finder_flags(&self) -> u16 {
        match self {
            Self::Folder(f) => f.finder_flags,
            Self::File(f) => f.finder_flags,
        }
    }

    fn thread_kind(&self) -> ThreadKind {
        match self {
            Self::Folder(_) => ThreadKind::Folder,
            Self::File(_) => ThreadKind::File,
        }
    }
}

/// The listing flags of an entry: `HIDDEN` when its stored name starts with
/// a dot or its Finder flags have `kIsInvisible`; `PACKAGE` for
/// `kHasBundle`, `ALIAS` for `kIsAlias`, and the label from `kColor`. The
/// Finder bits are taken as stored, for files and folders alike.
pub fn entry_flags(name_starts_with_dot: bool, finder_flags: u16) -> EntryFlags {
    let mut flags = EntryFlags::EMPTY;
    if name_starts_with_dot || finder_flags & FINDER_INVISIBLE != 0 {
        flags = flags.with(EntryFlags::HIDDEN);
    }
    if finder_flags & FINDER_HAS_BUNDLE != 0 {
        flags = flags.with(EntryFlags::PACKAGE);
    }
    if finder_flags & FINDER_ALIAS != 0 {
        flags = flags.with(EntryFlags::ALIAS);
    }
    // Three bits after the mask: 0 to 7.
    let label = (finder_flags & FINDER_COLOR) >> FINDER_COLOR_SHIFT;
    flags.with_label(u8::try_from(label).unwrap_or(0))
}

/// Which kind of node a thread record belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadKind {
    /// A folder thread.
    Folder,
    /// A file thread.
    File,
}

/// A thread record: the parent and name of a node, keyed by its ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Thread {
    /// Folder or file thread.
    pub kind: ThreadKind,
    /// The node's parent folder.
    pub parent_id: u32,
    /// The node's name in stored code units.
    pub name: Vec<u16>,
}

impl Thread {
    /// The name in display form.
    pub fn display_name(&self) -> String {
        let mut out = String::with_capacity(self.name.len());
        compose(&self.name, &mut out);
        out
    }
}

/// The record data of one catalog leaf record, read in place.
// Lives on the stack for one record; boxing the large variant would cost an
// allocation per listed entry.
#[allow(clippy::large_enum_variant)]
enum Data<'a> {
    Record(CatalogRecord),
    Thread {
        kind: ThreadKind,
        parent_id: u32,
        name_be: &'a [u8],
    },
}

fn record_type(data: &[u8]) -> Result<u16, CorruptKind> {
    let ty = be_u16(data, 0).ok_or(CorruptKind::Truncated {
        needed: 2,
        available: len64(data.len()),
    })?;
    match ty {
        FOLDER_RECORD | FILE_RECORD | FOLDER_THREAD | FILE_THREAD => Ok(ty),
        _ => Err(CorruptKind::InvalidValue {
            field: "catalog record type",
            value: u64::from(ty),
        }),
    }
}

fn is_thread(ty: u16) -> bool {
    matches!(ty, FOLDER_THREAD | FILE_THREAD)
}

fn need(data: &[u8], len: usize) -> Result<(), CorruptKind> {
    if data.len() < len {
        return Err(CorruptKind::Truncated {
            needed: len64(len),
            available: len64(data.len()),
        });
    }
    Ok(())
}

fn parse_dates(d: &[u8]) -> Option<Dates> {
    Some(Dates {
        created: be_u32(d, 12)?,
        content_modified: be_u32(d, 16)?,
        attributes_modified: be_u32(d, 20)?,
        accessed: be_u32(d, 24)?,
        backed_up: be_u32(d, 28)?,
    })
}

fn parse_bsd(d: &[u8]) -> Option<BsdInfo> {
    Some(BsdInfo {
        owner_id: be_u32(d, 32)?,
        group_id: be_u32(d, 36)?,
        admin_flags: u8_at(d, 40)?,
        owner_flags: u8_at(d, 41)?,
        file_mode: be_u16(d, 42)?,
        special: be_u32(d, 44)?,
    })
}

fn zero_id(id: u32) -> Result<u32, CorruptKind> {
    if id == 0 {
        return Err(CorruptKind::InvalidValue {
            field: "catalog node ID",
            value: 0,
        });
    }
    Ok(id)
}

fn parse_folder(d: &[u8]) -> Result<FolderRecord, CorruptKind> {
    need(d, FOLDER_RECORD_LEN)?;
    let short = CorruptKind::Truncated {
        needed: len64(FOLDER_RECORD_LEN),
        available: len64(d.len()),
    };
    let record = (|| {
        Some(FolderRecord {
            flags: be_u16(d, 2)?,
            valence: be_u32(d, 4)?,
            folder_id: be_u32(d, 8)?,
            dates: parse_dates(d)?,
            bsd: parse_bsd(d)?,
            finder_flags: be_u16(d, 56)?,
            finder_info: d.get(48..80)?.try_into().ok()?,
            text_encoding: be_u32(d, 80)?,
            folder_count: be_u32(d, 84)?,
        })
    })()
    .ok_or(short)?;
    zero_id(record.folder_id)?;
    Ok(record)
}

fn parse_file(d: &[u8]) -> Result<FileRecord, CorruptKind> {
    need(d, FILE_RECORD_LEN)?;
    let short = CorruptKind::Truncated {
        needed: len64(FILE_RECORD_LEN),
        available: len64(d.len()),
    };
    let record = (|| {
        Some(FileRecord {
            flags: be_u16(d, 2)?,
            file_id: be_u32(d, 8)?,
            dates: parse_dates(d)?,
            bsd: parse_bsd(d)?,
            file_type: be_u32(d, 48)?,
            creator: be_u32(d, 52)?,
            finder_flags: be_u16(d, 56)?,
            finder_info: d.get(48..80)?.try_into().ok()?,
            text_encoding: be_u32(d, 80)?,
            data_fork: ForkData::parse(d.get(88..)?)?,
            resource_fork: ForkData::parse(d.get(168..)?)?,
        })
    })()
    .ok_or(short)?;
    zero_id(record.file_id)?;
    Ok(record)
}

fn parse_data(d: &[u8]) -> Result<Data<'_>, CorruptKind> {
    let ty = record_type(d)?;
    match ty {
        FOLDER_RECORD => Ok(Data::Record(CatalogRecord::Folder(parse_folder(d)?))),
        FILE_RECORD => Ok(Data::Record(CatalogRecord::File(parse_file(d)?))),
        _ => {
            need(d, THREAD_HEADER_LEN)?;
            let short = CorruptKind::Truncated {
                needed: len64(THREAD_HEADER_LEN),
                available: len64(d.len()),
            };
            let parent_id = be_u32(d, 4).ok_or(short.clone())?;
            let units = be_u16(d, 8).ok_or(short)?;
            check_name_len("thread name length", units)?;
            let name_be = name_bytes(d, THREAD_HEADER_LEN, units)?;
            let kind = if ty == FOLDER_THREAD {
                ThreadKind::Folder
            } else {
                ThreadKind::File
            };
            Ok(Data::Thread {
                kind,
                parent_id,
                name_be,
            })
        }
    }
}

/// Code units of a stored big-endian name; a trailing odd byte is dropped.
fn units_of(name_be: &[u8]) -> Vec<u16> {
    let (pairs, _odd) = name_be.as_chunks::<2>();
    pairs.iter().map(|&pair| u16::from_be_bytes(pair)).collect()
}

/// The catalog B-tree with the volume's name order. Clone is cheap.
#[derive(Clone)]
pub struct Catalog {
    pub(super) tree: BTree,
    compare: NameCompare,
}

/// What one record of a lookup walk means.
// Lives on the stack for one record; boxing would allocate per record.
#[allow(clippy::large_enum_variant)]
enum FindStep {
    Found(CatalogRecord),
    Skip,
    Stop,
}

impl Catalog {
    /// `hfsx` is true when the volume header signature is `HX`. HFS+ always
    /// uses [`NameCompare::CaseFolding`] whatever the header says. HFSX takes
    /// it from the catalog header's key_compare_type; any other value is
    /// Corrupt (`InvalidValue { field: "key compare type" }`, `Location::Node(0)`).
    pub fn new(tree: BTree, hfsx: bool) -> Result<Self> {
        let compare = if hfsx {
            let b = tree.header().key_compare_type;
            NameCompare::from_key_compare_type(b).ok_or_else(|| {
                corrupt(
                    Location::Node(0),
                    CorruptKind::InvalidValue {
                        field: "key compare type",
                        value: u64::from(b),
                    },
                )
            })?
        } else {
            NameCompare::CaseFolding
        };
        Ok(Self { tree, compare })
    }

    /// The volume's name order.
    pub fn compare(&self) -> NameCompare {
        self.compare
    }

    /// A cursor at the first leaf record whose key is >= (parent_id, name).
    fn seek(&self, parent_id: u32, name: &[u16]) -> Result<LeafCursor> {
        let compare = self.compare;
        // The tree hands the comparator key bytes only, so a bad key here
        // cannot say which record it was.
        let mut cmp = |key: &[u8]| -> Result<Ordering> {
            let k = parse_key(key).map_err(|kind| corrupt(Location::Unspecified, kind))?;
            Ok(k.parent_id
                .cmp(&parent_id)
                .then_with(|| compare.compare_stored(k.name_be, name)))
        };
        self.tree.seek(&mut cmp)
    }

    /// The folder or file record keyed (parent_id, name), name in stored units.
    pub fn find(&self, parent_id: u32, name: &[u16]) -> Result<Option<CatalogRecord>> {
        if name.len() > MAX_NAME_UNITS {
            return Ok(None);
        }
        let compare = self.compare;
        let mut cursor = self.seek(parent_id, name)?;
        // Bounded by the cursor, which walks the leaf chain at most once.
        loop {
            let step = match cursor.next_record()? {
                None => return Ok(None),
                Some(rec) => find_step(rec, parent_id, name, compare),
            };
            match step {
                Ok(FindStep::Found(record)) => return Ok(Some(record)),
                Ok(FindStep::Skip) => {}
                Ok(FindStep::Stop) => return Ok(None),
                Err(kind) => return Err(corrupt(record_location(&cursor), kind)),
            }
        }
    }

    /// Same as [`find`](Self::find) from a host name: decompose, then find.
    /// Empty or longer than [`MAX_NAME_UNITS`] after decomposition: `Ok(None)`.
    pub fn lookup(&self, parent_id: u32, name: &str) -> Result<Option<CatalogRecord>> {
        // Every char gives at least one code unit, so this bounds the work
        // before decomposing.
        if name.is_empty() || name.chars().nth(MAX_NAME_UNITS).is_some() {
            return Ok(None);
        }
        let mut units = Vec::with_capacity(name.len());
        decompose(name, &mut units);
        if units.is_empty() || units.len() > MAX_NAME_UNITS {
            return Ok(None);
        }
        self.find(parent_id, &units)
    }

    /// The thread record of `id`, `Ok(None)` if there is none.
    pub fn thread(&self, id: u32) -> Result<Option<Thread>> {
        let mut cursor = self.seek(id, &[])?;
        let parsed = match cursor.next_record()? {
            None => return Ok(None),
            Some(rec) => thread_of(rec, id),
        };
        match parsed {
            Ok(thread) => Ok(thread),
            Err(kind) => Err(corrupt(record_location(&cursor), kind)),
        }
    }

    /// The folder or file record of `id` via its thread. `Ok(None)` if no thread.
    pub fn record(&self, id: u32) -> Result<Option<CatalogRecord>> {
        Ok(self.record_with_thread(id)?.map(|(_, record)| record))
    }

    /// Same as [`record`](Self::record), with the thread that led to it
    /// (the node's parent and stored name).
    pub fn record_with_thread(&self, id: u32) -> Result<Option<(Thread, CatalogRecord)>> {
        if id == 0 {
            return Ok(None);
        }
        let Some(thread) = self.thread(id)? else {
            return Ok(None);
        };
        let bad =
            |text: &'static str| corrupt(Location::Object(u64::from(id)), CorruptKind::Other(text));
        let Some(record) = self.find(thread.parent_id, &thread.name)? else {
            return Err(bad("thread names a record that does not exist"));
        };
        if record.id() != id {
            return Err(bad("thread names a record with a different ID"));
        }
        if record.thread_kind() != thread.kind {
            return Err(bad("thread kind does not match its record"));
        }
        Ok(Some((thread, record)))
    }

    /// (id, display name) of each folder from the root's child down to `id`
    /// itself. The root gives an empty Vec. `Ok(None)` if `id` has no thread.
    pub fn path(&self, id: u32) -> Result<Option<Vec<(u32, String)>>> {
        let Some(mut thread) = self.thread(id)? else {
            return Ok(None);
        };
        if id == ROOT_FOLDER_ID {
            return Ok(Some(Vec::new()));
        }
        let limit = self.tree.header().leaf_records.min(MAX_PATH_DEPTH);
        let mut out: Vec<(u32, String)> = Vec::new();
        let mut current = id;
        let mut steps: u32 = 0;
        // Bounded by `limit`.
        loop {
            let at = Location::Object(u64::from(current));
            if thread.parent_id == ROOT_PARENT_ID {
                return Err(corrupt(
                    at,
                    CorruptKind::Other("only the root folder may have the root's parent"),
                ));
            }
            if thread.parent_id == current {
                return Err(corrupt(
                    at,
                    CorruptKind::Loop {
                        what: "catalog path",
                    },
                ));
            }
            if current != id && thread.kind != ThreadKind::Folder {
                return Err(corrupt(
                    at,
                    CorruptKind::Other("an ancestor in the catalog path is not a folder"),
                ));
            }
            if steps >= limit {
                return Err(corrupt(
                    at,
                    CorruptKind::LimitExceeded {
                        what: "catalog path depth",
                        limit: u64::from(limit),
                    },
                ));
            }
            steps = steps.saturating_add(1);
            let parent = thread.parent_id;
            out.push((current, thread.display_name()));
            if parent == ROOT_FOLDER_ID {
                break;
            }
            current = parent;
            thread = self.thread(current)?.ok_or_else(|| {
                corrupt(
                    Location::Object(u64::from(current)),
                    CorruptKind::Other("a folder in the catalog path has no thread record"),
                )
            })?;
        }
        out.reverse();
        Ok(Some(out))
    }

    /// The volume name: the name in the root folder's thread.
    pub fn volume_name(&self) -> Result<String> {
        let at = Location::Object(u64::from(ROOT_FOLDER_ID));
        let thread = self.thread(ROOT_FOLDER_ID)?.ok_or_else(|| {
            corrupt(
                at,
                CorruptKind::Other("the root folder has no thread record"),
            )
        })?;
        if thread.kind != ThreadKind::Folder || thread.parent_id != ROOT_PARENT_ID {
            return Err(corrupt(
                at,
                CorruptKind::Other("the root folder's thread record is not a root folder thread"),
            ));
        }
        Ok(thread.display_name())
    }

    /// Opens folder `folder_id` for listing, records as stored: hard links
    /// are listed as the small files they are.
    ///
    /// Fails with `Error::NotADirectory` when `folder_id` is a file and
    /// `Error::NotFound` when it has no thread record.
    pub fn list(&self, folder_id: u32) -> Result<CatalogDir> {
        self.open_list(folder_id, None, None)
    }

    /// Same as [`list`](Self::list), resolving hard links through `links`
    /// and reporting compressed files at their uncompressed size.
    ///
    /// A resolved link keeps its own name and `HIDDEN` flag and takes the
    /// target's node ID, kind, data size and modification date. A link that
    /// cannot be resolved (or whose resolution fails to read) is listed with
    /// its own ID as a `File` with no size and no date; that is not a stream
    /// error. In the root, the private link folders are left out.
    ///
    /// A file with `UF_COMPRESSED` (or a link to one) reports the size in its
    /// decmpfs header, read from `attributes`: one attributes B-tree descent
    /// per such entry. When the header cannot be read (no attributes tree,
    /// no header, a bad one, a read error) the entry has no size.
    pub fn list_resolving(
        &self,
        folder_id: u32,
        links: &LinkResolver,
        attributes: Option<&Attributes>,
    ) -> Result<CatalogDir> {
        self.open_list(folder_id, Some(links.clone()), attributes.cloned())
    }

    fn open_list(
        &self,
        folder_id: u32,
        links: Option<LinkResolver>,
        attributes: Option<Attributes>,
    ) -> Result<CatalogDir> {
        let mut cursor = self.seek(folder_id, &[])?;
        let first = match cursor.next_record()? {
            None => {
                return Err(Error::NotFound {
                    node: u64::from(folder_id),
                });
            }
            Some(rec) => (|| {
                let key = parse_key(rec.key)?;
                if key.parent_id != folder_id {
                    return Ok(None);
                }
                record_type(rec.data).map(Some)
            })(),
        };
        match first {
            Ok(Some(FOLDER_THREAD)) => Ok(CatalogDir {
                cursor,
                folder_id,
                links,
                attributes,
                decmpfs: Vec::new(),
                name: String::new(),
                done: false,
            }),
            Ok(Some(FILE_THREAD)) => Err(Error::NotADirectory {
                node: u64::from(folder_id),
            }),
            Ok(_) => Err(Error::NotFound {
                node: u64::from(folder_id),
            }),
            Err(kind) => Err(corrupt(record_location(&cursor), kind)),
        }
    }
}

/// One record of a [`Catalog::find`] walk.
fn find_step(
    rec: Record<'_>,
    parent_id: u32,
    name: &[u16],
    compare: NameCompare,
) -> Result<FindStep, CorruptKind> {
    let key = parse_key(rec.key)?;
    let order = key
        .parent_id
        .cmp(&parent_id)
        .then_with(|| compare.compare_stored(key.name_be, name));
    if order != Ordering::Equal {
        return Ok(FindStep::Stop);
    }
    // With case folding, a name of only ignorable code points compares
    // Equal to the thread key (parent, "").
    if is_thread(record_type(rec.data)?) {
        return Ok(FindStep::Skip);
    }
    match parse_data(rec.data)? {
        Data::Record(record) => Ok(FindStep::Found(record)),
        Data::Thread { .. } => Ok(FindStep::Skip),
    }
}

/// The thread of `id` if `rec` is it.
fn thread_of(rec: Record<'_>, id: u32) -> Result<Option<Thread>, CorruptKind> {
    let key = parse_key(rec.key)?;
    if key.parent_id != id || !key.name_be.is_empty() {
        return Ok(None);
    }
    match parse_data(rec.data)? {
        Data::Record(_) => Ok(None),
        Data::Thread {
            kind,
            parent_id,
            name_be,
        } => {
            if parent_id == 0 {
                return Err(CorruptKind::InvalidValue {
                    field: "thread parent ID",
                    value: 0,
                });
            }
            Ok(Some(Thread {
                kind,
                parent_id,
                name: units_of(name_be),
            }))
        }
    }
}

/// A folder listing. Owns a [`LeafCursor`] (and, when it resolves hard
/// links, a [`LinkResolver`] and the attributes tree for compressed sizes),
/// so it is `Send` and does not borrow the [`Catalog`].
pub struct CatalogDir {
    cursor: LeafCursor,
    folder_id: u32,
    links: Option<LinkResolver>,
    /// Where decmpfs headers are read; only set when resolving.
    attributes: Option<Attributes>,
    /// The decmpfs header buffer, reused across entries.
    decmpfs: Vec<u8>,
    /// The current entry's display name, reused across entries.
    name: String,
    done: bool,
}

/// The fixed-size facts of one listed entry; the name is in `CatalogDir::name`.
#[derive(Clone, Copy)]
pub(super) struct Facts {
    pub(super) node: u32,
    pub(super) kind: NodeKind,
    pub(super) size: Option<u64>,
    /// The resource fork's size; 0 for a compressed file.
    pub(super) resource_size: Option<u64>,
    pub(super) modified: u32,
    pub(super) flags: EntryFlags,
    /// Set when the entry is a hard link, until it is resolved.
    pub(super) link: Option<Link>,
    /// A regular file with `UF_COMPRESSED`: its size is in its decmpfs header.
    pub(super) compressed: bool,
    /// `FInfo.fdType` of a file, `None` when zero, for a folder, and for a
    /// hard link until it is resolved (its own type is `hlnk` or `fdrp`).
    pub(super) finder_type: Option<[u8; 4]>,
}

/// A file record's type code, `None` when it is four zero bytes.
fn finder_type_of(f: &FileRecord) -> Option<[u8; 4]> {
    (f.file_type != 0).then(|| f.file_type.to_be_bytes())
}

impl Facts {
    /// The facts of a link entry once its target is known: `Ok(Some)` the
    /// target, anything else unresolvable.
    fn resolved(self, target: Result<Option<CatalogRecord>>) -> Self {
        match target {
            Ok(Some(CatalogRecord::Folder(f))) => Self {
                node: f.folder_id,
                kind: NodeKind::Directory,
                size: None,
                resource_size: None,
                modified: f.dates.content_modified,
                link: None,
                compressed: false,
                finder_type: None,
                ..self
            },
            Ok(Some(CatalogRecord::File(f))) => Self {
                node: f.file_id,
                kind: kind_of_mode(f.bsd.file_mode),
                size: Some(f.data_fork.logical_size),
                resource_size: Some(resource_size_of(&f)),
                modified: f.dates.content_modified,
                link: None,
                compressed: is_compressed(&f),
                finder_type: finder_type_of(&f),
                ..self
            },
            Ok(None) | Err(_) => Self {
                kind: NodeKind::File,
                size: None,
                resource_size: None,
                modified: 0,
                link: None,
                compressed: false,
                finder_type: None,
                ..self
            },
        }
    }
}

/// The resource fork size a listing and the details pane show for a file:
/// 0 for a compressed file, whose fork holds the compressed content.
pub fn resource_size_of(f: &FileRecord) -> u64 {
    if is_compressed(f) {
        0
    } else {
        f.resource_fork.logical_size
    }
}

/// What one record of a listing walk means.
pub(super) enum ListStep {
    Entry(Facts),
    Skip,
    End,
}

/// The kind a BSD `fileMode` gives a file record.
pub fn kind_of_mode(mode: u16) -> NodeKind {
    match mode & MODE_TYPE_MASK {
        MODE_SYMLINK => NodeKind::Symlink,
        // 0: no BSD info, as on volumes written by Mac OS 9.
        MODE_REGULAR | 0 => NodeKind::File,
        _ => NodeKind::Other,
    }
}

/// Reads one listing record; composes its name into `name` when it is an entry.
pub(super) fn list_step(
    rec: Record<'_>,
    folder_id: u32,
    name: &mut String,
) -> Result<ListStep, CorruptKind> {
    let key = parse_key(rec.key)?;
    if key.parent_id != folder_id {
        return Ok(ListStep::End);
    }
    if is_thread(record_type(rec.data)?) {
        return Ok(ListStep::Skip);
    }
    let record = match parse_data(rec.data)? {
        Data::Record(record) => record,
        Data::Thread { .. } => return Ok(ListStep::Skip),
    };
    let dotted = key.name_be.get(..2) == Some(&[0x00, 0x2E][..]);
    let flags = entry_flags(dotted, record.finder_flags());
    let facts = match record {
        CatalogRecord::Folder(f) => Facts {
            node: f.folder_id,
            kind: NodeKind::Directory,
            size: None,
            resource_size: None,
            modified: f.dates.content_modified,
            flags,
            link: None,
            compressed: false,
            finder_type: None,
        },
        CatalogRecord::File(f) => {
            let link = link_of(&f);
            Facts {
                node: f.file_id,
                kind: kind_of_mode(f.bsd.file_mode),
                size: Some(f.data_fork.logical_size),
                resource_size: Some(resource_size_of(&f)),
                modified: f.dates.content_modified,
                flags,
                link,
                compressed: is_compressed(&f),
                // A link's own type (`hlnk`, `fdrp`) says nothing of its
                // target; the target's comes with `resolved`.
                finder_type: if link.is_some() {
                    None
                } else {
                    finder_type_of(&f)
                },
            }
        }
    };
    name.clear();
    compose_stored(key.name_be, name);
    Ok(ListStep::Entry(facts))
}

impl CatalogDir {
    /// One step of the walk; errors finish the stream.
    fn step(&mut self) -> Result<ListStep> {
        let step = match self.cursor.next_record()? {
            None => return Ok(ListStep::End),
            Some(rec) => list_step(rec, self.folder_id, &mut self.name),
        };
        let step = step.map_err(|kind| corrupt(record_location(&self.cursor), kind))?;
        let ListStep::Entry(facts) = step else {
            return Ok(step);
        };
        let Some(links) = &self.links else {
            return Ok(ListStep::Entry(facts));
        };
        if links.hidden_in_listing(self.folder_id, facts.node) {
            return Ok(ListStep::Skip);
        }
        let mut facts = match facts.link {
            Some(link) => facts.resolved(links.resolve(link)),
            None => facts,
        };
        if facts.compressed {
            facts.size = self
                .attributes
                .as_ref()
                .and_then(|a| compressed_size(a, facts.node, &mut self.decmpfs));
        }
        Ok(ListStep::Entry(facts))
    }
}

impl DirStream for CatalogDir {
    fn next_batch(&mut self, out: &mut DirBatch, max: usize) -> Result<DirProgress> {
        if self.done {
            return Ok(DirProgress::Done);
        }
        let mut added = 0usize;
        // Bounded: every turn takes one record from the cursor, which ends.
        while added < max {
            match self.step() {
                Ok(ListStep::Entry(f)) => {
                    out.push(DirEntry {
                        node: NodeId(u64::from(f.node)),
                        name: &self.name,
                        kind: f.kind,
                        size: f.size,
                        resource_size: f.resource_size,
                        modified: hfs_time(f.modified),
                        flags: if f.compressed {
                            f.flags.with(EntryFlags::COMPRESSED)
                        } else {
                            f.flags
                        },
                    });
                    added = added.saturating_add(1);
                }
                Ok(ListStep::Skip) => {}
                Ok(ListStep::End) => {
                    self.done = true;
                    return Ok(DirProgress::Done);
                }
                Err(e) => {
                    self.done = true;
                    return Err(e);
                }
            }
        }
        Ok(DirProgress::More)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::btree::fixtures::{INDEX, LEAF, header, header_node, keyed, node, tree, var_index};

    const SIZE: usize = 4096;

    // ---- byte builders ----

    fn key_units(parent: u32, units: &[u16]) -> Vec<u8> {
        let mut k = parent.to_be_bytes().to_vec();
        k.extend_from_slice(&(units.len() as u16).to_be_bytes());
        for u in units {
            k.extend_from_slice(&u.to_be_bytes());
        }
        k
    }

    /// Key with `name` stored as its UTF-16 units, without decomposing.
    fn key(parent: u32, name: &str) -> Vec<u8> {
        key_units(parent, &name.encode_utf16().collect::<Vec<_>>())
    }

    fn put16(b: &mut [u8], at: usize, v: u16) {
        b[at..at + 2].copy_from_slice(&v.to_be_bytes());
    }

    fn put32(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_be_bytes());
    }

    const MODIFIED: u32 = 2_082_844_800 + 1_000; // 1970-01-01 00:16:40 UTC

    fn folder_data(id: u32, finder_flags: u16) -> Vec<u8> {
        let mut d = vec![0u8; FOLDER_RECORD_LEN];
        put16(&mut d, 0, FOLDER_RECORD);
        put32(&mut d, 4, 3); // valence
        put32(&mut d, 8, id);
        put32(&mut d, 12, 5); // created
        put32(&mut d, 16, MODIFIED);
        put32(&mut d, 32, 501);
        put16(&mut d, 42, 0o040_755);
        put16(&mut d, 56, finder_flags);
        put32(&mut d, 84, 1);
        d
    }

    fn file_data(id: u32, mode: u16, size: u64, finder_flags: u16) -> Vec<u8> {
        let mut d = vec![0u8; FILE_RECORD_LEN];
        put16(&mut d, 0, FILE_RECORD);
        put32(&mut d, 8, id);
        put32(&mut d, 16, MODIFIED);
        put16(&mut d, 42, mode);
        put32(&mut d, 48, u32::from_be_bytes(*b"TEXT"));
        put32(&mut d, 52, u32::from_be_bytes(*b"ttxt"));
        put16(&mut d, 56, finder_flags);
        d[88..96].copy_from_slice(&size.to_be_bytes());
        d[168..176].copy_from_slice(&7u64.to_be_bytes()); // resource fork size
        d
    }

    fn file_rec(id: u32) -> Vec<u8> {
        file_data(id, 0o100_644, u64::from(id) * 10, 0)
    }

    fn thread_data(ty: u16, parent: u32, name: &str) -> Vec<u8> {
        let units: Vec<u16> = name.encode_utf16().collect();
        let mut d = vec![0u8; THREAD_HEADER_LEN];
        put16(&mut d, 0, ty);
        put32(&mut d, 4, parent);
        put16(&mut d, 8, units.len() as u16);
        for u in units {
            d.extend_from_slice(&u.to_be_bytes());
        }
        d
    }

    fn fthread(parent: u32, name: &str) -> Vec<u8> {
        thread_data(FOLDER_THREAD, parent, name)
    }

    type Rec = (Vec<u8>, Vec<u8>);

    /// A catalog tree: root index node 1 over leaves 2.., `split` giving the
    /// number of records in each leaf.
    fn build_tree(recs: &[Rec], split: &[usize], key_compare: u8) -> BTree {
        assert_eq!(split.iter().sum::<usize>(), recs.len());
        let leaves = split.len() as u32;
        let mut h = header(2, 1, 2, 1 + leaves, 2 + leaves);
        h.node_size = SIZE as u16;
        h.max_key_length = 516;
        h.leaf_records = recs.len() as u32;
        h.key_compare_type = key_compare;
        let mut nodes = vec![header_node(&h)];
        let mut index = Vec::new();
        let mut leaf_nodes = Vec::new();
        let mut at = 0;
        for (i, &n) in split.iter().enumerate() {
            let number = 2 + i as u32;
            let chunk = &recs[at..at + n];
            at += n;
            let first_key = chunk.first().map(|r| r.0.clone()).unwrap_or_default();
            index.push(var_index(&first_key, number));
            let fwd = if i + 1 < split.len() { number + 1 } else { 0 };
            let bwd = if i == 0 { 0 } else { number - 1 };
            let body: Vec<Vec<u8>> = chunk.iter().map(|(k, d)| keyed(k, d)).collect();
            leaf_nodes.push(node(SIZE, LEAF, 1, fwd, bwd, &body));
        }
        nodes.push(node(SIZE, INDEX, 2, 0, 0, &index));
        nodes.extend(leaf_nodes);
        tree(&nodes).unwrap()
    }

    fn catalog(recs: &[Rec], split: &[usize]) -> Catalog {
        Catalog::new(build_tree(recs, split, 0), false).unwrap()
    }

    const CAFE: &str = "cafe\u{301}";

    /// The main fixture. Leaf 2 ends after the root's children; leaf 3 holds
    /// folder 16's thread and first children; leaf 4 the rest.
    fn main_records() -> (Vec<Rec>, Vec<usize>) {
        let recs = vec![
            // leaf 2
            (key(1, "Macintosh HD"), folder_data(2, 0)),
            (key(2, ""), fthread(1, "Macintosh HD")),
            (key(2, ".hidden"), file_rec(20)),
            (key(2, "a"), folder_data(16, 0)),
            (key(2, "b.txt"), file_rec(21)),
            (key(2, CAFE), file_rec(29)),
            (key(2, "link"), file_data(22, 0o120_755, 12, 0)),
            (key(2, "pipe"), file_data(23, 0o010_644, 0, 0)),
            (key(2, "readme"), file_rec(28)),
            (key(2, "zero"), file_data(24, 0, 99, FINDER_INVISIBLE)),
            // leaf 3
            (key(16, ""), fthread(2, "a")),
            (key(16, "c"), folder_data(17, 0)),
            (key(16, "d"), file_rec(25)),
            // leaf 4
            (key(16, "e"), file_rec(26)),
            (key(16, "f"), file_rec(27)),
            (key(17, ""), fthread(16, "c")),
            (key(17, "deep"), folder_data(18, 0)),
            (key(18, ""), fthread(17, "deep")),
            (key(18, "empty"), folder_data(19, 0)),
            (key(19, ""), fthread(18, "empty")),
            (key(20, ""), thread_data(FILE_THREAD, 2, ".hidden")),
            (key(21, ""), thread_data(FILE_THREAD, 2, "b.txt")),
        ];
        (recs, vec![10, 3, 9])
    }

    fn main_catalog() -> Catalog {
        let (recs, split) = main_records();
        catalog(&recs, &split)
    }

    // ---- listing helpers ----

    #[derive(Debug, Clone, PartialEq)]
    struct Owned {
        node: u64,
        name: String,
        kind: NodeKind,
        size: Option<u64>,
        modified: bool,
        hidden: bool,
    }

    fn owned(batch: &DirBatch) -> Vec<Owned> {
        batch
            .iter()
            .map(|e| Owned {
                node: e.node.0,
                name: e.name.to_string(),
                kind: e.kind,
                size: e.size,
                modified: e.modified == hfs_time(MODIFIED),
                hidden: e.flags.contains(EntryFlags::HIDDEN),
            })
            .collect()
    }

    fn list_all(c: &Catalog, id: u32, max: usize) -> Vec<Owned> {
        let mut dir = c.list(id).unwrap();
        let mut out = DirBatch::new();
        let mut calls = 0;
        loop {
            let before = out.len();
            let p = dir.next_batch(&mut out, max).unwrap();
            assert!(out.len() - before <= max);
            calls += 1;
            assert!(calls < 1000);
            if p == DirProgress::Done {
                break;
            }
        }
        // Done stays Done and leaves `out` alone.
        let n = out.len();
        assert_eq!(dir.next_batch(&mut out, 10).unwrap(), DirProgress::Done);
        assert_eq!(out.len(), n);
        owned(&out)
    }

    /// Lists until an error; returns the entries kept and the corruption.
    fn list_until_error(c: &Catalog, id: u32) -> (Vec<String>, Corruption) {
        let mut dir = c.list(id).unwrap();
        let mut out = DirBatch::new();
        let err = loop {
            match dir.next_batch(&mut out, 100) {
                Ok(DirProgress::More) => {}
                Ok(DirProgress::Done) => panic!("expected an error"),
                Err(e) => break e,
            }
        };
        let n = out.len();
        assert_eq!(dir.next_batch(&mut out, 100).unwrap(), DirProgress::Done);
        assert_eq!(out.len(), n);
        let names = out.iter().map(|e| e.name.to_string()).collect();
        (names, corruption(Err::<(), _>(err)))
    }

    fn corruption<T: std::fmt::Debug>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    fn e(node: u64, name: &str, kind: NodeKind, size: Option<u64>, hidden: bool) -> Owned {
        Owned {
            node,
            name: name.to_string(),
            kind,
            size,
            modified: true,
            hidden,
        }
    }

    // ---- listing ----

    #[test]
    fn list_spans_leaves_in_order() {
        let c = main_catalog();
        let want = vec![
            e(17, "c", NodeKind::Directory, None, false),
            e(25, "d", NodeKind::File, Some(250), false),
            e(26, "e", NodeKind::File, Some(260), false),
            e(27, "f", NodeKind::File, Some(270), false),
        ];
        for max in [1, 2, 3, 1000] {
            assert_eq!(list_all(&c, 16, max), want, "max {max}");
        }
    }

    #[test]
    fn list_root_kinds_and_flags() {
        let c = main_catalog();
        let want = vec![
            e(20, ".hidden", NodeKind::File, Some(200), true),
            e(16, "a", NodeKind::Directory, None, false),
            e(21, "b.txt", NodeKind::File, Some(210), false),
            e(29, "caf\u{e9}", NodeKind::File, Some(290), false),
            e(22, "link", NodeKind::Symlink, Some(12), false),
            e(23, "pipe", NodeKind::Other, Some(0), false),
            e(28, "readme", NodeKind::File, Some(280), false),
            e(24, "zero", NodeKind::File, Some(99), true),
        ];
        for max in [1, 2, 1000] {
            assert_eq!(list_all(&c, ROOT_FOLDER_ID, max), want, "max {max}");
        }
    }

    #[test]
    fn list_max_zero_and_empty_folder() {
        let c = main_catalog();
        let mut dir = c.list(16).unwrap();
        let mut out = DirBatch::new();
        assert_eq!(dir.next_batch(&mut out, 0).unwrap(), DirProgress::More);
        assert!(out.is_empty());
        assert_eq!(dir.next_batch(&mut out, 1).unwrap(), DirProgress::More);
        assert_eq!(out.len(), 1);
        assert_eq!(dir.next_batch(&mut out, 0).unwrap(), DirProgress::More);
        assert_eq!(out.len(), 1);
        // The listing continues where it was.
        assert_eq!(dir.next_batch(&mut out, 100).unwrap(), DirProgress::Done);
        assert_eq!(out.len(), 4);

        assert!(list_all(&c, 19, 10).is_empty());
        assert_eq!(list_all(&c, 18, 10).len(), 1);
    }

    #[test]
    fn list_not_a_directory_or_missing() {
        let c = main_catalog();
        assert!(matches!(c.list(20), Err(Error::NotADirectory { node: 20 })));
        // Past the last key.
        assert!(matches!(c.list(99), Err(Error::NotFound { node: 99 })));
        // Lands on another folder's thread.
        assert!(matches!(c.list(5), Err(Error::NotFound { node: 5 })));
        // A file record with no thread.
        assert!(matches!(c.list(25), Err(Error::NotFound { node: 25 })));
        assert!(matches!(c.list(0), Err(Error::NotFound { node: 0 })));
    }

    #[test]
    fn dir_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CatalogDir>();
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Catalog>();
    }

    // ---- lookup ----

    #[test]
    fn lookup_hfsplus_case_folding() {
        let c = main_catalog();
        assert_eq!(c.compare(), NameCompare::CaseFolding);
        let id = |name: &str| c.lookup(2, name).unwrap().map(|r| r.id());
        assert_eq!(id("README"), Some(28));
        assert_eq!(id("readme"), Some(28));
        assert_eq!(id("caf\u{e9}"), Some(29));
        assert_eq!(id(CAFE), Some(29));
        assert_eq!(id("a"), Some(16));
        assert_eq!(id(&"x".repeat(256)), None);
        assert_eq!(id(&"x".repeat(255)), None);
        assert_eq!(id(""), None);
        assert_eq!(id("nope"), None);
        assert_eq!(id("zzzz"), None);
        assert_eq!(c.lookup(16, "D").unwrap().map(|r| r.id()), Some(25));
        assert_eq!(c.lookup(99, "a").unwrap(), None);
        match c.lookup(2, "zero").unwrap() {
            Some(CatalogRecord::File(f)) => {
                assert_eq!(f.file_id, 24);
                assert_eq!(f.data_fork.logical_size, 99);
                assert_eq!(f.resource_fork.logical_size, 7);
                assert_eq!(f.file_type, u32::from_be_bytes(*b"TEXT"));
                assert_eq!(f.creator, u32::from_be_bytes(*b"ttxt"));
                assert_eq!(f.finder_flags, FINDER_INVISIBLE);
                assert_eq!(f.dates.content_modified, MODIFIED);
            }
            other => panic!("{other:?}"),
        }
        match c.lookup(2, "A").unwrap() {
            Some(CatalogRecord::Folder(f)) => {
                assert_eq!(f.folder_id, 16);
                assert_eq!(f.valence, 3);
                assert_eq!(f.folder_count, 1);
                assert_eq!(f.dates.created, 5);
                assert_eq!(f.bsd.owner_id, 501);
                assert_eq!(f.bsd.file_mode, 0o040_755);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn lookup_hfsx_binary() {
        let (recs, split) = main_records();
        let c = Catalog::new(build_tree(&recs, &split, 0xBC), true).unwrap();
        assert_eq!(c.compare(), NameCompare::Binary);
        assert_eq!(c.lookup(2, "README").unwrap(), None);
        assert_eq!(c.lookup(2, "readme").unwrap().map(|r| r.id()), Some(28));
        assert_eq!(c.lookup(2, "caf\u{e9}").unwrap().map(|r| r.id()), Some(29));

        let c = Catalog::new(build_tree(&recs, &split, 0xCF), true).unwrap();
        assert_eq!(c.lookup(2, "README").unwrap().map(|r| r.id()), Some(28));

        let bad = Catalog::new(build_tree(&recs, &split, 0x00), true);
        let corr = corruption(bad.map(|_| ()));
        assert_eq!(corr.location, Location::Node(0));
        assert_eq!(
            corr.kind,
            CorruptKind::InvalidValue {
                field: "key compare type",
                value: 0
            }
        );

        // Plain HFS+ case-folds whatever the header says.
        let c = Catalog::new(build_tree(&recs, &split, 0xBC), false).unwrap();
        assert_eq!(c.compare(), NameCompare::CaseFolding);
        assert_eq!(c.lookup(2, "README").unwrap().map(|r| r.id()), Some(28));
    }

    #[test]
    fn find_skips_equal_thread() {
        let recs = vec![
            (key(2, ""), fthread(1, "V")),
            (key_units(2, &[0x200C]), folder_data(30, 0)),
            (key(2, "b"), file_rec(31)),
        ];
        let c = catalog(&recs, &[3]);
        assert_eq!(c.find(2, &[0x200C]).unwrap().map(|r| r.id()), Some(30));
        assert_eq!(c.lookup(2, "\u{200C}").unwrap().map(|r| r.id()), Some(30));
        // Only the thread compares Equal: no record.
        let recs = vec![(key(2, ""), fthread(1, "V")), (key(2, "b"), file_rec(31))];
        let c = catalog(&recs, &[2]);
        assert_eq!(c.find(2, &[0x200C]).unwrap(), None);
        assert_eq!(c.find(2, &[]).unwrap(), None);
        assert_eq!(c.find(2, &[u16::from(b'x'); 256]).unwrap(), None);
    }

    // ---- threads, records, paths ----

    #[test]
    fn threads_and_records() {
        let c = main_catalog();
        let t = c.thread(16).unwrap().unwrap();
        assert_eq!(t.kind, ThreadKind::Folder);
        assert_eq!(t.parent_id, 2);
        assert_eq!(t.name, vec![u16::from(b'a')]);
        assert_eq!(t.display_name(), "a");
        let t = c.thread(20).unwrap().unwrap();
        assert_eq!(t.kind, ThreadKind::File);
        assert_eq!(t.display_name(), ".hidden");
        assert_eq!(c.thread(25).unwrap(), None);
        assert_eq!(c.thread(99).unwrap(), None);
        assert_eq!(c.thread(0).unwrap(), None);
        // ID 1 seeks to (1, "") and finds the root's folder record.
        assert_eq!(c.thread(1).unwrap(), None);

        assert!(
            matches!(c.record(16).unwrap(), Some(CatalogRecord::Folder(f)) if f.folder_id == 16)
        );
        assert!(matches!(c.record(20).unwrap(), Some(CatalogRecord::File(f)) if f.file_id == 20));
        assert!(matches!(c.record(2).unwrap(), Some(CatalogRecord::Folder(f)) if f.folder_id == 2));
        assert_eq!(c.record(0).unwrap(), None);
        assert_eq!(c.record(25).unwrap(), None);
        assert_eq!(c.record(99).unwrap(), None);
    }

    #[test]
    fn paths() {
        let c = main_catalog();
        let p = c.path(18).unwrap().unwrap();
        assert_eq!(
            p,
            vec![
                (16, "a".to_string()),
                (17, "c".to_string()),
                (18, "deep".to_string())
            ]
        );
        assert_eq!(c.path(2).unwrap(), Some(vec![]));
        assert_eq!(c.path(16).unwrap(), Some(vec![(16, "a".to_string())]));
        assert_eq!(c.path(20).unwrap(), Some(vec![(20, ".hidden".to_string())]));
        assert_eq!(c.path(99).unwrap(), None);
        assert_eq!(c.path(25).unwrap(), None);
        assert_eq!(c.volume_name().unwrap(), "Macintosh HD");
    }

    // ---- malformed input ----

    /// A root folder plus `extra` records, in one leaf.
    fn root_with(extra: Vec<Rec>) -> Catalog {
        let mut recs = vec![
            (key(1, "V"), folder_data(2, 0)),
            (key(2, ""), fthread(1, "V")),
        ];
        let n = recs.len() + extra.len();
        recs.extend(extra);
        catalog(&recs, &[n])
    }

    #[test]
    fn malformed_key_name_length() {
        let mut long = key(2, "b");
        long[4..6].copy_from_slice(&256u16.to_be_bytes());
        long.resize(6 + 512, 0);
        assert_eq!(CatalogKey::parse(&long), None);
        let c = root_with(vec![(key(2, "a"), file_rec(20)), (long, file_rec(21))]);
        let (names, corr) = list_until_error(&c, 2);
        assert_eq!(names, ["a"]);
        assert_eq!(corr.location, Location::Record { node: 2, record: 3 });
        assert!(matches!(
            corr.kind,
            CorruptKind::OutOfRange {
                field: "key name length",
                value: 256,
                ..
            }
        ));
        // In the comparator: lookups fail rather than guess.
        assert!(matches!(c.lookup(2, "b"), Err(Error::Corrupt(_))));
    }

    #[test]
    fn malformed_key_name_past_end() {
        let mut short = key(2, "bb");
        short[4..6].copy_from_slice(&5u16.to_be_bytes());
        assert_eq!(CatalogKey::parse(&short), None);
        assert_eq!(CatalogKey::parse(&[0, 0, 0]), None);
        // Trailing bytes after the name are fine.
        let mut padded = key(2, "b");
        padded.extend_from_slice(&[9, 9, 9]);
        let k = CatalogKey::parse(&padded).unwrap();
        assert_eq!((k.parent_id, k.name_be), (2, &[0u8, b'b'][..]));

        let c = root_with(vec![(key(2, "a"), file_rec(20)), (short, file_rec(21))]);
        let (names, corr) = list_until_error(&c, 2);
        assert_eq!(names, ["a"]);
        assert_eq!(corr.location, Location::Record { node: 2, record: 3 });
        assert!(matches!(
            corr.kind,
            CorruptKind::Truncated {
                needed: 16,
                available: 10
            }
        ));
    }

    #[test]
    fn malformed_record_type() {
        let mut bad = file_rec(21);
        put16(&mut bad, 0, 7);
        let c = root_with(vec![(key(2, "a"), file_rec(20)), (key(2, "b"), bad)]);
        let (names, corr) = list_until_error(&c, 2);
        assert_eq!(names, ["a"]);
        assert_eq!(corr.location, Location::Record { node: 2, record: 3 });
        assert_eq!(
            corr.kind,
            CorruptKind::InvalidValue {
                field: "catalog record type",
                value: 7
            }
        );
        assert!(c.lookup(2, "b").is_err());
        // A record too short for its type field.
        let c = root_with(vec![(key(2, "a"), file_rec(20)), (key(2, "b"), vec![0])]);
        let (_, corr) = list_until_error(&c, 2);
        assert!(matches!(
            corr.kind,
            CorruptKind::Truncated {
                needed: 2,
                available: 1
            }
        ));
    }

    #[test]
    fn malformed_truncated_records() {
        let mut folder = folder_data(21, 0);
        folder.truncate(87);
        let c = root_with(vec![(key(2, "a"), file_rec(20)), (key(2, "b"), folder)]);
        let (names, corr) = list_until_error(&c, 2);
        assert_eq!(names, ["a"]);
        assert_eq!(corr.location, Location::Record { node: 2, record: 3 });
        assert_eq!(
            corr.kind,
            CorruptKind::Truncated {
                needed: 88,
                available: 87
            }
        );
        assert!(c.lookup(2, "b").is_err());

        let mut file = file_rec(21);
        file.truncate(247);
        let c = root_with(vec![(key(2, "a"), file_rec(20)), (key(2, "b"), file)]);
        let (names, corr) = list_until_error(&c, 2);
        assert_eq!(names, ["a"]);
        assert_eq!(
            corr.kind,
            CorruptKind::Truncated {
                needed: 248,
                available: 247
            }
        );
        // Trailing bytes past the fixed size are allowed.
        let mut long = file_rec(21);
        long.extend_from_slice(&[1, 2, 3, 4]);
        let c = root_with(vec![(key(2, "b"), long)]);
        assert_eq!(list_all(&c, 2, 10).len(), 1);
    }

    #[test]
    fn records_carry_the_32_finder_info_bytes() {
        let pattern: Vec<u8> = (1..=32).collect();
        let mut file = file_rec(21);
        file[48..80].copy_from_slice(&pattern);
        let mut folder = folder_data(22, 0);
        folder[48..80].copy_from_slice(&pattern);
        assert_eq!(parse_file(&file).unwrap().finder_info[..], pattern[..]);
        assert_eq!(parse_folder(&folder).unwrap().finder_info[..], pattern[..]);
        // A record shorter than the finder info is still `Truncated`.
        assert!(matches!(
            parse_file(&file[..79]),
            Err(CorruptKind::Truncated { .. })
        ));
        assert!(matches!(
            parse_folder(&folder[..60]),
            Err(CorruptKind::Truncated { .. })
        ));
    }

    #[test]
    fn malformed_thread_name_past_data() {
        let mut t = fthread(2, "abc");
        t.truncate(t.len() - 2);
        let c = root_with(vec![(key(30, ""), t)]);
        let corr = corruption(c.thread(30));
        assert_eq!(corr.location, Location::Record { node: 2, record: 2 });
        assert_eq!(
            corr.kind,
            CorruptKind::Truncated {
                needed: 16,
                available: 14
            }
        );
        // A thread name length over 255.
        let mut t = fthread(2, "abc");
        put16(&mut t, 8, 256);
        let c = root_with(vec![(key(30, ""), t)]);
        assert!(matches!(
            corruption(c.thread(30)).kind,
            CorruptKind::OutOfRange {
                field: "thread name length",
                ..
            }
        ));
        // A thread header cut short.
        let c = root_with(vec![(key(30, ""), vec![0, 3, 0, 0, 0])]);
        assert!(c.thread(30).is_err());
        // Thread parent 0.
        let c = root_with(vec![(key(30, ""), fthread(0, "x"))]);
        assert!(matches!(
            corruption(c.thread(30)).kind,
            CorruptKind::InvalidValue {
                field: "thread parent ID",
                value: 0
            }
        ));
    }

    #[test]
    fn malformed_thread_cycle() {
        let c = root_with(vec![
            (key(30, ""), fthread(31, "a")),
            (key(31, ""), fthread(30, "b")),
        ]);
        let corr = corruption(c.path(30));
        assert!(matches!(
            corr.kind,
            CorruptKind::LimitExceeded {
                what: "catalog path depth",
                ..
            }
        ));
        // A thread whose parent is itself.
        let c = root_with(vec![(key(32, ""), fthread(32, "self"))]);
        let corr = corruption(c.path(32));
        assert_eq!(corr.location, Location::Object(32));
        assert_eq!(
            corr.kind,
            CorruptKind::Loop {
                what: "catalog path"
            }
        );
        // A missing ancestor.
        let c = root_with(vec![(key(33, ""), fthread(40, "orphan"))]);
        assert!(matches!(
            corruption(c.path(33)).location,
            Location::Object(40)
        ));
        // A non-root thread whose parent is the root's parent.
        let c = root_with(vec![(key(34, ""), fthread(ROOT_PARENT_ID, "x"))]);
        assert!(matches!(corruption(c.path(34)).kind, CorruptKind::Other(_)));
        // A file in the middle of a path.
        let c = root_with(vec![
            (key(35, ""), thread_data(FILE_THREAD, 2, "f")),
            (key(36, ""), fthread(35, "g")),
        ]);
        assert!(c.path(36).is_err());
    }

    #[test]
    fn malformed_record_mismatch() {
        // Thread 40 names (2, "x"), whose record has ID 41.
        let c = root_with(vec![
            (key(2, "x"), folder_data(41, 0)),
            (key(40, ""), fthread(2, "x")),
        ]);
        let corr = corruption(c.record(40));
        assert_eq!(corr.location, Location::Object(40));
        assert!(matches!(corr.kind, CorruptKind::Other(_)));
        // Kind mismatch: a file thread naming a folder record.
        let c = root_with(vec![
            (key(2, "x"), folder_data(40, 0)),
            (key(40, ""), thread_data(FILE_THREAD, 2, "x")),
        ]);
        assert!(matches!(
            corruption(c.record(40)).location,
            Location::Object(40)
        ));
        // The named record does not exist.
        let c = root_with(vec![(key(40, ""), fthread(2, "x"))]);
        assert!(matches!(
            corruption(c.record(40)).location,
            Location::Object(40)
        ));
    }

    #[test]
    fn malformed_zero_id_in_listing() {
        let c = root_with(vec![
            (key(2, "a"), file_rec(20)),
            (key(2, "b"), file_data(0, 0o100_644, 1, 0)),
            (key(2, "c"), file_rec(22)),
        ]);
        let (names, corr) = list_until_error(&c, 2);
        assert_eq!(names, ["a"]);
        assert_eq!(corr.location, Location::Record { node: 2, record: 3 });
        assert_eq!(
            corr.kind,
            CorruptKind::InvalidValue {
                field: "catalog node ID",
                value: 0
            }
        );
        let c = root_with(vec![(key(2, "a"), folder_data(0, 0))]);
        assert!(list_until_error(&c, 2).0.is_empty());
    }

    #[test]
    fn malformed_root_thread() {
        let c = catalog(&[(key(2, ""), fthread(5, "V"))], &[1]);
        assert!(c.volume_name().is_err());
        let c = catalog(&[(key(2, ""), thread_data(FILE_THREAD, 1, "V"))], &[1]);
        assert!(c.volume_name().is_err());
        let c = catalog(&[(key(3, ""), fthread(1, "V"))], &[1]);
        assert!(c.volume_name().is_err());
    }

    #[test]
    fn mode_kinds() {
        assert_eq!(kind_of_mode(0o120_777), NodeKind::Symlink);
        assert_eq!(kind_of_mode(0o100_644), NodeKind::File);
        assert_eq!(kind_of_mode(0), NodeKind::File);
        assert_eq!(kind_of_mode(0o644), NodeKind::File);
        assert_eq!(kind_of_mode(0o010_644), NodeKind::Other);
        assert_eq!(kind_of_mode(0o020_644), NodeKind::Other);
        assert_eq!(kind_of_mode(0o140_644), NodeKind::Other);
    }
}
