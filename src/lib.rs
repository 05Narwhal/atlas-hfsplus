// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! HFS+ and HFSX, read-only (big-endian).
//!
//! [`mount`] finds the volume in its source (bare or inside an HFS
//! wrapper), replays its journal in memory when macOS would, reads the
//! volume header, opens the extents overflow, catalog
//! and attributes B-trees and finds the hard link folders. [`HfsPlus`] then
//! answers the [`FileSystem`] calls from the catalog, and reads file content
//! (decompressing decmpfs files) and extended attributes through the same
//! trait.
//!
//! Damaged input is an error, never a panic: every length, offset and count
//! read from the disk is checked before use.
//!
//! ```no_run
//! use std::path::Path;
//! use std::sync::Arc;
//!
//! use atlas_fs_core::fs::{DirBatch, DirProgress, FileSystem};
//! use atlas_fs_core::io::file::FileSource;
//!
//! # fn main() -> Result<(), atlas_fs_core::error::Error> {
//! let source = Arc::new(FileSource::open(Path::new("volume.img"))?);
//! let volume = atlas_hfsplus::mount(source)?;
//! println!("{}", volume.info().name);
//!
//! let mut stream = volume.read_dir(volume.root())?;
//! let mut batch = DirBatch::new();
//! while stream.next_batch(&mut batch, 256)? == DirProgress::More {}
//! for entry in batch.iter() {
//!     println!("{}", entry.name);
//! }
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
// On-disk bytes are hostile. These lints make the usual ways of trusting
// them (unchecked indexing, arithmetic and casts, panicking shortcuts) fail
// the clippy gate. Unit tests and the test fixtures are exempt.
#![warn(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::unwrap_used
)]
#![cfg_attr(
    any(test, feature = "test-util"),
    allow(
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]

use std::ops::ControlFlow;
use std::sync::Arc;

use atlas_fs_core::error::{self, Error, Result};
use atlas_fs_core::fs::{
    AllocationMap, DirStream, EntryFlags, Extent, FileSystem, FinderCodes, Metadata, NodeId,
    NodeKind, Owner, ScanEntry, ScanSummary, Stream, VolumeIdentity, VolumeInfo, XattrInfo,
};
use atlas_fs_core::io::block::{BlockSource, Slice};

use allocation::ALLOCATION_FILE_ID;
use attributes::{Attributes, Xattr};
use btree::BTree;
use catalog::{
    BsdInfo, Catalog, CatalogRecord, FileRecord, ROOT_FOLDER_ID, entry_flags, kind_of_mode,
    resource_size_of,
};
use compression::{DECMPFS_XATTR, MAX_DECMPFS_XATTR, compressed_size, is_compressed};
use extents::{ForkData, ForkMap, ForkSource, ForkType, Geometry, fork_map};
use journal::JournalState;
use links::{LinkResolver, link_of};
use volume_header::{Located, VolumeHeader, hfs_time, locate, read_header};

pub mod allocation;
pub mod attributes;
pub mod btree;
// `be_i16` still has no caller outside tests.
#[allow(dead_code)]
#[doc(hidden)]
pub mod bytes;
pub mod catalog;
pub mod compression;
pub mod extents;
#[cfg(test)]
mod fuzz_seeds;
pub mod journal;
pub mod links;
mod scan;
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod test_volume;
pub mod unicode;
pub mod volume_header;

/// CNID of the extents overflow file.
const EXTENTS_FILE_ID: u32 = 3;
/// CNID of the catalog file.
const CATALOG_FILE_ID: u32 = 4;
/// Longest symbolic link target `metadata` reads, in bytes.
const MAX_LINK_TARGET: u64 = 1024;

/// A mounted HFS+ or HFSX volume.
pub struct HfsPlus {
    /// The HFS+ volume itself: the source, or a slice of it when wrapped.
    volume: Arc<dyn BlockSource>,
    geometry: Geometry,
    /// The extents overflow B-tree, for mapping forks past eight extents.
    extents: BTree,
    catalog: Catalog,
    /// The attributes B-tree; `None` when the volume has no attributes file.
    attributes: Option<Attributes>,
    links: LinkResolver,
    info: VolumeInfo,
    journal: JournalState,
    /// The allocation file's fork descriptor, from the header as replayed.
    /// Mapped only by `allocation_map`, so a damaged one fails that alone.
    allocation_file: ForkData,
    /// Byte offset of the volume in the mount source: 0 unless wrapped.
    volume_offset: u64,
    /// From the header as replayed.
    identity: VolumeIdentity,
}

// `FileSystem` requires it; this names the failure here if a field breaks it.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HfsPlus>();
};

/// Mounts the HFS+ or HFSX volume in `source`: the whole source, or the
/// volume embedded in a classic HFS wrapper.
///
/// Fails when the volume header, a B-tree header (extents, catalog, or an
/// attributes file that is present), the catalog's name order or the root
/// folder's thread record is damaged.
pub fn mount(source: Arc<dyn BlockSource>) -> Result<HfsPlus> {
    let located = locate(&*source)?;
    let raw_header = read_header(&*volume_in(Arc::clone(&source), &located)?)?;
    // The journal overlay works in device coordinates (journal block
    // numbers count from the partition's start), so it goes under the
    // volume slice. The header is read again through it: a replayed
    // journal usually carries a newer one.
    let opened = journal::open(source, located.offset, &raw_header)?;
    let volume = volume_in(opened.device, &located)?;
    let header = read_header(&*volume)?;
    let geometry = Geometry {
        block_size: header.block_size,
        total_blocks: header.total_blocks,
    };

    let extents_map = fork_map(
        geometry,
        EXTENTS_FILE_ID,
        ForkType::Data,
        &header.extents_file,
        None,
    )?;
    let extents = BTree::open(
        Arc::new(ForkSource::new(Arc::clone(&volume), Arc::new(extents_map))),
        "extents overflow B-tree",
    )?;
    let catalog_map = fork_map(
        geometry,
        CATALOG_FILE_ID,
        ForkType::Data,
        &header.catalog_file,
        Some(&extents),
    )?;
    let catalog_tree = BTree::open(
        Arc::new(ForkSource::new(Arc::clone(&volume), Arc::new(catalog_map))),
        "catalog B-tree",
    )?;
    let catalog = Catalog::new(catalog_tree, header.is_hfsx())?;
    let attributes = Attributes::open(
        Arc::clone(&volume),
        geometry,
        &header.attributes_file,
        &extents,
    )?;
    let name = catalog.volume_name()?;
    let links = LinkResolver::new(catalog.clone())?;

    let block_size = u64::from(header.block_size);
    let mut info = VolumeInfo::new(name);
    info.format = Some(if header.is_hfsx() { "HFSX" } else { "HFS+" });
    info.total_bytes = u64::from(header.total_blocks).checked_mul(block_size);
    info.free_bytes = u64::from(header.free_blocks).checked_mul(block_size);
    info.resource_forks = true;
    info.journal = Some(opened.state.clone());

    Ok(HfsPlus {
        volume,
        geometry,
        extents,
        catalog,
        attributes,
        links,
        info,
        journal: opened.state,
        allocation_file: header.allocation_file,
        volume_offset: located.offset,
        identity: identity_of(&header),
    })
}

/// The identity of the volume `header` describes.
///
/// `id`: the 64-bit volume identifier (`finderInfo[6]` and `[7]`), the
/// creation date and the block size; a volume whose identifier is zero is
/// still told apart by the last two. `change`: the write count, the
/// modification date and the file and folder counts. All big-endian.
fn identity_of(header: &VolumeHeader) -> VolumeIdentity {
    let [.., id_high, id_low] = header.finder_info;
    let put = |words: [u32; 4]| {
        let mut out = [0u8; 16];
        for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *bytes = word.to_be_bytes();
        }
        out
    };
    VolumeIdentity {
        id: put([id_high, id_low, header.create_date, header.block_size]),
        change: put([
            header.write_count,
            header.modify_date,
            header.file_count,
            header.folder_count,
        ]),
    }
}

/// True when a listing of `parent` leaves out its child `id`
/// ([`LinkResolver::hidden_in_listing`], which `CatalogDir` also applies).
fn hidden_in_listing(links: &LinkResolver, parent: u32, id: u32) -> bool {
    links.hidden_in_listing(parent, id)
}

/// The HFS+ volume `located` names in `device`: the device itself, or a
/// slice of it when wrapped.
fn volume_in(device: Arc<dyn BlockSource>, located: &Located) -> Result<Arc<dyn BlockSource>> {
    Ok(if located.offset == 0 && located.len == device.len() {
        device
    } else {
        Arc::new(Slice::new(device, located.offset, located.len)?)
    })
}

/// The CNID a node ID names. 0 and IDs past `u32` are `NotFound`.
fn cnid(node: NodeId) -> Result<u32> {
    match u32::try_from(node.0) {
        Ok(id) if id != 0 => Ok(id),
        _ => Err(Error::NotFound { node: node.0 }),
    }
}

impl HfsPlus {
    /// What mounting did with the volume's journal: nothing to do, replayed
    /// in memory, or left alone because it could not be replayed (the volume
    /// is then read as it is on disk).
    pub fn journal(&self) -> &JournalState {
        &self.journal
    }

    /// The target stored in a symbolic link's data fork, when it is 1 to
    /// [`MAX_LINK_TARGET`] bytes and can be read. Never fails: a target that
    /// cannot be read is simply not reported.
    fn link_target(&self, file: &FileRecord) -> Option<String> {
        let len = file.data_fork.logical_size;
        if !(1..=MAX_LINK_TARGET).contains(&len) {
            return None;
        }
        let map = fork_map(
            self.geometry,
            file.file_id,
            ForkType::Data,
            &file.data_fork,
            Some(&self.extents),
        )
        .ok()?;
        let fork = ForkSource::new(Arc::clone(&self.volume), Arc::new(map));
        let mut buf = vec![0u8; usize::try_from(len).ok()?];
        fork.read_at(0, &mut buf).ok()?;
        Some(String::from_utf8_lossy(&buf).into_owned())
    }

    /// A fork of `file` as a block source.
    fn fork(&self, file: &FileRecord, fork_type: ForkType) -> Result<Arc<dyn BlockSource>> {
        let fork = match fork_type {
            ForkType::Data => &file.data_fork,
            ForkType::Resource => &file.resource_fork,
        };
        let map = fork_map(
            self.geometry,
            file.file_id,
            fork_type,
            fork,
            Some(&self.extents),
        )?;
        Ok(Arc::new(ForkSource::new(
            Arc::clone(&self.volume),
            Arc::new(map),
        )))
    }

    /// The size a listing reports for `file`: the decmpfs size of a
    /// compressed file (`None` when its header cannot be read), else the
    /// data fork's logical size.
    fn file_size(&self, file: &FileRecord) -> Option<u64> {
        if !is_compressed(file) {
            return Some(file.data_fork.logical_size);
        }
        let attributes = self.attributes.as_ref()?;
        compressed_size(attributes, file.file_id, &mut Vec::with_capacity(16))
    }

    /// The record `node` describes, after hard links.
    fn shown(&self, node: NodeId) -> Result<Shown> {
        let id = cnid(node)?;
        let record = self
            .catalog
            .record(id)?
            .ok_or(Error::NotFound { node: node.0 })?;
        Ok(self.through_links(record))
    }

    /// The decompressed content of a compressed file.
    fn open_compressed(&self, file: &FileRecord) -> Result<Arc<dyn BlockSource>> {
        let missing = || {
            Error::Corrupt(error::Corruption::new(
                "decmpfs",
                error::Location::Object(u64::from(file.file_id)),
                error::CorruptKind::Other("compressed file has no com.apple.decmpfs attribute"),
            ))
        };
        let attributes = self.attributes.as_ref().ok_or_else(missing)?;
        let xattr = attributes
            .get(file.file_id, DECMPFS_XATTR, MAX_DECMPFS_XATTR)?
            .ok_or_else(missing)?;
        compression::open(&xattr, file.file_id, &mut || {
            self.fork(file, ForkType::Resource)
        })
    }

    /// The CNID whose attributes `node` has (the hard link's target, when it
    /// resolves) and whether that is a compressed file.
    fn xattr_owner(&self, node: NodeId) -> Result<(u32, bool)> {
        let shown = self.shown(node)?;
        let compressed =
            matches!(&shown, Shown::Record(CatalogRecord::File(f)) if is_compressed(f));
        let id = match shown {
            Shown::Record(record) => record.id(),
            Shown::UnresolvedLink(link) => link.file_id,
        };
        Ok((id, compressed))
    }

    /// What a listing shows for `record`: a hard link's target, or the link
    /// itself when it cannot be resolved. Any other record is itself.
    fn through_links(&self, record: CatalogRecord) -> Shown {
        let CatalogRecord::File(file) = record else {
            return Shown::Record(record);
        };
        let Some(link) = link_of(&file) else {
            return Shown::Record(record);
        };
        // As in a listing, a target that cannot be read counts as missing,
        // so `lookup` and `metadata` agree with the listing's node IDs.
        match self.links.resolve(link) {
            Ok(Some(target)) => Shown::Record(target),
            Ok(None) | Err(_) => Shown::UnresolvedLink(file),
        }
    }
}

/// The type and creator codes of `file`, as stored; `None` when both are 0.
fn finder_codes_of(file: &FileRecord) -> Option<FinderCodes> {
    (file.file_type != 0 || file.creator != 0).then(|| FinderCodes {
        file_type: file.file_type.to_be_bytes(),
        creator: file.creator.to_be_bytes(),
    })
}

/// The 32 bytes of Finder information as stored; `None` when all are zero.
fn finder_info_of(bytes: [u8; 32]) -> Option<[u8; 32]> {
    bytes.iter().any(|&b| b != 0).then_some(bytes)
}

/// The owner and mode of a node; `None` when the volume recorded no mode
/// (Mac OS 9 wrote zeros).
fn owner_of(bsd: &BsdInfo) -> Option<Owner> {
    (bsd.file_mode != 0).then_some(Owner {
        user: bsd.owner_id,
        group: bsd.group_id,
        mode: bsd.file_mode,
    })
}

/// The runs of `map` as bytes of the mount source (the volume starts at
/// `volume_offset`), in fork order, each merged into the one before when
/// they touch on disk. Whole allocation blocks, past the logical size too.
fn disk_runs(
    map: &ForkMap,
    file_id: u32,
    block_size: u32,
    volume_offset: u64,
) -> Result<Vec<Extent>> {
    let overflow = || {
        Error::Corrupt(error::Corruption::new(
            "fork",
            error::Location::Object(u64::from(file_id)),
            error::CorruptKind::Overflow {
                field: "extent byte offset",
            },
        ))
    };
    let bs = u64::from(block_size);
    let runs = map.runs();
    let mut out: Vec<Extent> = Vec::with_capacity(runs.size_hint().0);
    // Bounded by the fork map's run count.
    for (_, run) in runs {
        let len = u64::from(run.block_count)
            .checked_mul(bs)
            .ok_or_else(overflow)?;
        if len == 0 {
            continue;
        }
        let offset = u64::from(run.start_block)
            .checked_mul(bs)
            .and_then(|b| b.checked_add(volume_offset))
            .ok_or_else(overflow)?;
        if let Some(prev) = out.last_mut()
            && prev.offset.checked_add(prev.len) == Some(offset)
        {
            prev.len = prev.len.checked_add(len).ok_or_else(overflow)?;
            continue;
        }
        out.push(Extent { offset, len });
    }
    Ok(out)
}

/// The record that describes a node, after hard links.
enum Shown {
    /// The node's own record, or the target of the hard link it is.
    Record(CatalogRecord),
    /// A hard link whose target cannot be found: the link's own record.
    UnresolvedLink(FileRecord),
}

impl FileSystem for HfsPlus {
    fn info(&self) -> VolumeInfo {
        self.info.clone()
    }

    fn root(&self) -> NodeId {
        NodeId(u64::from(ROOT_FOLDER_ID))
    }

    fn metadata(&self, node: NodeId) -> Result<Metadata> {
        let id = cnid(node)?;
        let Some((thread, record)) = self.catalog.record_with_thread(id)? else {
            return Err(Error::NotFound { node: node.0 });
        };
        // The flags are the entry's own (its name and Finder flags), as in
        // a listing, even when a hard link describes its target.
        let flags = entry_flags(thread.name.first() == Some(&0x2E), record.finder_flags());
        let mut meta = Metadata {
            node,
            kind: NodeKind::File,
            size: None,
            created: None,
            modified: None,
            accessed: None,
            flags,
            link_target: None,
            resource_size: None,
            finder_codes: None,
            owner: None,
            attributes: None,
            finder_info: None,
        };
        let shown = self.through_links(record);
        // The attribute count belongs to the shown record, as `xattrs` does.
        let compressed =
            matches!(&shown, Shown::Record(CatalogRecord::File(f)) if is_compressed(f));
        let id = match &shown {
            Shown::Record(record) => record.id(),
            Shown::UnresolvedLink(link) => link.file_id,
        };
        if let Some(attributes) = &self.attributes {
            // A list that cannot be read leaves the count unknown; it does
            // not fail the call.
            meta.attributes = attributes.list(id).ok().and_then(|list| {
                let n = list
                    .iter()
                    .filter(|x| !(compressed && x.name == DECMPFS_XATTR))
                    .count();
                u32::try_from(n).ok()
            });
        }
        match shown {
            Shown::Record(CatalogRecord::Folder(f)) => {
                meta.kind = NodeKind::Directory;
                meta.finder_info = finder_info_of(f.finder_info);
                meta.owner = owner_of(&f.bsd);
                meta.created = hfs_time(f.dates.created);
                meta.modified = hfs_time(f.dates.content_modified);
                meta.accessed = hfs_time(f.dates.accessed);
            }
            Shown::Record(CatalogRecord::File(f)) => {
                meta.kind = kind_of_mode(f.bsd.file_mode);
                meta.size = self.file_size(&f);
                meta.resource_size = Some(resource_size_of(&f));
                meta.finder_codes = finder_codes_of(&f);
                meta.finder_info = finder_info_of(f.finder_info);
                meta.owner = owner_of(&f.bsd);
                if compressed {
                    meta.flags = meta.flags.with(EntryFlags::COMPRESSED);
                }
                meta.created = hfs_time(f.dates.created);
                meta.modified = hfs_time(f.dates.content_modified);
                meta.accessed = hfs_time(f.dates.accessed);
                if meta.kind == NodeKind::Symlink {
                    meta.link_target = self.link_target(&f);
                }
            }
            // An unresolvable hard link: the link file itself, size unknown.
            Shown::UnresolvedLink(link) => {
                meta.finder_codes = finder_codes_of(&link);
                meta.finder_info = finder_info_of(link.finder_info);
                meta.owner = owner_of(&link.bsd);
                meta.created = hfs_time(link.dates.created);
                meta.modified = hfs_time(link.dates.content_modified);
                meta.accessed = hfs_time(link.dates.accessed);
            }
        }
        Ok(meta)
    }

    fn lookup(&self, dir: NodeId, name: &str) -> Result<Option<NodeId>> {
        let dir_id = cnid(dir)?;
        let Some(record) = self.catalog.lookup(dir_id, name)? else {
            return Ok(None);
        };
        if hidden_in_listing(&self.links, dir_id, record.id()) {
            return Ok(None);
        }
        let id = match self.through_links(record) {
            Shown::Record(target) => target.id(),
            Shown::UnresolvedLink(link) => link.file_id,
        };
        Ok(Some(NodeId(u64::from(id))))
    }

    fn read_dir(&self, dir: NodeId) -> Result<Box<dyn DirStream>> {
        let listing =
            self.catalog
                .list_resolving(cnid(dir)?, &self.links, self.attributes.as_ref())?;
        Ok(Box::new(listing))
    }

    /// The allocated blocks of a fork: the data fork, or the resource fork
    /// for `Stream::Resource` and for `Stream::Data` of a compressed file
    /// (where decmpfs keeps the chunks; inline compression has none). Hard
    /// links report their target, an unresolvable one its own forks.
    fn extents(&self, node: NodeId, stream: Stream) -> Result<Vec<Extent>> {
        let file = match self.shown(node)? {
            Shown::Record(CatalogRecord::Folder(_)) => {
                return Err(Error::IsADirectory { node: node.0 });
            }
            Shown::Record(CatalogRecord::File(file)) | Shown::UnresolvedLink(file) => file,
        };
        let (fork_type, fork) = match stream {
            Stream::Data if !is_compressed(&file) => (ForkType::Data, &file.data_fork),
            Stream::Data | Stream::Resource => (ForkType::Resource, &file.resource_fork),
        };
        let map = fork_map(
            self.geometry,
            file.file_id,
            fork_type,
            fork,
            Some(&self.extents),
        )?;
        disk_runs(
            &map,
            file.file_id,
            self.geometry.block_size,
            self.volume_offset,
        )
    }

    /// The content of `stream` of `node`, following file hard links to their
    /// target: the data fork of a plain file, the decompressed content of a
    /// compressed one (never its resource fork as such). A symbolic link
    /// gives its data fork, which holds the target path. An unresolvable
    /// hard link gives its own (normally empty) data fork, as it is listed
    /// as a file.
    ///
    /// `Stream::Resource` is the resource fork of a plain file. A compressed
    /// file and a file whose fork is empty have none (`Error::NoSuchStream`).
    ///
    /// Fails with `Error::IsADirectory` for a folder (or a directory hard
    /// link), `Error::NotFound` for a missing node, `Error::Unsupported` for
    /// a compression type this reader does not decode, and `Corrupt` for a
    /// compressed file whose decmpfs attribute is missing or damaged.
    fn open(&self, node: NodeId, stream: Stream) -> Result<Arc<dyn BlockSource>> {
        // An unresolvable hard link opens its own forks and never counts as
        // compressed; only a resolved record does.
        let (file, compressed) = match self.shown(node)? {
            Shown::Record(CatalogRecord::Folder(_)) => {
                return Err(Error::IsADirectory { node: node.0 });
            }
            Shown::Record(CatalogRecord::File(file)) => {
                let compressed = is_compressed(&file);
                (file, compressed)
            }
            Shown::UnresolvedLink(file) => (file, false),
        };
        match stream {
            Stream::Data if compressed => self.open_compressed(&file),
            Stream::Data => self.fork(&file, ForkType::Data),
            Stream::Resource => {
                if compressed || file.resource_fork.logical_size == 0 {
                    return Err(Error::NoSuchStream { node: node.0 });
                }
                self.fork(&file, ForkType::Resource)
            }
        }
    }

    /// The extended attributes of `node` (of the target, for a hard link):
    /// names and value sizes, in the attributes B-tree's order. On a
    /// compressed file `com.apple.decmpfs` is left out, as macOS does: it
    /// is how the filesystem stores the content, not an attribute the user
    /// set. Empty when the volume has no attributes file.
    fn xattrs(&self, node: NodeId) -> Result<Vec<XattrInfo>> {
        let (id, compressed) = self.xattr_owner(node)?;
        let Some(attributes) = &self.attributes else {
            return Ok(Vec::new());
        };
        let mut list = attributes.list(id)?;
        if compressed {
            list.retain(|x| x.name != DECMPFS_XATTR);
        }
        Ok(list
            .into_iter()
            .map(|Xattr { name, size }| XattrInfo { name, size })
            .collect())
    }

    /// The whole value of attribute `name` of `node` (of the target, for a
    /// hard link). `com.apple.decmpfs` of a compressed file is not there, as
    /// in [`Self::xattrs`]. At most `max` bytes are read or allocated.
    fn read_xattr(&self, node: NodeId, name: &str, max: usize) -> Result<Option<Vec<u8>>> {
        let (id, compressed) = self.xattr_owner(node)?;
        if compressed && name == DECMPFS_XATTR {
            return Ok(None);
        }
        let Some(attributes) = &self.attributes else {
            return Ok(None);
        };
        let mut out = Vec::new();
        match attributes.read_head(id, name, max, &mut out)? {
            None => Ok(None),
            Some(full) if full > u64::try_from(max).unwrap_or(u64::MAX) => Err(Error::Unsupported(
                "extended attribute longer than the limit".into(),
            )),
            Some(_) => Ok(Some(out)),
        }
    }

    /// Reads the allocation file through the journal overlay and counts its
    /// set bits per bucket (see [`allocation::density`]).
    fn allocation_map(&self, buckets: usize) -> Result<AllocationMap> {
        let Geometry {
            block_size,
            total_blocks,
        } = self.geometry;
        let len = u64::from(total_blocks)
            .checked_mul(u64::from(block_size))
            .ok_or_else(|| {
                Error::Corrupt(error::Corruption::new(
                    "volume header",
                    error::Location::Unspecified,
                    error::CorruptKind::Overflow {
                        field: "volume bytes",
                    },
                ))
            })?;
        let map = fork_map(
            self.geometry,
            ALLOCATION_FILE_ID,
            ForkType::Data,
            &self.allocation_file,
            Some(&self.extents),
        )?;
        let bitmap = ForkSource::new(Arc::clone(&self.volume), Arc::new(map));
        let density = allocation::density(&bitmap, total_blocks, buckets)?;
        Ok(AllocationMap::new(self.volume_offset, len, density))
    }

    /// Reads the attributes and catalog B-trees front to back, each once
    /// (see the `scan` module).
    fn scan(&self, visit: &mut dyn FnMut(ScanEntry<'_>) -> ControlFlow<()>) -> Result<ScanSummary> {
        scan::scan(self, visit)
    }

    fn identity(&self) -> Option<VolumeIdentity> {
        Some(self.identity)
    }
}

#[cfg(test)]
mod tests {
    use super::links::{DIR_PRIVATE_FOLDER, FILE_PRIVATE_FOLDER};
    use super::test_volume::{
        BLOCK, CREATED, FileSpec, JournalSetup, MIN_JOURNAL, MODIFIED, ROOT, TestJournal, TestList,
        VolumeBuilder, XattrValue, changed_blocks, source, wrap,
    };
    use super::*;
    use atlas_fs_core::error::{CorruptKind, Corruption};
    use atlas_fs_core::fs::{DirBatch, DirProgress, EntryFlags, Timestamp};

    // ---- helpers ----

    #[derive(Debug, Clone, PartialEq)]
    struct Entry {
        node: u64,
        name: String,
        kind: NodeKind,
        size: Option<u64>,
        modified: Option<Timestamp>,
        hidden: bool,
    }

    fn e(node: u64, name: &str, kind: NodeKind, size: Option<u64>, modified: u32) -> Entry {
        Entry {
            node,
            name: name.to_string(),
            kind,
            size,
            modified: hfs_time(modified),
            hidden: false,
        }
    }

    fn hidden(entry: Entry) -> Entry {
        Entry {
            hidden: true,
            ..entry
        }
    }

    fn list(fs: &dyn FileSystem, dir: u64) -> Vec<Entry> {
        let mut stream = fs.read_dir(NodeId(dir)).unwrap();
        let mut batch = DirBatch::new();
        let mut calls = 0;
        // Small pages, so link resolution runs across batch boundaries.
        while stream.next_batch(&mut batch, 2).unwrap() == DirProgress::More {
            calls += 1;
            assert!(calls < 1000);
        }
        batch
            .iter()
            .map(|d| Entry {
                node: d.node.0,
                name: d.name.to_string(),
                kind: d.kind,
                size: d.size,
                modified: d.modified,
                hidden: d.flags.contains(EntryFlags::HIDDEN),
            })
            .collect()
    }

    fn mounted(builder: &VolumeBuilder) -> HfsPlus {
        mount(source(builder.build())).unwrap()
    }

    /// The volume's facts without its journal outcome, which differs
    /// between a replayed mount and a clean one.
    fn volume_only(fs: &HfsPlus) -> VolumeInfo {
        VolumeInfo {
            journal: None,
            ..fs.info()
        }
    }

    fn lookup(fs: &HfsPlus, dir: u32, name: &str) -> Option<u64> {
        fs.lookup(NodeId(u64::from(dir)), name)
            .unwrap()
            .map(|n| n.0)
    }

    fn corruption<T>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            Err(e) => panic!("expected Corrupt, got {e:?}"),
            Ok(_) => panic!("expected Corrupt, got Ok"),
        }
    }

    const INODE_MODIFIED: u32 = MODIFIED + 100;
    const DIR_MODIFIED: u32 = MODIFIED;

    /// A root with ordinary entries, both private folders, links that
    /// resolve and links that do not, and symbolic links.
    fn main_volume() -> VolumeBuilder {
        let mut v = VolumeBuilder::new("Test Volume");
        v.folder(ROOT, "Applications", 16)
            .file(ROOT, "readme.txt", 17, FileSpec::regular(100))
            .file(ROOT, ".DS_Store", 18, FileSpec::regular(6))
            .file(ROOT, "Invisible", 19, FileSpec::regular(7).invisible())
            .folder(ROOT, "Docs", 20)
            .file(20, "a.txt", 21, FileSpec::regular(1))
            // File link targets.
            .folder_units(ROOT, &FILE_PRIVATE_FOLDER, 30, 0x5000)
            .file(
                30,
                "iNode100",
                31,
                FileSpec::regular(555).modified(INODE_MODIFIED),
            )
            .folder(30, "iNode101", 32)
            .file(30, "iNode102", 33, FileSpec::file_link(100))
            .file(30, "iNode103", 34, FileSpec::symlink(b"../target"))
            // Directory link targets.
            .folder_units(ROOT, &DIR_PRIVATE_FOLDER, 40, 0x5000)
            .folder(40, "dir_200", 41)
            .file(41, "inside.txt", 42, FileSpec::regular(9))
            .file(40, "dir_201", 43, FileSpec::regular(3))
            // Links.
            .file(ROOT, "hard.txt", 50, FileSpec::file_link(100))
            .file(ROOT, "hard dir", 51, FileSpec::dir_link(200))
            .file(ROOT, "missing", 52, FileSpec::file_link(999))
            .file(ROOT, "to folder", 53, FileSpec::file_link(101))
            .file(ROOT, "chain", 54, FileSpec::file_link(102))
            .file(ROOT, "dir to file", 55, FileSpec::dir_link(201))
            .file(ROOT, ".hidden link", 56, FileSpec::file_link(100))
            .file(
                ROOT,
                "invisible link",
                57,
                FileSpec::file_link(100).invisible(),
            )
            .file(ROOT, "link to symlink", 58, FileSpec::file_link(103))
            .file(20, "hard in docs", 59, FileSpec::file_link(100))
            // Symbolic links.
            .file(ROOT, "sym", 60, FileSpec::symlink(b"/usr/bin/env"))
            .file(ROOT, "sym long", 61, FileSpec::symlink(&[b'x'; 1025]))
            .file(ROOT, "sym max", 62, FileSpec::symlink(&[b'y'; 1024]))
            .file(ROOT, "sym empty", 63, FileSpec::symlink(b""))
            .file(ROOT, "sym bad utf8", 64, FileSpec::symlink(b"a\xFFb"));
        v
    }

    // ---- mount and info ----

    #[test]
    fn mount_info_and_root() {
        let bytes = main_volume().build();
        let total = bytes.len() as u64;
        let fs = mount(source(bytes)).unwrap();
        let info = fs.info();
        assert_eq!(info.name, "Test Volume");
        assert_eq!(info.format, Some("HFS+"));
        assert_eq!(info.total_bytes, Some(total));
        assert_eq!(info.free_bytes, Some(BLOCK as u64));
        assert_eq!(fs.root(), NodeId(2));
        assert_eq!(fs.journal(), &JournalState::Clean);
        assert!(info.resource_forks);
        assert_eq!(info.journal, Some(JournalState::Clean));
        assert_eq!(fs.links.file_folder(), Some(30));
        assert_eq!(fs.links.dir_folder(), Some(40));

        let root = fs.metadata(fs.root()).unwrap();
        assert_eq!(root.kind, NodeKind::Directory);
        assert_eq!(root.size, None);
        assert_eq!(root.created, hfs_time(CREATED));
    }

    #[test]
    fn root_listing() {
        let fs = mounted(&main_volume());
        let file = NodeKind::File;
        let link_mod = MODIFIED + 7;
        let want = vec![
            hidden(e(18, ".DS_Store", file, Some(6), MODIFIED)),
            hidden(e(31, ".hidden link", file, Some(555), INODE_MODIFIED)),
            e(16, "Applications", NodeKind::Directory, None, MODIFIED),
            e(54, "chain", file, None, 0),
            e(55, "dir to file", file, None, 0),
            e(20, "Docs", NodeKind::Directory, None, MODIFIED),
            e(41, "hard dir", NodeKind::Directory, None, DIR_MODIFIED),
            e(31, "hard.txt", file, Some(555), INODE_MODIFIED),
            hidden(e(19, "Invisible", file, Some(7), MODIFIED)),
            hidden(e(31, "invisible link", file, Some(555), INODE_MODIFIED)),
            e(34, "link to symlink", NodeKind::Symlink, Some(9), MODIFIED),
            e(52, "missing", file, None, 0),
            e(17, "readme.txt", file, Some(100), MODIFIED),
            e(60, "sym", NodeKind::Symlink, Some(12), MODIFIED),
            e(64, "sym bad utf8", NodeKind::Symlink, Some(3), MODIFIED),
            e(63, "sym empty", NodeKind::Symlink, Some(0), MODIFIED),
            e(61, "sym long", NodeKind::Symlink, Some(1025), MODIFIED),
            e(62, "sym max", NodeKind::Symlink, Some(1024), MODIFIED),
            e(53, "to folder", file, None, 0),
        ];
        let got = list(&fs, 2);
        assert_eq!(got, want);
        // The link's own date is not used for a resolved link.
        assert!(got.iter().all(|x| x.modified != hfs_time(link_mod)));
    }

    #[test]
    fn listing_resolves_links_outside_the_root() {
        let fs = mounted(&main_volume());
        assert_eq!(
            list(&fs, 20),
            vec![
                e(21, "a.txt", NodeKind::File, Some(1), MODIFIED),
                e(
                    31,
                    "hard in docs",
                    NodeKind::File,
                    Some(555),
                    INODE_MODIFIED
                ),
            ]
        );
        // The private folders themselves can be listed by ID.
        let names: Vec<String> = list(&fs, 30).into_iter().map(|x| x.name).collect();
        assert_eq!(names, ["iNode100", "iNode101", "iNode102", "iNode103"]);
    }

    // ---- hard links ----

    #[test]
    fn file_hard_link() {
        let fs = mounted(&main_volume());
        assert_eq!(lookup(&fs, ROOT, "hard.txt"), Some(31));
        assert_eq!(lookup(&fs, ROOT, "HARD.TXT"), Some(31));
        assert_eq!(lookup(&fs, 20, "hard in docs"), Some(31));

        // The link's own ID describes the target, keeping the node asked for.
        let m = fs.metadata(NodeId(50)).unwrap();
        assert_eq!(m.node, NodeId(50));
        assert_eq!(m.kind, NodeKind::File);
        assert_eq!(m.size, Some(555));
        assert_eq!(m.modified, hfs_time(INODE_MODIFIED));
        assert_eq!(m.created, hfs_time(CREATED));
        assert_eq!(m.flags, EntryFlags::EMPTY);
        assert_eq!(m.link_target, None);
        // The flags stay the link's own.
        assert_eq!(fs.metadata(NodeId(56)).unwrap().flags, EntryFlags::HIDDEN);
        assert_eq!(fs.metadata(NodeId(57)).unwrap().flags, EntryFlags::HIDDEN);
        // The target ID describes the same file.
        let t = fs.metadata(NodeId(31)).unwrap();
        assert_eq!((t.kind, t.size, t.modified), (m.kind, m.size, m.modified));
    }

    #[test]
    fn hard_link_to_a_symlink() {
        let fs = mounted(&main_volume());
        assert_eq!(lookup(&fs, ROOT, "link to symlink"), Some(34));
        let m = fs.metadata(NodeId(58)).unwrap();
        assert_eq!(m.kind, NodeKind::Symlink);
        assert_eq!(m.link_target.as_deref(), Some("../target"));
    }

    #[test]
    fn directory_hard_link() {
        let fs = mounted(&main_volume());
        assert_eq!(lookup(&fs, ROOT, "hard dir"), Some(41));
        let m = fs.metadata(NodeId(51)).unwrap();
        assert_eq!(m.kind, NodeKind::Directory);
        assert_eq!(m.size, None);
        assert_eq!(
            list(&fs, 41),
            vec![e(42, "inside.txt", NodeKind::File, Some(9), MODIFIED)]
        );
        assert_eq!(lookup(&fs, 41, "inside.txt"), Some(42));
        // The link file itself is not a folder.
        assert!(matches!(
            fs.read_dir(NodeId(51)),
            Err(Error::NotADirectory { node: 51 })
        ));
    }

    #[test]
    fn private_folders_are_hidden_from_the_root() {
        let fs = mounted(&main_volume());
        let names: Vec<String> = list(&fs, 2).into_iter().map(|x| x.name).collect();
        assert!(!names.iter().any(|n| n.contains("Private")));
        let file_folder = String::from_utf16(&FILE_PRIVATE_FOLDER).unwrap();
        let dir_folder = String::from_utf16(&DIR_PRIVATE_FOLDER).unwrap();
        assert_eq!(lookup(&fs, ROOT, &file_folder), None);
        assert_eq!(lookup(&fs, ROOT, &dir_folder), None);
        // A folder of the same name elsewhere is an ordinary folder.
        let mut v = main_volume();
        v.folder_units(20, &FILE_PRIVATE_FOLDER, 70, 0);
        let fs = mounted(&v);
        assert_eq!(lookup(&fs, 20, &file_folder), Some(70));
        assert!(list(&fs, 20).iter().any(|x| x.node == 70));
    }

    #[test]
    fn unresolvable_links() {
        let fs = mounted(&main_volume());
        // Missing iNode, an iNode that is a folder, a link to a link, and a
        // directory link whose target is a file.
        for (name, id) in [
            ("missing", 52),
            ("to folder", 53),
            ("chain", 54),
            ("dir to file", 55),
        ] {
            assert_eq!(lookup(&fs, ROOT, name), Some(id), "{name}");
            let m = fs.metadata(NodeId(id)).unwrap();
            assert_eq!(m.kind, NodeKind::File, "{name}");
            assert_eq!(m.size, None, "{name}");
            assert_eq!(m.modified, hfs_time(MODIFIED + 7), "{name}");
            assert!(matches!(
                fs.read_dir(NodeId(id)),
                Err(Error::NotADirectory { .. })
            ));
        }
    }

    #[test]
    fn links_without_private_folders() {
        let mut v = VolumeBuilder::new("Plain");
        v.file(ROOT, "file link", 16, FileSpec::file_link(1))
            .file(ROOT, "dir link", 17, FileSpec::dir_link(2))
            .file(ROOT, "plain", 18, FileSpec::regular(4));
        let fs = mounted(&v);
        assert_eq!(fs.links.file_folder(), None);
        assert_eq!(fs.links.dir_folder(), None);
        assert_eq!(
            list(&fs, 2),
            vec![
                e(17, "dir link", NodeKind::File, None, 0),
                e(16, "file link", NodeKind::File, None, 0),
                e(18, "plain", NodeKind::File, Some(4), MODIFIED),
            ]
        );
        assert_eq!(lookup(&fs, ROOT, "file link"), Some(16));
    }

    #[test]
    fn private_folder_name_on_a_file_is_not_a_private_folder() {
        let mut v = VolumeBuilder::new("Odd");
        v.file_units(ROOT, &FILE_PRIVATE_FOLDER, 16, FileSpec::regular(1))
            .file(ROOT, "link", 17, FileSpec::file_link(1));
        let fs = mounted(&v);
        assert_eq!(fs.links.file_folder(), None);
        let got = list(&fs, 2);
        assert_eq!(got.len(), 2);
        // NUL folds to 0xFFFF, so the odd file sorts last.
        assert_eq!(got[0], e(17, "link", NodeKind::File, None, 0));
        assert_eq!(got[1].node, 16);
    }

    // ---- resource forks, Finder codes, owner, attributes ----

    const FORK: &[u8] = b"0123456789";

    fn code(c: &[u8; 4]) -> u32 {
        u32::from_be_bytes(*c)
    }

    /// (name, resource size, compressed) of each entry of `dir`.
    fn forks(fs: &HfsPlus, dir: u64) -> Vec<(String, Option<u64>, bool)> {
        let mut stream = fs.read_dir(NodeId(dir)).unwrap();
        let mut batch = DirBatch::new();
        while stream.next_batch(&mut batch, 2).unwrap() == DirProgress::More {}
        batch
            .iter()
            .map(|d| {
                (
                    d.name.to_string(),
                    d.resource_size,
                    d.flags.contains(EntryFlags::COMPRESSED),
                )
            })
            .collect()
    }

    fn facts_volume() -> VolumeBuilder {
        let mut v = VolumeBuilder::new("Facts");
        v.file(ROOT, "forked", 16, FileSpec::regular(4))
            .resource_fork(16, FORK)
            .xattr(16, "user.a", XattrValue::Inline(b"1".to_vec()))
            .xattr(16, "user.b", XattrValue::Inline(b"22".to_vec()))
            .file(ROOT, "plain", 17, FileSpec::regular(3))
            .folder(ROOT, "dir", 18)
            .file(
                ROOT,
                "coded",
                19,
                FileSpec {
                    file_type: code(b"PNGf"),
                    creator: code(b"prvw"),
                    mode: 0o100_640,
                    group: 80,
                    ..FileSpec::regular(1)
                },
            )
            .file(
                ROOT,
                "bare",
                20,
                FileSpec {
                    file_type: 0,
                    creator: 0,
                    mode: 0,
                    ..FileSpec::regular(1)
                },
            )
            .file(
                ROOT,
                "half",
                21,
                FileSpec {
                    file_type: 0,
                    creator: code(b"abcd"),
                    ..FileSpec::regular(1)
                },
            )
            .folder_units(ROOT, &FILE_PRIVATE_FOLDER, 30, 0x5000)
            .file(30, "iNode100", 31, FileSpec::regular(555))
            .resource_fork(31, b"abc")
            .file(ROOT, "hard", 50, FileSpec::file_link(100))
            .file(ROOT, "dangling", 52, FileSpec::file_link(999));
        v
    }

    #[test]
    fn resource_sizes_in_listing_and_metadata() {
        let fs = mounted(&facts_volume());
        let got = forks(&fs, 2);
        let of = |name: &str| got.iter().find(|x| x.0 == name).unwrap().clone();
        assert_eq!(of("forked"), ("forked".into(), Some(10), false));
        assert_eq!(of("plain").1, Some(0));
        assert_eq!(of("dir").1, None);
        // A hard link reports its target's fork; one without a target none.
        assert_eq!(of("hard").1, Some(3));
        assert_eq!(of("dangling").1, None);
        for (node, want) in [
            (16, Some(10)),
            (17, Some(0)),
            (18, None),
            (50, Some(3)),
            (31, Some(3)),
            (52, None),
        ] {
            assert_eq!(
                fs.metadata(NodeId(node)).unwrap().resource_size,
                want,
                "node {node}"
            );
        }
    }

    #[test]
    fn finder_codes_and_owner_round_trip() {
        let fs = mounted(&facts_volume());
        let m = fs.metadata(NodeId(19)).unwrap();
        assert_eq!(
            m.finder_codes,
            Some(FinderCodes {
                file_type: *b"PNGf",
                creator: *b"prvw"
            })
        );
        assert_eq!(
            m.owner,
            Some(Owner {
                user: 501,
                group: 80,
                mode: 0o100_640
            })
        );
        // Both codes zero, and no mode: nothing to show.
        let m = fs.metadata(NodeId(20)).unwrap();
        assert_eq!((m.finder_codes, m.owner), (None, None));
        // One zero code is still a pair.
        let half = fs.metadata(NodeId(21)).unwrap().finder_codes.unwrap();
        assert_eq!(half.file_type, [0; 4]);
        assert_eq!(half.creator, *b"abcd");
        // A folder has an owner and no codes.
        let m = fs.metadata(NodeId(18)).unwrap();
        assert_eq!(m.finder_codes, None);
        assert_eq!(
            m.owner,
            Some(Owner {
                user: 501,
                group: 0,
                mode: 0o040_755
            })
        );
        // A hard link describes its target.
        let link = fs.metadata(NodeId(50)).unwrap();
        let target = fs.metadata(NodeId(31)).unwrap();
        assert_eq!(link.owner, target.owner);
        assert_eq!(link.finder_codes, target.finder_codes);
    }

    #[test]
    fn attribute_counts() {
        let fs = mounted(&facts_volume());
        assert_eq!(fs.metadata(NodeId(16)).unwrap().attributes, Some(2));
        assert_eq!(fs.metadata(NodeId(17)).unwrap().attributes, Some(0));
        assert_eq!(fs.metadata(NodeId(18)).unwrap().attributes, Some(0));
        // No attributes file: unknown, not zero.
        let fs = mounted(&main_volume());
        assert_eq!(fs.metadata(NodeId(17)).unwrap().attributes, None);
    }

    #[test]
    fn open_data_and_resource_streams() {
        let fs = mounted(&facts_volume());
        let read = |n: u64, stream: Stream| {
            let src = fs.open(NodeId(n), stream).unwrap();
            let mut buf = vec![0u8; src.len() as usize];
            src.read_at(0, &mut buf).unwrap();
            buf
        };
        // A file with a resource fork gives it. Its data fork claims 4 bytes
        // and maps none: opening still works (it reads no content) and the
        // damage shows in `read_at`.
        assert_eq!(read(16, Stream::Resource), FORK);
        let data = fs.open(NodeId(16), Stream::Data).unwrap();
        assert_eq!(data.len(), 4);
        assert!(data.read_at(0, &mut [0u8; 4]).is_err());
        // A hard link gives its target's fork.
        assert_eq!(read(50, Stream::Resource), b"abc");
        // An empty fork is no stream; neither is a folder's, nor is a
        // missing node's one.
        for node in [17, 52] {
            assert!(
                matches!(
                    fs.open(NodeId(node), Stream::Resource),
                    Err(Error::NoSuchStream { .. })
                ),
                "node {node}"
            );
        }
        assert!(matches!(
            fs.open(NodeId(18), Stream::Resource),
            Err(Error::IsADirectory { node: 18 })
        ));
        assert!(matches!(
            fs.open(NodeId(18), Stream::Data),
            Err(Error::IsADirectory { node: 18 })
        ));
        assert!(matches!(
            fs.open(NodeId(9999), Stream::Resource),
            Err(Error::NotFound { node: 9999 })
        ));
    }

    #[test]
    fn finder_info_round_trips() {
        let mut v = facts_volume();
        v.folder_units(ROOT, &test_volume::units("vis"), 40, 0x4000);
        let fs = mounted(&v);
        let info = |n: u64| fs.metadata(NodeId(n)).unwrap().finder_info;
        // FInfo: the type and creator codes then the Finder flags.
        let mut want = [0u8; 32];
        want[..4].copy_from_slice(b"PNGf");
        want[4..8].copy_from_slice(b"prvw");
        assert_eq!(info(19), Some(want));
        // All 32 bytes zero: nothing to show.
        assert_eq!(info(20), None);
        assert_eq!(info(18), None);
        // DInfo of a folder: its Finder flags sit at byte 8.
        let mut want = [0u8; 32];
        want[8..10].copy_from_slice(&0x4000u16.to_be_bytes());
        assert_eq!(info(40), Some(want));
        // A hard link describes its target; one without a target itself.
        assert_eq!(info(50), info(31));
        assert_eq!(&info(31).unwrap()[..4], b"TEXT");
        assert_eq!(&info(52).unwrap()[..4], b"hlnk");
    }

    #[test]
    fn plain_volume_has_no_compressed_flag() {
        let fs = mounted(&facts_volume());
        assert!(forks(&fs, 2).iter().all(|x| !x.2));
        assert!(
            !fs.metadata(NodeId(16))
                .unwrap()
                .flags
                .contains(EntryFlags::COMPRESSED)
        );
    }

    // ---- symbolic links ----

    #[test]
    fn symlink_targets() {
        let fs = mounted(&main_volume());
        let target = |id: u64| fs.metadata(NodeId(id)).unwrap().link_target;
        let m = fs.metadata(NodeId(60)).unwrap();
        assert_eq!(m.kind, NodeKind::Symlink);
        assert_eq!(m.size, Some(12));
        assert_eq!(m.link_target.as_deref(), Some("/usr/bin/env"));
        assert_eq!(target(61), None);
        assert_eq!(target(62), Some("y".repeat(1024)));
        assert_eq!(target(63), None);
        assert_eq!(target(64).as_deref(), Some("a\u{FFFD}b"));
        // Not a symlink: no target, even with data.
        assert_eq!(target(17), None);
        // A symlink is not followed as a folder.
        assert!(matches!(
            fs.read_dir(NodeId(60)),
            Err(Error::NotADirectory { node: 60 })
        ));
    }

    #[test]
    fn symlink_with_unreadable_fork() {
        // Logical size 10 but no extents: reading fails, metadata does not.
        let mut v = VolumeBuilder::new("V");
        let spec = FileSpec {
            mode: test_volume::MODE_SYMLINK,
            ..FileSpec::regular(10)
        };
        v.file(ROOT, "broken", 16, spec);
        let fs = mounted(&v);
        let m = fs.metadata(NodeId(16)).unwrap();
        assert_eq!(m.kind, NodeKind::Symlink);
        assert_eq!(m.link_target, None);
    }

    // ---- names ----

    #[test]
    fn hfsplus_lookup_folds_case() {
        let fs = mounted(&main_volume());
        assert_eq!(lookup(&fs, ROOT, "README.TXT"), Some(17));
        assert_eq!(lookup(&fs, ROOT, "readme.txt"), Some(17));
        assert_eq!(lookup(&fs, ROOT, "nope"), None);
        assert_eq!(lookup(&fs, 9999, "readme.txt"), None);
    }

    #[test]
    fn hfsx_binary_lookup_is_case_sensitive() {
        let mut v = VolumeBuilder::new("X");
        v.hfsx(0xBC)
            .file(ROOT, "Readme", 16, FileSpec::regular(1))
            .file(ROOT, "readme", 17, FileSpec::regular(2))
            .folder_units(ROOT, &FILE_PRIVATE_FOLDER, 18, 0)
            .file(18, "iNode5", 19, FileSpec::regular(3))
            .file(ROOT, "link", 20, FileSpec::file_link(5));
        let fs = mounted(&v);
        assert_eq!(fs.info().format, Some("HFSX"));
        assert_eq!(lookup(&fs, ROOT, "Readme"), Some(16));
        assert_eq!(lookup(&fs, ROOT, "readme"), Some(17));
        assert_eq!(lookup(&fs, ROOT, "README"), None);
        assert_eq!(lookup(&fs, ROOT, "link"), Some(19));
        let nodes: Vec<u64> = list(&fs, 2).into_iter().map(|x| x.node).collect();
        assert_eq!(nodes, [16, 19, 17]);

        // HFSX with case folding behaves like HFS+.
        let mut v = VolumeBuilder::new("X");
        v.hfsx(0xCF).file(ROOT, "Readme", 16, FileSpec::regular(1));
        let fs = mounted(&v);
        assert_eq!(lookup(&fs, ROOT, "README"), Some(16));
    }

    // ---- wrapped, journal ----

    #[test]
    fn journal_of_a_non_journaling_mount_is_ignored() {
        // Journaled and not unmounted cleanly, but lastMountedVersion is not
        // 'HFSJ' (and the journal info block is not one): macOS would not
        // replay, so neither does the mount.
        let mut v = main_volume();
        v.journal_dirty();
        let fs = mounted(&v);
        assert_eq!(fs.journal(), &JournalState::Clean);
        assert_eq!(fs.info().name, "Test Volume");
    }

    /// Allocation blocks of the journaled test volumes.
    const JOURNALED_BLOCKS: u32 = 160;

    /// Volume A, and volume B: A plus a file with content in a folder and a
    /// file in the root, so the catalog moves and the header counts change.
    /// Both journaled with one layout, after `edit`.
    fn a_and_b(edit: impl Fn(&mut JournalSetup)) -> (VolumeBuilder, VolumeBuilder) {
        let mut a = VolumeBuilder::new("Journaled");
        a.journaled(JOURNALED_BLOCKS, MIN_JOURNAL)
            .file(ROOT, "old.txt", 16, FileSpec::regular(5))
            .folder(ROOT, "Docs", 17);
        edit(a.journal_setup());
        let mut b = a.clone();
        b.file(
            17,
            "new.txt",
            18,
            FileSpec {
                content: Some(b"replayed".to_vec()),
                ..FileSpec::regular(0)
            },
        )
        .file(ROOT, "zz.txt", 19, FileSpec::regular(9));
        (a, b)
    }

    /// `a_image` with the blocks where `b_image` differs journaled at device
    /// byte `journal_base`: the first changed block as one transaction, the
    /// rest as a second of two block lists.
    fn journal_b_into_a(
        a_image: &[u8],
        b_image: &[u8],
        journal_base: usize,
        little_endian: bool,
        damaged: bool,
    ) -> Vec<u8> {
        let blocks = changed_blocks(a_image, b_image, 0, 512);
        assert!(blocks.len() > 3, "{} changed blocks", blocks.len());
        let (head, rest) = blocks.split_at(1);
        let (r1, r2) = rest.split_at(rest.len() / 2);
        let mut first = TestList::first(41, head.to_vec());
        first.bad_checksum = damaged;
        let j = TestJournal {
            little_endian,
            lists: vec![
                first,
                TestList::first(42, r1.to_vec()),
                TestList::more(42, r2.to_vec()),
            ],
            ..TestJournal::new(MIN_JOURNAL)
        };
        let mut image = a_image.to_vec();
        j.write(&mut image, journal_base);
        image
    }

    fn file_content(fs: &HfsPlus, node: u64) -> Vec<u8> {
        let data = fs.open(NodeId(node), Stream::Data).unwrap();
        let mut buf = vec![0u8; data.len() as usize];
        data.read_at(0, &mut buf).unwrap();
        buf
    }

    #[test]
    fn dirty_journal_replays_to_the_newer_volume() {
        let (a, b) = a_and_b(|_| {});
        let (a_image, b_image) = (a.build(), b.build());
        let old = mounted(&a);
        let new = mounted(&b);
        assert_eq!(old.journal(), &JournalState::Clean);
        assert_ne!(old.info().free_bytes, new.info().free_bytes);
        for le in [false, true] {
            let image =
                journal_b_into_a(&a_image, &b_image, a.journal_offset() as usize, le, false);
            let fs = mount(source(image)).unwrap();
            assert_eq!(
                fs.journal(),
                &JournalState::Replayed {
                    transactions: 2,
                    stopped_early: false
                }
            );
            // The header (free blocks, catalog extents) comes from the journal.
            assert_eq!(volume_only(&fs), volume_only(&new));
            assert_eq!(fs.info().journal.as_ref(), Some(fs.journal()));
            assert!(matches!(
                fs.info().journal,
                Some(JournalState::Replayed { .. })
            ));
            assert_eq!(list(&fs, 2), list(&new, 2));
            assert_eq!(list(&fs, 17), list(&new, 17));
            assert_eq!(file_content(&fs, 18), b"replayed");
        }
    }

    #[test]
    fn wrapped_volume_journal_counts_blocks_from_the_wrapper() {
        let (a, b) = a_and_b(|_| {});
        let (a_image, b_image) = (wrap(&a.build()), wrap(&b.build()));
        // The embedded volume starts 8 sectors and one block in.
        let base = 8 * 512 + BLOCK + a.journal_offset() as usize;
        let image = journal_b_into_a(&a_image, &b_image, base, true, false);
        let fs = mount(source(image)).unwrap();
        assert!(matches!(
            fs.journal(),
            JournalState::Replayed {
                transactions: 2,
                ..
            }
        ));
        let new = mounted(&b);
        assert_eq!(volume_only(&fs), volume_only(&new));
        assert_eq!(list(&fs, 17), list(&new, 17));
        assert_eq!(file_content(&fs, 18), b"replayed");
    }

    #[test]
    fn unreplayable_journal_reads_the_disk_as_is() {
        let (a, b) = a_and_b(|_| {});
        let image = journal_b_into_a(
            &a.build(),
            &b.build(),
            a.journal_offset() as usize,
            false,
            true,
        );
        let fs = mount(source(image)).unwrap();
        assert!(matches!(fs.journal(), JournalState::NotReplayed { .. }));
        let old = mounted(&a);
        assert_eq!(volume_only(&fs), volume_only(&old));
        assert_eq!(list(&fs, 17), list(&old, 17));
    }

    #[test]
    fn journal_left_by_a_non_journaling_mount_is_not_replayed() {
        let (a, b) = a_and_b(|j| j.last_mounted = u32::from_be_bytes(*b"10.0"));
        let image = journal_b_into_a(
            &a.build(),
            &b.build(),
            a.journal_offset() as usize,
            false,
            false,
        );
        let fs = mount(source(image)).unwrap();
        assert_eq!(fs.journal(), &JournalState::Clean);
        let old = mounted(&a);
        assert_eq!(volume_only(&fs), volume_only(&old));
        assert_eq!(list(&fs, 2), list(&old, 2));
    }

    // ---- malformed ----

    #[test]
    fn missing_root_thread_fails_mount() {
        let mut v = VolumeBuilder::new("V");
        v.without_root_thread()
            .file(ROOT, "a", 16, FileSpec::regular(1));
        let c = corruption(mount(source(v.build())));
        assert_eq!(c.structure, "catalog B-tree");
        assert!(matches!(c.kind, CorruptKind::Other(_)));
    }

    #[test]
    fn not_hfs_plus_fails_mount() {
        assert!(matches!(
            mount(source(vec![0u8; 8192])),
            Err(Error::Corrupt(_))
        ));
        assert!(mount(source(Vec::new())).is_err());
        // A source cut short before the catalog: its header node is past the end.
        let mut bytes = main_volume().build();
        bytes.truncate(BLOCK * 3);
        assert!(mount(source(bytes)).is_err());
    }

    #[test]
    fn node_ids_out_of_range() {
        let fs = mounted(&main_volume());
        for node in [0, u64::from(u32::MAX) + 1, u64::MAX] {
            let n = NodeId(node);
            assert!(matches!(fs.metadata(n), Err(Error::NotFound { node: x }) if x == node));
            assert!(matches!(fs.read_dir(n), Err(Error::NotFound { node: x }) if x == node));
            assert!(matches!(fs.lookup(n, "a"), Err(Error::NotFound { node: x }) if x == node));
        }
        assert!(matches!(
            fs.metadata(NodeId(9999)),
            Err(Error::NotFound { node: 9999 })
        ));
        assert!(matches!(
            fs.read_dir(NodeId(9999)),
            Err(Error::NotFound { node: 9999 })
        ));
        assert!(matches!(
            fs.read_dir(NodeId(17)),
            Err(Error::NotADirectory { node: 17 })
        ));
    }

    // ---- compression and extended attributes ----

    mod compressed {
        use super::super::compression::fixtures::{
            content, header, lzfse, offsets_fork, pieces, zlib, zlib_fork,
        };
        use super::*;

        const UF_COMPRESSED: u8 = 0x20;

        struct Fixture {
            fs: HfsPlus,
            small: Vec<u8>,
            big: Vec<u8>,
            lz: Vec<u8>,
            fork_xattr: Vec<u8>,
        }

        fn with_payload(ty: u32, size: usize, payload: &[u8]) -> Vec<u8> {
            let mut x = header(ty, size as u64);
            x.extend_from_slice(payload);
            x
        }

        fn fixture() -> Fixture {
            let small = content(3000);
            let big = content(150 * 1024);
            let lz = content(70 * 1024);
            let fork_xattr = content(9 * BLOCK + 123);
            let big_fork = zlib_fork(&pieces(&big).iter().map(|p| zlib(p)).collect::<Vec<_>>());
            let lz_fork = offsets_fork(&pieces(&lz).iter().map(|p| lzfse(p)).collect::<Vec<_>>());
            let mut bad_magic = header(3, 10);
            bad_magic[0] = b'X';
            let mut v = VolumeBuilder::new("Compressed");
            v.compressed_file(
                ROOT,
                "small.txt",
                80,
                &with_payload(3, small.len(), &zlib(&small)),
                None,
            )
            .compressed_file(
                ROOT,
                "big.bin",
                81,
                &header(4, big.len() as u64),
                Some(&big_fork),
            )
            .compressed_file(
                ROOT,
                "lzfse.bin",
                82,
                &header(12, lz.len() as u64),
                Some(&lz_fork),
            )
            .file(ROOT, "no header", 83, FileSpec::regular(0))
            .owner_flags(83, UF_COMPRESSED)
            .compressed_file(ROOT, "bad magic", 84, &bad_magic, None)
            .compressed_file(ROOT, "lzbitmap", 85, &header(14, 5000), None)
            .file(
                ROOT,
                "plain.txt",
                86,
                FileSpec {
                    content: Some(b"hello".to_vec()),
                    ..FileSpec::regular(0)
                },
            )
            .xattr(86, "user.a", XattrValue::Inline(b"abc".to_vec()))
            .xattr(86, "big", XattrValue::Fork(fork_xattr.clone()))
            .xattr(86, "com.apple.FinderInfo", XattrValue::Inline(vec![0; 32]))
            .xattr(80, "user.tag", XattrValue::Inline(b"t".to_vec()))
            // A hard link to a compressed file.
            .folder_units(ROOT, &FILE_PRIVATE_FOLDER, 30, 0x5000)
            .compressed_file(
                30,
                "iNode300",
                31,
                &with_payload(3, small.len(), &zlib(&small)),
                None,
            )
            .file(ROOT, "link", 87, FileSpec::file_link(300))
            // The flag on a symlink is ignored.
            .file(ROOT, "sym", 88, FileSpec::symlink(b"small.txt"))
            .owner_flags(88, UF_COMPRESSED)
            .folder(ROOT, "dir", 89)
            .xattr(89, "user.folder", XattrValue::Inline(b"f".to_vec()));
            Fixture {
                fs: mounted(&v),
                small,
                big,
                lz,
                fork_xattr,
            }
        }

        fn read_all(src: &dyn BlockSource) -> Vec<u8> {
            let mut buf = vec![0u8; src.len() as usize];
            src.read_at(0, &mut buf).unwrap();
            buf
        }

        #[test]
        fn scan_matches_the_walk_with_compressed_files() {
            let f = fixture();
            let summary = scan_tests::assert_scan_matches_walk(&f.fs);
            assert_eq!(summary.skipped, 0);
        }

        #[test]
        fn listing_reports_uncompressed_sizes() {
            let f = fixture();
            let sizes: Vec<(String, u64, Option<u64>)> = list(&f.fs, 2)
                .into_iter()
                .map(|x| (x.name, x.node, x.size))
                .collect();
            let want = |name: &str, node: u64, size: Option<u64>| (name.to_string(), node, size);
            assert_eq!(
                sizes,
                vec![
                    want("bad magic", 84, None),
                    want("big.bin", 81, Some(f.big.len() as u64)),
                    want("dir", 89, None),
                    want("link", 31, Some(f.small.len() as u64)),
                    want("lzbitmap", 85, Some(5000)),
                    want("lzfse.bin", 82, Some(f.lz.len() as u64)),
                    want("no header", 83, None),
                    want("plain.txt", 86, Some(5)),
                    want("small.txt", 80, Some(f.small.len() as u64)),
                    want("sym", 88, Some(9)),
                ]
            );
            // metadata agrees with the listing, through the link too.
            for (_, node, size) in sizes {
                let m = f.fs.metadata(NodeId(node)).unwrap();
                assert_eq!(m.size, size, "node {node}");
            }
            assert_eq!(
                f.fs.metadata(NodeId(87)).unwrap().size,
                Some(f.small.len() as u64)
            );
        }

        #[test]
        fn compressed_files_show_an_empty_fork_and_the_flag() {
            let f = fixture();
            let got = forks(&f.fs, 2);
            let of = |name: &str| got.iter().find(|x| x.0 == name).unwrap().clone();
            // The resource fork of chunks is not a fork the user made.
            for name in ["big.bin", "lzfse.bin", "small.txt", "link", "lzbitmap"] {
                assert_eq!(of(name).1, Some(0), "{name}");
                assert!(of(name).2, "{name}");
            }
            // The flag on a symlink is ignored; plain files and folders.
            assert!(!of("sym").2);
            assert!(!of("plain.txt").2);
            assert_eq!(of("dir").1, None);
            for node in [80, 81, 82, 87, 31] {
                let m = f.fs.metadata(NodeId(node)).unwrap();
                assert!(m.flags.contains(EntryFlags::COMPRESSED), "node {node}");
                assert_eq!(m.resource_size, Some(0), "node {node}");
            }
            assert!(
                !f.fs
                    .metadata(NodeId(86))
                    .unwrap()
                    .flags
                    .contains(EntryFlags::COMPRESSED)
            );
            // Counts leave decmpfs out: small.txt keeps user.tag, big.bin
            // and the linked iNode have nothing else.
            let count = |n: u64| f.fs.metadata(NodeId(n)).unwrap().attributes;
            assert_eq!(count(80), Some(1));
            assert_eq!(count(81), Some(0));
            assert_eq!(count(87), Some(0));
            // A plain file keeps all its attributes.
            assert_eq!(count(86), Some(3));
            // The chunks are still in the resource fork on disk.
            assert!(
                !f.fs
                    .extents(NodeId(81), Stream::Resource)
                    .unwrap()
                    .is_empty()
            );
        }

        #[test]
        fn open_decompresses() {
            let f = fixture();
            let open = |n: u64| read_all(&*f.fs.open(NodeId(n), Stream::Data).unwrap());
            assert_eq!(open(80), f.small);
            assert_eq!(open(81), f.big);
            assert_eq!(open(82), f.lz);
            // A hard link opens its target.
            assert_eq!(open(87), f.small);
            assert_eq!(open(31), f.small);
            assert_eq!(open(86), b"hello");
            assert_eq!(open(88), b"small.txt");
            // A read across chunks of the resource fork kind.
            let src = f.fs.open(NodeId(81), Stream::Data).unwrap();
            let mut buf = vec![0u8; 1000];
            src.read_at(65_000, &mut buf).unwrap();
            assert_eq!(buf, f.big[65_000..66_000]);
        }

        #[test]
        fn open_failures() {
            let f = fixture();
            assert!(matches!(
                f.fs.open(NodeId(89), Stream::Data),
                Err(Error::IsADirectory { node: 89 })
            ));
            assert!(matches!(
                f.fs.open(NodeId(2), Stream::Data),
                Err(Error::IsADirectory { node: 2 })
            ));
            assert!(matches!(
                f.fs.open(NodeId(2), Stream::Resource),
                Err(Error::IsADirectory { node: 2 })
            ));
            assert!(matches!(
                f.fs.open(NodeId(85), Stream::Data),
                Err(Error::Unsupported(_))
            ));
            assert!(matches!(
                f.fs.open(NodeId(83), Stream::Data),
                Err(Error::Corrupt(_))
            ));
            assert!(matches!(
                f.fs.open(NodeId(84), Stream::Data),
                Err(Error::Corrupt(_))
            ));
            assert!(matches!(
                f.fs.open(NodeId(9999), Stream::Data),
                Err(Error::NotFound { node: 9999 })
            ));
            assert!(matches!(
                f.fs.open(NodeId(0), Stream::Data),
                Err(Error::NotFound { node: 0 })
            ));
        }

        #[test]
        fn open_resource_of_a_compressed_file_is_no_stream() {
            let f = fixture();
            // The chunks in the resource fork are storage, not a user's fork;
            // a hard link to a compressed file is the same.
            for node in [80, 81, 82, 87, 31] {
                assert!(
                    matches!(
                        f.fs.open(NodeId(node), Stream::Resource),
                        Err(Error::NoSuchStream { .. })
                    ),
                    "node {node}"
                );
            }
            // A plain file without a fork, and a folder.
            assert!(matches!(
                f.fs.open(NodeId(86), Stream::Resource),
                Err(Error::NoSuchStream { node: 86 })
            ));
            assert!(matches!(
                f.fs.open(NodeId(89), Stream::Resource),
                Err(Error::IsADirectory { node: 89 })
            ));
            assert!(matches!(
                f.fs.open(NodeId(9999), Stream::Resource),
                Err(Error::NotFound { node: 9999 })
            ));
        }

        #[test]
        fn read_xattr_returns_whole_values() {
            let f = fixture();
            let get = |n: u64, name: &str, max: usize| f.fs.read_xattr(NodeId(n), name, max);
            assert_eq!(get(86, "user.a", 64).unwrap(), Some(b"abc".to_vec()));
            // Exactly the size fits; one byte less does not.
            assert_eq!(get(86, "user.a", 3).unwrap(), Some(b"abc".to_vec()));
            assert!(matches!(get(86, "user.a", 2), Err(Error::Unsupported(_))));
            assert!(matches!(get(86, "user.a", 0), Err(Error::Unsupported(_))));
            // A value kept in a fork of its own, across extents records.
            let size = f.fork_xattr.len();
            assert_eq!(get(86, "big", size).unwrap(), Some(f.fork_xattr.clone()));
            assert!(matches!(
                get(86, "big", size - 1),
                Err(Error::Unsupported(_))
            ));
            // Absent names, on a file and on a node without attributes.
            assert_eq!(get(86, "user.nope", 64).unwrap(), None);
            assert_eq!(get(88, "user.a", 64).unwrap(), None);
            // A folder has attributes too.
            assert_eq!(get(89, "user.folder", 8).unwrap(), Some(b"f".to_vec()));
            // A compressed file keeps its others and hides decmpfs, as
            // `xattrs` does; so does a hard link to one.
            assert_eq!(get(80, "user.tag", 8).unwrap(), Some(b"t".to_vec()));
            for node in [80, 81, 87] {
                assert_eq!(get(node, DECMPFS_XATTR, 1 << 20).unwrap(), None);
                assert!(
                    f.fs.xattrs(NodeId(node))
                        .unwrap()
                        .iter()
                        .all(|x| x.name != DECMPFS_XATTR)
                );
            }
            // The same attribute on a plain file would be readable.
            assert!(matches!(
                get(9999, "user.a", 8),
                Err(Error::NotFound { node: 9999 })
            ));
        }

        #[test]
        fn xattrs_list_names_and_sizes() {
            let f = fixture();
            let got: Vec<(String, u64)> =
                f.fs.xattrs(NodeId(86))
                    .unwrap()
                    .into_iter()
                    .map(|x| (x.name, x.size))
                    .collect();
            assert_eq!(
                got,
                vec![
                    ("big".to_string(), f.fork_xattr.len() as u64),
                    ("com.apple.FinderInfo".to_string(), 32),
                    ("user.a".to_string(), 3),
                ]
            );
            // Through a hard link, the target's attributes.
            let names: Vec<String> =
                f.fs.xattrs(NodeId(87))
                    .unwrap()
                    .into_iter()
                    .map(|x| x.name)
                    .collect();
            assert!(names.is_empty(), "{names:?}");
            // A compressed file leaves decmpfs out and keeps the others.
            let names: Vec<String> =
                f.fs.xattrs(NodeId(80))
                    .unwrap()
                    .into_iter()
                    .map(|x| x.name)
                    .collect();
            assert_eq!(names, ["user.tag"]);
            // The attribute is still there for the decompressor.
            let attributes = f.fs.attributes.as_ref().unwrap();
            assert!(
                attributes
                    .get(80, DECMPFS_XATTR, MAX_DECMPFS_XATTR)
                    .unwrap()
                    .is_some()
            );
            assert_eq!(f.fs.xattrs(NodeId(89)).unwrap()[0].name, "user.folder");
            assert!(f.fs.xattrs(NodeId(2)).unwrap().is_empty());
            assert!(matches!(
                f.fs.xattrs(NodeId(9999)),
                Err(Error::NotFound { .. })
            ));
            // The fork-based value, nine blocks and more: eight extents in
            // the fork record, the rest in an extents record.
            let attributes = f.fs.attributes.as_ref().unwrap();
            assert_eq!(
                attributes.get(86, "big", 1 << 20).unwrap().unwrap(),
                f.fork_xattr
            );
        }

        #[test]
        fn volume_without_attributes_file() {
            let fs = mounted(&main_volume());
            assert!(fs.attributes.is_none());
            assert!(fs.xattrs(NodeId(17)).unwrap().is_empty());
            assert_eq!(fs.read_xattr(NodeId(17), "user.a", 64).unwrap(), None);
            // A compressed flag with no attributes file: no size, and
            // opening it fails.
            let mut v = VolumeBuilder::new("V");
            v.file(ROOT, "c", 16, FileSpec::regular(0))
                .owner_flags(16, UF_COMPRESSED)
                .file(ROOT, "p", 17, FileSpec::regular(4));
            let fs = mounted(&v);
            assert!(fs.attributes.is_none());
            assert_eq!(
                list(&fs, 2).into_iter().map(|x| x.size).collect::<Vec<_>>(),
                [None, Some(4)]
            );
            assert_eq!(fs.metadata(NodeId(16)).unwrap().size, None);
            assert!(matches!(
                fs.open(NodeId(16), Stream::Data),
                Err(Error::Corrupt(_))
            ));
        }

        #[test]
        fn damaged_attributes_file_fails_mount() {
            let mut v = VolumeBuilder::new("V");
            v.file(ROOT, "a", 16, FileSpec::regular(1)).xattr(
                16,
                "x",
                XattrValue::Inline(b"1".to_vec()),
            );
            let mut bytes = v.build();
            // The attributes tree's header node: its first block is named by
            // the volume header at 1024 + 352 + 16.
            let at = 1024 + 352 + 16;
            let start = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            // Break the node size in its header record.
            bytes[start * BLOCK + 14 + 18] = 0;
            bytes[start * BLOCK + 14 + 19] = 3;
            let c = corruption(mount(source(bytes)));
            assert_eq!(c.structure, "attributes B-tree");
        }
    }

    // ---- extents and the allocation map ----

    mod disk_strip {
        use super::super::btree::fixtures::{LEAF, header, header_node, keyed, node};
        use super::super::extents::{ExtentDescriptor, ForkData};
        use super::super::links::DIR_PRIVATE_FOLDER;
        use super::*;
        use atlas_fs_core::fs::MAX_ALLOCATION_BUCKETS;

        /// Byte of the allocation file's fork descriptor in an image.
        const ALLOCATION_FORK: usize = 1024 + 112;
        /// Byte of `totalBlocks` in an image.
        const TOTAL_BLOCKS: usize = 1024 + 44;
        /// Where the wrapper of [`wrap`] puts the volume.
        const WRAPPED_AT: u64 = 8 * 512 + BLOCK as u64;

        fn be32(image: &[u8], at: usize) -> u32 {
            u32::from_be_bytes(image[at..at + 4].try_into().unwrap())
        }

        /// The densities, from the image's own bitmap one bit at a time.
        fn naive_density(image: &[u8], buckets: usize) -> Vec<u8> {
            let total = u64::from(be32(image, TOTAL_BLOCKS));
            let bitmap = be32(image, ALLOCATION_FORK + 16) as usize * BLOCK;
            let n = buckets.clamp(1, MAX_ALLOCATION_BUCKETS).min(total as usize) as u64;
            (0..n)
                .map(|i| {
                    let (lo, hi) = (i * total / n, (i + 1) * total / n);
                    let used = (lo..hi)
                        .filter(|&b| image[bitmap + (b / 8) as usize] & (0x80 >> (b % 8)) != 0)
                        .count() as u64;
                    (used * 255).div_ceil(hi - lo) as u8
                })
                .collect()
        }

        fn fork_bytes(f: &ForkData) -> Vec<u8> {
            let mut b = f.logical_size.to_be_bytes().to_vec();
            b.extend_from_slice(&f.clump_size.to_be_bytes());
            b.extend_from_slice(&f.total_blocks.to_be_bytes());
            for e in &f.extents {
                b.extend_from_slice(&e.start_block.to_be_bytes());
                b.extend_from_slice(&e.block_count.to_be_bytes());
            }
            b
        }

        /// Replaces the one place `old` is stored in `image` with `new`.
        fn patch_fork(image: &mut [u8], old: &ForkData, new: &ForkData) {
            let (old, new) = (fork_bytes(old), fork_bytes(new));
            let at: Vec<usize> = image
                .windows(old.len())
                .enumerate()
                .filter(|(_, w)| *w == old.as_slice())
                .map(|(i, _)| i)
                .collect();
            assert_eq!(at.len(), 1, "fork descriptor not unique");
            image[at[0]..at[0] + new.len()].copy_from_slice(&new);
        }

        /// A fork of `logical` bytes over `extents` (blocks: start, count),
        /// the first eight in the descriptor; `total` counts every block.
        fn fork(logical: u64, total: u32, extents: &[(u32, u32)]) -> ForkData {
            let mut f = ForkData {
                logical_size: logical,
                total_blocks: total,
                ..ForkData::default()
            };
            for (slot, &(start_block, block_count)) in f.extents.iter_mut().zip(extents) {
                *slot = ExtentDescriptor {
                    start_block,
                    block_count,
                };
            }
            f
        }

        /// An extents overflow record: file ID, fork type byte, start block,
        /// extents (start, count).
        type OverflowRecord<'a> = (u32, u8, u32, &'a [(u32, u32)]);

        /// Replaces the extents overflow B-tree (block 1) with one leaf of
        /// `records`, in key order. 512-byte nodes, so the header and leaf
        /// fit the block.
        fn install_overflow(image: &mut [u8], records: &[OverflowRecord<'_>]) {
            let mut h = header(1, 1, 1, 1, 8);
            h.max_key_length = 10;
            h.leaf_records = records.len() as u32;
            h.free_nodes = 6;
            let leaf: Vec<Vec<u8>> = records
                .iter()
                .map(|&(id, fork_type, start, extents)| {
                    let mut k = vec![fork_type, 0];
                    k.extend_from_slice(&id.to_be_bytes());
                    k.extend_from_slice(&start.to_be_bytes());
                    let data = &fork_bytes(&fork(0, 0, extents))[16..];
                    keyed(&k, data)
                })
                .collect();
            let mut tree = header_node(&h);
            tree.extend_from_slice(&node(512, LEAF, 1, 0, 0, &leaf));
            tree.resize(BLOCK, 0);
            image[BLOCK..2 * BLOCK].copy_from_slice(&tree);
        }

        fn data_fork(fs: &HfsPlus, id: u32) -> ForkData {
            match fs.catalog.record(id).unwrap().unwrap() {
                CatalogRecord::File(f) => f.data_fork,
                CatalogRecord::Folder(_) => panic!("{id} is a folder"),
            }
        }

        /// `(offset, len)` of every extent, in allocation blocks from `base`.
        fn blocks(extents: &[Extent], base: u64) -> Vec<(u64, u64)> {
            let bs = BLOCK as u64;
            extents
                .iter()
                .map(|e| {
                    assert_eq!((e.offset - base) % bs, 0);
                    assert_eq!(e.len % bs, 0);
                    assert_ne!(e.len, 0);
                    ((e.offset - base) / bs, e.len / bs)
                })
                .collect()
        }

        fn content(blocks: usize, tail: usize, fill: u8) -> FileSpec {
            FileSpec {
                content: Some(vec![fill; blocks * BLOCK - tail]),
                ..FileSpec::regular(0)
            }
        }

        const FRAG: u32 = 16;
        const PLAIN: u32 = 17;
        const EMPTY: u32 = 18;
        const DIR: u32 = 19;
        const RSRC_COMPRESSED: u32 = 20;
        const INLINE_COMPRESSED: u32 = 21;
        const SYM: u32 = 22;
        const TARGET: u32 = 31;
        const HARD: u32 = 50;
        const BROKEN: u32 = 51;
        const HARD_DIR: u32 = 52;

        /// File 16 of 12 blocks, the rest small: content, a resource fork,
        /// compressed files, links.
        fn strip_volume() -> VolumeBuilder {
            let mut v = VolumeBuilder::new("Strip");
            v.file(ROOT, "frag", FRAG, content(12, 5, 0xF1))
                .file(ROOT, "plain", PLAIN, content(3, BLOCK - 1, 0xA1))
                .resource_fork(PLAIN, &[0xB2; BLOCK + 7])
                .file(ROOT, "empty", EMPTY, FileSpec::regular(0))
                .folder(ROOT, "dir", DIR)
                .compressed_file(
                    ROOT,
                    "rsrc compressed",
                    RSRC_COMPRESSED,
                    b"fpmc\x04\0\0\0",
                    Some(&[0xC3; 2 * BLOCK + 1]),
                )
                .compressed_file(ROOT, "inline compressed", INLINE_COMPRESSED, b"x", None)
                .file(ROOT, "sym", SYM, FileSpec::symlink(b"/a/b"))
                .folder_units(ROOT, &FILE_PRIVATE_FOLDER, 30, 0x5000)
                .file(30, "iNode100", TARGET, content(2, 1, 0xD4))
                .resource_fork(TARGET, &[0xE5; 10])
                .file(ROOT, "hard", HARD, FileSpec::file_link(100))
                .file(ROOT, "broken", BROKEN, FileSpec::file_link(999))
                .folder_units(ROOT, &DIR_PRIVATE_FOLDER, 40, 0x5000)
                .folder(40, "dir_200", 41)
                .file(ROOT, "hard dir", HARD_DIR, FileSpec::dir_link(200));
            v
        }

        /// File 16's twelve blocks in a scrambled logical order, relative
        /// to its first block: eight extents in the catalog, three more in
        /// an overflow record.
        const FIRST_EIGHT: [(u32, u32); 8] = [
            (8, 2),
            (0, 1),
            (1, 1),
            (10, 1),
            (6, 1),
            (2, 1),
            (3, 1),
            (4, 1),
        ];
        const OVERFLOW: [(u32, u32); 3] = [(5, 1), (11, 1), (7, 1)];

        /// `strip_volume` with file 16 fragmented as above.
        fn fragmented() -> (Vec<u8>, u32) {
            let builder = strip_volume();
            let mut image = builder.build();
            let old = data_fork(&mounted(&builder), FRAG);
            let s = old.extents[0].start_block;
            assert_eq!(old.extents[0].block_count, 12);
            let shift = |r: &[(u32, u32)]| -> Vec<(u32, u32)> {
                r.iter().map(|&(b, c)| (s + b, c)).collect()
            };
            let new = fork(old.logical_size, 12, &shift(&FIRST_EIGHT));
            patch_fork(&mut image, &old, &new);
            install_overflow(&mut image, &[(FRAG, 0x00, 9, &shift(&OVERFLOW))]);
            (image, s)
        }

        fn extents(fs: &HfsPlus, node: u32, stream: Stream) -> Vec<(u64, u64)> {
            blocks(&fs.extents(NodeId(u64::from(node)), stream).unwrap(), 0)
        }

        #[test]
        fn fragmented_file_in_logical_order_merging_disk_neighbours() {
            let (image, s) = fragmented();
            let fs = mount(source(image.clone())).unwrap();
            let s = u64::from(s);
            // 0+1 merge, 2..5 merge across the overflow record, the rest
            // stay apart even where they touch in the other direction.
            let want: Vec<(u64, u64)> = [(8, 2), (0, 2), (10, 1), (6, 1), (2, 4), (11, 1), (7, 1)]
                .iter()
                .map(|&(b, c)| (s + b, c))
                .collect();
            let got = fs.extents(NodeId(u64::from(FRAG)), Stream::Data).unwrap();
            assert_eq!(blocks(&got, 0), want);
            // The content reads back in the same order: the image's bytes
            // along the extents, cut to the logical size.
            let mut along: Vec<u8> = got
                .iter()
                .flat_map(|e| image[e.offset as usize..(e.offset + e.len) as usize].to_vec())
                .collect();
            along.truncate(12 * BLOCK - 5);
            assert_eq!(file_content(&fs, u64::from(FRAG)), along);
            // The padding of the last physical block is in logical block 10.
            assert!(along[BLOCK * 10..BLOCK * 11].ends_with(&[0; 5]));
            assert!(along[BLOCK * 11..].iter().all(|&b| b == 0xF1));
        }

        #[test]
        fn file_extents_by_stream() {
            let builder = strip_volume();
            let fs = mounted(&builder);
            let first = |id: u32| u64::from(data_fork(&fs, id).extents[0].start_block);
            // A plain file: one run of whole blocks, past its logical size.
            assert_eq!(extents(&fs, PLAIN, Stream::Data), [(first(PLAIN), 3)]);
            // Its resource fork, placed right after the data.
            assert_eq!(
                extents(&fs, PLAIN, Stream::Resource),
                [(first(PLAIN) + 3, 2)]
            );
            // Empty forks, with or without a size.
            assert!(extents(&fs, EMPTY, Stream::Data).is_empty());
            assert!(extents(&fs, EMPTY, Stream::Resource).is_empty());
            // A symlink: its data fork.
            assert_eq!(extents(&fs, SYM, Stream::Data), [(first(SYM), 1)]);
            // Compressed: Data is where the compressed bytes are.
            let rsrc = extents(&fs, RSRC_COMPRESSED, Stream::Resource);
            assert_eq!(rsrc.len(), 1);
            assert_eq!(rsrc[0].1, 3);
            assert_eq!(extents(&fs, RSRC_COMPRESSED, Stream::Data), rsrc);
            assert!(extents(&fs, INLINE_COMPRESSED, Stream::Data).is_empty());
            assert!(extents(&fs, INLINE_COMPRESSED, Stream::Resource).is_empty());
            // A hard link reports its target, for both streams.
            for stream in [Stream::Data, Stream::Resource] {
                assert_eq!(extents(&fs, HARD, stream), extents(&fs, TARGET, stream));
            }
            assert_eq!(extents(&fs, HARD, Stream::Data), [(first(TARGET), 2)]);
            assert_eq!(extents(&fs, HARD, Stream::Resource).len(), 1);
            // An unresolvable link reports its own (empty) forks.
            assert!(extents(&fs, BROKEN, Stream::Data).is_empty());
        }

        #[test]
        fn extents_of_folders_and_missing_nodes() {
            let fs = mounted(&strip_volume());
            for stream in [Stream::Data, Stream::Resource] {
                for id in [u64::from(ROOT), u64::from(DIR), u64::from(HARD_DIR)] {
                    assert!(
                        matches!(fs.extents(NodeId(id), stream), Err(Error::IsADirectory { node }) if node == id),
                        "{id}"
                    );
                }
                for id in [0, 9999, u64::MAX] {
                    assert!(
                        matches!(fs.extents(NodeId(id), stream), Err(Error::NotFound { node }) if node == id),
                        "{id}"
                    );
                }
            }
        }

        #[test]
        fn extents_past_the_volume_fail() {
            let builder = strip_volume();
            let mut image = builder.build();
            let old = data_fork(&mounted(&builder), PLAIN);
            let total = be32(&image, TOTAL_BLOCKS);
            patch_fork(
                &mut image,
                &old,
                &fork(old.logical_size, 3, &[(total - 1, 3)]),
            );
            let fs = mount(source(image)).unwrap();
            let c = corruption(fs.extents(NodeId(u64::from(PLAIN)), Stream::Data));
            assert!(matches!(c.kind, CorruptKind::OutOfRange { .. }), "{c:?}");
        }

        #[test]
        fn allocation_map_matches_the_bitmap() {
            for builder in [main_volume(), strip_volume()] {
                let image = builder.build();
                let total = be32(&image, TOTAL_BLOCKS) as usize;
                let fs = mount(source(image.clone())).unwrap();
                for buckets in [0, 1, 2, 3, 7, total - 1, total, total + 5, 4096, usize::MAX] {
                    let map = fs.allocation_map(buckets).unwrap();
                    assert_eq!(map.start, 0);
                    assert_eq!(map.len, image.len() as u64);
                    assert_eq!(map.density, naive_density(&image, buckets), "{buckets}");
                    assert_eq!(map.density.len(), buckets.clamp(1, total));
                }
                // Every block its own bucket: all used but the spare last one.
                let mut want = vec![255; total];
                want[total - 1] = 0;
                assert_eq!(fs.allocation_map(total).unwrap().density, want);
            }
        }

        #[test]
        fn allocation_map_of_a_volume_with_more_blocks_than_buckets() {
            // 5000 blocks, 4096 buckets: most are one block, some two, and
            // they do not fall on bitmap byte edges.
            let mut v = VolumeBuilder::new("Big");
            v.journaled(5000, MIN_JOURNAL)
                .file(ROOT, "a", 16, content(3, 0, 1));
            let image = v.build();
            let fs = mount(source(image.clone())).unwrap();
            for buckets in [3, 1000, 4095, 4096, 4097, 10_000] {
                let map = fs.allocation_map(buckets).unwrap();
                assert_eq!(map.density.len(), buckets.min(4096));
                assert_eq!(map.density, naive_density(&image, buckets), "{buckets}");
            }
            let d = fs.allocation_map(4096).unwrap().density;
            assert!(d.contains(&0) && d.contains(&255));
        }

        #[test]
        fn wrapped_volume_reports_source_bytes() {
            let (image, _) = fragmented();
            let bare = mount(source(image.clone())).unwrap();
            let wrapped = mount(source(wrap(&image))).unwrap();
            let (b, w) = (
                bare.allocation_map(64).unwrap(),
                wrapped.allocation_map(64).unwrap(),
            );
            assert_eq!(w.start, WRAPPED_AT);
            assert_eq!((w.len, &w.density), (b.len, &b.density));
            for id in [FRAG, PLAIN, SYM, HARD, RSRC_COMPRESSED] {
                for stream in [Stream::Data, Stream::Resource] {
                    let node = NodeId(u64::from(id));
                    let want = bare.extents(node, stream).unwrap();
                    let got = wrapped.extents(node, stream).unwrap();
                    assert_eq!(blocks(&got, WRAPPED_AT), blocks(&want, 0), "{id}");
                }
            }
        }

        #[test]
        fn allocation_map_reads_the_replayed_bitmap() {
            let (a, b) = a_and_b(|_| {});
            let (a_image, b_image) = (a.build(), b.build());
            let old = mounted(&a).allocation_map(MAX_ALLOCATION_BUCKETS).unwrap();
            let new = mounted(&b).allocation_map(MAX_ALLOCATION_BUCKETS).unwrap();
            assert_ne!(old, new);
            for le in [false, true] {
                let image =
                    journal_b_into_a(&a_image, &b_image, a.journal_offset() as usize, le, false);
                let fs = mount(source(image)).unwrap();
                assert!(matches!(fs.journal(), JournalState::Replayed { .. }));
                assert_eq!(fs.allocation_map(MAX_ALLOCATION_BUCKETS).unwrap(), new);
            }
            // Wrapped, the replayed bitmap at the wrapper's offset.
            let base = WRAPPED_AT as usize + a.journal_offset() as usize;
            let image = journal_b_into_a(&wrap(&a_image), &wrap(&b_image), base, true, false);
            let map = mount(source(image)).unwrap().allocation_map(4096).unwrap();
            assert_eq!(map.start, WRAPPED_AT);
            assert_eq!(map.density, new.density);
        }

        #[test]
        fn allocation_file_through_overflow_extents() {
            // Nine blocks: the bitmap's own, seven more in the header, and
            // one in an extents overflow record.
            let mut image = main_volume().build();
            let start = be32(&image, ALLOCATION_FORK + 16);
            let want = naive_density(&image, 4096);
            let mut extents = vec![(start, 1)];
            extents.extend((2..9).map(|b| (b, 1)));
            let f = fork(9 * BLOCK as u64, 9, &extents);
            image[ALLOCATION_FORK..ALLOCATION_FORK + 80].copy_from_slice(&fork_bytes(&f));
            install_overflow(&mut image, &[(ALLOCATION_FILE_ID, 0x00, 8, &[(9, 1)])]);
            let fs = mount(source(image)).unwrap();
            assert_eq!(fs.allocation_map(4096).unwrap().density, want);
        }

        #[test]
        fn damaged_allocation_file_fails_only_the_map() {
            let base = main_volume().build();
            let total = be32(&base, TOTAL_BLOCKS);
            let start = be32(&base, ALLOCATION_FORK + 16);
            let needed = u64::from(total).div_ceil(8);

            // Shorter than the bitmap needs.
            let mut image = base.clone();
            let f = fork(needed - 1, 1, &[(start, 1)]);
            image[ALLOCATION_FORK..ALLOCATION_FORK + 80].copy_from_slice(&fork_bytes(&f));
            let fs = mount(source(image)).unwrap();
            let c = corruption(fs.allocation_map(16));
            assert_eq!(c.structure, "allocation file");
            assert_eq!(
                c.kind,
                CorruptKind::Truncated {
                    needed,
                    available: needed - 1
                }
            );
            assert!(fs.extents(NodeId(60), Stream::Data).is_ok());

            // An overflow extent past the volume's blocks.
            let mut image = base.clone();
            let f = fork(BLOCK as u64, 2, &[(start, 1)]);
            image[ALLOCATION_FORK..ALLOCATION_FORK + 80].copy_from_slice(&fork_bytes(&f));
            install_overflow(&mut image, &[(ALLOCATION_FILE_ID, 0x00, 1, &[(total, 1)])]);
            let fs = mount(source(image)).unwrap();
            let c = corruption(fs.allocation_map(16));
            assert!(matches!(c.kind, CorruptKind::OutOfRange { .. }), "{c:?}");
            assert_eq!(list(&fs, 2).len(), 19);

            // An overflow record that is missing altogether.
            let mut image = base.clone();
            image[ALLOCATION_FORK..ALLOCATION_FORK + 80].copy_from_slice(&fork_bytes(&f));
            let fs = mount(source(image)).unwrap();
            assert!(matches!(fs.allocation_map(16), Err(Error::Corrupt(_))));

            // The bitmap's block cut off the end of the source.
            let mut image = base.clone();
            image.truncate(start as usize * BLOCK);
            let fs = mount(source(image)).unwrap();
            assert!(fs.allocation_map(16).is_err());
            assert_eq!(fs.metadata(NodeId(17)).unwrap().size, Some(100));
        }

        #[test]
        fn allocation_map_on_a_compressed_volume_with_attributes() {
            // The attributes B-tree sits before the allocation file.
            let builder = strip_volume();
            let image = builder.build();
            let fs = mounted(&builder);
            assert!(fs.attributes.is_some());
            assert_eq!(
                fs.allocation_map(4096).unwrap().density,
                naive_density(&image, 4096)
            );
        }
    }

    // ---- scan and identity ----

    pub(super) mod scan_tests {
        use std::time::Instant;

        use super::super::btree::fixtures::{
            LEAF as LEAF_KIND, MAP_RECORD_AT, header as tree_header, header_node, keyed, node,
            var_index,
        };
        use super::super::catalog::CatalogKey;
        use super::super::compression::fixtures::header as decmpfs;
        use super::super::test_volume::{folder_data, key, units};
        use super::*;
        use atlas_fs_core::fs::default_scan;
        use atlas_fs_core::io::block::MemSource;

        fn line(e: &ScanEntry<'_>) -> String {
            // The walk goes through listings, which carry no type code; the
            // scan's type codes are tested in `scan.rs`.
            let e = ScanEntry {
                finder_type: None,
                ..*e
            };
            format!("{e:?}")
        }

        /// Every entry `scan` reports, as sorted lines, and its summary.
        fn scanned(fs: &HfsPlus) -> (Vec<String>, ScanSummary) {
            let mut out = Vec::new();
            let summary = fs
                .scan(&mut |e| {
                    out.push(line(&e));
                    ControlFlow::Continue(())
                })
                .unwrap();
            out.sort();
            (out, summary)
        }

        /// Every entry a `read_dir` walk from the root reports, each folder
        /// listed once, as sorted lines.
        fn walked(fs: &HfsPlus) -> Vec<String> {
            let mut out = Vec::new();
            let summary = default_scan(fs, &mut |e| {
                out.push(line(&e));
                ControlFlow::Continue(())
            })
            .unwrap();
            assert_eq!(summary.skipped, 0, "the walk itself hit damage");
            out.sort();
            out
        }

        /// The specification of `scan`: the same entries as the walk.
        pub(in super::super) fn assert_scan_matches_walk(fs: &HfsPlus) -> ScanSummary {
            let (scan, summary) = scanned(fs);
            let walk = walked(fs);
            assert_eq!(scan.join("\n"), walk.join("\n"));
            assert_eq!(summary.entries, scan.len() as u64);
            summary
        }

        #[test]
        fn scan_matches_the_walk_with_links() {
            let summary = assert_scan_matches_walk(&mounted(&main_volume()));
            assert_eq!(summary.skipped, 0);
        }

        #[test]
        fn scan_matches_the_walk_with_forks_and_codes() {
            let summary = assert_scan_matches_walk(&mounted(&facts_volume()));
            assert_eq!(summary.skipped, 0);
        }

        #[test]
        fn scan_of_hfsx_matches_the_walk() {
            let mut v = main_volume();
            v.hfsx(0xBC);
            assert_scan_matches_walk(&mounted(&v));
        }

        /// Plain files, a compressed file, a file hard link with two names,
        /// a directory hard link with two names, and 40 files in `Trash`
        /// (800), whose records run past its thread into leaves of their own.
        fn scan_volume() -> VolumeBuilder {
            let mut v = VolumeBuilder::new("Scan");
            v.folder(ROOT, "Docs", 16)
                .file(16, "a.txt", 17, FileSpec::regular(10))
                .file(ROOT, "b.txt", 18, FileSpec::regular(20).invisible())
                .compressed_file(ROOT, "packed", 19, &decmpfs(3, 1234), None)
                .folder_units(ROOT, &FILE_PRIVATE_FOLDER, 30, 0x5000)
                .file(
                    30,
                    "iNode7",
                    31,
                    FileSpec::regular(77).modified(MODIFIED + 9),
                )
                .file(ROOT, "one", 50, FileSpec::file_link(7))
                .file(16, ".two", 51, FileSpec::file_link(7))
                .folder_units(ROOT, &DIR_PRIVATE_FOLDER, 40, 0x5000)
                .folder(40, "dir_9", 41)
                .file(41, "deep.txt", 42, FileSpec::regular(5))
                .file(ROOT, "linked dir", 52, FileSpec::dir_link(9))
                .file(16, "again", 53, FileSpec::dir_link(9))
                .folder(ROOT, "Trash", 800);
            for i in 0..40u32 {
                v.file(
                    800,
                    &format!("gone {i:02}"),
                    900 + i,
                    FileSpec::regular(u64::from(i)),
                );
            }
            v
        }

        /// Byte offset of the catalog B-tree in a `test_volume` image.
        fn catalog_at(image: &[u8]) -> usize {
            let start = u32::from_be_bytes(image[1024 + 288..1024 + 292].try_into().unwrap());
            start as usize * BLOCK
        }

        /// Frees the catalog leaves after the one holding Trash's thread,
        /// as deleting their files would: their bits are cleared (when
        /// `clear_bits`) and the leaf chain ends before them. The records
        /// stay in the nodes. Returns the names left in the freed nodes.
        fn free_trash_leaves(image: &mut [u8], clear_bits: bool) -> Vec<String> {
            let at = catalog_at(image);
            let total = u32::from_be_bytes(image[at + 14 + 22..at + 14 + 26].try_into().unwrap());
            let tree = BTree::open(
                Arc::new(MemSource::new(
                    image[at..at + total as usize * BLOCK].to_vec(),
                )),
                "t",
            )
            .unwrap();
            let mut thread_leaf = None;
            let mut freed = Vec::new();
            for n in 2..total {
                let node = tree.read_node(n).unwrap();
                assert_eq!(node.kind(), btree::NodeKind::Leaf);
                for i in 0..node.len() {
                    let rec = node.record(i).unwrap();
                    let k = CatalogKey::parse(rec.key).unwrap();
                    if k.parent_id == 800 && k.name_be.is_empty() {
                        thread_leaf = Some(n);
                    } else if thread_leaf.is_some_and(|t| t < n) && k.parent_id == 800 {
                        let mut name = String::new();
                        unicode::compose_stored(k.name_be, &mut name);
                        freed.push(name);
                    }
                }
                if let Some(t) = thread_leaf
                    && t < n
                {
                    if clear_bits {
                        let byte = at + MAP_RECORD_AT + n as usize / 8;
                        image[byte] &= !(0x80u8 >> (n % 8));
                    }
                    if n == t + 1 {
                        let last = at + t as usize * BLOCK;
                        image[last..last + 4].copy_from_slice(&0u32.to_be_bytes());
                    }
                }
            }
            freed
        }

        #[test]
        fn scan_matches_the_walk_and_skips_freed_nodes() {
            let mut image = scan_volume().build();
            let freed = free_trash_leaves(&mut image, true);
            assert!(freed.len() >= 10, "{freed:?}");
            let fs = mount(source(image)).unwrap();
            let summary = assert_scan_matches_walk(&fs);
            assert_eq!(summary.skipped, 0);
            let (lines, _) = scanned(&fs);
            for name in &freed {
                assert!(!lines.iter().any(|l| l.contains(&format!("{name:?}"))));
            }
            // Some of Trash is live, in the thread's own leaf.
            assert!(lines.iter().any(|l| l.contains("gone 00")));
            // The two names of each hard link, and the compressed size.
            assert_eq!(
                lines
                    .iter()
                    .filter(|l| l.contains("node: NodeId(31)"))
                    .count(),
                2
            );
            assert_eq!(
                lines
                    .iter()
                    .filter(|l| l.contains("node: NodeId(41)"))
                    .count(),
                2
            );
            assert!(
                lines
                    .iter()
                    .any(|l| l.contains("\"packed\"") && l.contains("Some(1234)"))
            );
        }

        #[test]
        fn freed_nodes_are_left_out_by_the_bitmap_alone() {
            // Unlinked from the chain but still marked used: the scan reads
            // them (stale entries), so the bitmap is what keeps them out.
            let mut image = scan_volume().build();
            let freed = free_trash_leaves(&mut image, false);
            let fs = mount(source(image)).unwrap();
            let (lines, _) = scanned(&fs);
            assert_eq!(lines.len(), walked(&fs).len() + freed.len());
        }

        #[test]
        fn scan_counts_a_damaged_record_and_keeps_the_rest() {
            let builder = scan_volume();
            let clean = mounted(&builder);
            let (all, _) = scanned(&clean);
            let mut image = builder.build();
            // The record type of "b.txt"'s file record becomes 9.
            let k = key(ROOT, &units("b.txt"));
            let mut rec = (k.len() as u16).to_be_bytes().to_vec();
            rec.extend_from_slice(&k);
            let at = image.windows(rec.len()).position(|w| w == rec).unwrap() + rec.len();
            image[at..at + 2].copy_from_slice(&9u16.to_be_bytes());
            let fs = mount(source(image)).unwrap();
            let (lines, summary) = scanned(&fs);
            assert_eq!(summary.skipped, 1);
            let expect: Vec<String> = all
                .into_iter()
                .filter(|l| !l.contains("\"b.txt\""))
                .collect();
            assert_eq!(lines, expect);
        }

        #[test]
        fn scan_stops_on_break_in_either_pass() {
            let fs = mounted(&main_volume());
            let (all, _) = scanned(&fs);
            for stop in 1..=all.len() {
                let mut seen = 0;
                let summary = fs
                    .scan(&mut |_| {
                        seen += 1;
                        if seen == stop {
                            ControlFlow::Break(())
                        } else {
                            ControlFlow::Continue(())
                        }
                    })
                    .unwrap();
                assert_eq!(seen, stop);
                assert_eq!(summary.entries, stop as u64);
            }
        }

        #[test]
        fn scan_of_a_volume_without_links_or_attributes() {
            let mut v = VolumeBuilder::new("Bare");
            v.folder(ROOT, "a", 16)
                .file(16, "b", 17, FileSpec::regular(1));
            let summary = assert_scan_matches_walk(&mounted(&v));
            assert_eq!(
                summary,
                ScanSummary {
                    entries: 2,
                    skipped: 0
                }
            );
            let empty = VolumeBuilder::new("Empty");
            assert_eq!(
                assert_scan_matches_walk(&mounted(&empty)),
                ScanSummary::default()
            );
        }

        fn put32(image: &mut [u8], at: usize, v: u32) {
            image[1024 + at..1024 + at + 4].copy_from_slice(&v.to_be_bytes());
        }

        #[test]
        fn identity_from_the_header() {
            let mut image = main_volume().build();
            put32(&mut image, 104, 0x0102_0304); // finderInfo[6]
            put32(&mut image, 108, 0x0506_0708); // finderInfo[7]
            put32(&mut image, 16, 0x1112_1314); // createDate
            put32(&mut image, 68, 0x2122_2324); // writeCount
            put32(&mut image, 20, 0x3132_3334); // modifyDate
            put32(&mut image, 32, 0x4142_4344); // fileCount
            put32(&mut image, 36, 0x5152_5354); // folderCount
            let id = mount(source(image.clone())).unwrap().identity().unwrap();
            assert_eq!(
                id,
                VolumeIdentity {
                    id: [
                        1, 2, 3, 4, 5, 6, 7, 8, 0x11, 0x12, 0x13, 0x14, 0, 0, 0x10, 0
                    ],
                    change: [
                        0x21, 0x22, 0x23, 0x24, 0x31, 0x32, 0x33, 0x34, 0x41, 0x42, 0x43, 0x44,
                        0x51, 0x52, 0x53, 0x54
                    ],
                }
            );

            // Another mount: the identity stays, the change marker moves.
            put32(&mut image, 68, 0x2122_2325);
            let again = mount(source(image.clone())).unwrap().identity().unwrap();
            assert_eq!(again.id, id.id);
            assert_ne!(again.change, id.change);

            // No volume identifier: still an identity, from the rest.
            put32(&mut image, 104, 0);
            put32(&mut image, 108, 0);
            let bare = mount(source(image)).unwrap().identity().unwrap();
            assert_eq!(bare.id[..8], [0; 8]);
            assert_eq!(bare.id[8..], id.id[8..]);
        }

        // ---- throughput ----

        fn file_record(id: u32, size: u64) -> Vec<u8> {
            let mut d = vec![0u8; 248];
            d[0..2].copy_from_slice(&2u16.to_be_bytes());
            d[8..12].copy_from_slice(&id.to_be_bytes());
            d[16..20].copy_from_slice(&MODIFIED.to_be_bytes());
            d[42..44].copy_from_slice(&0o100_644u16.to_be_bytes());
            d[48..52].copy_from_slice(b"TEXT");
            d[88..96].copy_from_slice(&size.to_be_bytes());
            d
        }

        fn thread_record(ty: u16, parent: u32, name: &[u16]) -> Vec<u8> {
            let mut d = vec![0u8; 10];
            d[0..2].copy_from_slice(&ty.to_be_bytes());
            d[4..8].copy_from_slice(&parent.to_be_bytes());
            d[8..10].copy_from_slice(&(name.len() as u16).to_be_bytes());
            for u in name {
                d.extend_from_slice(&u.to_be_bytes());
            }
            d
        }

        /// Packs `records` (keyed, in key order) into nodes of one level:
        /// returns the nodes' record lists.
        fn pack(records: Vec<Vec<u8>>) -> Vec<Vec<Vec<u8>>> {
            let mut nodes: Vec<Vec<Vec<u8>>> = vec![Vec::new()];
            let mut used = 14 + 2;
            for r in records {
                if used + r.len() + 2 > BLOCK {
                    nodes.push(Vec::new());
                    used = 14 + 2;
                }
                used += r.len() + 2;
                nodes.last_mut().unwrap().push(r);
            }
            nodes
        }

        /// The key bytes of a keyed record.
        fn key_of(rec: &[u8]) -> &[u8] {
            let len = usize::from(u16::from_be_bytes([rec[0], rec[1]]));
            &rec[2..2 + len]
        }

        /// A volume with `files` files in the root. The test builder keeps
        /// the catalog to one index node, so this lays out a deeper catalog
        /// itself and puts it after a built empty volume.
        fn big_volume(files: u32) -> Vec<u8> {
            let name = |i: u32| units(&format!("f{i:07}"));
            let mut records = Vec::with_capacity(2 * files as usize + 2);
            records.push(keyed(&key(1, &units("Bench")), &folder_data(ROOT, 0)));
            records.push(keyed(
                &key(ROOT, &[]),
                &thread_record(3, 1, &units("Bench")),
            ));
            for i in 0..files {
                records.push(keyed(
                    &key(ROOT, &name(i)),
                    &file_record(16 + i, u64::from(i)),
                ));
            }
            for i in 0..files {
                records.push(keyed(&key(16 + i, &[]), &thread_record(4, ROOT, &name(i))));
            }
            let leaf_count = records.len();
            let leaves = pack(records);

            // Number the nodes: header 0, leaves from 1, index levels after.
            let mut nodes: Vec<Vec<u8>> = Vec::new();
            let mut level: Vec<(Vec<u8>, u32)> = Vec::new();
            let count = leaves.len() as u32;
            for (i, recs) in leaves.iter().enumerate() {
                let number = 1 + i as u32;
                let forward = if number < count { number + 1 } else { 0 };
                let backward = number - 1;
                nodes.push(node(BLOCK, LEAF_KIND, 1, forward, backward, recs));
                level.push((key_of(&recs[0]).to_vec(), number));
            }
            let mut next = 1 + count;
            let mut depth = 1u16;
            while level.len() > 1 || depth == 1 {
                depth += 1;
                let recs: Vec<Vec<u8>> = level.iter().map(|(k, c)| var_index(k, *c)).collect();
                let mut upper = Vec::new();
                for group in pack(recs) {
                    nodes.push(node(BLOCK, 0, depth as u8, 0, 0, &group));
                    upper.push((key_of(&group[0]).to_vec(), next));
                    next += 1;
                }
                level = upper;
            }
            let mut h = tree_header(depth, next - 1, 1, count, next);
            h.node_size = BLOCK as u16;
            h.max_key_length = 516;
            h.leaf_records = leaf_count as u32;
            let mut catalog = header_node(&h);
            for n in nodes {
                catalog.extend_from_slice(&n);
            }

            let mut image = VolumeBuilder::new("Bench").build();
            let start = (image.len() / BLOCK) as u32;
            let blocks = (catalog.len() / BLOCK) as u32;
            image.extend_from_slice(&catalog);
            image.resize(image.len() + BLOCK, 0); // spare block
            put32(&mut image, 44, start + blocks + 1); // totalBlocks
            let fork = 272;
            image[1024 + fork..1024 + fork + 8]
                .copy_from_slice(&(catalog.len() as u64).to_be_bytes());
            put32(&mut image, fork + 12, blocks);
            put32(&mut image, fork + 16, start);
            put32(&mut image, fork + 20, blocks);
            image
        }

        #[test]
        fn big_catalog_layout_scans_like_the_walk() {
            let fs = mount(source(big_volume(3000))).unwrap();
            let summary = assert_scan_matches_walk(&fs);
            assert_eq!(summary.entries, 3000);
        }

        #[test]
        #[ignore = "timing; run in release with --ignored --nocapture"]
        // Printing the measurement is this test's purpose.
        #[allow(clippy::print_stdout)]
        fn hfsplus_scan_throughput() {
            const FILES: u32 = 200_000;
            let built = Instant::now();
            let image = big_volume(FILES);
            let build_time = built.elapsed();
            let bytes = image.len();
            let fs = mount(Arc::new(MemSource::new(image))).unwrap();
            let mut best = f64::INFINITY;
            let mut entries = 0;
            for _ in 0..5 {
                let mut names = 0usize;
                let started = Instant::now();
                let summary = fs
                    .scan(&mut |e| {
                        names += e.name.len();
                        ControlFlow::Continue(())
                    })
                    .unwrap();
                best = best.min(started.elapsed().as_secs_f64());
                entries = summary.entries;
                assert!(names > 0);
            }
            assert_eq!(entries, u64::from(FILES));
            println!(
                "hfsplus_scan_throughput: {entries} entries in {:.1} ms (best of 5) = {:.0} entries/s; \
                 catalog image {} MiB built in {:.0} ms",
                best * 1e3,
                entries as f64 / best,
                bytes >> 20,
                build_time.as_secs_f64() * 1e3,
            );
        }
    }

    #[test]
    fn hfs_plus_is_send_sync_and_streams_are_send() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<HfsPlus>();
        let fs: Arc<dyn FileSystem> = Arc::new(mounted(&main_volume()));
        let stream = fs.read_dir(fs.root()).unwrap();
        let handle = std::thread::spawn(move || {
            let mut stream = stream;
            let mut batch = DirBatch::new();
            while stream.next_batch(&mut batch, 100).unwrap() == DirProgress::More {}
            batch.len()
        });
        assert_eq!(handle.join().unwrap(), 19);
    }
}
