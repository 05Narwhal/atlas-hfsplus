// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The volume scan for the search index: sequential passes over the B-trees.
//!
//! Three steps, each a front-to-back read of a B-tree's allocated leaves
//! ([`BTree::scan_leaves`](super::btree::BTree::scan_leaves)):
//!
//! 1. The attributes tree, for the uncompressed size of every compressed
//!    file (its inline `com.apple.decmpfs` header).
//! 2. The catalog. Every record a listing would show is reported as it is
//!    met, through the listing's own record reader, except hard links: they
//!    are kept aside, and the link targets in the private folders are noted.
//! 3. The hard links, resolved against the targets the catalog pass noted.
//!
//! So a hard link costs no catalog descent and a compressed file no
//! attributes descent, which is what makes a volume with millions of either
//! scannable in seconds.

use std::collections::HashMap;
use std::ops::ControlFlow;

use super::HfsPlus;
use super::attributes::{Attributes, Value, parse_key as parse_attr_key, parse_value};
use super::btree::Node;
use super::catalog::{CatalogKey, Facts, ListStep, ROOT_FOLDER_ID, ROOT_PARENT_ID, list_step};
use super::compression::{DECMPFS_XATTR, DecmpfsHeader, HEADER_LEN};
use super::links::{DIR_TARGET_PREFIX, FILE_TARGET_PREFIX, Link, LinkKind, TargetName};
use super::unicode::NameCompare;
use super::volume_header::hfs_time;
use atlas_fs_core::error::Result;
use atlas_fs_core::fs::{EntryFlags, NodeId, NodeKind, ScanEntry, ScanSummary};

/// Most compressed sizes step 1 keeps; files past it have no size.
pub const MAX_COMPRESSED: usize = 50_000_000;

/// Most hard links step 2 keeps for step 3; links past it are reported
/// unresolved. Also the most targets each private folder's map keeps.
pub const MAX_PENDING_LINKS: usize = 50_000_000;

/// What a link shows of its target: the facts a listing takes from it.
#[derive(Clone, Copy)]
struct Target {
    node: u32,
    kind: NodeKind,
    size: Option<u64>,
    resource_size: Option<u64>,
    modified: u32,
    compressed: bool,
    finder_type: Option<[u8; 4]>,
}

/// A hard link met in the catalog pass, reported in step 3.
struct PendingLink {
    parent: u32,
    /// Start of the display name in the name arena.
    name_start: usize,
    /// Length of the display name in bytes (at most 255 units of 3 bytes).
    name_len: u16,
    link: Link,
    /// The link's own flags (`HIDDEN`), which it keeps.
    flags: EntryFlags,
    /// The link file's own ID, reported when the target is missing.
    own_id: u32,
}

/// The state of one scan.
struct Scan<'v> {
    visit: &'v mut dyn FnMut(ScanEntry<'_>) -> ControlFlow<()>,
    summary: ScanSummary,
    /// Set once the visitor returned `Break`.
    stopped: bool,
}

impl Scan<'_> {
    /// Passes one entry to the visitor and counts it.
    fn report(&mut self, entry: ScanEntry<'_>) -> ControlFlow<()> {
        self.summary.entries = self.summary.entries.saturating_add(1);
        let flow = (self.visit)(entry);
        if flow.is_break() {
            self.stopped = true;
        }
        flow
    }

    fn skip(&mut self, n: u64) {
        self.summary.skipped = self.summary.skipped.saturating_add(n);
    }
}

/// [`FileSystem::scan`](atlas_fs_core::fs::FileSystem::scan) for HFS+ and HFSX.
pub(super) fn scan(
    fs: &HfsPlus,
    visit: &mut dyn FnMut(ScanEntry<'_>) -> ControlFlow<()>,
) -> Result<ScanSummary> {
    let mut scan = Scan {
        visit,
        summary: ScanSummary::default(),
        stopped: false,
    };
    let sizes = match &fs.attributes {
        Some(attributes) => compressed_sizes(attributes, &mut scan),
        None => HashMap::new(),
    };
    let mut pass = CatalogPass::new(fs, &sizes);
    let leaves = fs
        .catalog
        .tree
        .scan_leaves(&mut |node: &Node| Ok(pass.leaf(node, &mut scan)))?;
    scan.skip(u64::from(leaves.skipped));
    if scan.stopped {
        return Ok(scan.summary);
    }
    pass.resolve_links(&mut scan);
    Ok(scan.summary)
}

/// Step 1: `file_id -> uncompressed size` from every inline decmpfs header.
///
/// A record that fails to parse is ignored (the file then has no size, as
/// in a listing). Damaged nodes are counted as skipped; a tree that cannot
/// be scanned at all costs the sizes it had not reached and counts as one
/// skipped structure, since a listing would not fail over it either.
fn compressed_sizes(attributes: &Attributes, scan: &mut Scan<'_>) -> HashMap<u32, u64> {
    let mut sizes: HashMap<u32, u64> = HashMap::new();
    let mut name = [0u8; 2 * DECMPFS_XATTR.len()];
    for (pair, unit) in name
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(DECMPFS_XATTR.encode_utf16())
    {
        *pair = unit.to_be_bytes();
    }
    let result = attributes.tree.scan_leaves(&mut |node: &Node| {
        // Bounded by the node's record count.
        for index in 0..node.len() {
            let Ok(rec) = node.record(index) else {
                continue;
            };
            let Ok(key) = parse_attr_key(rec.key) else {
                continue;
            };
            // A lookup reads the record (file, name, start block 0).
            if key.name_be != name.as_slice() || key.start_block != 0 {
                continue;
            }
            let Ok(Some(Value::Inline(value))) = parse_value(rec.data) else {
                continue;
            };
            let head = value.get(..HEADER_LEN).unwrap_or(value);
            let Ok(header) = DecmpfsHeader::parse(head, key.file_id) else {
                continue;
            };
            if sizes.len() >= MAX_COMPRESSED {
                return Ok(ControlFlow::Break(()));
            }
            // The first record wins, as the first one a lookup finds.
            sizes.entry(key.file_id).or_insert(header.uncompressed_size);
        }
        Ok(ControlFlow::Continue(()))
    });
    match result {
        Ok(leaves) => scan.skip(u64::from(leaves.skipped)),
        Err(_) => scan.skip(1),
    }
    sizes
}

/// Step 2 and 3: the catalog pass and the links it set aside.
struct CatalogPass<'a> {
    fs: &'a HfsPlus,
    sizes: &'a HashMap<u32, u64>,
    compare: NameCompare,
    file_folder: Option<u32>,
    dir_folder: Option<u32>,
    /// Targets of file links by link reference (`iNode<n>`).
    file_targets: HashMap<u32, Target>,
    /// Targets of directory links by link reference (`dir_<n>`).
    dir_targets: HashMap<u32, Target>,
    pending: Vec<PendingLink>,
    /// The display names of `pending`, back to back.
    arena: String,
    /// The current record's display name, reused.
    name: String,
}

impl<'a> CatalogPass<'a> {
    fn new(fs: &'a HfsPlus, sizes: &'a HashMap<u32, u64>) -> Self {
        Self {
            fs,
            sizes,
            compare: fs.catalog.compare(),
            file_folder: fs.links.file_folder(),
            dir_folder: fs.links.dir_folder(),
            file_targets: HashMap::new(),
            dir_targets: HashMap::new(),
            pending: Vec::new(),
            arena: String::new(),
            name: String::new(),
        }
    }

    /// The size a listing shows for an entry with `facts`.
    fn size_of(&self, facts: &Facts) -> Option<u64> {
        if facts.compressed {
            self.sizes.get(&facts.node).copied()
        } else {
            facts.size
        }
    }

    /// Every record of one catalog leaf.
    fn leaf(&mut self, node: &Node, scan: &mut Scan<'_>) -> ControlFlow<()> {
        // Bounded by the node's record count.
        for index in 0..node.len() {
            let Ok(rec) = node.record(index) else {
                scan.skip(1);
                continue;
            };
            let Some(key) = CatalogKey::parse(rec.key) else {
                scan.skip(1);
                continue;
            };
            let parent = key.parent_id;
            // The listing's own reader: same kind, size, flags and name.
            let facts = match list_step(rec, parent, &mut self.name) {
                Ok(ListStep::Entry(facts)) => facts,
                // Thread records (`End` cannot happen: the parent matches).
                Ok(ListStep::Skip | ListStep::End) => continue,
                Err(_) => {
                    scan.skip(1);
                    continue;
                }
            };
            self.record(parent, key.name_be, facts, scan)?;
        }
        ControlFlow::Continue(())
    }

    /// One folder or file record.
    fn record(
        &mut self,
        parent: u32,
        name_be: &[u8],
        facts: Facts,
        scan: &mut Scan<'_>,
    ) -> ControlFlow<()> {
        // The root itself (keyed by its parent, 1) is never reported.
        if parent == ROOT_PARENT_ID || facts.node == ROOT_FOLDER_ID {
            return ControlFlow::Continue(());
        }
        if Some(parent) == self.file_folder {
            // A file link target: a file that is not itself a link.
            if facts.kind != NodeKind::Directory && facts.link.is_none() {
                self.note_target(LinkKind::File, name_be, &facts);
            }
            return ControlFlow::Continue(());
        }
        if Some(parent) == self.dir_folder {
            if facts.kind == NodeKind::Directory {
                self.note_target(LinkKind::Directory, name_be, &facts);
            }
            return ControlFlow::Continue(());
        }
        if super::hidden_in_listing(&self.fs.links, parent, facts.node) {
            return ControlFlow::Continue(());
        }
        if let Some(link) = facts.link {
            if self.pending.len() < MAX_PENDING_LINKS
                && let Some(pending) = self.keep_link(parent, link, &facts)
            {
                self.pending.push(pending);
                return ControlFlow::Continue(());
            }
            return scan.report(unresolved(parent, &self.name, &facts));
        }
        let flags = if facts.compressed {
            facts.flags.with(EntryFlags::COMPRESSED)
        } else {
            facts.flags
        };
        let entry = ScanEntry {
            node: NodeId(u64::from(facts.node)),
            parent: NodeId(u64::from(parent)),
            name: &self.name,
            kind: facts.kind,
            size: self.size_of(&facts),
            resource_size: facts.resource_size,
            modified: hfs_time(facts.modified),
            flags,
            finder_type: facts.finder_type,
        };
        scan.report(entry)
    }

    /// Notes a link target named `<prefix><n>` under link reference `n`.
    fn note_target(&mut self, kind: LinkKind, name_be: &[u8], facts: &Facts) {
        let (prefix, targets) = match kind {
            LinkKind::File => (FILE_TARGET_PREFIX, &mut self.file_targets),
            LinkKind::Directory => (DIR_TARGET_PREFIX, &mut self.dir_targets),
        };
        let Some(number) = target_number(name_be, prefix, self.compare) else {
            return;
        };
        if targets.len() >= MAX_PENDING_LINKS {
            return;
        }
        let size = if facts.compressed {
            self.sizes.get(&facts.node).copied()
        } else {
            facts.size
        };
        // The first record wins, as in a lookup.
        targets.entry(number).or_insert(Target {
            node: facts.node,
            kind: facts.kind,
            size,
            resource_size: facts.resource_size,
            modified: facts.modified,
            compressed: facts.compressed,
            finder_type: facts.finder_type,
        });
    }

    /// Copies the current name into the arena for a link reported later.
    fn keep_link(&mut self, parent: u32, link: Link, facts: &Facts) -> Option<PendingLink> {
        let name_len = u16::try_from(self.name.len()).ok()?;
        let name_start = self.arena.len();
        self.arena.push_str(&self.name);
        Some(PendingLink {
            parent,
            name_start,
            name_len,
            link,
            flags: facts.flags,
            own_id: facts.node,
        })
    }

    /// Step 3: every link set aside, through its target when there is one.
    fn resolve_links(self, scan: &mut Scan<'_>) {
        let Self {
            pending,
            arena,
            file_targets,
            dir_targets,
            ..
        } = self;
        // Bounded by the pending list.
        for link in &pending {
            let end = link.name_start.saturating_add(usize::from(link.name_len));
            let name = arena.get(link.name_start..end).unwrap_or("");
            let targets = match link.link.kind {
                LinkKind::File => &file_targets,
                LinkKind::Directory => &dir_targets,
            };
            let parent = NodeId(u64::from(link.parent));
            let entry = match targets.get(&link.link.reference) {
                // As `Facts::resolved`: the link's name and flags, the
                // target's node, kind, sizes, date and type code.
                Some(t) => ScanEntry {
                    node: NodeId(u64::from(t.node)),
                    parent,
                    name,
                    kind: t.kind,
                    size: t.size,
                    resource_size: t.resource_size,
                    modified: hfs_time(t.modified),
                    flags: if t.compressed {
                        link.flags.with(EntryFlags::COMPRESSED)
                    } else {
                        link.flags
                    },
                    finder_type: t.finder_type,
                },
                None => ScanEntry {
                    node: NodeId(u64::from(link.own_id)),
                    parent,
                    name,
                    kind: NodeKind::File,
                    size: None,
                    resource_size: None,
                    modified: None,
                    flags: link.flags,
                    finder_type: None,
                },
            };
            if scan.report(entry).is_break() {
                return;
            }
        }
    }
}

/// A link reported as a listing shows one it cannot resolve: its own ID, a
/// file, no size and no date.
fn unresolved<'n>(parent: u32, name: &'n str, facts: &Facts) -> ScanEntry<'n> {
    ScanEntry {
        node: NodeId(u64::from(facts.node)),
        parent: NodeId(u64::from(parent)),
        name,
        kind: NodeKind::File,
        size: None,
        resource_size: None,
        modified: None,
        flags: facts.flags,
        finder_type: None,
    }
}

/// The link reference `n` of a target called `<prefix><n>`: the name a
/// lookup for that reference would find, so with the volume's case rules
/// and no leading zeros.
fn target_number(name_be: &[u8], prefix: &[u8], compare: NameCompare) -> Option<u32> {
    let digits = name_be.get(prefix.len().checked_mul(2)?..)?;
    if digits.is_empty() {
        return None;
    }
    let mut number: u32 = 0;
    // Bounded by the name's length; a u32 overflows after ten digits.
    for pair in digits.chunks(2) {
        let [0, d @ b'0'..=b'9'] = pair else {
            return None;
        };
        number = number
            .checked_mul(10)?
            .checked_add(u32::from(d.checked_sub(b'0')?))?;
    }
    let name = TargetName::new(prefix, number)?;
    (compare.compare_stored(name_be, name.units()) == std::cmp::Ordering::Equal).then_some(number)
}

#[cfg(test)]
mod tests {
    use super::super::mount;
    use super::super::test_volume::{finder_volume, source};
    use super::*;
    use atlas_fs_core::fs::{DirBatch, DirProgress, FileSystem};

    fn be(name: &str) -> Vec<u8> {
        name.encode_utf16().flat_map(u16::to_be_bytes).collect()
    }

    #[test]
    fn target_numbers() {
        let fold = NameCompare::CaseFolding;
        let bin = NameCompare::Binary;
        let n = |name: &str, c| target_number(&be(name), FILE_TARGET_PREFIX, c);
        assert_eq!(n("iNode0", bin), Some(0));
        assert_eq!(n("iNode123", bin), Some(123));
        assert_eq!(n("iNode4294967295", bin), Some(u32::MAX));
        assert_eq!(n("iNode4294967296", bin), None);
        assert_eq!(n("iNode012", bin), None);
        assert_eq!(n("iNode", bin), None);
        assert_eq!(n("iNode1a", bin), None);
        assert_eq!(n("iNode-1", bin), None);
        assert_eq!(n("xNode1", bin), None);
        assert_eq!(n("INODE7", bin), None);
        // A case-folding volume finds `iNode7` under any case.
        assert_eq!(n("INODE7", fold), Some(7));
        assert_eq!(n("iNode٣", fold), None); // ARABIC-INDIC DIGIT THREE
        let mut odd = be("iNode1");
        odd.push(b'2');
        assert_eq!(target_number(&odd, FILE_TARGET_PREFIX, bin), None);
        assert_eq!(
            target_number(&be("dir_42"), DIR_TARGET_PREFIX, bin),
            Some(42)
        );
        assert_eq!(target_number(&be("dir"), DIR_TARGET_PREFIX, bin), None);
    }

    /// `name -> (flags, finder_type)` of every entry the scan reports.
    fn scanned(fs: &dyn FileSystem) -> HashMap<String, (EntryFlags, Option<[u8; 4]>)> {
        let mut out = HashMap::new();
        fs.scan(&mut |e| {
            out.insert(e.name.to_owned(), (e.flags, e.finder_type));
            ControlFlow::Continue(())
        })
        .unwrap();
        out
    }

    #[test]
    fn finder_bits_and_type_codes_from_the_catalog() {
        let fs = mount(source(finder_volume().build())).unwrap();
        let entries = scanned(&fs);
        let get = |name: &str| *entries.get(name).unwrap_or_else(|| panic!("{name}"));

        let (flags, ty) = get("Bundle Folder");
        assert_eq!(flags, EntryFlags::PACKAGE.with_label(2));
        assert_eq!(ty, None, "a folder has no type code");

        let (flags, ty) = get("ReadMe");
        assert_eq!(flags, EntryFlags::EMPTY.with_label(7));
        assert_eq!(ty, Some(*b"TEXT"));

        let (flags, ty) = get("My Alias");
        assert_eq!(flags, EntryFlags::ALIAS.with(EntryFlags::HIDDEN));
        assert_eq!(flags.label(), 0);
        assert_eq!(ty, None, "a zero type code is none");

        // The bundle bit is kept raw on a file.
        let (flags, ty) = get("SimpleText");
        assert_eq!(flags, EntryFlags::PACKAGE);
        assert_eq!(ty, Some(*b"APPL"));

        // A hard link: its own flags, its target's type code.
        let (flags, ty) = get("Picture link");
        assert_eq!(flags, EntryFlags::EMPTY.with_label(5));
        assert_eq!(ty, Some(*b"PICT"));
        assert!(!entries.contains_key("iNode7"));

        assert_eq!(get("deep"), (EntryFlags::EMPTY, Some(*b"TEXT")));
        assert_eq!(get(".Spotlight-V100").0, EntryFlags::HIDDEN);

        // A listing reports the same flags.
        let mut stream = fs.read_dir(fs.root()).unwrap();
        let mut batch = DirBatch::new();
        let mut calls = 0;
        while stream.next_batch(&mut batch, 2).unwrap() == DirProgress::More {
            calls += 1;
            assert!(calls < 100);
        }
        assert!(batch.len() >= 6);
        for e in batch.iter() {
            assert_eq!(e.flags, get(e.name).0, "{}", e.name);
        }
        // And so does `metadata`.
        assert_eq!(
            fs.metadata(NodeId(16)).unwrap().flags,
            EntryFlags::PACKAGE.with_label(2)
        );
        assert_eq!(
            fs.metadata(NodeId(18)).unwrap().flags,
            EntryFlags::ALIAS.with(EntryFlags::HIDDEN)
        );
    }
}
