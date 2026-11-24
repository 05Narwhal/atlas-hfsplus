// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! B-tree node reader and iterators.
//!
//! Format-neutral: this module knows the node layout, the header record and
//! the key-length rules of TN1150, and nothing about any key format. Callers
//! search with their own comparator, so the catalog, extents and attributes
//! trees of HFS+ (and the trees of classic HFS) all read through it.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::ops::ControlFlow;
use std::sync::Arc;

use super::bytes::{be_u16, be_u32, u8_at};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::io::block::BlockSource;

/// Size of the node descriptor (`BTNodeDescriptor`) at the start of every node.
pub const NODE_DESCRIPTOR_LEN: usize = 14;

/// Size of the header record (`BTHeaderRec`), the first record of node 0.
pub const HEADER_RECORD_LEN: usize = 106;

/// `kBTBigKeysMask`: key lengths are `u16` rather than `u8`.
pub const BIG_KEYS_MASK: u32 = 0x0000_0002;

/// `kBTVariableIndexKeysMask`: index keys occupy their own length rather
/// than `max_key_length` bytes.
pub const VARIABLE_INDEX_KEYS_MASK: u32 = 0x0000_0004;

/// Smallest node size accepted.
const MIN_NODE_SIZE: u16 = 512;

/// Largest node size accepted.
const MAX_NODE_SIZE: u16 = 32768;

/// Hard cap on tree depth, whatever the header says.
const MAX_DEPTH: u16 = 32;

/// Bytes read from node 0 to learn the node size: descriptor plus header record.
const HEADER_PREFIX_LEN: usize = NODE_DESCRIPTOR_LEN + HEADER_RECORD_LEN;

/// Index of the map record (the node allocation bitmap) in the header node.
const MAP_RECORD: u16 = 2;

/// Bytes [`BTree::scan_leaves`] asks the fork for per `read_bulk` call,
/// before rounding down to a whole number of nodes (at least one).
pub const SCAN_PIECE: usize = 1 << 20;

/// Nodes [`BTree::scan_leaves`] lets fail to read before it gives up with
/// the read error.
pub const MAX_SKIPPED: u32 = 4096;

fn corrupt(structure: &'static str, location: Location, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(structure, location, kind))
}

/// Length as `u64` for error values.
fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// The B-tree header record (`BTHeaderRec`), read from node 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderRecord {
    /// Number of levels; 0 for an empty tree.
    pub depth: u16,
    /// Node number of the root, 0 for an empty tree.
    pub root_node: u32,
    /// Records in all leaf nodes.
    pub leaf_records: u32,
    /// Node number of the first leaf.
    pub first_leaf: u32,
    /// Node number of the last leaf.
    pub last_leaf: u32,
    /// Node size in bytes.
    pub node_size: u16,
    /// Largest key length, not counting the key length field.
    pub max_key_length: u16,
    /// Nodes in the tree, used or free.
    pub total_nodes: u32,
    /// Free nodes.
    pub free_nodes: u32,
    /// `btreeType` (0 for HFS+ control files).
    pub btree_type: u8,
    /// `keyCompareType` (HFSX catalog: `0xCF` case folding, `0xBC` binary).
    pub key_compare_type: u8,
    /// Attribute bits ([`BIG_KEYS_MASK`], [`VARIABLE_INDEX_KEYS_MASK`]).
    pub attributes: u32,
}

impl HeaderRecord {
    /// Parses a header record from the start of `b`. `None` if `b` is
    /// shorter than [`HEADER_RECORD_LEN`]. Values are not validated here.
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < HEADER_RECORD_LEN {
            return None;
        }
        Some(Self {
            depth: be_u16(b, 0)?,
            root_node: be_u32(b, 2)?,
            leaf_records: be_u32(b, 6)?,
            first_leaf: be_u32(b, 10)?,
            last_leaf: be_u32(b, 14)?,
            node_size: be_u16(b, 18)?,
            max_key_length: be_u16(b, 20)?,
            total_nodes: be_u32(b, 22)?,
            free_nodes: be_u32(b, 26)?,
            // 30: reserved1 (u16), 32: clumpSize (u32).
            btree_type: u8_at(b, 36)?,
            key_compare_type: u8_at(b, 37)?,
            attributes: be_u32(b, 38)?,
        })
    }

    fn rules(&self) -> KeyRules {
        KeyRules {
            big_keys: self.attributes & BIG_KEYS_MASK != 0,
            variable_index_keys: self.attributes & VARIABLE_INDEX_KEYS_MASK != 0,
            max_key_length: self.max_key_length,
        }
    }
}

/// What a node needs from the header to split records into key and data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KeyRules {
    big_keys: bool,
    variable_index_keys: bool,
    max_key_length: u16,
}

/// The kind of a node (`kind` in the node descriptor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// `kBTLeafNode` (-1): records with keys and data.
    Leaf,
    /// `kBTIndexNode` (0): records with keys and child node numbers.
    Index,
    /// `kBTHeaderNode` (1): node 0.
    Header,
    /// `kBTMapNode` (2): allocation bitmap continuation.
    Map,
}

impl NodeKind {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0xFF => Some(Self::Leaf),
            0x00 => Some(Self::Index),
            0x01 => Some(Self::Header),
            0x02 => Some(Self::Map),
            _ => None,
        }
    }

    /// The on-disk byte, for error values.
    fn to_byte(self) -> u8 {
        match self {
            Self::Leaf => 0xFF,
            Self::Index => 0x00,
            Self::Header => 0x01,
            Self::Map => 0x02,
        }
    }
}

/// One keyed record: the key (after its length field) and the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    /// Key bytes, not including the key length field.
    pub key: &'a [u8],
    /// Data bytes: the leaf payload, or the child node number in an index node.
    /// Includes any trailing pad byte.
    pub data: &'a [u8],
}

/// One validated node.
///
/// Construction checks the descriptor and the offset table; record access
/// reads in place from the node buffer.
pub struct Node {
    number: u32,
    buf: Box<[u8]>,
    kind: NodeKind,
    height: u8,
    forward: u32,
    backward: u32,
    count: u16,
    rules: KeyRules,
    structure: &'static str,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("number", &self.number)
            .field("kind", &self.kind)
            .field("height", &self.height)
            .field("forward", &self.forward)
            .field("backward", &self.backward)
            .field("records", &self.count)
            .finish_non_exhaustive()
    }
}

impl Node {
    /// Validates `buf` as node `number`.
    fn parse(
        number: u32,
        buf: Box<[u8]>,
        rules: KeyRules,
        structure: &'static str,
    ) -> Result<Self> {
        let at = Location::Node(number);
        let size = buf.len();
        let short = || {
            corrupt(
                structure,
                at,
                CorruptKind::Truncated {
                    needed: len64(NODE_DESCRIPTOR_LEN),
                    available: len64(size),
                },
            )
        };
        let forward = be_u32(&buf, 0).ok_or_else(short)?;
        let backward = be_u32(&buf, 4).ok_or_else(short)?;
        let kind_byte = u8_at(&buf, 8).ok_or_else(short)?;
        let height = u8_at(&buf, 9).ok_or_else(short)?;
        let count = be_u16(&buf, 10).ok_or_else(short)?;
        if size < NODE_DESCRIPTOR_LEN {
            return Err(short());
        }
        let kind = NodeKind::from_byte(kind_byte).ok_or_else(|| {
            corrupt(
                structure,
                at,
                CorruptKind::InvalidValue {
                    field: "node kind",
                    value: u64::from(kind_byte),
                },
            )
        })?;

        // The offset table holds count + 1 entries and must fit after the
        // descriptor.
        let table_len = usize::from(count)
            .checked_add(1)
            .and_then(|n| n.checked_mul(2));
        let table_start = table_len
            .and_then(|t| size.checked_sub(t))
            .filter(|&start| start >= NODE_DESCRIPTOR_LEN)
            .ok_or_else(|| {
                let fit = size
                    .saturating_sub(NODE_DESCRIPTOR_LEN)
                    .checked_div(2)
                    .unwrap_or(0)
                    .saturating_sub(1);
                corrupt(
                    structure,
                    at,
                    CorruptKind::OutOfRange {
                        field: "record count",
                        value: u64::from(count),
                        limit_name: "records that fit in the node",
                        limit: len64(fit),
                    },
                )
            })?;

        let node = Self {
            number,
            buf,
            kind,
            height,
            forward,
            backward,
            count,
            rules,
            structure,
        };

        let mut prev: Option<usize> = None;
        for i in 0..=count {
            let is_free = i == count;
            let loc = if is_free {
                at
            } else {
                Location::Record {
                    node: number,
                    record: i,
                }
            };
            let field = if is_free {
                "free space offset"
            } else {
                "record offset"
            };
            let off = node.offset(i).ok_or_else(short)?;
            if off < NODE_DESCRIPTOR_LEN {
                return Err(corrupt(
                    structure,
                    loc,
                    CorruptKind::InvalidValue {
                        field,
                        value: len64(off),
                    },
                ));
            }
            if off > table_start {
                return Err(corrupt(
                    structure,
                    loc,
                    CorruptKind::OutOfRange {
                        field,
                        value: len64(off),
                        limit_name: "offset table start",
                        limit: len64(table_start),
                    },
                ));
            }
            if let Some(p) = prev {
                // Records are non-empty and ascending; free space may start
                // right at the end of the last record.
                let ok = if is_free { off >= p } else { off > p };
                if !ok {
                    return Err(corrupt(
                        structure,
                        loc,
                        CorruptKind::Other("record offsets are not in ascending order"),
                    ));
                }
            }
            prev = Some(off);
        }
        Ok(node)
    }

    /// Offset-table entry `i` (0..=count). Bounds were checked at parse.
    fn offset(&self, i: u16) -> Option<usize> {
        let back = usize::from(i).checked_add(1)?.checked_mul(2)?;
        let at = self.buf.len().checked_sub(back)?;
        be_u16(&self.buf, at).map(usize::from)
    }

    fn into_buf(self) -> Box<[u8]> {
        self.buf
    }

    /// The node's number in the tree.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// The node's kind.
    pub fn kind(&self) -> NodeKind {
        self.kind
    }

    /// Level in the tree: 1 for leaves, 0 for header and map nodes.
    pub fn height(&self) -> u8 {
        self.height
    }

    /// Next node of the same kind and level, 0 for none.
    pub fn forward(&self) -> u32 {
        self.forward
    }

    /// Previous node of the same kind and level, 0 for none.
    pub fn backward(&self) -> u32 {
        self.backward
    }

    /// Number of records.
    pub fn len(&self) -> u16 {
        self.count
    }

    /// True when the node holds no records.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn record_error(&self, index: u16, kind: CorruptKind) -> Error {
        corrupt(
            self.structure,
            Location::Record {
                node: self.number,
                record: index,
            },
            kind,
        )
    }

    /// Record `index` as raw bytes (header and map nodes have no keys).
    pub fn raw_record(&self, index: u16) -> Result<&[u8]> {
        if index >= self.count {
            return Err(corrupt(
                self.structure,
                Location::Node(self.number),
                CorruptKind::OutOfRange {
                    field: "record index",
                    value: u64::from(index),
                    limit_name: "record count",
                    limit: u64::from(self.count),
                },
            ));
        }
        let start = self.offset(index);
        let end = index.checked_add(1).and_then(|next| self.offset(next));
        start
            .zip(end)
            .and_then(|(s, e)| self.buf.get(s..e))
            .ok_or_else(|| self.record_error(index, CorruptKind::Other("record offset invalid")))
    }

    /// Record `index` split into key and data per the tree's key-length rules.
    pub fn record(&self, index: u16) -> Result<Record<'_>> {
        if !matches!(self.kind, NodeKind::Leaf | NodeKind::Index) {
            return Err(self.record_error(
                index,
                CorruptKind::Other("header and map node records have no key"),
            ));
        }
        let rec = self.raw_record(index)?;
        let rules = self.rules;
        let (len_field, key_len) = if rules.big_keys {
            (2usize, be_u16(rec, 0))
        } else {
            (1usize, u8_at(rec, 0).map(u16::from))
        };
        let key_len = key_len.ok_or_else(|| {
            self.record_error(
                index,
                CorruptKind::Truncated {
                    needed: len64(len_field),
                    available: len64(rec.len()),
                },
            )
        })?;
        let room = rec.len().saturating_sub(len_field);
        let past_record = |value: u64| {
            self.record_error(
                index,
                CorruptKind::OutOfRange {
                    field: "key length",
                    value,
                    limit_name: "record length",
                    limit: len64(room),
                },
            )
        };
        let key_end = len_field
            .checked_add(usize::from(key_len))
            .filter(|&e| e <= rec.len())
            .ok_or_else(|| past_record(u64::from(key_len)))?;

        // Index keys of a tree without variable index keys occupy
        // max_key_length bytes whatever their own length.
        let occupied = if self.kind == NodeKind::Index && !rules.variable_index_keys {
            if key_len > rules.max_key_length {
                return Err(self.record_error(
                    index,
                    CorruptKind::OutOfRange {
                        field: "key length",
                        value: u64::from(key_len),
                        limit_name: "max key length",
                        limit: u64::from(rules.max_key_length),
                    },
                ));
            }
            rules.max_key_length
        } else {
            key_len
        };
        // TN1150: data starts on an even offset; a pad byte follows an odd
        // key length field plus key.
        let data_start = len_field
            .checked_add(usize::from(occupied))
            .and_then(|n| n.checked_add(n & 1))
            .filter(|&s| s <= rec.len())
            .ok_or_else(|| past_record(u64::from(occupied)))?;

        match (rec.get(len_field..key_end), rec.get(data_start..)) {
            (Some(key), Some(data)) => Ok(Record { key, data }),
            _ => Err(past_record(u64::from(key_len))),
        }
    }
}

/// A B-tree read from one fork.
#[derive(Clone)]
pub struct BTree {
    file: Arc<dyn BlockSource>,
    header: HeaderRecord,
    structure: &'static str,
}

impl BTree {
    /// Reads and validates the header node (node 0). `structure` names the tree in errors ("catalog B-tree").
    pub fn open(file: Arc<dyn BlockSource>, structure: &'static str) -> Result<Self> {
        let at = Location::Node(0);
        let available = file.len();
        if available < len64(HEADER_PREFIX_LEN) {
            return Err(corrupt(
                structure,
                at,
                CorruptKind::Truncated {
                    needed: len64(HEADER_PREFIX_LEN),
                    available,
                },
            ));
        }
        let mut prefix = [0u8; HEADER_PREFIX_LEN];
        file.read_at(0, &mut prefix)?;
        let header = prefix
            .get(NODE_DESCRIPTOR_LEN..)
            .and_then(HeaderRecord::parse)
            .ok_or_else(|| {
                corrupt(
                    structure,
                    at,
                    CorruptKind::Truncated {
                        needed: len64(HEADER_PREFIX_LEN),
                        available,
                    },
                )
            })?;
        let size = header.node_size;
        if !size.is_power_of_two() || !(MIN_NODE_SIZE..=MAX_NODE_SIZE).contains(&size) {
            return Err(corrupt(
                structure,
                at,
                CorruptKind::InvalidValue {
                    field: "node size",
                    value: u64::from(size),
                },
            ));
        }
        let tree = Self {
            file,
            header,
            structure,
        };
        let node = tree.read_node(0)?;
        if node.kind() != NodeKind::Header {
            return Err(tree.bad_kind(&node));
        }
        // The header record must be a whole record of the header node.
        let rec = node.raw_record(0)?;
        if rec.len() < HEADER_RECORD_LEN {
            return Err(corrupt(
                structure,
                Location::Record { node: 0, record: 0 },
                CorruptKind::Truncated {
                    needed: len64(HEADER_RECORD_LEN),
                    available: len64(rec.len()),
                },
            ));
        }
        Ok(tree)
    }

    /// The header record.
    pub fn header(&self) -> &HeaderRecord {
        &self.header
    }

    fn bad_kind(&self, node: &Node) -> Error {
        corrupt(
            self.structure,
            Location::Node(node.number()),
            CorruptKind::InvalidValue {
                field: "node kind",
                value: u64::from(node.kind().to_byte()),
            },
        )
    }

    /// Reads and validates node `number`.
    pub fn read_node(&self, number: u32) -> Result<Node> {
        self.load_node(number, None)
    }

    /// Reads node `number`, reusing `buf` when it has the node size.
    fn load_node(&self, number: u32, buf: Option<Box<[u8]>>) -> Result<Node> {
        let h = &self.header;
        let at = Location::Node(number);
        if number >= h.total_nodes {
            return Err(corrupt(
                self.structure,
                at,
                CorruptKind::OutOfRange {
                    field: "node number",
                    value: u64::from(number),
                    limit_name: "total nodes",
                    limit: u64::from(h.total_nodes),
                },
            ));
        }
        let size = u64::from(h.node_size);
        let start = u64::from(number).checked_mul(size);
        let end = start.and_then(|s| s.checked_add(size));
        let len = self.file.len();
        let (Some(start), Some(end)) = (start, end) else {
            return Err(corrupt(
                self.structure,
                at,
                CorruptKind::Overflow { field: "node end" },
            ));
        };
        if end > len {
            return Err(corrupt(
                self.structure,
                at,
                CorruptKind::OutOfRange {
                    field: "node end",
                    value: end,
                    limit_name: "fork length",
                    limit: len,
                },
            ));
        }
        let node_size = usize::from(h.node_size);
        let mut buf = match buf {
            Some(b) if b.len() == node_size => b,
            _ => vec![0u8; node_size].into_boxed_slice(),
        };
        self.file.read_at(start, &mut buf)?;
        Node::parse(number, buf, h.rules(), self.structure)
    }

    fn end_cursor(&self) -> LeafCursor {
        LeafCursor {
            tree: self.clone(),
            node: None,
            index: 0,
            visited: 0,
            last: None,
        }
    }

    /// Descends from the root to the first leaf record whose key is >= the target.
    /// `cmp` receives a record's key bytes (after the key length field) and returns that key's ordering relative to the target.
    /// The cursor may be positioned past the last record of a leaf; next_record then moves along the chain.
    pub fn seek(&self, cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>) -> Result<LeafCursor> {
        let h = &self.header;
        if h.root_node == 0 {
            return Ok(self.end_cursor());
        }
        if h.depth == 0 {
            return Err(corrupt(
                self.structure,
                Location::Node(0),
                CorruptKind::InvalidValue {
                    field: "tree depth",
                    value: 0,
                },
            ));
        }
        if h.depth > MAX_DEPTH {
            return Err(corrupt(
                self.structure,
                Location::Node(0),
                CorruptKind::LimitExceeded {
                    what: "tree depth",
                    limit: u64::from(MAX_DEPTH),
                },
            ));
        }

        let mut node = self.read_node(h.root_node)?;
        let mut height = h.depth;
        // Bounded: height falls by one per level, from at most MAX_DEPTH.
        while height > 1 {
            self.check_level(&node, height)?;
            let index = upper_bound(&node, cmp)?.saturating_sub(1);
            let rec = node.record(index)?;
            let child = be_u32(rec.data, 0).ok_or_else(|| {
                node.record_error(
                    index,
                    CorruptKind::Truncated {
                        needed: 4,
                        available: len64(rec.data.len()),
                    },
                )
            })?;
            node = self.load_node(child, Some(node.into_buf()))?;
            height = height.saturating_sub(1);
        }
        self.check_level(&node, 1)?;
        let index = lower_bound(&node, cmp)?;
        Ok(LeafCursor {
            tree: self.clone(),
            node: Some(node),
            index,
            visited: 1,
            last: None,
        })
    }

    /// Checks that `node` sits at `height`: the height matches and the kind
    /// is Leaf at height 1, Index above.
    fn check_level(&self, node: &Node, height: u16) -> Result<()> {
        if u16::from(node.height()) != height {
            return Err(corrupt(
                self.structure,
                Location::Node(node.number()),
                CorruptKind::InvalidValue {
                    field: "node height",
                    value: u64::from(node.height()),
                },
            ));
        }
        let want = if height == 1 {
            NodeKind::Leaf
        } else {
            NodeKind::Index
        };
        if node.kind() != want {
            return Err(self.bad_kind(node));
        }
        if want == NodeKind::Index && node.is_empty() {
            return Err(corrupt(
                self.structure,
                Location::Node(node.number()),
                CorruptKind::Other("index node has no records"),
            ));
        }
        Ok(())
    }

    /// A cursor at the first record of the first leaf. An empty tree (root_node 0) yields a cursor that is already at the end.
    pub fn first(&self) -> Result<LeafCursor> {
        if self.header.root_node == 0 {
            return Ok(self.end_cursor());
        }
        let node = self.read_node(self.header.first_leaf)?;
        if node.kind() != NodeKind::Leaf {
            return Err(self.bad_kind(&node));
        }
        Ok(LeafCursor {
            tree: self.clone(),
            node: Some(node),
            index: 0,
            visited: 1,
            last: None,
        })
    }

    /// Calls `visit` with every allocated leaf node, in node-number order.
    ///
    /// Reads the fork front to back in pieces of about [`SCAN_PIECE`] bytes
    /// and looks only at the nodes the tree's own allocation bitmap marks as
    /// used: a freed node still holds its old records. Nodes the bitmap does
    /// not cover count as used. Index, header and map nodes are passed over;
    /// a node that fails validation is skipped and counted. A piece that
    /// fails to read is read again node by node, so one bad sector costs
    /// one node; a node that still fails is skipped and counted.
    ///
    /// Fails when node 0 cannot be read, with the read error once more than
    /// [`MAX_SKIPPED`] nodes have failed to read, and with the visitor's
    /// error. `Break` from the visitor ends the pass with `Ok`.
    pub fn scan_leaves(
        &self,
        visit: &mut dyn FnMut(&Node) -> Result<ControlFlow<()>>,
    ) -> Result<LeafScan> {
        let h = self.header;
        let header_node = self.read_node(0)?;
        let map = self.node_map(&header_node);
        let node_size = usize::from(h.node_size);
        let size64 = u64::from(h.node_size);

        // Only whole nodes inside the fork are read. A fork shorter than
        // `total_nodes` nodes is damage: the used nodes past its end (and a
        // partial node at its end) cannot be read and count as skipped.
        let in_fork = self.file.len().checked_div(size64).unwrap_or(0);
        let readable = u32::try_from(in_fork).map_or(h.total_nodes, |n| n.min(h.total_nodes));
        let mut scan = LeafScan {
            nodes: 0,
            skipped: map.used_between(readable, h.total_nodes),
        };

        let per_piece = SCAN_PIECE.checked_div(node_size).unwrap_or(1).max(1);
        let piece_nodes = u32::try_from(per_piece).unwrap_or(u32::MAX);
        let mut piece = vec![0u8; per_piece.saturating_mul(node_size)];
        // Node 0's buffer becomes the reused node buffer.
        let mut spare = Some(header_node.into_buf());
        let mut failed: u32 = 0;
        let mut first: u32 = 0;
        // Bounded: `first` grows by at least one node per turn up to
        // `readable`, and each piece walks at most `piece_nodes` slices.
        while first < readable {
            let count = piece_nodes.min(readable.saturating_sub(first));
            let end = first.saturating_add(count);
            if !(first..end).any(|n| map.in_use(n)) {
                first = end;
                continue;
            }
            let len = usize::try_from(count)
                .ok()
                .and_then(|c| c.checked_mul(node_size));
            let offset = u64::from(first).checked_mul(size64);
            let (Some(offset), Some(buf)) = (offset, len.and_then(|l| piece.get_mut(..l))) else {
                return Err(corrupt(
                    self.structure,
                    Location::Node(first),
                    CorruptKind::Overflow { field: "node end" },
                ));
            };
            let whole = self.file.read_bulk(offset, buf).is_ok();
            let mut number = first;
            for slice in buf.chunks_exact_mut(node_size) {
                let this = number;
                number = number.saturating_add(1);
                if !map.in_use(this) {
                    continue;
                }
                if !whole {
                    let at = u64::from(this).saturating_mul(size64);
                    if let Err(e) = self.file.read_bulk(at, slice) {
                        scan.skipped = scan.skipped.saturating_add(1);
                        failed = failed.saturating_add(1);
                        if failed > MAX_SKIPPED {
                            return Err(e);
                        }
                        continue;
                    }
                }
                if self.scan_node(this, slice, &mut spare, &mut scan, visit)?
                    == ControlFlow::Break(())
                {
                    return Ok(scan);
                }
            }
            first = end;
        }
        Ok(scan)
    }

    /// One node of a leaf scan, from its bytes: passed to `visit` when it is
    /// a valid leaf, counted when it is damaged, else passed over.
    fn scan_node(
        &self,
        number: u32,
        bytes: &[u8],
        spare: &mut Option<Box<[u8]>>,
        scan: &mut LeafScan,
        visit: &mut dyn FnMut(&Node) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        // The kind byte first, so index nodes are not copied and checked
        // only to be passed over.
        match bytes.get(8).copied().and_then(NodeKind::from_byte) {
            Some(NodeKind::Leaf) => {}
            Some(_) => return Ok(ControlFlow::Continue(())),
            None => {
                scan.skipped = scan.skipped.saturating_add(1);
                return Ok(ControlFlow::Continue(()));
            }
        }
        let mut buf = match spare.take() {
            Some(b) if b.len() == bytes.len() => b,
            _ => vec![0u8; bytes.len()].into_boxed_slice(),
        };
        buf.copy_from_slice(bytes);
        let Ok(node) = Node::parse(number, buf, self.header.rules(), self.structure) else {
            scan.skipped = scan.skipped.saturating_add(1);
            return Ok(ControlFlow::Continue(()));
        };
        scan.nodes = scan.nodes.saturating_add(1);
        let flow = visit(&node);
        *spare = Some(node.into_buf());
        flow
    }

    /// The node allocation bitmap: the header node's map record, then the
    /// records of the map nodes chained from it by forward links.
    ///
    /// Keeps at most `total_nodes / 8 + 1` bytes and follows at most
    /// `total_nodes` map nodes. A link that repeats, a node that cannot be
    /// read or is not a map node ends the chain; the nodes the bitmap then
    /// does not cover count as used.
    fn node_map(&self, header_node: &Node) -> NodeMap {
        let total = self.header.total_nodes;
        let cap = usize::try_from(total.checked_div(8).unwrap_or(0))
            .unwrap_or(usize::MAX)
            .saturating_add(1);
        let mut bits: Vec<u8> = Vec::new();
        let take = |rec: &[u8], bits: &mut Vec<u8>| {
            let room = cap.saturating_sub(bits.len());
            bits.extend_from_slice(rec.get(..room).unwrap_or(rec));
        };
        let Ok(rec) = header_node.raw_record(MAP_RECORD) else {
            return NodeMap { bits };
        };
        take(rec, &mut bits);
        let mut seen: HashSet<u32> = HashSet::new();
        seen.insert(0);
        let mut next = header_node.forward();
        let mut buf: Option<Box<[u8]>> = None;
        let mut hops: u32 = 0;
        // Bounded: at most `total` hops, and each map record adds at least
        // one byte (records are non-empty) until `cap` is reached.
        while next != 0 && bits.len() < cap && hops < total && seen.insert(next) {
            hops = hops.saturating_add(1);
            let Ok(node) = self.load_node(next, buf.take()) else {
                break;
            };
            if node.kind() != NodeKind::Map {
                break;
            }
            let Ok(rec) = node.raw_record(0) else {
                break;
            };
            take(rec, &mut bits);
            next = node.forward();
            buf = Some(node.into_buf());
        }
        NodeMap { bits }
    }
}

/// What [`BTree::scan_leaves`] covered.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LeafScan {
    /// Leaf nodes passed to the visitor.
    pub nodes: u32,
    /// Nodes marked in use that could not be read or failed validation.
    pub skipped: u32,
}

/// A B-tree's node allocation bitmap: one bit per node, most significant
/// bit first, set when the node is in use.
struct NodeMap {
    bits: Vec<u8>,
}

impl NodeMap {
    /// True when node `number` is in use or past the end of the bitmap.
    fn in_use(&self, number: u32) -> bool {
        let byte = usize::try_from(number >> 3)
            .ok()
            .and_then(|i| self.bits.get(i));
        match byte {
            Some(b) => b & (0x80u8 >> (number & 7)) != 0,
            None => true,
        }
    }

    /// Nodes in use from `from` up to (not including) `to`.
    fn used_between(&self, from: u32, to: u32) -> u32 {
        let covered = u32::try_from(self.bits.len())
            .ok()
            .and_then(|n| n.checked_mul(8))
            .unwrap_or(u32::MAX);
        // Past the bitmap every node counts as used.
        let past = to.saturating_sub(from.max(covered));
        // Bounded by the bitmap's length in bits.
        let inside = (from..to.min(covered)).filter(|&n| self.in_use(n)).count();
        past.saturating_add(u32::try_from(inside).unwrap_or(u32::MAX))
    }
}

/// Index of the first record whose key is greater than the target
/// (`node.len()` if none).
fn upper_bound(node: &Node, cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>) -> Result<u16> {
    let (mut lo, mut hi) = (0u16, node.len());
    while lo < hi {
        let mid = lo.midpoint(hi);
        if cmp(node.record(mid)?.key)? == Ordering::Greater {
            hi = mid;
        } else {
            lo = mid.saturating_add(1);
        }
    }
    Ok(lo)
}

/// Index of the first record whose key is >= the target (`node.len()` if none).
fn lower_bound(node: &Node, cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>) -> Result<u16> {
    let (mut lo, mut hi) = (0u16, node.len());
    while lo < hi {
        let mid = lo.midpoint(hi);
        if cmp(node.record(mid)?.key)? == Ordering::Less {
            lo = mid.saturating_add(1);
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// A position in the leaf chain.
///
/// Owns a [`BTree`] clone and one node buffer, so it can move between
/// threads. After an error while moving to the next leaf, the cursor is at
/// the end; a record that fails to split is skipped by the next call.
pub struct LeafCursor {
    tree: BTree,
    node: Option<Node>,
    index: u16,
    /// Leaf nodes loaded so far, bounded by the tree's node count.
    visited: u32,
    /// (node, record) of the record the last `next_record` call handed out.
    last: Option<(u32, u16)>,
}

impl LeafCursor {
    /// (node number, record index) of the record most recently returned by
    /// [`next_record`](Self::next_record), or of the record whose split made
    /// that call fail. `None` before the first record and once the cursor is
    /// at the end.
    pub fn position(&self) -> Option<(u32, u16)> {
        self.last
    }

    /// The record at the cursor, then advances. Ok(None) at the end of the leaf chain.
    pub fn next_record(&mut self) -> Result<Option<Record<'_>>> {
        self.last = None;
        // Bounded: each advance increments `visited`, which is capped at
        // total_nodes.
        loop {
            let Some(node) = &self.node else {
                return Ok(None);
            };
            if self.index < node.len() {
                break;
            }
            let (current, next) = (node.number(), node.forward());
            if next == 0 {
                self.node = None;
                return Ok(None);
            }
            self.advance(current, next)?;
        }
        let index = self.index;
        self.index = index.saturating_add(1);
        match &self.node {
            Some(node) => {
                self.last = Some((node.number(), index));
                node.record(index).map(Some)
            }
            None => Ok(None),
        }
    }

    fn advance(&mut self, current: u32, next: u32) -> Result<()> {
        let tree = &self.tree;
        let limit = tree.header.total_nodes;
        let buf = self.node.take().map(Node::into_buf);
        if next == current {
            return Err(corrupt(
                tree.structure,
                Location::Node(current),
                CorruptKind::Loop { what: "leaf chain" },
            ));
        }
        if self.visited >= limit {
            return Err(corrupt(
                tree.structure,
                Location::Node(current),
                CorruptKind::LimitExceeded {
                    what: "leaf chain",
                    limit: u64::from(limit),
                },
            ));
        }
        let node = tree.load_node(next, buf)?;
        if node.kind() != NodeKind::Leaf {
            return Err(tree.bad_kind(&node));
        }
        self.visited = self.visited.saturating_add(1);
        self.index = 0;
        self.node = Some(node);
        Ok(())
    }
}

/// Byte-level builders for B-tree fixtures, shared with other modules' tests.
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod fixtures {
    use super::*;
    use atlas_fs_core::io::block::MemSource;

    /// Node kind bytes.
    pub const LEAF: i8 = -1;
    pub const INDEX: i8 = 0;
    pub const HEADER: i8 = 1;
    pub const MAP: i8 = 2;

    /// A node of `size` bytes: descriptor, records from offset 14, and the
    /// backward offset table with the free-space entry.
    pub fn node(
        size: usize,
        kind: i8,
        height: u8,
        forward: u32,
        backward: u32,
        records: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut b = vec![0u8; size];
        b[0..4].copy_from_slice(&forward.to_be_bytes());
        b[4..8].copy_from_slice(&backward.to_be_bytes());
        b[8] = kind as u8;
        b[9] = height;
        b[10..12].copy_from_slice(&(records.len() as u16).to_be_bytes());
        let mut offsets = Vec::new();
        let mut at = NODE_DESCRIPTOR_LEN;
        for r in records {
            b[at..at + r.len()].copy_from_slice(r);
            offsets.push(at as u16);
            at += r.len();
        }
        offsets.push(at as u16);
        for (i, o) in offsets.iter().enumerate() {
            let pos = size - 2 * (i + 1);
            b[pos..pos + 2].copy_from_slice(&o.to_be_bytes());
        }
        assert!(at <= size - 2 * offsets.len(), "fixture node overflows");
        b
    }

    /// Writes offset-table entry `i` of a built node.
    pub fn set_offset(node: &mut [u8], i: usize, value: u16) {
        let pos = node.len() - 2 * (i + 1);
        node[pos..pos + 2].copy_from_slice(&value.to_be_bytes());
    }

    /// A keyed record with a `u16` key length, padded so data is even.
    pub fn keyed(key: &[u8], data: &[u8]) -> Vec<u8> {
        let mut r = (key.len() as u16).to_be_bytes().to_vec();
        r.extend_from_slice(key);
        if r.len() % 2 == 1 {
            r.push(0);
        }
        r.extend_from_slice(data);
        r
    }

    /// An index record whose key field occupies `max_key_length` bytes.
    pub fn fixed_index(key: &[u8], max_key_length: usize, child: u32) -> Vec<u8> {
        let mut r = (key.len() as u16).to_be_bytes().to_vec();
        r.extend_from_slice(key);
        r.resize(2 + max_key_length, 0);
        if r.len() % 2 == 1 {
            r.push(0);
        }
        r.extend_from_slice(&child.to_be_bytes());
        r
    }

    /// An index record with a variable-length key.
    pub fn var_index(key: &[u8], child: u32) -> Vec<u8> {
        keyed(key, &child.to_be_bytes())
    }

    /// The 106 header record bytes.
    pub fn header_record(h: &HeaderRecord) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_RECORD_LEN];
        b[0..2].copy_from_slice(&h.depth.to_be_bytes());
        b[2..6].copy_from_slice(&h.root_node.to_be_bytes());
        b[6..10].copy_from_slice(&h.leaf_records.to_be_bytes());
        b[10..14].copy_from_slice(&h.first_leaf.to_be_bytes());
        b[14..18].copy_from_slice(&h.last_leaf.to_be_bytes());
        b[18..20].copy_from_slice(&h.node_size.to_be_bytes());
        b[20..22].copy_from_slice(&h.max_key_length.to_be_bytes());
        b[22..26].copy_from_slice(&h.total_nodes.to_be_bytes());
        b[26..30].copy_from_slice(&h.free_nodes.to_be_bytes());
        b[36] = h.btree_type;
        b[37] = h.key_compare_type;
        b[38..42].copy_from_slice(&h.attributes.to_be_bytes());
        b
    }

    /// Byte offset of the map record in a node built by [`header_node`].
    pub const MAP_RECORD_AT: usize = NODE_DESCRIPTOR_LEN + HEADER_RECORD_LEN + 128;

    /// Node 0: header record, 128-byte user data record, map record. The
    /// map marks nodes `0..total_nodes` in use, as far as it reaches.
    pub fn header_node(h: &HeaderRecord) -> Vec<u8> {
        let size = usize::from(h.node_size);
        let len = size - MAP_RECORD_AT - 8;
        let mut map = vec![0u8; len];
        for n in 0..(h.total_nodes as usize).min(len * 8) {
            map[n / 8] |= 0x80 >> (n % 8);
        }
        node(
            size,
            HEADER,
            0,
            0,
            0,
            &[header_record(h), vec![0; 128], map],
        )
    }

    /// Sets the use bit of node `number` in the map record of a header node
    /// built by [`header_node`].
    pub fn set_node_used(header: &mut [u8], number: usize, used: bool) {
        let at = MAP_RECORD_AT + number / 8;
        let bit = 0x80u8 >> (number % 8);
        if used {
            header[at] |= bit;
        } else {
            header[at] &= !bit;
        }
    }

    /// A header with the given shape, `u16` keys and variable index keys.
    pub fn header(depth: u16, root: u32, first: u32, last: u32, total: u32) -> HeaderRecord {
        HeaderRecord {
            depth,
            root_node: root,
            leaf_records: 0,
            first_leaf: first,
            last_leaf: last,
            node_size: 512,
            max_key_length: 16,
            total_nodes: total,
            free_nodes: 0,
            btree_type: 0,
            key_compare_type: 0,
            attributes: BIG_KEYS_MASK | VARIABLE_INDEX_KEYS_MASK,
        }
    }

    /// Lays `nodes` out back to back.
    pub fn image(nodes: &[Vec<u8>]) -> Arc<dyn BlockSource> {
        Arc::new(MemSource::new(nodes.concat()))
    }

    /// Opens a tree from `nodes` (node 0 first).
    pub fn tree(nodes: &[Vec<u8>]) -> Result<BTree> {
        BTree::open(image(nodes), "test B-tree")
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use atlas_fs_core::io::block::MemSource;

    const SIZE: usize = 512;

    fn k(v: u32) -> Vec<u8> {
        v.to_be_bytes().to_vec()
    }

    fn leaf_rec(v: u32) -> Vec<u8> {
        keyed(&k(v), &[0xD0, v as u8])
    }

    fn leaf(fwd: u32, bwd: u32, keys: &[u32]) -> Vec<u8> {
        let recs: Vec<Vec<u8>> = keys.iter().map(|&v| leaf_rec(v)).collect();
        node(SIZE, LEAF, 1, fwd, bwd, &recs)
    }

    fn key_cmp(target: u32) -> impl FnMut(&[u8]) -> Result<Ordering> {
        move |key: &[u8]| Ok(be_u32(key, 0).unwrap().cmp(&target))
    }

    fn seek_key(t: &BTree, target: u32) -> Option<u32> {
        let mut cursor = t.seek(&mut key_cmp(target)).unwrap();
        cursor
            .next_record()
            .unwrap()
            .map(|r| be_u32(r.key, 0).unwrap())
    }

    fn all_keys(mut c: LeafCursor) -> Vec<u32> {
        let mut out = Vec::new();
        while let Some(r) = c.next_record().unwrap() {
            assert_eq!(r.data, &[0xD0, be_u32(r.key, 0).unwrap() as u8]);
            out.push(be_u32(r.key, 0).unwrap());
        }
        out
    }

    fn err<T>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            Err(e) => panic!("expected Corrupt, got {e:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    fn one_leaf() -> BTree {
        let h = header(1, 1, 1, 1, 2);
        tree(&[header_node(&h), leaf(0, 0, &[10, 20, 30])]).unwrap()
    }

    /// Root index (node 1) over leaves 2 and 3.
    fn two_level(fixed: bool) -> BTree {
        let mut h = header(2, 1, 2, 3, 4);
        let index = if fixed {
            h.attributes = BIG_KEYS_MASK;
            h.max_key_length = 6;
            vec![fixed_index(&k(10), 6, 2), fixed_index(&k(40), 6, 3)]
        } else {
            vec![var_index(&k(10), 2), var_index(&k(40), 3)]
        };
        tree(&[
            header_node(&h),
            node(SIZE, INDEX, 2, 0, 0, &index),
            leaf(3, 0, &[10, 20, 30]),
            leaf(0, 2, &[40, 50]),
        ])
        .unwrap()
    }

    #[test]
    fn header_parse() {
        let mut h = header(3, 7, 4, 9, 64);
        h.leaf_records = 1234;
        h.free_nodes = 11;
        h.btree_type = 0x80;
        h.key_compare_type = 0xBC;
        h.max_key_length = 516;
        let t = tree(&[header_node(&h), vec![0; SIZE * 63]]).unwrap();
        assert_eq!(*t.header(), h);
        assert_eq!(HeaderRecord::parse(&header_record(&h)), Some(h));
        assert_eq!(HeaderRecord::parse(&header_record(&h)[..105]), None);
        let n = t.read_node(0).unwrap();
        assert_eq!(n.kind(), NodeKind::Header);
        assert_eq!(n.len(), 3);
        assert_eq!(n.raw_record(0).unwrap().len(), HEADER_RECORD_LEN);
        assert!(n.record(0).is_err());
    }

    #[test]
    fn node_accessors_and_split() {
        let t = two_level(false);
        let n = t.read_node(3).unwrap();
        assert_eq!(
            (n.number(), n.kind(), n.height(), n.forward(), n.backward()),
            (3, NodeKind::Leaf, 1, 0, 2)
        );
        assert_eq!(n.len(), 2);
        let r = n.record(1).unwrap();
        assert_eq!(r.key, &k(50)[..]);
        assert_eq!(r.data, &[0xD0, 50]);
        assert!(n.record(2).is_err());
        // Odd key: one pad byte before the data.
        let h = header(1, 1, 1, 1, 2);
        let t = tree(&[
            header_node(&h),
            node(SIZE, LEAF, 1, 0, 0, &[keyed(&[1, 2, 3], &[9, 9])]),
        ])
        .unwrap();
        let n = t.read_node(1).unwrap();
        let r = n.record(0).unwrap();
        assert_eq!((r.key, r.data), (&[1u8, 2, 3][..], &[9u8, 9][..]));
    }

    #[test]
    fn small_keys_without_big_keys_mask() {
        // u8 key length, as classic HFS: 1 + 3 = 4 bytes, even, no pad.
        let mut h = header(1, 1, 1, 1, 2);
        h.attributes = 0;
        let rec = vec![3, 0xA, 0xB, 0xC, 0x77, 0x88];
        let t = tree(&[header_node(&h), node(SIZE, LEAF, 1, 0, 0, &[rec])]).unwrap();
        let n = t.read_node(1).unwrap();
        let r = n.record(0).unwrap();
        assert_eq!(
            (r.key, r.data),
            (&[0xAu8, 0xB, 0xC][..], &[0x77u8, 0x88][..])
        );
    }

    #[test]
    fn small_key_pad_rule_classic_hfs() {
        // Bytes written out by hand, not through a builder, so the test does
        // not share the pad rule it checks.
        let mut h = header(2, 1, 2, 2, 3);
        h.attributes = 0;
        h.max_key_length = 37;
        // Catalog key with an empty name: keyLength 6, reserved, parentID 2,
        // name length 0. 1 + 6 = 7 is odd: one pad byte, data at 8.
        let odd = vec![6, 0, 0, 0, 0, 2, 0, 0xEE, 0xD1, 0xD2];
        // Extents key: keyLength 7. 1 + 7 = 8 is even: data at 8.
        let even = vec![7, 0xFF, 0, 0, 0, 20, 0, 3, 0xD3, 0xD4];
        // Index record of a tree without variable index keys: keyLength 6
        // but the key occupies max_key_length (37) bytes; 1 + 37 = 38 is
        // even, so the child node number is at 38.
        let mut index = vec![0u8; 42];
        index[..7].copy_from_slice(&[6, 0, 0, 0, 0, 1, 0]);
        index[38..42].copy_from_slice(&2u32.to_be_bytes());
        // Fill the unused key bytes so a wrong split would show.
        for b in &mut index[7..38] {
            *b = 0xAA;
        }
        let t = tree(&[
            header_node(&h),
            node(SIZE, INDEX, 2, 0, 0, &[index]),
            node(SIZE, LEAF, 1, 0, 0, &[odd, even]),
        ])
        .unwrap();

        let leaf = t.read_node(2).unwrap();
        let r = leaf.record(0).unwrap();
        assert_eq!(r.key, &[0u8, 0, 0, 0, 2, 0][..]);
        assert_eq!(r.data, &[0xD1u8, 0xD2][..]);
        let r = leaf.record(1).unwrap();
        assert_eq!(r.key, &[0xFFu8, 0, 0, 0, 20, 0, 3][..]);
        assert_eq!(r.data, &[0xD3u8, 0xD4][..]);

        let index = t.read_node(1).unwrap();
        let r = index.record(0).unwrap();
        assert_eq!(r.key, &[0u8, 0, 0, 0, 1, 0][..]);
        assert_eq!(r.data, &2u32.to_be_bytes()[..]);

        // The seek follows that child pointer to the leaf.
        let mut cmp = |_: &[u8]| Ok(Ordering::Less);
        let mut c = t.seek(&mut cmp).unwrap();
        assert_eq!(c.next_record().unwrap(), None);
        let mut cmp = |key: &[u8]| Ok(key.cmp(&[0u8, 0, 0, 0, 2, 0][..]));
        let mut c = t.seek(&mut cmp).unwrap();
        assert_eq!(c.next_record().unwrap().unwrap().data, &[0xD1u8, 0xD2][..]);
        assert_eq!(c.position(), Some((2, 0)));
    }

    #[test]
    fn seek_one_leaf() {
        let t = one_leaf();
        assert_eq!(seek_key(&t, 5), Some(10));
        assert_eq!(seek_key(&t, 20), Some(20));
        assert_eq!(seek_key(&t, 25), Some(30));
        assert_eq!(seek_key(&t, 35), None);
        assert_eq!(all_keys(t.seek(&mut key_cmp(15)).unwrap()), [20, 30]);
    }

    #[test]
    fn seek_two_levels() {
        for fixed in [false, true] {
            let t = two_level(fixed);
            assert_eq!(seek_key(&t, 5), Some(10), "fixed {fixed}");
            assert_eq!(seek_key(&t, 10), Some(10));
            assert_eq!(seek_key(&t, 20), Some(20));
            // Past the end of leaf 2: the cursor moves to leaf 3.
            assert_eq!(seek_key(&t, 35), Some(40));
            assert_eq!(seek_key(&t, 40), Some(40));
            assert_eq!(seek_key(&t, 45), Some(50));
            assert_eq!(seek_key(&t, 60), None);
            assert_eq!(all_keys(t.seek(&mut key_cmp(25)).unwrap()), [30, 40, 50]);
        }
    }

    #[test]
    fn comparator_errors_propagate() {
        let t = two_level(false);
        let mut cmp = |_: &[u8]| -> Result<Ordering> { Err(Error::unsupported("x")) };
        assert!(matches!(t.seek(&mut cmp), Err(Error::Unsupported(_))));
    }

    #[test]
    fn cursor_across_three_leaves() {
        let h = header(1, 1, 1, 3, 4);
        let t = tree(&[
            header_node(&h),
            leaf(2, 0, &[1, 2]),
            leaf(3, 1, &[3]),
            leaf(0, 2, &[4, 5, 6]),
        ])
        .unwrap();
        assert_eq!(all_keys(t.first().unwrap()), [1, 2, 3, 4, 5, 6]);
        let mut c = t.first().unwrap();
        while c.next_record().unwrap().is_some() {}
        assert!(c.next_record().unwrap().is_none());
    }

    #[test]
    fn cursor_position() {
        // Seek positions the cursor without returning a record.
        let t = one_leaf();
        let mut c = t.seek(&mut key_cmp(20)).unwrap();
        assert_eq!(c.position(), None);
        c.next_record().unwrap().unwrap();
        assert_eq!(c.position(), Some((1, 1)));
        c.next_record().unwrap().unwrap();
        assert_eq!(c.position(), Some((1, 2)));
        assert!(c.next_record().unwrap().is_none());
        assert_eq!(c.position(), None);
        assert!(c.next_record().unwrap().is_none());
        assert_eq!(c.position(), None);
        // Across a leaf boundary the node number changes and the index restarts.
        let mut c = two_level(false).seek(&mut key_cmp(30)).unwrap();
        c.next_record().unwrap().unwrap();
        assert_eq!(c.position(), Some((2, 2)));
        c.next_record().unwrap().unwrap();
        assert_eq!(c.position(), Some((3, 0)));
        // A record that fails to split is the position of the failed call.
        let h = header(1, 1, 1, 1, 2);
        let mut bad = leaf_rec(20);
        bad[0..2].copy_from_slice(&200u16.to_be_bytes());
        let t = tree(&[
            header_node(&h),
            node(SIZE, LEAF, 1, 0, 0, &[leaf_rec(10), bad]),
        ])
        .unwrap();
        let mut c = t.first().unwrap();
        c.next_record().unwrap().unwrap();
        assert!(c.next_record().is_err());
        assert_eq!(c.position(), Some((1, 1)));
        // The empty tree's cursor has no position.
        let h = header(0, 0, 0, 0, 1);
        let mut c = tree(&[header_node(&h)]).unwrap().first().unwrap();
        assert!(c.next_record().unwrap().is_none());
        assert_eq!(c.position(), None);
    }

    #[test]
    fn empty_tree() {
        let h = header(0, 0, 0, 0, 1);
        let t = tree(&[header_node(&h)]).unwrap();
        assert!(t.first().unwrap().next_record().unwrap().is_none());
        assert!(
            t.seek(&mut key_cmp(1))
                .unwrap()
                .next_record()
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn empty_leaf_moves_on() {
        let h = header(1, 1, 1, 2, 3);
        let t = tree(&[header_node(&h), leaf(2, 0, &[]), leaf(0, 1, &[7])]).unwrap();
        assert_eq!(all_keys(t.first().unwrap()), [7]);
    }

    #[test]
    fn malformed_offsets_descending() {
        let h = header(1, 1, 1, 1, 2);
        let mut n = leaf(0, 0, &[10, 20]);
        set_offset(&mut n, 0, 22);
        set_offset(&mut n, 1, 14);
        let t = tree(&[header_node(&h), n]).unwrap();
        let c = err(t.read_node(1));
        assert_eq!(c.location, Location::Record { node: 1, record: 1 });
        assert!(matches!(c.kind, CorruptKind::Other(_)));
    }

    #[test]
    fn malformed_offset_past_node_end() {
        let h = header(1, 1, 1, 1, 2);
        let mut n = leaf(0, 0, &[10, 20]);
        set_offset(&mut n, 1, SIZE as u16 + 10);
        let t = tree(&[header_node(&h), n]).unwrap();
        let c = err(t.read_node(1));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "record offset",
                ..
            }
        ));
        // Free space offset inside the offset table, and below the last record.
        let mut n = leaf(0, 0, &[10, 20]);
        set_offset(&mut n, 2, SIZE as u16 - 2);
        let t = tree(&[header_node(&h), n]).unwrap();
        assert!(matches!(err(t.read_node(1)).location, Location::Node(1)));
        let mut n = leaf(0, 0, &[10, 20]);
        set_offset(&mut n, 2, 20);
        let t = tree(&[header_node(&h), n]).unwrap();
        assert!(matches!(err(t.read_node(1)).kind, CorruptKind::Other(_)));
        // Offset below the descriptor.
        let mut n = leaf(0, 0, &[10]);
        set_offset(&mut n, 0, 4);
        let t = tree(&[header_node(&h), n]).unwrap();
        assert!(matches!(
            err(t.read_node(1)).kind,
            CorruptKind::InvalidValue { .. }
        ));
    }

    #[test]
    fn malformed_record_count() {
        let h = header(1, 1, 1, 1, 2);
        let mut n = leaf(0, 0, &[10]);
        n[10..12].copy_from_slice(&300u16.to_be_bytes());
        let t = tree(&[header_node(&h), n]).unwrap();
        assert!(matches!(
            err(t.read_node(1)).kind,
            CorruptKind::OutOfRange {
                field: "record count",
                ..
            }
        ));
        let mut n = leaf(0, 0, &[10]);
        n[10..12].copy_from_slice(&u16::MAX.to_be_bytes());
        let t = tree(&[header_node(&h), n]).unwrap();
        assert!(t.read_node(1).is_err());
    }

    #[test]
    fn malformed_kind() {
        let h = header(1, 1, 1, 1, 2);
        let mut n = leaf(0, 0, &[10]);
        n[8] = 7;
        let t = tree(&[header_node(&h), n]).unwrap();
        assert!(matches!(
            err(t.read_node(1)).kind,
            CorruptKind::InvalidValue {
                field: "node kind",
                value: 7
            }
        ));
    }

    #[test]
    fn malformed_key_length_past_record() {
        let h = header(1, 1, 1, 1, 2);
        let mut rec = leaf_rec(10);
        rec[0..2].copy_from_slice(&200u16.to_be_bytes());
        let t = tree(&[header_node(&h), node(SIZE, LEAF, 1, 0, 0, &[rec])]).unwrap();
        let n = t.read_node(1).unwrap();
        let c = err(n.record(0));
        assert_eq!(c.location, Location::Record { node: 1, record: 0 });
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "key length",
                ..
            }
        ));
        // A one-byte record cannot hold a u16 key length.
        let t = tree(&[header_node(&h), node(SIZE, LEAF, 1, 0, 0, &[vec![0]])]).unwrap();
        let n = t.read_node(1).unwrap();
        assert!(matches!(
            err(n.record(0)).kind,
            CorruptKind::Truncated { .. }
        ));
        // Fixed index key longer than max_key_length.
        let mut h = header(2, 1, 2, 2, 3);
        h.attributes = BIG_KEYS_MASK;
        h.max_key_length = 2;
        let t = tree(&[
            header_node(&h),
            node(SIZE, INDEX, 2, 0, 0, &[var_index(&k(10), 2)]),
            leaf(0, 0, &[10]),
        ])
        .unwrap();
        assert!(t.seek(&mut key_cmp(10)).is_err());
    }

    #[test]
    fn malformed_forward_loop() {
        let h = header(1, 1, 1, 2, 3);
        let t = tree(&[header_node(&h), leaf(2, 0, &[1]), leaf(1, 1, &[2])]).unwrap();
        let mut c = t.first().unwrap();
        let mut seen = 0;
        let e = loop {
            match c.next_record() {
                Ok(Some(_)) => seen += 1,
                Ok(None) => panic!("loop not detected"),
                Err(e) => break e,
            }
            assert!(seen < 10);
        };
        match e {
            Error::Corrupt(c) => assert!(matches!(
                c.kind,
                CorruptKind::LimitExceeded { .. } | CorruptKind::Loop { .. }
            )),
            other => panic!("{other:?}"),
        }
        // After the error the cursor is at the end.
        assert!(c.next_record().unwrap().is_none());
        // A leaf pointing at itself.
        let h = header(1, 1, 1, 1, 2);
        let t = tree(&[header_node(&h), leaf(1, 0, &[1])]).unwrap();
        let mut c = t.first().unwrap();
        c.next_record().unwrap();
        assert!(matches!(
            err(c.next_record()).kind,
            CorruptKind::Loop { .. }
        ));
    }

    #[test]
    fn malformed_forward_to_non_leaf() {
        let h = header(2, 1, 2, 2, 3);
        let t = tree(&[
            header_node(&h),
            node(SIZE, INDEX, 2, 0, 0, &[var_index(&k(1), 2)]),
            leaf(1, 0, &[1]),
        ])
        .unwrap();
        let mut c = t.first().unwrap();
        c.next_record().unwrap();
        assert!(matches!(
            err(c.next_record()).kind,
            CorruptKind::InvalidValue {
                field: "node kind",
                ..
            }
        ));
    }

    #[test]
    fn malformed_child_height() {
        let h = header(2, 1, 2, 2, 3);
        let mut bad_leaf = leaf(0, 0, &[10]);
        bad_leaf[9] = 2;
        let t = tree(&[
            header_node(&h),
            node(SIZE, INDEX, 2, 0, 0, &[var_index(&k(10), 2)]),
            bad_leaf,
        ])
        .unwrap();
        let c = err(t.seek(&mut key_cmp(10)));
        assert_eq!(c.location, Location::Node(2));
        assert!(matches!(
            c.kind,
            CorruptKind::InvalidValue {
                field: "node height",
                ..
            }
        ));
        // Root height disagreeing with the header depth.
        let h = header(3, 1, 2, 2, 3);
        let t = tree(&[
            header_node(&h),
            node(SIZE, INDEX, 2, 0, 0, &[var_index(&k(10), 2)]),
            leaf(0, 0, &[10]),
        ])
        .unwrap();
        assert_eq!(err(t.seek(&mut key_cmp(10))).location, Location::Node(1));
        // A child of the right height but the wrong kind.
        let h = header(2, 1, 2, 2, 3);
        let mut wrong = node(SIZE, INDEX, 1, 0, 0, &[var_index(&k(10), 2)]);
        wrong[9] = 1;
        let t = tree(&[
            header_node(&h),
            node(SIZE, INDEX, 2, 0, 0, &[var_index(&k(10), 2)]),
            wrong,
        ])
        .unwrap();
        assert!(matches!(
            err(t.seek(&mut key_cmp(10))).kind,
            CorruptKind::InvalidValue {
                field: "node kind",
                ..
            }
        ));
        // Depth over the hard cap.
        let h = header(33, 1, 1, 1, 2);
        let t = tree(&[header_node(&h), leaf(0, 0, &[1])]).unwrap();
        assert!(matches!(
            err(t.seek(&mut key_cmp(1))).kind,
            CorruptKind::LimitExceeded { .. }
        ));
    }

    #[test]
    fn malformed_node_number() {
        let t = one_leaf();
        let c = err(t.read_node(2));
        assert_eq!(c.location, Location::Node(2));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "node number",
                ..
            }
        ));
        assert!(t.read_node(u32::MAX).is_err());
        // Child pointer past total_nodes.
        let h = header(2, 1, 2, 2, 3);
        let t = tree(&[
            header_node(&h),
            node(SIZE, INDEX, 2, 0, 0, &[var_index(&k(10), 99)]),
            leaf(0, 0, &[10]),
        ])
        .unwrap();
        assert!(t.seek(&mut key_cmp(10)).is_err());
        // total_nodes claims more nodes than the fork holds.
        let h = header(1, 1, 1, 1, 5);
        let t = tree(&[header_node(&h), leaf(0, 0, &[10])]).unwrap();
        assert!(matches!(
            err(t.read_node(3)).kind,
            CorruptKind::OutOfRange {
                field: "node end",
                ..
            }
        ));
    }

    #[test]
    fn malformed_node_size() {
        for size in [1000u16, 256, 0, 65535] {
            let mut h = header(1, 1, 1, 1, 2);
            h.node_size = size;
            let mut n = header_record(&h);
            let mut hn = vec![0u8; 14];
            hn[8] = 1;
            hn.append(&mut n);
            hn.resize(2048, 0);
            let c = err(BTree::open(image(&[hn]), "t"));
            assert!(matches!(
                c.kind,
                CorruptKind::InvalidValue {
                    field: "node size",
                    ..
                }
            ));
        }
        // Too short to hold the header at all.
        assert!(matches!(
            err(BTree::open(image(&[vec![0; 50]]), "t")).kind,
            CorruptKind::Truncated { .. }
        ));
        // Node 0 that is not a header node.
        let h = header(1, 1, 1, 1, 2);
        let mut hn = header_node(&h);
        hn[8] = 0xFF;
        assert!(BTree::open(image(&[hn, leaf(0, 0, &[1])]), "t").is_err());
    }

    #[test]
    fn cursor_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<LeafCursor>();
        assert_send::<BTree>();
        assert_send::<Node>();
    }

    // ---- scan_leaves ----

    /// Node numbers and keys of every leaf `scan_leaves` visits.
    fn scanned(t: &BTree) -> (Vec<(u32, Vec<u32>)>, LeafScan) {
        let mut seen = Vec::new();
        let summary = t
            .scan_leaves(&mut |n: &Node| {
                let keys = (0..n.len())
                    .map(|i| be_u32(n.record(i).unwrap().key, 0).unwrap())
                    .collect();
                seen.push((n.number(), keys));
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        (seen, summary)
    }

    /// A header node whose map record is exactly `map`, linked to `forward`.
    fn header_with_map(h: &HeaderRecord, map: &[u8], forward: u32) -> Vec<u8> {
        node(
            SIZE,
            HEADER,
            0,
            forward,
            0,
            &[header_record(h), vec![0; 128], map.to_vec()],
        )
    }

    fn map_node(forward: u32, map: &[u8]) -> Vec<u8> {
        node(SIZE, MAP, 0, forward, 0, &[map.to_vec()])
    }

    #[test]
    fn scan_skips_a_freed_leaf_that_still_holds_records() {
        let h = header(1, 1, 1, 3, 4);
        let mut hn = header_node(&h);
        set_node_used(&mut hn, 2, false);
        let t = tree(&[
            hn,
            leaf(3, 0, &[1, 2]),
            leaf(0, 0, &[3]), // freed: still a valid leaf
            leaf(0, 1, &[4]),
        ])
        .unwrap();
        let (seen, summary) = scanned(&t);
        assert_eq!(seen, vec![(1, vec![1, 2]), (3, vec![4])]);
        assert_eq!(
            summary,
            LeafScan {
                nodes: 2,
                skipped: 0
            }
        );
    }

    #[test]
    fn scan_passes_over_index_header_and_map_nodes() {
        let (seen, summary) = scanned(&two_level(false));
        assert_eq!(seen, vec![(2, vec![10, 20, 30]), (3, vec![40, 50])]);
        assert_eq!(summary.skipped, 0);
    }

    #[test]
    fn scan_follows_the_map_node_chain() {
        // Header map: nodes 0, 1, 2 and 7. Map node 1: node 9. Map node 2:
        // node 23. Nodes 8 and 22 are free leaves.
        let h = header(1, 7, 7, 23, 24);
        let mut nodes = vec![
            header_with_map(&h, &[0b1110_0001], 1),
            map_node(2, &[0b0100_0000]),
            map_node(0, &[0b0000_0001]),
        ];
        nodes.resize(24, vec![0; SIZE]);
        for n in [7, 8, 9, 22, 23] {
            nodes[n] = leaf(0, 0, &[n as u32]);
        }
        let (seen, summary) = scanned(&tree(&nodes).unwrap());
        assert_eq!(seen, vec![(7, vec![7]), (9, vec![9]), (23, vec![23])]);
        assert_eq!(
            summary,
            LeafScan {
                nodes: 3,
                skipped: 0
            }
        );
    }

    #[test]
    fn scan_treats_nodes_past_a_short_bitmap_as_used() {
        // A one-byte map covers nodes 0..8; nodes 8 and 9 are past it.
        let h = header(1, 2, 2, 9, 10);
        let mut nodes = vec![header_with_map(&h, &[0b1010_0000], 0)];
        nodes.resize(10, vec![0; SIZE]);
        for n in [1, 2, 8, 9] {
            nodes[n] = leaf(0, 0, &[n as u32]);
        }
        let (seen, _) = scanned(&tree(&nodes).unwrap());
        assert_eq!(seen, vec![(2, vec![2]), (8, vec![8]), (9, vec![9])]);
    }

    #[test]
    fn scan_ends_a_looping_or_broken_map_chain() {
        // 0 -> 1 -> 2 -> 1: the chain ends at the repeat, so nodes 24..32
        // are past the bitmap and count as used.
        let h = header(1, 3, 3, 25, 32);
        let mut nodes = vec![
            header_with_map(&h, &[0b1111_0000], 1),
            map_node(2, &[0b0000_0000]),
            map_node(1, &[0b0000_0000]),
        ];
        nodes.resize(32, vec![0; SIZE]);
        for n in [3, 12, 25] {
            nodes[n] = leaf(0, 0, &[n as u32]);
        }
        let (seen, _) = scanned(&tree(&nodes).unwrap());
        assert_eq!(seen, vec![(3, vec![3]), (25, vec![25])]);

        // A chain into a node that is not a map node ends there.
        let mut nodes2 = nodes.clone();
        nodes2[0] = header_with_map(&h, &[0b1001_0000], 3);
        let (seen, _) = scanned(&tree(&nodes2).unwrap());
        // Node 3 is used per the header byte; 12 and 25 are past the map.
        assert_eq!(seen, vec![(3, vec![3]), (12, vec![12]), (25, vec![25])]);

        // A self link and a link past the tree end the chain too.
        for forward in [0u32, 1, 999] {
            let mut nodes3 = nodes.clone();
            nodes3[1] = map_node(forward, &[0]);
            nodes3[0] = header_with_map(&h, &[0b1001_0000], 1);
            let (seen, _) = scanned(&tree(&nodes3).unwrap());
            assert!(seen.contains(&(3, vec![3])));
            assert!(!seen.iter().any(|(n, _)| *n == 12), "{forward}");
        }
    }

    #[test]
    fn scan_counts_a_corrupt_leaf_and_reads_the_rest() {
        let h = header(1, 1, 1, 4, 5);
        let mut bad = leaf(0, 0, &[2]);
        set_offset(&mut bad, 0, 3); // record offset inside the descriptor
        let mut bad_kind = leaf(0, 0, &[3]);
        bad_kind[8] = 7;
        let t = tree(&[
            header_node(&h),
            leaf(0, 0, &[1]),
            bad,
            bad_kind,
            leaf(0, 0, &[4]),
        ])
        .unwrap();
        let (seen, summary) = scanned(&t);
        assert_eq!(seen, vec![(1, vec![1]), (4, vec![4])]);
        assert_eq!(
            summary,
            LeafScan {
                nodes: 2,
                skipped: 2
            }
        );
    }

    #[test]
    fn scan_stops_on_break_and_on_a_visitor_error() {
        let h = header(1, 1, 1, 3, 4);
        let t = tree(&[
            header_node(&h),
            leaf(0, 0, &[1]),
            leaf(0, 0, &[2]),
            leaf(0, 0, &[3]),
        ])
        .unwrap();
        let mut seen = Vec::new();
        let summary = t
            .scan_leaves(&mut |n: &Node| {
                seen.push(n.number());
                Ok(if n.number() == 2 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                })
            })
            .unwrap();
        assert_eq!(seen, vec![1, 2]);
        assert_eq!(summary.nodes, 2);

        let r = t.scan_leaves(&mut |_| Err(Error::Unsupported("stop".into())));
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn scan_reads_only_whole_nodes_inside_the_fork() {
        // The header claims 6 nodes; the fork holds 3 and a half.
        let h = header(1, 1, 1, 2, 6);
        let mut bytes = [header_node(&h), leaf(0, 0, &[1]), leaf(0, 0, &[2])].concat();
        bytes.extend_from_slice(&leaf(0, 0, &[3])[..SIZE / 2]);
        let t = BTree::open(Arc::new(MemSource::new(bytes)), "t").unwrap();
        let (seen, summary) = scanned(&t);
        assert_eq!(seen, vec![(1, vec![1]), (2, vec![2])]);
        assert_eq!(summary.skipped, 3);
    }

    /// A source whose reads fail when they touch `bad` (byte range).
    struct Failing {
        inner: MemSource,
        bad: std::ops::Range<u64>,
        reads: std::sync::atomic::AtomicUsize,
    }

    impl BlockSource for Failing {
        fn len(&self) -> u64 {
            self.inner.len()
        }
        fn sector_size(&self) -> u32 {
            512
        }
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
            self.reads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let end = offset + buf.len() as u64;
            if offset < self.bad.end && self.bad.start < end {
                return Err(Error::Io {
                    context: atlas_fs_core::error::IoContext::Read {
                        offset,
                        len: buf.len() as u64,
                    },
                    source: std::io::Error::other("bad sector"),
                });
            }
            self.inner.read_at(offset, buf)
        }
    }

    fn failing(nodes: &[Vec<u8>], bad: std::ops::Range<u64>) -> BTree {
        BTree::open(
            Arc::new(Failing {
                inner: MemSource::new(nodes.concat()),
                bad,
                reads: Default::default(),
            }),
            "t",
        )
        .unwrap()
    }

    #[test]
    fn scan_retries_a_failed_piece_node_by_node() {
        let h = header(1, 1, 1, 4, 5);
        let nodes = [
            header_node(&h),
            leaf(0, 0, &[1]),
            leaf(0, 0, &[2]),
            leaf(0, 0, &[3]),
            leaf(0, 0, &[4]),
        ];
        // One sector inside node 2.
        let bad = 2 * SIZE as u64 + 100..2 * SIZE as u64 + 101;
        let (seen, summary) = scanned(&failing(&nodes, bad));
        assert_eq!(seen, vec![(1, vec![1]), (3, vec![3]), (4, vec![4])]);
        assert_eq!(
            summary,
            LeafScan {
                nodes: 3,
                skipped: 1
            }
        );
    }

    #[test]
    fn scan_gives_up_after_too_many_unreadable_nodes() {
        let total = MAX_SKIPPED as usize + 10;
        let h = header(1, 1, 1, 1, total as u32);
        let mut nodes = vec![vec![0u8; SIZE]; total];
        nodes[0] = header_node(&h);
        // The header node's map covers 2048 nodes; the rest count as used.
        let t = failing(&nodes, SIZE as u64..u64::MAX);
        let r = t.scan_leaves(&mut |_| Ok(ControlFlow::Continue(())));
        assert!(matches!(r, Err(Error::Io { .. })), "{r:?}");
    }

    #[test]
    fn scan_reads_in_pieces_and_reuses_buffers() {
        // More nodes than one piece holds, every one a leaf.
        let per_piece = SCAN_PIECE / SIZE;
        let total = per_piece + 3;
        let h = header(1, 1, 1, total as u32 - 1, total as u32);
        let mut hn = header_node(&h);
        // Mark the nodes past the 256-byte map in use explicitly by leaving
        // them past it; free node 5.
        set_node_used(&mut hn, 5, false);
        let mut nodes = vec![hn];
        for n in 1..total {
            nodes.push(leaf(0, 0, &[n as u32]));
        }
        let source = Arc::new(Failing {
            inner: MemSource::new(nodes.concat()),
            bad: 0..0,
            reads: Default::default(),
        });
        let t = BTree::open(source.clone(), "t").unwrap();
        let before = source.reads.load(std::sync::atomic::Ordering::Relaxed);
        let (seen, summary) = scanned(&t);
        assert_eq!(seen.len(), total - 2);
        assert!(!seen.iter().any(|(n, _)| *n == 5));
        assert!(seen.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(summary.skipped, 0);
        // Node 0 (once, for the bitmap), then two pieces.
        let reads = source.reads.load(std::sync::atomic::Ordering::Relaxed) - before;
        assert_eq!(reads, 3);
    }
}
