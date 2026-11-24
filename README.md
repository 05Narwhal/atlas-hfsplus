# atlas-hfsplus

A read-only HFS+ and HFSX reader. It implements `atlas_fs_core::fs::FileSystem`, so a mounted volume lists
folders, looks up names, reads file forks and extended attributes, and scans the whole catalog through the
same trait as the other `atlas-*` readers.

- Finds the volume bare or inside a classic HFS wrapper.
- Replays a dirty journal in memory (the image is never written).
- Catalog, extents overflow and attributes B-trees, hard links to files and folders, symbolic links.
- decmpfs compressed files (zlib, LZVN, LZFSE) and extended attributes.
- No `unsafe`. Every length, offset and count read from the disk is checked; damage is an error, not a panic.

The B-tree, fork, catalog, compression and Unicode modules are public because other formats (HFS, APFS,
disk images) build on them.

```rust,no_run
use std::sync::Arc;
use atlas_fs_core::fs::FileSystem;
use atlas_fs_core::io::block::BlockSource;
use atlas_fs_core::io::file::FileSource;

let source: Arc<dyn BlockSource> = Arc::new(FileSource::open("volume.img".as_ref())?);
let volume = atlas_hfsplus::mount(source)?;
let mut dir = volume.read_dir(volume.root())?;
# Ok::<(), atlas_fs_core::error::Error>(())
```

The `test-util` feature exposes the test volume builder and fixtures; it is for tests of crates that use this
one and is not part of the stable API.

## Licence

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)) or MIT license
([LICENSE-MIT](LICENSE-MIT)) at your option.
