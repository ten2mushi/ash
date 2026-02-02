//! Shared memory layout computation for N-feature grids.
//!
//! Memory layout:
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Header (64 bytes, cache-line aligned)                          │
//! │  ├─ magic: u32                                                  │
//! │  ├─ version: AtomicU64                                          │
//! │  ├─ grid_dim: u32                                               │
//! │  ├─ cell_size: f32                                              │
//! │  ├─ capacity: u32                                               │
//! │  ├─ feature_dim: u16                                            │
//! │  ├─ num_blocks: AtomicU32                                       │
//! │  └─ padding                                                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Map (lock-free hash table)                               │
//! │  └─ entries: [AtomicU64; capacity * 2]                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Coords (reverse lookup)                                  │
//! │  └─ coords: [BlockCoord; capacity]                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Feature Data (SoA layout, 32-byte aligned)                     │
//! │  └─ For each feature dimension:                                 │
//! │       values: [f32; capacity * grid_dim³]                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Block Versions (per-block seqlock)                             │
//! │  └─ versions: [AtomicU64; capacity]                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use core::sync::atomic::{AtomicU32, AtomicU64};

/// Magic number for shared memory validation.
pub const SHARED_MAGIC: u32 = 0x41534833; // "ASH3" (version 3 for N-feature support)

/// Header for the shared memory region.
///
/// This structure is cache-line aligned (64 bytes) to prevent false sharing.
#[repr(C, align(64))]
pub struct SharedHeader {
    /// Magic number for validation.
    pub magic: u32,
    /// Grid dimension (cells per block edge).
    pub grid_dim: u32,
    /// Cell size in world units.
    pub cell_size: f32,
    /// Maximum number of blocks.
    pub capacity: u32,
    /// Number of features per cell.
    pub feature_dim: u16,
    /// Padding for alignment.
    _padding1: u16,
    /// Global epoch counter (incremented on structural changes).
    pub global_version: AtomicU64,
    /// Current number of allocated blocks.
    pub num_blocks: AtomicU32,
    /// Padding to fill cache line.
    _padding2: [u8; 28],
}

impl SharedHeader {
    /// Create a new header with the given configuration.
    pub const fn new(grid_dim: u32, cell_size: f32, capacity: u32, feature_dim: u16) -> Self {
        Self {
            magic: SHARED_MAGIC,
            grid_dim,
            cell_size,
            capacity,
            feature_dim,
            _padding1: 0,
            global_version: AtomicU64::new(0),
            num_blocks: AtomicU32::new(0),
            _padding2: [0; 28],
        }
    }

    /// Validate the header.
    pub fn validate(&self) -> bool {
        self.magic == SHARED_MAGIC
            && self.grid_dim > 0
            && self.cell_size > 0.0
            && self.capacity > 0
            && self.feature_dim > 0
    }

    /// Validate that feature dimension matches expected N.
    pub fn validate_feature_dim<const N: usize>(&self) -> bool {
        self.validate() && self.feature_dim as usize == N
    }

    /// Get cells per block.
    pub const fn cells_per_block(&self) -> usize {
        (self.grid_dim as usize)
            .saturating_mul(self.grid_dim as usize)
            .saturating_mul(self.grid_dim as usize)
    }

    /// Get block size in world units.
    pub fn block_size(&self) -> f32 {
        self.grid_dim as f32 * self.cell_size
    }
}

/// Layout offsets for shared memory region with N features.
#[derive(Debug, Clone, Copy)]
pub struct SharedLayout {
    /// Offset to block map entries.
    pub block_map_offset: usize,
    /// Offset to block coordinates.
    pub coords_offset: usize,
    /// Offset to feature arrays (SoA layout).
    /// feature_offsets[i] is the offset to feature i's data.
    pub feature_offsets: [usize; 16], // Support up to 16 features
    /// Offset to per-block versions.
    pub versions_offset: usize,
    /// Total size of the shared memory region.
    pub total_size: usize,
    /// Number of features stored.
    pub feature_count: usize,
}

impl SharedLayout {
    /// Compute the layout for a given configuration.
    pub const fn compute<const N: usize>(grid_dim: u32, capacity: usize) -> Self {
        let header_size = core::mem::size_of::<SharedHeader>();
        let cells_per_block = (grid_dim as usize) * (grid_dim as usize) * (grid_dim as usize);

        // Block map: 2x capacity for ~50% load factor
        let block_map_entries = capacity * 2;
        let block_map_size = block_map_entries * core::mem::size_of::<AtomicU64>();

        // Block coordinates
        let coords_size = capacity * 12; // sizeof BlockCoord

        // Feature values (SoA layout, 32-byte aligned)
        let values_per_feature = capacity * cells_per_block;
        let bytes_per_feature = values_per_feature * core::mem::size_of::<f32>();

        // Per-block versions
        let versions_size = capacity * core::mem::size_of::<AtomicU64>();

        // Compute offsets with alignment
        let block_map_offset = header_size;
        let coords_offset = block_map_offset + block_map_size;

        // 32-byte align the first feature
        let mut current_offset = (coords_offset + coords_size + 31) & !31;

        let mut feature_offsets = [0usize; 16];
        let mut i = 0;
        while i < N && i < 16 {
            feature_offsets[i] = current_offset;
            current_offset += bytes_per_feature;
            // 32-byte align each feature array
            current_offset = (current_offset + 31) & !31;
            i += 1;
        }

        let versions_offset = current_offset;
        let total_size = versions_offset + versions_size;

        Self {
            block_map_offset,
            coords_offset,
            feature_offsets,
            versions_offset,
            total_size,
            feature_count: N,
        }
    }

    /// Get the offset for a specific feature dimension.
    #[inline]
    pub const fn feature_offset(&self, feature_idx: usize) -> usize {
        self.feature_offsets[feature_idx]
    }
}

/// Compute the total size needed for a shared memory region with N features.
pub const fn compute_shared_size<const N: usize>(grid_dim: u32, capacity: usize) -> usize {
    SharedLayout::compute::<N>(grid_dim, capacity).total_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_header() {
        let header = SharedHeader::new(8, 0.1, 1000, 1);
        assert!(header.validate());
        assert!(header.validate_feature_dim::<1>());
        assert!(!header.validate_feature_dim::<4>());
        assert_eq!(header.cells_per_block(), 512);
        assert!((header.block_size() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_shared_layout_n1() {
        let layout = SharedLayout::compute::<1>(8, 1000);

        // Verify offsets are properly ordered
        assert!(layout.block_map_offset < layout.coords_offset);
        assert!(layout.coords_offset < layout.feature_offsets[0]);
        assert!(layout.feature_offsets[0] < layout.versions_offset);
        assert!(layout.versions_offset < layout.total_size);

        // Verify values are 32-byte aligned
        assert_eq!(layout.feature_offsets[0] % 32, 0);
    }

    #[test]
    fn test_shared_layout_n4() {
        let layout = SharedLayout::compute::<4>(8, 1000);

        // Verify all 4 feature offsets are valid and 32-byte aligned
        for i in 0..4 {
            assert!(layout.feature_offsets[i] > 0);
            assert_eq!(layout.feature_offsets[i] % 32, 0);
            if i > 0 {
                assert!(layout.feature_offsets[i] > layout.feature_offsets[i - 1]);
            }
        }

        // N=4 should be larger than N=1
        let layout_n1 = SharedLayout::compute::<1>(8, 1000);
        assert!(layout.total_size > layout_n1.total_size);
    }

    #[test]
    fn test_compute_shared_size_n1() {
        let size = compute_shared_size::<1>(8, 1000);

        // At least 2MB for 1000 blocks with 512 cells each
        assert!(size > 2_000_000);
        assert!(size < 3_000_000);
    }

    #[test]
    fn test_compute_shared_size_n4() {
        let size_n1 = compute_shared_size::<1>(8, 1000);
        let size_n4 = compute_shared_size::<4>(8, 1000);

        // N=4 should be roughly 4x the data size
        assert!(size_n4 > size_n1 * 3);
        assert!(size_n4 < size_n1 * 5);
    }
}
