//! Writer for shared memory grid (used by neural_ash).
//!
//! Provides safe concurrent write access using seqlock protocol.

use core::sync::atomic::{AtomicU64, Ordering};

use ash_core::BlockCoord;

use super::layout::{SharedHeader, SharedLayout};
use crate::error::{AshIoError, Result};

/// Writer for shared memory grid with N features.
///
/// Provides safe concurrent write access using seqlock protocol
/// for reader synchronization.
///
/// # Type Parameter
///
/// `N` - Number of features per cell
///
/// # Safety
///
/// - Only one writer should exist per shared memory region
/// - The region must remain valid for the lifetime of this writer
pub struct SharedGridWriter<const N: usize> {
    /// Pointer to the shared memory region.
    base: *mut u8,
    /// Cached layout information.
    layout: SharedLayout,
    /// Grid dimension.
    grid_dim: u32,
    /// Cell size.
    cell_size: f32,
    /// Maximum capacity.
    capacity: usize,
    /// Cells per block.
    cells_per_block: usize,
}

// Safety: SharedGridWriter coordinates access via seqlock
unsafe impl<const N: usize> Send for SharedGridWriter<N> {}

impl<const N: usize> SharedGridWriter<N> {
    /// Initialize a shared memory region and create a writer.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a writable memory region of at least `size` bytes
    /// - The memory must be zeroed or previously initialized
    /// - Only one writer should use this region
    pub unsafe fn initialize(
        ptr: *mut u8,
        size: usize,
        grid_dim: u32,
        cell_size: f32,
        capacity: usize,
    ) -> Result<Self> {
        let layout = SharedLayout::compute::<N>(grid_dim, capacity);

        if size < layout.total_size {
            return Err(AshIoError::SharedMemory {
                message: "shared memory region too small",
            });
        }

        let cells_per_block = (grid_dim as usize).pow(3);

        // Initialize header
        let header_ptr = ptr as *mut SharedHeader;
        *header_ptr = SharedHeader::new(grid_dim, cell_size, capacity as u32, N as u16);

        // Zero the block map
        let block_map_ptr = ptr.add(layout.block_map_offset) as *mut AtomicU64;
        for i in 0..(capacity * 2) {
            (*block_map_ptr.add(i)) = AtomicU64::new(0);
        }

        // Initialize version counters to 0 (even = not writing)
        let versions_ptr = ptr.add(layout.versions_offset) as *mut AtomicU64;
        for i in 0..capacity {
            (*versions_ptr.add(i)) = AtomicU64::new(0);
        }

        // Initialize feature values to UNTRAINED_SENTINEL
        for feature_idx in 0..N {
            let values_ptr = ptr.add(layout.feature_offset(feature_idx)) as *mut f32;
            for i in 0..(capacity * cells_per_block) {
                *values_ptr.add(i) = ash_core::UNTRAINED_SENTINEL;
            }
        }

        Ok(Self {
            base: ptr,
            layout,
            grid_dim,
            cell_size,
            capacity,
            cells_per_block,
        })
    }

    /// Attach to an existing shared memory region.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid, initialized shared memory region
    /// - Only one writer should use this region
    pub unsafe fn attach(ptr: *mut u8) -> Result<Self> {
        let header = &*(ptr as *const SharedHeader);

        if !header.validate_feature_dim::<N>() {
            return Err(AshIoError::SharedMemory {
                message: "invalid shared memory header or feature dimension mismatch",
            });
        }

        let grid_dim = header.grid_dim;
        let cell_size = header.cell_size;
        let capacity = header.capacity as usize;
        let cells_per_block = header.cells_per_block();
        let layout = SharedLayout::compute::<N>(grid_dim, capacity);

        Ok(Self {
            base: ptr,
            layout,
            grid_dim,
            cell_size,
            capacity,
            cells_per_block,
        })
    }

    /// Get mutable reference to the header.
    #[inline]
    fn header_mut(&self) -> &mut SharedHeader {
        unsafe { &mut *(self.base as *mut SharedHeader) }
    }

    /// Get mutable pointer to block map.
    #[inline]
    fn block_map_mut(&self) -> *mut AtomicU64 {
        unsafe { self.base.add(self.layout.block_map_offset) as *mut AtomicU64 }
    }

    /// Get mutable pointer to coordinates.
    #[inline]
    fn coords_mut(&self) -> *mut BlockCoord {
        unsafe { self.base.add(self.layout.coords_offset) as *mut BlockCoord }
    }

    /// Get mutable pointer to a feature's values.
    #[inline]
    fn feature_values_mut(&self, feature_idx: usize) -> *mut f32 {
        unsafe { self.base.add(self.layout.feature_offset(feature_idx)) as *mut f32 }
    }

    /// Get mutable pointer to versions.
    #[inline]
    fn versions_mut(&self) -> *mut AtomicU64 {
        unsafe { self.base.add(self.layout.versions_offset) as *mut AtomicU64 }
    }

    /// Allocate a new block.
    ///
    /// Returns the block index if successful, None if capacity exceeded
    /// or block already exists.
    pub fn allocate_block(&mut self, coord: BlockCoord) -> Option<usize> {
        let morton = ash_core::morton_encode_signed(coord);
        let hash = morton as u32;
        let map_capacity = self.capacity * 2;
        let mut idx = (morton as usize) % map_capacity;

        let header = self.header_mut();
        let current_blocks = header.num_blocks.load(Ordering::Relaxed) as usize;

        if current_blocks >= self.capacity {
            return None;
        }

        // Find empty slot or existing entry
        for _ in 0..map_capacity {
            let entry_ptr = unsafe { &*self.block_map_mut().add(idx) };
            let entry = entry_ptr.load(Ordering::Acquire);
            let state = entry >> 62;
            let entry_hash = entry as u32;

            match state {
                0 => {
                    // Empty slot - try to claim it
                    let block_idx = current_blocks;
                    let new_entry = (2u64 << 62) | ((block_idx as u64 & 0x3FFF_FFFF) << 32) | hash as u64;

                    if entry_ptr.compare_exchange(0, new_entry, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                        // Successfully claimed
                        unsafe {
                            *self.coords_mut().add(block_idx) = coord;
                        }
                        header.num_blocks.fetch_add(1, Ordering::Release);
                        return Some(block_idx);
                    }
                    // CAS failed, retry at same slot
                }
                2 if entry_hash == hash => {
                    // Block already exists
                    return None;
                }
                _ => {
                    idx = (idx + 1) % map_capacity;
                }
            }
        }

        None // Table full
    }

    /// Write all N feature values for a cell using seqlock protocol.
    ///
    /// This ensures readers see consistent data.
    pub fn write_values(&mut self, block_idx: usize, cell_idx: usize, values: [f32; N]) {
        let version_ptr = unsafe { &*self.versions_mut().add(block_idx) };

        // 1. Increment version to ODD (signals write in progress)
        let old_version = version_ptr.fetch_add(1, Ordering::Release);
        debug_assert!(old_version & 1 == 0, "Writer detected concurrent write");

        // 2. Memory fence before writing
        core::sync::atomic::fence(Ordering::Release);

        // 3. Write all feature values
        let value_idx = block_idx * self.cells_per_block + cell_idx;
        for i in 0..N {
            unsafe {
                *self.feature_values_mut(i).add(value_idx) = values[i];
            }
        }

        // 4. Memory fence after writing
        core::sync::atomic::fence(Ordering::Release);

        // 5. Increment version to EVEN (signals write complete)
        version_ptr.fetch_add(1, Ordering::Release);
    }

    /// Write entire block's worth of feature values.
    ///
    /// More efficient than individual cell writes.
    pub fn write_block(&mut self, block_idx: usize, values: &[[f32; N]]) {
        debug_assert_eq!(values.len(), self.cells_per_block);

        let version_ptr = unsafe { &*self.versions_mut().add(block_idx) };

        // 1. Start write (odd version)
        let old_version = version_ptr.fetch_add(1, Ordering::Release);
        debug_assert!(old_version & 1 == 0, "Writer detected concurrent write");

        // 2. Memory fence
        core::sync::atomic::fence(Ordering::Release);

        // 3. Write all values
        let base_idx = block_idx * self.cells_per_block;
        for (cell_idx, cell_values) in values.iter().enumerate() {
            let value_idx = base_idx + cell_idx;
            for i in 0..N {
                unsafe {
                    *self.feature_values_mut(i).add(value_idx) = cell_values[i];
                }
            }
        }

        // 4. Memory fence
        core::sync::atomic::fence(Ordering::Release);

        // 5. Complete write (even version)
        version_ptr.fetch_add(1, Ordering::Release);
    }

    /// Write a single feature for an entire block.
    ///
    /// Efficient for updating one feature dimension at a time.
    pub fn write_block_feature(&mut self, block_idx: usize, feature_idx: usize, values: &[f32]) {
        debug_assert_eq!(values.len(), self.cells_per_block);

        let version_ptr = unsafe { &*self.versions_mut().add(block_idx) };

        // Seqlock protocol
        let old_version = version_ptr.fetch_add(1, Ordering::Release);
        debug_assert!(old_version & 1 == 0);

        core::sync::atomic::fence(Ordering::Release);

        let base_idx = block_idx * self.cells_per_block;
        let feature_ptr = self.feature_values_mut(feature_idx);
        unsafe {
            core::ptr::copy_nonoverlapping(
                values.as_ptr(),
                feature_ptr.add(base_idx),
                self.cells_per_block,
            );
        }

        core::sync::atomic::fence(Ordering::Release);
        version_ptr.fetch_add(1, Ordering::Release);
    }

    /// Flush writes (memory fence).
    #[inline]
    pub fn flush(&self) {
        core::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Get the number of allocated blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.header_mut().num_blocks.load(Ordering::Relaxed) as usize
    }

    /// Get grid dimension.
    #[inline]
    pub fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    /// Get cell size.
    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Get capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::layout::compute_shared_size;
    use super::super::view::SharedGridView;

    #[test]
    fn test_writer_initialize() {
        let size = compute_shared_size::<1>(8, 100);
        let mut buffer = vec![0u8; size];

        let writer = unsafe {
            SharedGridWriter::<1>::initialize(buffer.as_mut_ptr(), size, 8, 0.1, 100)
        };

        assert!(writer.is_ok());
        let writer = writer.unwrap();
        assert_eq!(writer.num_blocks(), 0);
        assert_eq!(writer.grid_dim(), 8);
    }

    #[test]
    fn test_writer_allocate_block() {
        let size = compute_shared_size::<1>(8, 100);
        let mut buffer = vec![0u8; size];

        let mut writer = unsafe {
            SharedGridWriter::<1>::initialize(buffer.as_mut_ptr(), size, 8, 0.1, 100).unwrap()
        };

        let coord = BlockCoord::new(1, 2, 3);
        let block_idx = writer.allocate_block(coord);

        assert!(block_idx.is_some());
        assert_eq!(block_idx.unwrap(), 0);
        assert_eq!(writer.num_blocks(), 1);

        // Duplicate should fail
        let dup = writer.allocate_block(coord);
        assert!(dup.is_none());
    }

    #[test]
    fn test_writer_write_values() {
        let size = compute_shared_size::<4>(4, 10);
        let mut buffer = vec![0u8; size];

        let mut writer = unsafe {
            SharedGridWriter::<4>::initialize(buffer.as_mut_ptr(), size, 4, 0.1, 10).unwrap()
        };

        let block_idx = writer.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();

        // Write values
        let values = [1.0, 2.0, 3.0, 4.0];
        writer.write_values(block_idx, 0, values);

        // Create reader view and verify
        let view = unsafe { SharedGridView::<4>::from_ptr(buffer.as_ptr()) };
        assert!(view.is_some());

        let view = view.unwrap();
        let read_values = view.read_values_seqlock(block_idx, 0);
        assert!(read_values.is_some());
        assert_eq!(read_values.unwrap(), values);
    }

    #[test]
    fn test_writer_view_roundtrip() {
        let size = compute_shared_size::<1>(8, 100);
        let mut buffer = vec![0u8; size];

        // Create writer and add a block
        let mut writer = unsafe {
            SharedGridWriter::<1>::initialize(buffer.as_mut_ptr(), size, 8, 0.1, 100).unwrap()
        };

        let coord = BlockCoord::new(0, 0, 0);
        let block_idx = writer.allocate_block(coord).unwrap();

        // Write some values
        let cells_per_block = 8 * 8 * 8;
        for cell_idx in 0..cells_per_block {
            writer.write_values(block_idx, cell_idx, [cell_idx as f32 * 0.01]);
        }

        // Create view and verify
        let view = unsafe { SharedGridView::<1>::from_ptr(buffer.as_ptr()).unwrap() };

        assert_eq!(view.num_blocks(), 1);
        assert!(view.get_block_index(coord).is_some());

        // Read back values
        for cell_idx in 0..10 {
            let values = view.read_values_seqlock(block_idx, cell_idx);
            assert!(values.is_some());
            let [val] = values.unwrap();
            assert!((val - cell_idx as f32 * 0.01).abs() < 1e-6);
        }
    }
}
