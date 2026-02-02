//! Read-only view into shared memory for real-time queries.
//!
//! Provides lock-free, wait-free access to grid data that is being
//! concurrently updated by a writer (e.g., neural_ash).

use core::sync::atomic::{AtomicU64, Ordering};

use ash_core::{
    decompose_point, resolve_corner, trilinear_interpolate, trilinear_with_gradient,
    BlockCoord, CellCoord, CellValueProvider, InterpolationResult, Point3, UNTRAINED_SENTINEL,
};

use super::layout::{SharedHeader, SharedLayout};

/// Read-only view into shared memory for N-feature queries.
///
/// This structure provides lock-free, wait-free access to grid data
/// that is being concurrently updated by a writer.
///
/// # Type Parameter
///
/// `N` - Number of features per cell
///
/// # Safety
///
/// The shared memory region must be:
/// - Properly initialized by the writer
/// - Mapped into both processes' address spaces
/// - Not unmapped while this view exists
pub struct SharedGridView<const N: usize> {
    /// Pointer to the shared memory region.
    base: *const u8,
    /// Cached layout information.
    layout: SharedLayout,
    /// Cached header values (immutable after init).
    grid_dim: u32,
    cell_size: f32,
    capacity: usize,
    cells_per_block: usize,
}

// Safety: SharedGridView only performs atomic reads
unsafe impl<const N: usize> Send for SharedGridView<N> {}
unsafe impl<const N: usize> Sync for SharedGridView<N> {}

impl<const N: usize> SharedGridView<N> {
    /// Create a view from a shared memory pointer.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid, initialized shared memory region
    /// - The region must remain valid for the lifetime of this view
    /// - The region must have been initialized with the correct feature dimension N
    pub unsafe fn from_ptr(ptr: *const u8) -> Option<Self> {
        let header = &*(ptr as *const SharedHeader);

        if !header.validate_feature_dim::<N>() {
            return None;
        }

        let grid_dim = header.grid_dim;
        let cell_size = header.cell_size;
        let capacity = header.capacity as usize;
        let cells_per_block = header.cells_per_block();
        let layout = SharedLayout::compute::<N>(grid_dim, capacity);

        Some(Self {
            base: ptr,
            layout,
            grid_dim,
            cell_size,
            capacity,
            cells_per_block,
        })
    }

    /// Get the header.
    #[inline]
    fn header(&self) -> &SharedHeader {
        unsafe { &*(self.base as *const SharedHeader) }
    }

    /// Get the block map entries.
    #[inline]
    fn block_map(&self) -> &[AtomicU64] {
        unsafe {
            let ptr = self.base.add(self.layout.block_map_offset) as *const AtomicU64;
            core::slice::from_raw_parts(ptr, self.capacity * 2)
        }
    }

    /// Get pointer to a specific feature's values.
    #[inline]
    fn feature_values_ptr(&self, feature_idx: usize) -> *const f32 {
        unsafe { self.base.add(self.layout.feature_offset(feature_idx)) as *const f32 }
    }

    /// Get the per-block versions.
    #[inline]
    fn versions(&self) -> &[AtomicU64] {
        unsafe {
            let ptr = self.base.add(self.layout.versions_offset) as *const AtomicU64;
            core::slice::from_raw_parts(ptr, self.capacity)
        }
    }

    /// Look up a block index by coordinate.
    #[inline]
    pub fn get_block_index(&self, coord: BlockCoord) -> Option<usize> {
        let morton = ash_core::morton_encode_signed(coord);
        let hash = morton as u32;
        let map = self.block_map();
        let map_capacity = self.capacity * 2;
        let mut idx = (morton as usize) % map_capacity;

        for _ in 0..map_capacity {
            let entry = map[idx].load(Ordering::Acquire);
            let state = entry >> 62;
            let entry_hash = entry as u32;

            match state {
                0 => return None, // Empty
                2 if entry_hash == hash => {
                    // Occupied with matching hash
                    let block_idx = ((entry >> 32) & 0x3FFF_FFFF) as usize;
                    return Some(block_idx);
                }
                _ => idx = (idx + 1) % map_capacity,
            }
        }
        None
    }

    /// Read all N feature values for a cell with seqlock protection.
    ///
    /// Returns `None` if the block doesn't exist or is being written.
    #[inline]
    pub fn read_values_seqlock(&self, block_idx: usize, cell_idx: usize) -> Option<[f32; N]> {
        let versions = self.versions();

        // Seqlock read protocol
        loop {
            // 1. Read version (must be even = not being written)
            let v1 = versions[block_idx].load(Ordering::Acquire);
            if v1 & 1 != 0 {
                // Odd = write in progress, spin
                core::hint::spin_loop();
                continue;
            }

            // 2. Read all feature values
            let value_idx = block_idx * self.cells_per_block + cell_idx;
            let mut values = [0.0f32; N];
            for i in 0..N {
                let ptr = self.feature_values_ptr(i);
                values[i] = unsafe { *ptr.add(value_idx) };
            }

            // 3. Memory fence
            core::sync::atomic::fence(Ordering::Acquire);

            // 4. Re-read version (must match)
            let v2 = versions[block_idx].load(Ordering::Relaxed);
            if v1 == v2 {
                return Some(values);
            }
            // Version changed, retry
            core::hint::spin_loop();
        }
    }

    /// Read all N feature values without seqlock (for single-writer scenarios).
    ///
    /// This is faster but only safe when no concurrent writes are happening.
    #[inline]
    pub fn read_values_unchecked(&self, block_idx: usize, cell_idx: usize) -> [f32; N] {
        let value_idx = block_idx * self.cells_per_block + cell_idx;
        let mut values = [0.0f32; N];
        for i in 0..N {
            let ptr = self.feature_values_ptr(i);
            values[i] = unsafe { *ptr.add(value_idx) };
        }
        values
    }

    /// Query all N feature values at a world-space point.
    pub fn query(&self, point: Point3) -> Option<[f32; N]> {
        let (block, cell, local) = decompose_point(point, self.cell_size, self.grid_dim);

        let result: InterpolationResult<N> = trilinear_interpolate(self, block, cell, local)?;
        Some(result.values)
    }

    /// Query with gradient for all N features.
    pub fn query_with_gradient(&self, point: Point3) -> Option<([f32; N], [[f32; 3]; N])> {
        let (block, cell, local) = decompose_point(point, self.cell_size, self.grid_dim);

        let (result, cell_grads) = trilinear_with_gradient(self, block, cell, local)?;

        let inv_cell = 1.0 / self.cell_size;
        let mut world_grads = [[0.0f32; 3]; N];
        for i in 0..N {
            world_grads[i] = [
                cell_grads[i][0] * inv_cell,
                cell_grads[i][1] * inv_cell,
                cell_grads[i][2] * inv_cell,
            ];
        }

        Some((result.values, world_grads))
    }

    /// Get the number of allocated blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.header().num_blocks.load(Ordering::Relaxed) as usize
    }

    /// Get the grid dimension.
    #[inline]
    pub fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    /// Get the cell size.
    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

// SDF-specific methods for N=1 views
impl SharedGridView<1> {
    /// Query the SDF value at a world-space point (convenience for N=1).
    #[inline]
    pub fn query_sdf(&self, point: Point3) -> Option<f32> {
        self.query(point).map(|[v]| v)
    }

    /// Check if a point is in collision.
    #[inline]
    pub fn in_collision(&self, point: Point3, threshold: f32) -> bool {
        self.query(point)
            .map(|[sdf]| sdf < threshold)
            .unwrap_or(false)
    }
}

impl<const N: usize> CellValueProvider<N> for SharedGridView<N> {
    fn get_corner_values(
        &self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
    ) -> Option<[f32; N]> {
        let (resolved_block, resolved_cell) =
            resolve_corner(block, cell, corner, self.grid_dim);

        let block_idx = self.get_block_index(resolved_block)?;
        let cell_idx = resolved_cell.flat_index(self.grid_dim);

        // Use seqlock for safe concurrent access
        let values = self.read_values_seqlock(block_idx, cell_idx)?;

        // Check for untrained sentinel in first feature
        if values[0] >= UNTRAINED_SENTINEL * 0.5 {
            return None;
        }

        Some(values)
    }

    #[inline]
    fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    #[inline]
    fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::layout::SharedHeader;

    fn create_test_shared_memory<const N: usize>(
        grid_dim: u32,
        cell_size: f32,
        capacity: usize,
    ) -> (Vec<u8>, SharedLayout) {
        let layout = SharedLayout::compute::<N>(grid_dim, capacity);
        let mut buffer = vec![0u8; layout.total_size];

        // Initialize header
        let header_ptr = buffer.as_mut_ptr() as *mut SharedHeader;
        unsafe {
            *header_ptr = SharedHeader::new(grid_dim, cell_size, capacity as u32, N as u16);
        }

        (buffer, layout)
    }

    #[test]
    fn test_shared_view_from_ptr_valid() {
        let (buffer, _) = create_test_shared_memory::<1>(8, 0.1, 100);

        let view = unsafe { SharedGridView::<1>::from_ptr(buffer.as_ptr()) };
        assert!(view.is_some());

        let view = view.unwrap();
        assert_eq!(view.grid_dim(), 8);
        assert!((view.cell_size() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_shared_view_from_ptr_invalid_magic() {
        let buffer = vec![0u8; 1024];
        // Don't initialize header properly

        let view = unsafe { SharedGridView::<1>::from_ptr(buffer.as_ptr()) };
        assert!(view.is_none());
    }

    #[test]
    fn test_shared_view_feature_dim_mismatch() {
        // Create with N=1
        let (buffer, _) = create_test_shared_memory::<1>(8, 0.1, 100);

        // Try to create view with N=4
        let view = unsafe { SharedGridView::<4>::from_ptr(buffer.as_ptr()) };
        assert!(view.is_none());
    }

    #[test]
    fn test_shared_view_n4() {
        let (buffer, _) = create_test_shared_memory::<4>(8, 0.1, 100);

        let view = unsafe { SharedGridView::<4>::from_ptr(buffer.as_ptr()) };
        assert!(view.is_some());
    }
}
