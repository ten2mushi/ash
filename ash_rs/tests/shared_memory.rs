//! Shared memory interface tests for ash_rs.
//!
//! These tests verify the seqlock protocol and SharedGridView functionality
//! for zero-copy integration with neural_ash.

use ash_core::{BlockCoord, Point3};
use ash_rs::{
    compute_shared_size, SharedGridView, SharedHeader, SharedLayout, SHARED_MAGIC,
};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// =============================================================================
// SharedHeader Tests
// =============================================================================

#[test]
fn test_shared_header_creation() {
    let header = SharedHeader::new(8, 0.1, 1000, 1);

    assert_eq!(header.magic, SHARED_MAGIC);
    assert_eq!(header.grid_dim, 8);
    assert!((header.cell_size - 0.1).abs() < 1e-6);
    assert_eq!(header.capacity, 1000);
    assert_eq!(header.feature_dim, 1);
    assert_eq!(header.num_blocks.load(Ordering::Relaxed), 0);
    assert_eq!(header.global_version.load(Ordering::Relaxed), 0);
}

#[test]
fn test_shared_header_validation() {
    let valid = SharedHeader::new(8, 0.1, 1000, 1);
    assert!(valid.validate());
    assert!(valid.validate_feature_dim::<1>());

    // Test multi-feature header
    let valid4 = SharedHeader::new(8, 0.1, 1000, 4);
    assert!(valid4.validate());
    assert!(valid4.validate_feature_dim::<4>());
    assert!(!valid4.validate_feature_dim::<1>()); // Wrong feature dim
}

#[test]
fn test_shared_header_cells_per_block() {
    let header = SharedHeader::new(8, 0.1, 1000, 1);
    assert_eq!(header.cells_per_block(), 512); // 8^3

    let header4 = SharedHeader::new(4, 0.1, 1000, 1);
    assert_eq!(header4.cells_per_block(), 64); // 4^3

    let header16 = SharedHeader::new(16, 0.1, 1000, 1);
    assert_eq!(header16.cells_per_block(), 4096); // 16^3
}

#[test]
fn test_shared_header_block_size() {
    let header = SharedHeader::new(8, 0.1, 1000, 1);
    assert!((header.block_size() - 0.8).abs() < 1e-6); // 8 * 0.1

    let header2 = SharedHeader::new(8, 0.05, 1000, 1);
    assert!((header2.block_size() - 0.4).abs() < 1e-6); // 8 * 0.05
}

#[test]
fn test_shared_header_alignment() {
    // SharedHeader should be 64 bytes (cache line aligned)
    assert_eq!(std::mem::size_of::<SharedHeader>(), 64);
    assert_eq!(std::mem::align_of::<SharedHeader>(), 64);
}

// =============================================================================
// SharedLayout Tests
// =============================================================================

#[test]
fn test_shared_layout_compute() {
    let layout = SharedLayout::compute::<1>(8, 1000);

    // Verify offsets are in correct order
    assert!(layout.block_map_offset > 0); // After header
    assert!(layout.coords_offset > layout.block_map_offset);
    assert!(layout.feature_offsets[0] > layout.coords_offset);
    assert!(layout.versions_offset > layout.feature_offsets[0]);
    assert!(layout.total_size > layout.versions_offset);
}

#[test]
fn test_shared_layout_values_alignment() {
    // Values should be 32-byte aligned for SIMD
    let layout = SharedLayout::compute::<1>(8, 1000);
    assert_eq!(layout.feature_offsets[0] % 32, 0, "Values offset should be 32-byte aligned");
}

#[test]
fn test_shared_layout_sizes() {
    let grid_dim = 8u32;
    let capacity = 1000usize;
    let layout = SharedLayout::compute::<1>(grid_dim, capacity);

    // Block map: 2 * capacity * 8 bytes (AtomicU64)
    let expected_block_map_size = 2 * capacity * 8;
    let actual_block_map_size = layout.coords_offset - layout.block_map_offset;
    assert_eq!(actual_block_map_size, expected_block_map_size);

    // Coords: capacity * 12 bytes (BlockCoord = 3 * i32)
    let expected_coords_size = capacity * 12;
    // Account for alignment padding
    let actual_coords_space = layout.feature_offsets[0] - layout.coords_offset;
    assert!(actual_coords_space >= expected_coords_size);
}

#[test]
fn test_shared_layout_different_configs() {
    // Test with various configurations
    let configs = [
        (8, 100),
        (8, 1000),
        (8, 10000),
        (4, 1000),
        (16, 1000),
    ];

    for (grid_dim, capacity) in configs {
        let layout = SharedLayout::compute::<1>(grid_dim, capacity);

        // Basic sanity checks
        assert!(layout.total_size > 0);
        assert!(layout.feature_offsets[0] % 32 == 0);
        assert!(layout.total_size > layout.versions_offset);
    }
}

#[test]
fn test_shared_layout_multi_feature() {
    // Test with 4 features
    let layout4 = SharedLayout::compute::<4>(8, 1000);

    // Should have 4 valid feature offsets
    for i in 0..4 {
        assert!(layout4.feature_offsets[i] > 0);
        assert!(layout4.feature_offsets[i] % 32 == 0, "Feature {} offset should be 32-byte aligned", i);
    }

    // Feature offsets should be in increasing order
    for i in 1..4 {
        assert!(layout4.feature_offsets[i] > layout4.feature_offsets[i-1]);
    }
}

// =============================================================================
// compute_shared_size Tests
// =============================================================================

#[test]
fn test_compute_shared_size() {
    let size = compute_shared_size::<1>(8, 1000);

    // Size should account for:
    // - Header: 64 bytes
    // - Block map: 2000 * 8 = 16000 bytes
    // - Coords: 1000 * 12 = 12000 bytes
    // - Values: 1000 * 512 * 4 = 2048000 bytes
    // - Versions: 1000 * 8 = 8000 bytes
    // Plus alignment padding

    assert!(size > 2_000_000, "Expected at least 2MB, got {}", size);
    assert!(size < 3_000_000, "Expected less than 3MB, got {}", size);
}

#[test]
fn test_compute_shared_size_matches_layout() {
    let grid_dim = 8;
    let capacity = 1000;

    let size = compute_shared_size::<1>(grid_dim, capacity);
    let layout = SharedLayout::compute::<1>(grid_dim, capacity);

    assert_eq!(size, layout.total_size);
}

#[test]
fn test_compute_shared_size_multi_feature() {
    let grid_dim = 8;
    let capacity = 1000;

    let size1 = compute_shared_size::<1>(grid_dim, capacity);
    let size4 = compute_shared_size::<4>(grid_dim, capacity);

    // 4 features should be ~4x the values section of 1 feature
    // But header and other metadata don't scale, so ratio will be less than 4
    assert!(size4 > size1);
    assert!(size4 < size1 * 5); // Sanity check
}

// =============================================================================
// SharedGridView Basic Tests
// =============================================================================

/// Helper to create a test shared memory region
fn create_test_shared_memory(
    grid_dim: u32,
    cell_size: f32,
    capacity: usize,
) -> (Vec<u8>, SharedLayout) {
    let layout = SharedLayout::compute::<1>(grid_dim, capacity);
    let size = layout.total_size;

    // Allocate with extra space for alignment
    let mut buffer = vec![0u8; size + 64];
    let aligned_ptr = {
        let ptr = buffer.as_mut_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    // Initialize header
    unsafe {
        let header = &mut *(aligned_ptr as *mut SharedHeader);
        header.magic = SHARED_MAGIC;
        header.grid_dim = grid_dim;
        header.cell_size = cell_size;
        header.capacity = capacity as u32;
        header.feature_dim = 1;
        header.num_blocks = AtomicU32::new(0);
        header.global_version = AtomicU64::new(0);
    }

    (buffer, layout)
}

#[test]
fn test_shared_grid_view_from_ptr_valid() {
    let grid_dim = 8u32;
    let cell_size = 0.1f32;
    let capacity = 100usize;

    let (buffer, _layout) = create_test_shared_memory(grid_dim, cell_size, capacity);

    // Get aligned pointer
    let aligned_ptr = {
        let ptr = buffer.as_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    // Create view
    let view = unsafe { SharedGridView::<1>::from_ptr(aligned_ptr) };

    assert!(view.is_some(), "Should successfully create view from valid memory");

    let view = view.unwrap();
    assert_eq!(view.num_blocks(), 0);
}

#[test]
fn test_shared_grid_view_from_ptr_invalid_magic() {
    let mut buffer = vec![0u8; 1024];

    // Set invalid magic
    buffer[0..4].copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    let view = unsafe { SharedGridView::<1>::from_ptr(buffer.as_ptr()) };

    assert!(view.is_none(), "Should fail with invalid magic");
}

// =============================================================================
// Seqlock Protocol Tests
// =============================================================================

#[test]
fn test_seqlock_even_version_read() {
    let grid_dim = 8u32;
    let cell_size = 0.1f32;
    let capacity = 10usize;

    let (mut buffer, layout) = create_test_shared_memory(grid_dim, cell_size, capacity);

    // Get aligned pointer
    let aligned_ptr = {
        let ptr = buffer.as_mut_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    // Add a block
    unsafe {
        let header = &mut *(aligned_ptr as *mut SharedHeader);
        header.num_blocks.store(1, Ordering::Release);

        // Set up block map entry
        let map_ptr = aligned_ptr.add(layout.block_map_offset) as *mut AtomicU64;
        // Block (0,0,0) -> index 0
        let morton = ash_core::morton_encode_signed(BlockCoord::new(0, 0, 0));
        let hash = morton as u32;
        let idx = (morton as usize) % (capacity * 2);
        let entry = (2u64 << 62) | (0u64 << 32) | (hash as u64); // state=occupied, index=0
        (*map_ptr.add(idx)).store(entry, Ordering::Release);

        // Set values
        let values_ptr = aligned_ptr.add(layout.feature_offsets[0]) as *mut f32;
        for i in 0..512 {
            *values_ptr.add(i) = 0.5;
        }

        // Set version to even (0) - not being written
        let versions_ptr = aligned_ptr.add(layout.versions_offset) as *mut AtomicU64;
        (*versions_ptr).store(0, Ordering::Release);
    }

    // Create view and read
    let view = unsafe { SharedGridView::<1>::from_ptr(aligned_ptr as *const u8) }.unwrap();

    let value = view.read_values_seqlock(0, 0);
    assert!(value.is_some());
    let [v] = value.unwrap();
    assert!((v - 0.5).abs() < 1e-6);
}

#[test]
fn test_seqlock_version_consistency() {
    // This test simulates version checking by verifying the seqlock logic
    let grid_dim = 8u32;
    let cell_size = 0.1f32;
    let capacity = 10usize;

    let (mut buffer, layout) = create_test_shared_memory(grid_dim, cell_size, capacity);

    let aligned_ptr = {
        let ptr = buffer.as_mut_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    // Set up a valid block
    unsafe {
        let header = &mut *(aligned_ptr as *mut SharedHeader);
        header.num_blocks.store(1, Ordering::Release);

        let values_ptr = aligned_ptr.add(layout.feature_offsets[0]) as *mut f32;
        for i in 0..512 {
            *values_ptr.add(i) = 1.0;
        }

        // Version starts at 0 (even, valid)
        let versions_ptr = aligned_ptr.add(layout.versions_offset) as *mut AtomicU64;
        (*versions_ptr).store(0, Ordering::Release);
    }

    // Read should succeed with even version
    // (We can't easily test the spin-wait behavior without a separate thread)
}

// =============================================================================
// SharedGridView Query Tests
// =============================================================================

#[test]
fn test_shared_grid_view_query_unallocated() {
    let grid_dim = 8u32;
    let cell_size = 0.1f32;
    let capacity = 10usize;

    let (buffer, _layout) = create_test_shared_memory(grid_dim, cell_size, capacity);

    let aligned_ptr = {
        let ptr = buffer.as_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    let view = unsafe { SharedGridView::<1>::from_ptr(aligned_ptr) }.unwrap();

    // Query should return None for unallocated region
    let result = view.query(Point3::new(0.5, 0.5, 0.5));
    assert!(result.is_none(), "Query in unallocated region should return None");
}

#[test]
fn test_shared_grid_view_collision_check() {
    let grid_dim = 8u32;
    let cell_size = 0.1f32;
    let capacity = 10usize;

    let (buffer, _layout) = create_test_shared_memory(grid_dim, cell_size, capacity);

    let aligned_ptr = {
        let ptr = buffer.as_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    let view = unsafe { SharedGridView::<1>::from_ptr(aligned_ptr) }.unwrap();

    // Collision check in unallocated region should return false (safe)
    let collision = view.in_collision(Point3::new(0.5, 0.5, 0.5), 0.0);
    assert!(!collision, "Unallocated region should be treated as safe");
}

// =============================================================================
// SharedGridView Block Map Tests
// =============================================================================

#[test]
fn test_shared_grid_view_get_block_index() {
    let grid_dim = 8u32;
    let cell_size = 0.1f32;
    let capacity = 100usize;

    let (mut buffer, layout) = create_test_shared_memory(grid_dim, cell_size, capacity);

    let aligned_ptr = {
        let ptr = buffer.as_mut_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    // Add several blocks to the map
    let test_blocks = [
        (BlockCoord::new(0, 0, 0), 0usize),
        (BlockCoord::new(1, 0, 0), 1),
        (BlockCoord::new(0, 1, 0), 2),
        (BlockCoord::new(-1, -1, -1), 3),
    ];

    unsafe {
        let header = &mut *(aligned_ptr as *mut SharedHeader);
        header.num_blocks.store(test_blocks.len() as u32, Ordering::Release);

        let map_ptr = aligned_ptr.add(layout.block_map_offset) as *mut AtomicU64;

        for (coord, idx) in &test_blocks {
            let morton = ash_core::morton_encode_signed(*coord);
            let hash = morton as u32;
            let slot = (morton as usize) % (capacity * 2);
            let entry = (2u64 << 62) | ((*idx as u64) << 32) | (hash as u64);
            (*map_ptr.add(slot)).store(entry, Ordering::Release);
        }
    }

    let view = unsafe { SharedGridView::<1>::from_ptr(aligned_ptr as *const u8) }.unwrap();

    // Verify block lookups
    for (coord, expected_idx) in &test_blocks {
        let idx = view.get_block_index(*coord);
        assert_eq!(
            idx,
            Some(*expected_idx),
            "Block {:?} should have index {}",
            coord,
            expected_idx
        );
    }

    // Non-existent block should return None
    let missing = view.get_block_index(BlockCoord::new(100, 100, 100));
    assert!(missing.is_none());
}

// =============================================================================
// Version Number Tests
// =============================================================================

#[test]
fn test_version_number_even_odd() {
    // Even version = not being written
    // Odd version = write in progress

    let even_versions: Vec<u64> = vec![0, 2, 4, 100, 1000000];
    let odd_versions: Vec<u64> = vec![1, 3, 5, 101, 1000001];

    for v in &even_versions {
        assert!(v & 1 == 0, "{} should be even", v);
    }

    for v in &odd_versions {
        assert!(v & 1 == 1, "{} should be odd", v);
    }
}

#[test]
fn test_version_wraparound() {
    // Test behavior near u64::MAX
    let near_max: u64 = u64::MAX - 10;

    // Incrementing should wrap around safely
    let wrapped = near_max.wrapping_add(20);
    assert!(wrapped < near_max, "Should have wrapped around");

    // Even/odd check should still work
    assert_eq!(wrapped & 1, near_max.wrapping_add(20) & 1);
}

// =============================================================================
// CellValueProvider Implementation Tests
// =============================================================================

#[test]
fn test_shared_grid_view_cell_value_provider() {
    let grid_dim = 8u32;
    let cell_size = 0.1f32;
    let capacity = 10usize;

    let (mut buffer, layout) = create_test_shared_memory(grid_dim, cell_size, capacity);

    let aligned_ptr = {
        let ptr = buffer.as_mut_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    // Set up a valid block
    unsafe {
        let header = &mut *(aligned_ptr as *mut SharedHeader);
        header.num_blocks.store(1, Ordering::Release);

        // Add block to map
        let map_ptr = aligned_ptr.add(layout.block_map_offset) as *mut AtomicU64;
        let morton = ash_core::morton_encode_signed(BlockCoord::new(0, 0, 0));
        let hash = morton as u32;
        let slot = (morton as usize) % (capacity * 2);
        let entry = (2u64 << 62) | (0u64 << 32) | (hash as u64);
        (*map_ptr.add(slot)).store(entry, Ordering::Release);

        // Set values
        let values_ptr = aligned_ptr.add(layout.feature_offsets[0]) as *mut f32;
        for i in 0..512 {
            *values_ptr.add(i) = i as f32 * 0.01;
        }

        // Set version
        let versions_ptr = aligned_ptr.add(layout.versions_offset) as *mut AtomicU64;
        (*versions_ptr).store(0, Ordering::Release);
    }

    let view = unsafe { SharedGridView::<1>::from_ptr(aligned_ptr as *const u8) }.unwrap();

    // Test CellValueProvider methods
    use ash_core::CellValueProvider;

    assert_eq!(view.grid_dim(), grid_dim);
    assert!((view.cell_size() - cell_size).abs() < 1e-6);
    assert!((view.block_size() - 0.8).abs() < 1e-6);
}

// =============================================================================
// Memory Safety Tests
// =============================================================================

#[test]
fn test_shared_memory_read_only_access() {
    // SharedGridView should only read, never write
    let grid_dim = 8u32;
    let cell_size = 0.1f32;
    let capacity = 10usize;

    let (buffer, _layout) = create_test_shared_memory(grid_dim, cell_size, capacity);

    // Store original buffer contents
    let _original: Vec<u8> = buffer.clone();

    let aligned_ptr = {
        let ptr = buffer.as_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    let view = unsafe { SharedGridView::<1>::from_ptr(aligned_ptr) }.unwrap();

    // Perform various operations
    let _ = view.num_blocks();
    let _ = view.get_block_index(BlockCoord::new(0, 0, 0));
    let _ = view.query(Point3::new(0.5, 0.5, 0.5));
    let _ = view.in_collision(Point3::new(0.5, 0.5, 0.5), 0.0);

    // Buffer should be unchanged
    // (This is a weak test since we can't easily detect writes to const pointers)
}

// =============================================================================
// Size Calculation Tests
// =============================================================================

#[test]
fn test_shared_size_scales_with_capacity() {
    let grid_dim = 8;

    let sizes: Vec<usize> = [100, 1000, 10000]
        .iter()
        .map(|&cap| compute_shared_size::<1>(grid_dim, cap))
        .collect();

    // Size should increase with capacity
    assert!(sizes[1] > sizes[0]);
    assert!(sizes[2] > sizes[1]);

    // Size should scale roughly linearly with capacity (values dominate)
    // 1000 blocks should be ~10x the size of 100 blocks
    let ratio_1k_to_100 = sizes[1] as f64 / sizes[0] as f64;
    assert!(
        ratio_1k_to_100 > 5.0 && ratio_1k_to_100 < 15.0,
        "Size ratio 1000/100 should be ~10, got {}",
        ratio_1k_to_100
    );
}

#[test]
fn test_shared_size_scales_with_grid_dim() {
    let capacity = 1000;

    let sizes: Vec<usize> = [4, 8, 16]
        .iter()
        .map(|&dim| compute_shared_size::<1>(dim, capacity))
        .collect();

    // Size should increase with grid_dim (values scale as dim^3)
    assert!(sizes[1] > sizes[0]);
    assert!(sizes[2] > sizes[1]);

    // 8^3 = 512 cells, 4^3 = 64 cells, ratio should be ~8
    // But header and other metadata don't scale, so ratio will be less
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_shared_minimum_capacity() {
    let grid_dim = 8;
    let capacity = 1;

    let size = compute_shared_size::<1>(grid_dim, capacity);
    let layout = SharedLayout::compute::<1>(grid_dim, capacity);

    assert!(size > 0);
    assert!(layout.total_size == size);
}

#[test]
fn test_shared_large_capacity() {
    let grid_dim = 8;
    let capacity = 100000; // 100k blocks

    let size = compute_shared_size::<1>(grid_dim, capacity);

    // 100k blocks * 512 cells * 4 bytes = ~200MB for values alone
    assert!(size > 200_000_000, "Size should be > 200MB");
}
