//! Comprehensive Test Suite for ash_core
//!
//! This test file follows the "Tests as Definition: The Yoneda Way" philosophy.
//! Each test serves as part of the behavioral specification - together they define
//! what ash_core does so completely that any implementation passing all tests
//! must be functionally equivalent to the original.
//!
//! # Test Categories
//!
//! 1. **Type Invariants** - Properties that must hold for all instances
//! 2. **Roundtrip Properties** - Encode/decode, decompose/compose pairs
//! 3. **Mathematical Invariants** - Weight sums, gradient formulas, etc.
//! 4. **Boundary Conditions** - Edge cases at limits of valid input
//! 5. **Error Conditions** - Proper handling of invalid inputs
//! 6. **Numerical Verification** - Analytical vs numerical gradient checks
//! 7. **SDF-Specific Properties** - Sphere, linear, and other SDF behaviors
//! 8. **Table Invariants** - Marching cubes table correctness

use ash_core::prelude::*;
use ash_core::*;

// =============================================================================
// Test Helpers and Mock Providers
// =============================================================================

/// Mock provider implementing a linear SDF: f(x,y,z) = x + y + z
/// This provides a known analytical solution for gradient verification.
///
/// IMPORTANT: The trilinear_interpolate function calls get_corner_value with
/// corner=(0,0,0) after resolving the cell position. So we ignore the corner
/// parameter and compute the SDF at the cell origin.
struct LinearSdfProvider {
    grid_dim: u32,
    cell_size: f32,
}

impl CellValueProvider<1> for LinearSdfProvider {
    fn get_corner_values(
        &self,
        block: BlockCoord,
        cell: CellCoord,
        _corner: (u32, u32, u32),
    ) -> Option<[f32; 1]> {
        // The interpolation code resolves corners to adjacent cells, then
        // queries with corner=(0,0,0). So we compute the SDF at cell origin.
        let block_size = self.block_size();
        let x = block.x as f32 * block_size + cell.x as f32 * self.cell_size;
        let y = block.y as f32 * block_size + cell.y as f32 * self.cell_size;
        let z = block.z as f32 * block_size + cell.z as f32 * self.cell_size;
        Some([x + y + z])
    }

    fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

/// Mock provider implementing a sphere SDF centered at a given point.
///
/// IMPORTANT: The trilinear_interpolate function calls get_corner_value with
/// corner=(0,0,0) after resolving the cell position. So we ignore the corner
/// parameter and compute the SDF at the cell origin.
struct SphereSdfProvider {
    grid_dim: u32,
    cell_size: f32,
    center: Point3,
    radius: f32,
}

impl CellValueProvider<1> for SphereSdfProvider {
    fn get_corner_values(
        &self,
        block: BlockCoord,
        cell: CellCoord,
        _corner: (u32, u32, u32),
    ) -> Option<[f32; 1]> {
        // The interpolation code resolves corners to adjacent cells, then
        // queries with corner=(0,0,0). So we compute the SDF at cell origin.
        let block_size = self.block_size();
        let x = block.x as f32 * block_size + cell.x as f32 * self.cell_size;
        let y = block.y as f32 * block_size + cell.y as f32 * self.cell_size;
        let z = block.z as f32 * block_size + cell.z as f32 * self.cell_size;

        let point = Point3::new(x, y, z);
        let diff = point - self.center;
        Some([diff.length() - self.radius])
    }

    fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

/// Mock provider that returns the UNTRAINED_SENTINEL for specific cells.
struct UntrainedRegionProvider {
    grid_dim: u32,
    cell_size: f32,
    untrained_blocks: Vec<BlockCoord>,
}

impl CellValueProvider<1> for UntrainedRegionProvider {
    fn get_corner_values(
        &self,
        block: BlockCoord,
        _cell: CellCoord,
        _corner: (u32, u32, u32),
    ) -> Option<[f32; 1]> {
        if self.untrained_blocks.contains(&block) {
            Some([UNTRAINED_SENTINEL])
        } else {
            Some([0.5])
        }
    }

    fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

/// Mock provider that returns None for missing blocks.
struct SparseProvider {
    grid_dim: u32,
    cell_size: f32,
    existing_blocks: Vec<BlockCoord>,
}

impl CellValueProvider<1> for SparseProvider {
    fn get_corner_values(
        &self,
        block: BlockCoord,
        _cell: CellCoord,
        _corner: (u32, u32, u32),
    ) -> Option<[f32; 1]> {
        if self.existing_blocks.contains(&block) {
            Some([0.5])
        } else {
            None
        }
    }

    fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

/// Mock provider with constant corner values for controlled testing.
///
/// This provider is designed to work with the marching cubes algorithm.
/// Since trilinear_interpolate uses resolve_corner and then queries with
/// corner=(0,0,0), we need to track which corner was originally requested
/// based on the resolved cell position.
///
/// For marching cubes, we want to control the 8 corner values of a single cell.
/// The simplest approach is to return values based on the cell's position
/// relative to the origin cell.
struct ConstantProvider {
    grid_dim: u32,
    cell_size: f32,
    values: [f32; 8],
}

impl CellValueProvider<1> for ConstantProvider {
    fn get_corner_values(
        &self,
        _block: BlockCoord,
        cell: CellCoord,
        _corner: (u32, u32, u32),
    ) -> Option<[f32; 1]> {
        // For cells within the first 2x2x2 region, map to our 8 values
        // This allows testing marching cubes on cell (0,0,0)
        if cell.x <= 1 && cell.y <= 1 && cell.z <= 1 {
            // Map cell position to corner index
            // Cell (0,0,0) -> corner 0, (1,0,0) -> corner 1, etc.
            let corner_idx = match (cell.x, cell.y, cell.z) {
                (0, 0, 0) => 0,
                (1, 0, 0) => 1,
                (1, 1, 0) => 2,
                (0, 1, 0) => 3,
                (0, 0, 1) => 4,
                (1, 0, 1) => 5,
                (1, 1, 1) => 6,
                (0, 1, 1) => 7,
                _ => return Some([1.0]), // Outside should be positive
            };
            Some([self.values[corner_idx]])
        } else {
            // Default for cells outside the 2x2x2 region
            Some([1.0])
        }
    }

    fn grid_dim(&self) -> u32 {
        self.grid_dim
    }

    fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

/// Mock gradient accumulator for testing backward pass.
struct TestGradientAccumulator {
    accumulated: Vec<(BlockCoord, CellCoord, (u32, u32, u32), f32)>,
}

impl TestGradientAccumulator {
    fn new() -> Self {
        Self {
            accumulated: Vec::new(),
        }
    }

    fn total_gradient(&self) -> f32 {
        self.accumulated.iter().map(|(_, _, _, g)| g).sum()
    }
}

impl GradientAccumulator<1> for TestGradientAccumulator {
    fn accumulate_gradient(
        &mut self,
        block: BlockCoord,
        cell: CellCoord,
        corner: (u32, u32, u32),
        weight: f32,
        upstream_grad: [f32; 1],
    ) {
        self.accumulated
            .push((block, cell, corner, weight * upstream_grad[0]));
    }
}

// =============================================================================
// SECTION 1: Point3 Type Tests
// =============================================================================

mod point3_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Construction and Basic Properties
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_point_with_new() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn should_create_point_with_splat() {
        let p = Point3::splat(5.0);
        assert_eq!(p.x, 5.0);
        assert_eq!(p.y, 5.0);
        assert_eq!(p.z, 5.0);
    }

    #[test]
    fn should_have_default_at_origin() {
        let p = Point3::default();
        assert_eq!(p, Point3::new(0.0, 0.0, 0.0));
    }

    // -------------------------------------------------------------------------
    // Arithmetic Operations
    // -------------------------------------------------------------------------

    #[test]
    fn should_add_points_componentwise() {
        let a = Point3::new(1.0, 2.0, 3.0);
        let b = Point3::new(4.0, 5.0, 6.0);
        let sum = a + b;
        assert_eq!(sum, Point3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn should_subtract_points_componentwise() {
        let a = Point3::new(4.0, 5.0, 6.0);
        let b = Point3::new(1.0, 2.0, 3.0);
        let diff = a - b;
        assert_eq!(diff, Point3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn should_multiply_by_scalar_right() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let scaled = p * 2.0;
        assert_eq!(scaled, Point3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn should_multiply_by_scalar_left() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let scaled = 2.0 * p;
        assert_eq!(scaled, Point3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn should_divide_by_scalar() {
        let p = Point3::new(2.0, 4.0, 6.0);
        let divided = p / 2.0;
        assert_eq!(divided, Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn should_negate_point() {
        let p = Point3::new(1.0, -2.0, 3.0);
        let neg = -p;
        assert_eq!(neg, Point3::new(-1.0, 2.0, -3.0));
    }

    // -------------------------------------------------------------------------
    // Vector Operations
    // -------------------------------------------------------------------------

    #[test]
    fn should_compute_dot_product() {
        let a = Point3::new(1.0, 2.0, 3.0);
        let b = Point3::new(4.0, 5.0, 6.0);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(a.dot(b), 32.0);
    }

    #[test]
    fn should_have_zero_dot_product_for_orthogonal_vectors() {
        let x = Point3::new(1.0, 0.0, 0.0);
        let y = Point3::new(0.0, 1.0, 0.0);
        assert_eq!(x.dot(y), 0.0);
    }

    #[test]
    fn should_compute_cross_product() {
        let x = Point3::new(1.0, 0.0, 0.0);
        let y = Point3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert_eq!(z, Point3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn should_have_anticommutative_cross_product() {
        let a = Point3::new(1.0, 2.0, 3.0);
        let b = Point3::new(4.0, 5.0, 6.0);
        let cross_ab = a.cross(b);
        let cross_ba = b.cross(a);
        assert_eq!(cross_ab, -cross_ba);
    }

    #[test]
    fn should_compute_length_squared() {
        let p = Point3::new(3.0, 4.0, 0.0);
        assert_eq!(p.length_squared(), 25.0);
    }

    #[test]
    fn should_compute_length() {
        let p = Point3::new(3.0, 4.0, 0.0);
        assert_eq!(p.length(), 5.0);
    }

    #[test]
    fn should_normalize_to_unit_length() {
        let p = Point3::new(3.0, 4.0, 0.0);
        let n = p.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert!((n.x - 0.6).abs() < 1e-6);
        assert!((n.y - 0.8).abs() < 1e-6);
    }

    #[test]
    fn should_handle_normalize_of_zero_vector() {
        let zero = Point3::splat(0.0);
        let n = zero.normalize();
        // Should return zero vector, not NaN
        assert_eq!(n, Point3::splat(0.0));
    }

    // -------------------------------------------------------------------------
    // Interpolation
    // -------------------------------------------------------------------------

    #[test]
    fn should_lerp_at_t_zero_returns_start() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 10.0, 10.0);
        assert_eq!(a.lerp(b, 0.0), a);
    }

    #[test]
    fn should_lerp_at_t_one_returns_end() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 10.0, 10.0);
        assert_eq!(a.lerp(b, 1.0), b);
    }

    #[test]
    fn should_lerp_at_midpoint() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 10.0, 10.0);
        let mid = a.lerp(b, 0.5);
        assert_eq!(mid, Point3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn should_lerp_extrapolate_beyond_one() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 0.0, 0.0);
        let extrapolated = a.lerp(b, 2.0);
        assert_eq!(extrapolated, Point3::new(20.0, 0.0, 0.0));
    }

    // -------------------------------------------------------------------------
    // Component-wise Operations
    // -------------------------------------------------------------------------

    #[test]
    fn should_compute_componentwise_min() {
        let a = Point3::new(1.0, 5.0, 3.0);
        let b = Point3::new(4.0, 2.0, 6.0);
        let min = a.min(b);
        assert_eq!(min, Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn should_compute_componentwise_max() {
        let a = Point3::new(1.0, 5.0, 3.0);
        let b = Point3::new(4.0, 2.0, 6.0);
        let max = a.max(b);
        assert_eq!(max, Point3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn should_compute_componentwise_abs() {
        let p = Point3::new(-1.0, 2.0, -3.0);
        let abs = p.abs();
        assert_eq!(abs, Point3::new(1.0, 2.0, 3.0));
    }

    // -------------------------------------------------------------------------
    // Conversions
    // -------------------------------------------------------------------------

    #[test]
    fn should_convert_to_and_from_array() {
        let arr = [1.0f32, 2.0, 3.0];
        let p: Point3 = arr.into();
        let back: [f32; 3] = p.into();
        assert_eq!(arr, back);
    }

    #[test]
    fn should_convert_from_tuple() {
        let tuple = (1.0f32, 2.0f32, 3.0f32);
        let p: Point3 = tuple.into();
        assert_eq!(p, Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn should_return_array_via_as_array() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p.as_array(), [1.0, 2.0, 3.0]);
    }

    // -------------------------------------------------------------------------
    // Special Values
    // -------------------------------------------------------------------------

    #[test]
    fn should_handle_infinity() {
        let p = Point3::new(f32::INFINITY, f32::NEG_INFINITY, 0.0);
        assert!(p.x.is_infinite());
        assert!(p.y.is_infinite());
        assert!(p.z.is_finite());
    }

    #[test]
    fn should_handle_nan_in_length() {
        let p = Point3::new(f32::NAN, 0.0, 0.0);
        assert!(p.length().is_nan());
    }
}

// =============================================================================
// SECTION 2: BlockCoord Type Tests
// =============================================================================

mod block_coord_tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn should_create_with_new() {
        let b = BlockCoord::new(1, -2, 3);
        assert_eq!(b.x, 1);
        assert_eq!(b.y, -2);
        assert_eq!(b.z, 3);
    }

    #[test]
    fn should_have_default_at_origin() {
        let b = BlockCoord::default();
        assert_eq!(b, BlockCoord::new(0, 0, 0));
    }

    #[test]
    fn should_convert_to_and_from_array() {
        let arr = [1i32, -2, 3];
        let b: BlockCoord = arr.into();
        let back: [i32; 3] = b.into();
        assert_eq!(arr, back);
    }

    #[test]
    fn should_implement_hash_consistently() {
        let mut set = HashSet::new();
        let b1 = BlockCoord::new(1, 2, 3);
        let b2 = BlockCoord::new(1, 2, 3);
        let b3 = BlockCoord::new(1, 2, 4);

        set.insert(b1);
        // b2 should be found since it equals b1
        assert!(set.contains(&b2));
        // b3 should not be found
        assert!(!set.contains(&b3));
    }

    #[test]
    fn should_handle_extreme_coordinates() {
        let max_block = BlockCoord::new(i32::MAX, i32::MAX, i32::MAX);
        let min_block = BlockCoord::new(i32::MIN, i32::MIN, i32::MIN);
        // Should not panic
        assert_eq!(max_block.as_array(), [i32::MAX, i32::MAX, i32::MAX]);
        assert_eq!(min_block.as_array(), [i32::MIN, i32::MIN, i32::MIN]);
    }
}

// =============================================================================
// SECTION 3: CellCoord Type Tests
// =============================================================================

mod cell_coord_tests {
    use super::*;

    #[test]
    fn should_create_with_new() {
        let c = CellCoord::new(1, 2, 3);
        assert_eq!(c.x, 1);
        assert_eq!(c.y, 2);
        assert_eq!(c.z, 3);
    }

    #[test]
    fn should_compute_flat_index_correctly() {
        let grid_dim = 8u32;

        // Origin
        assert_eq!(CellCoord::new(0, 0, 0).flat_index(grid_dim), 0);

        // Along x-axis
        assert_eq!(CellCoord::new(1, 0, 0).flat_index(grid_dim), 1);
        assert_eq!(CellCoord::new(7, 0, 0).flat_index(grid_dim), 7);

        // Along y-axis
        assert_eq!(CellCoord::new(0, 1, 0).flat_index(grid_dim), 8);
        assert_eq!(CellCoord::new(0, 7, 0).flat_index(grid_dim), 56);

        // Along z-axis
        assert_eq!(CellCoord::new(0, 0, 1).flat_index(grid_dim), 64);
        assert_eq!(CellCoord::new(0, 0, 7).flat_index(grid_dim), 448);

        // Maximum cell
        assert_eq!(CellCoord::new(7, 7, 7).flat_index(grid_dim), 511);
    }

    #[test]
    fn should_roundtrip_flat_index() {
        let grid_dim = 8u32;
        for z in 0..grid_dim {
            for y in 0..grid_dim {
                for x in 0..grid_dim {
                    let c = CellCoord::new(x, y, z);
                    let idx = c.flat_index(grid_dim);
                    let back = CellCoord::from_flat_index(idx, grid_dim);
                    assert_eq!(c, back, "Roundtrip failed for ({}, {}, {})", x, y, z);
                }
            }
        }
    }

    #[test]
    fn should_work_with_different_grid_dimensions() {
        for grid_dim in [1, 2, 4, 8, 16, 32] {
            let max_cell = CellCoord::new(grid_dim - 1, grid_dim - 1, grid_dim - 1);
            let expected_max_idx = (grid_dim * grid_dim * grid_dim - 1) as usize;
            assert_eq!(max_cell.flat_index(grid_dim), expected_max_idx);
        }
    }
}

// =============================================================================
// SECTION 4: LocalCoord Type Tests
// =============================================================================

mod local_coord_tests {
    use super::*;

    #[test]
    fn should_create_with_new() {
        let l = LocalCoord::new(0.25, 0.5, 0.75);
        assert_eq!(l.u, 0.25);
        assert_eq!(l.v, 0.5);
        assert_eq!(l.w, 0.75);
    }

    #[test]
    fn should_validate_valid_coords() {
        assert!(LocalCoord::new(0.0, 0.0, 0.0).is_valid());
        assert!(LocalCoord::new(1.0, 1.0, 1.0).is_valid());
        assert!(LocalCoord::new(0.5, 0.5, 0.5).is_valid());
    }

    #[test]
    fn should_invalidate_out_of_range_coords() {
        assert!(!LocalCoord::new(-0.1, 0.5, 0.5).is_valid());
        assert!(!LocalCoord::new(0.5, 1.1, 0.5).is_valid());
        assert!(!LocalCoord::new(0.5, 0.5, -0.001).is_valid());
    }

    #[test]
    fn should_clamp_to_valid_range() {
        let l = LocalCoord::new(-0.5, 0.5, 1.5);
        let clamped = l.clamped();
        assert_eq!(clamped.u, 0.0);
        assert_eq!(clamped.v, 0.5);
        assert_eq!(clamped.w, 1.0);
    }

    #[test]
    fn should_convert_to_and_from_array() {
        let arr = [0.25f32, 0.5, 0.75];
        let l: LocalCoord = arr.into();
        let back: [f32; 3] = l.into();
        assert_eq!(arr, back);
    }
}

// =============================================================================
// SECTION 5: Corner Index Tests
// =============================================================================

mod corner_index_tests {
    use super::*;

    #[test]
    fn should_map_indices_to_correct_corners() {
        assert_eq!(corner_from_index(0), (0, 0, 0));
        assert_eq!(corner_from_index(1), (1, 0, 0));
        assert_eq!(corner_from_index(2), (1, 1, 0));
        assert_eq!(corner_from_index(3), (0, 1, 0));
        assert_eq!(corner_from_index(4), (0, 0, 1));
        assert_eq!(corner_from_index(5), (1, 0, 1));
        assert_eq!(corner_from_index(6), (1, 1, 1));
        assert_eq!(corner_from_index(7), (0, 1, 1));
    }

    #[test]
    fn should_roundtrip_all_corner_indices() {
        for i in 0..8 {
            let corner = corner_from_index(i);
            let back = index_from_corner(corner);
            assert_eq!(i, back, "Roundtrip failed for index {}", i);
        }
    }

    #[test]
    fn should_handle_index_wrapping_with_mask() {
        // Indices >= 8 should wrap due to & 7 mask
        assert_eq!(corner_from_index(8), corner_from_index(0));
        assert_eq!(corner_from_index(9), corner_from_index(1));
        assert_eq!(corner_from_index(255), corner_from_index(7));
    }

    #[test]
    fn should_match_marching_cubes_convention() {
        // Verify the corner layout matches the MC CORNER_OFFSETS table
        use ash_core::marching_cubes::CORNER_OFFSETS;
        for i in 0..8 {
            let from_trait = corner_from_index(i);
            let from_table = CORNER_OFFSETS[i];
            assert_eq!(from_trait, from_table, "Mismatch at corner {}", i);
        }
    }
}

// =============================================================================
// SECTION 6: Coordinate Decomposition/Composition Tests
// =============================================================================

mod coords_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Decompose/Compose Roundtrip Properties
    // -------------------------------------------------------------------------

    #[test]
    fn should_roundtrip_origin() {
        let cell_size = 0.1;
        let grid_dim = 8;
        let point = Point3::new(0.0, 0.0, 0.0);

        let (block, cell, local) = decompose_point(point, cell_size, grid_dim);
        let reconstructed = compose_point(block, cell, local, cell_size, grid_dim);

        let diff = (point - reconstructed).length();
        assert!(diff < 1e-5, "Roundtrip failed at origin: diff = {}", diff);
    }

    #[test]
    fn should_roundtrip_positive_points() {
        let cell_size = 0.1;
        let grid_dim = 8;

        let test_points = [
            Point3::new(0.05, 0.05, 0.05),
            Point3::new(0.5, 0.5, 0.5),
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(10.5, 20.3, 30.7),
        ];

        for point in &test_points {
            let (block, cell, local) = decompose_point(*point, cell_size, grid_dim);
            let reconstructed = compose_point(block, cell, local, cell_size, grid_dim);

            let diff = (*point - reconstructed).length();
            assert!(
                diff < 1e-4,
                "Roundtrip failed for {:?}: got {:?}, diff = {}",
                point,
                reconstructed,
                diff
            );
        }
    }

    #[test]
    fn should_roundtrip_negative_points() {
        let cell_size = 0.1;
        let grid_dim = 8;

        let test_points = [
            Point3::new(-0.05, 0.0, 0.0),
            Point3::new(-0.5, -0.5, -0.5),
            Point3::new(-1.0, -2.0, -3.0),
            Point3::new(-10.5, -20.3, -30.7),
        ];

        for point in &test_points {
            let (block, cell, local) = decompose_point(*point, cell_size, grid_dim);
            let reconstructed = compose_point(block, cell, local, cell_size, grid_dim);

            let diff = (*point - reconstructed).length();
            assert!(
                diff < 1e-4,
                "Roundtrip failed for {:?}: got {:?}, diff = {}",
                point,
                reconstructed,
                diff
            );
        }
    }

    #[test]
    fn should_roundtrip_mixed_sign_points() {
        let cell_size = 0.1;
        let grid_dim = 8;

        let test_points = [
            Point3::new(-0.5, 0.5, -0.5),
            Point3::new(100.0, -100.0, 50.0),
            Point3::new(-0.001, 0.001, -0.001),
        ];

        for point in &test_points {
            let (block, cell, local) = decompose_point(*point, cell_size, grid_dim);
            let reconstructed = compose_point(block, cell, local, cell_size, grid_dim);

            let diff = (*point - reconstructed).length();
            assert!(
                diff < 1e-4,
                "Roundtrip failed for {:?}: got {:?}, diff = {}",
                point,
                reconstructed,
                diff
            );
        }
    }

    // -------------------------------------------------------------------------
    // Decomposition Correctness
    // -------------------------------------------------------------------------

    #[test]
    fn should_decompose_to_correct_block_for_positive_coords() {
        let cell_size = 0.1;
        let grid_dim = 8; // block_size = 0.8

        // Point in block (0,0,0): [0, 0.8)
        let (block, _, _) = decompose_point(Point3::new(0.5, 0.5, 0.5), cell_size, grid_dim);
        assert_eq!(block, BlockCoord::new(0, 0, 0));

        // Point in block (1,0,0): [0.8, 1.6)
        let (block, _, _) = decompose_point(Point3::new(0.9, 0.5, 0.5), cell_size, grid_dim);
        assert_eq!(block, BlockCoord::new(1, 0, 0));

        // Point in block (2,3,4)
        let (block, _, _) = decompose_point(Point3::new(1.7, 2.5, 3.3), cell_size, grid_dim);
        assert_eq!(block, BlockCoord::new(2, 3, 4));
    }

    #[test]
    fn should_decompose_to_correct_block_for_negative_coords() {
        let cell_size = 0.1;
        let grid_dim = 8; // block_size = 0.8

        // Point just below origin should be in block (-1, -1, -1)
        let (block, cell, _) = decompose_point(Point3::new(-0.05, -0.05, -0.05), cell_size, grid_dim);
        assert_eq!(block, BlockCoord::new(-1, -1, -1));
        // Should be in the last cell of that block
        assert_eq!(cell, CellCoord::new(7, 7, 7));
    }

    #[test]
    fn should_produce_local_coords_in_valid_range() {
        let cell_size = 0.1;
        let grid_dim = 8;

        let test_points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.05, 0.05, 0.05),
            Point3::new(-0.05, -0.05, -0.05),
            Point3::new(100.123, -50.456, 25.789),
        ];

        let eps = 1e-4;
        for point in &test_points {
            let (_, _, local) = decompose_point(*point, cell_size, grid_dim);

            assert!(
                local.u >= -eps && local.u <= 1.0 + eps,
                "Local u out of range for {:?}: {}",
                point,
                local.u
            );
            assert!(
                local.v >= -eps && local.v <= 1.0 + eps,
                "Local v out of range for {:?}: {}",
                point,
                local.v
            );
            assert!(
                local.w >= -eps && local.w <= 1.0 + eps,
                "Local w out of range for {:?}: {}",
                point,
                local.w
            );
        }
    }

    // -------------------------------------------------------------------------
    // Block Boundary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_handle_points_exactly_on_block_boundary() {
        let cell_size = 0.1;
        let grid_dim = 8;
        let block_size = cell_size * grid_dim as f32;

        // Exactly on block boundary
        let (block, cell, local) = decompose_point(
            Point3::new(block_size, 0.0, 0.0),
            cell_size,
            grid_dim,
        );

        // Should be at the start of block (1,0,0)
        assert_eq!(block.x, 1);
        assert_eq!(cell.x, 0);
        assert!(local.u.abs() < 1e-4);
    }

    #[test]
    fn should_handle_points_exactly_on_cell_boundary() {
        let cell_size = 0.1;
        let grid_dim = 8;

        // Exactly on cell boundary within block
        let (block, cell, local) = decompose_point(
            Point3::new(cell_size * 3.0, 0.0, 0.0),
            cell_size,
            grid_dim,
        );

        assert_eq!(block, BlockCoord::new(0, 0, 0));
        assert_eq!(cell.x, 3);
        assert!(local.u.abs() < 1e-4);
    }

    // -------------------------------------------------------------------------
    // Corner Resolution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_resolve_corner_within_same_block() {
        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(3, 3, 3);
        let grid_dim = 8;

        // All corners within same block
        for corner_idx in 0..8 {
            let corner = corner_from_index(corner_idx);
            let (resolved_block, resolved_cell) = resolve_corner(block, cell, corner, grid_dim);

            if corner == (0, 0, 0) {
                assert_eq!(resolved_block, block);
                assert_eq!(resolved_cell, cell);
            } else {
                // Other corners should be adjacent cells in same block
                assert_eq!(resolved_block, block);
            }
        }
    }

    #[test]
    fn should_resolve_corner_crossing_block_boundary() {
        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(7, 7, 7); // Last cell in block
        let grid_dim = 8;

        // Corner (1,1,1) should cross into block (1,1,1)
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, (1, 1, 1), grid_dim);
        assert_eq!(resolved_block, BlockCoord::new(1, 1, 1));
        assert_eq!(resolved_cell, CellCoord::new(0, 0, 0));

        // Corner (1,0,0) should cross only in x
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, (1, 0, 0), grid_dim);
        assert_eq!(resolved_block, BlockCoord::new(1, 0, 0));
        assert_eq!(resolved_cell, CellCoord::new(0, 7, 7));
    }

    #[test]
    fn should_resolve_corner_from_negative_block() {
        let block = BlockCoord::new(-1, -1, -1);
        let cell = CellCoord::new(7, 7, 7);
        let grid_dim = 8;

        // Corner (1,1,1) should cross into block (0,0,0)
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, (1, 1, 1), grid_dim);
        assert_eq!(resolved_block, BlockCoord::new(0, 0, 0));
        assert_eq!(resolved_cell, CellCoord::new(0, 0, 0));
    }

    // -------------------------------------------------------------------------
    // Neighbor Block Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_return_27_neighbor_blocks() {
        let block = BlockCoord::new(5, 10, -3);
        let neighbors = get_neighbor_blocks(block);

        assert_eq!(neighbors.len(), 27);
    }

    #[test]
    fn should_include_center_block_at_index_13() {
        let block = BlockCoord::new(5, 10, -3);
        let neighbors = get_neighbor_blocks(block);

        // Index 13 is the center (1 + 3*1 + 9*1 = 13)
        assert_eq!(neighbors[13], block);
    }

    #[test]
    fn should_include_all_corners_of_3x3x3_neighborhood() {
        let block = BlockCoord::new(0, 0, 0);
        let neighbors = get_neighbor_blocks(block);

        // Check corners
        assert!(neighbors.contains(&BlockCoord::new(-1, -1, -1)));
        assert!(neighbors.contains(&BlockCoord::new(1, -1, -1)));
        assert!(neighbors.contains(&BlockCoord::new(-1, 1, -1)));
        assert!(neighbors.contains(&BlockCoord::new(1, 1, -1)));
        assert!(neighbors.contains(&BlockCoord::new(-1, -1, 1)));
        assert!(neighbors.contains(&BlockCoord::new(1, -1, 1)));
        assert!(neighbors.contains(&BlockCoord::new(-1, 1, 1)));
        assert!(neighbors.contains(&BlockCoord::new(1, 1, 1)));
    }

    // -------------------------------------------------------------------------
    // Cell/Corner Position Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_compute_cell_origin_correctly() {
        let cell_size = 0.1;
        let grid_dim = 8;

        // Origin block, origin cell
        let origin = cell_origin(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            cell_size,
            grid_dim,
        );
        assert!((origin - Point3::new(0.0, 0.0, 0.0)).length() < 1e-6);

        // Cell (2, 3, 4) in block (0, 0, 0)
        let pos = cell_origin(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(2, 3, 4),
            cell_size,
            grid_dim,
        );
        assert!((pos - Point3::new(0.2, 0.3, 0.4)).length() < 1e-6);

        // Block (1, 0, 0), cell (0, 0, 0)
        let pos = cell_origin(
            BlockCoord::new(1, 0, 0),
            CellCoord::new(0, 0, 0),
            cell_size,
            grid_dim,
        );
        assert!((pos - Point3::new(0.8, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn should_compute_corner_position_correctly() {
        let cell_size = 0.1;
        let grid_dim = 8;
        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        // Corner (0,0,0) should be at origin
        let pos = corner_position(block, cell, (0, 0, 0), cell_size, grid_dim);
        assert!((pos - Point3::new(0.0, 0.0, 0.0)).length() < 1e-6);

        // Corner (1,1,1) should be offset by cell_size in each direction
        let pos = corner_position(block, cell, (1, 1, 1), cell_size, grid_dim);
        assert!((pos - Point3::new(0.1, 0.1, 0.1)).length() < 1e-6);
    }
}

// =============================================================================
// SECTION 7: Trilinear Interpolation Weight Tests
// =============================================================================

mod trilinear_weight_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Weight Sum Invariant
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_weights_sum_to_one_at_origin() {
        let weights = compute_trilinear_weights(LocalCoord::new(0.0, 0.0, 0.0));
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum = {}", sum);
    }

    #[test]
    fn should_have_weights_sum_to_one_at_corner_111() {
        let weights = compute_trilinear_weights(LocalCoord::new(1.0, 1.0, 1.0));
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum = {}", sum);
    }

    #[test]
    fn should_have_weights_sum_to_one_at_center() {
        let weights = compute_trilinear_weights(LocalCoord::new(0.5, 0.5, 0.5));
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum = {}", sum);
    }

    #[test]
    fn should_have_weights_sum_to_one_for_random_positions() {
        let test_positions = [
            (0.1, 0.2, 0.3),
            (0.9, 0.1, 0.5),
            (0.333, 0.666, 0.999),
            (0.001, 0.999, 0.5),
        ];

        for (u, v, w) in test_positions {
            let weights = compute_trilinear_weights(LocalCoord::new(u, v, w));
            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Sum = {} for ({}, {}, {})",
                sum,
                u,
                v,
                w
            );
        }
    }

    // -------------------------------------------------------------------------
    // Weight Non-Negativity Invariant
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_all_weights_non_negative() {
        let test_positions = [
            LocalCoord::new(0.0, 0.0, 0.0),
            LocalCoord::new(1.0, 1.0, 1.0),
            LocalCoord::new(0.5, 0.5, 0.5),
            LocalCoord::new(0.1, 0.9, 0.3),
            LocalCoord::new(0.0, 0.5, 1.0),
        ];

        for local in test_positions {
            let weights = compute_trilinear_weights(local);
            for (i, &w) in weights.iter().enumerate() {
                assert!(w >= 0.0, "Weight {} is negative: {} for {:?}", i, w, local);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Weight Bound Invariant
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_all_weights_at_most_one() {
        let test_positions = [
            LocalCoord::new(0.0, 0.0, 0.0),
            LocalCoord::new(1.0, 1.0, 1.0),
            LocalCoord::new(0.5, 0.5, 0.5),
            LocalCoord::new(0.1, 0.9, 0.3),
        ];

        for local in test_positions {
            let weights = compute_trilinear_weights(local);
            for (i, &w) in weights.iter().enumerate() {
                assert!(w <= 1.0, "Weight {} exceeds 1: {} for {:?}", i, w, local);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Corner Weight Properties
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_single_weight_one_at_corner_000() {
        let weights = compute_trilinear_weights(LocalCoord::new(0.0, 0.0, 0.0));
        assert!((weights[0] - 1.0).abs() < 1e-6);
        for i in 1..8 {
            assert!(weights[i].abs() < 1e-6, "Weight {} should be 0", i);
        }
    }

    #[test]
    fn should_have_single_weight_one_at_corner_100() {
        let weights = compute_trilinear_weights(LocalCoord::new(1.0, 0.0, 0.0));
        assert!((weights[1] - 1.0).abs() < 1e-6);
        for i in [0, 2, 3, 4, 5, 6, 7] {
            assert!(weights[i].abs() < 1e-6, "Weight {} should be 0", i);
        }
    }

    #[test]
    fn should_have_single_weight_one_at_corner_110() {
        let weights = compute_trilinear_weights(LocalCoord::new(1.0, 1.0, 0.0));
        assert!((weights[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_have_single_weight_one_at_corner_010() {
        let weights = compute_trilinear_weights(LocalCoord::new(0.0, 1.0, 0.0));
        assert!((weights[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_have_single_weight_one_at_corner_001() {
        let weights = compute_trilinear_weights(LocalCoord::new(0.0, 0.0, 1.0));
        assert!((weights[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_have_single_weight_one_at_corner_101() {
        let weights = compute_trilinear_weights(LocalCoord::new(1.0, 0.0, 1.0));
        assert!((weights[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_have_single_weight_one_at_corner_111() {
        let weights = compute_trilinear_weights(LocalCoord::new(1.0, 1.0, 1.0));
        assert!((weights[6] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_have_single_weight_one_at_corner_011() {
        let weights = compute_trilinear_weights(LocalCoord::new(0.0, 1.0, 1.0));
        assert!((weights[7] - 1.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Midpoint Properties
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_equal_weights_at_center() {
        let weights = compute_trilinear_weights(LocalCoord::new(0.5, 0.5, 0.5));
        for &w in &weights {
            assert!((w - 0.125).abs() < 1e-6, "Weight should be 0.125, got {}", w);
        }
    }

    #[test]
    fn should_have_four_equal_weights_on_face_center() {
        // Center of z=0 face
        let weights = compute_trilinear_weights(LocalCoord::new(0.5, 0.5, 0.0));

        // Corners 0,1,2,3 should have weight 0.25
        for i in 0..4 {
            assert!(
                (weights[i] - 0.25).abs() < 1e-6,
                "Weight {} should be 0.25, got {}",
                i,
                weights[i]
            );
        }
        // Corners 4,5,6,7 should have weight 0
        for i in 4..8 {
            assert!(
                weights[i].abs() < 1e-6,
                "Weight {} should be 0, got {}",
                i,
                weights[i]
            );
        }
    }

    #[test]
    fn should_have_two_equal_weights_on_edge_center() {
        // Center of edge along x at (0.5, 0, 0)
        let weights = compute_trilinear_weights(LocalCoord::new(0.5, 0.0, 0.0));

        assert!((weights[0] - 0.5).abs() < 1e-6);
        assert!((weights[1] - 0.5).abs() < 1e-6);
        for i in 2..8 {
            assert!(weights[i].abs() < 1e-6);
        }
    }
}

// =============================================================================
// SECTION 8: Trilinear Interpolation Value Tests
// =============================================================================

mod trilinear_interpolation_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Linear SDF Interpolation
    // -------------------------------------------------------------------------

    #[test]
    fn should_interpolate_linear_sdf_at_cell_corner() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.0, 0.0, 0.0);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        // The mock returns f(x,y,z) = x + y + z in world units
        // At corner (0,0,0) of cell (2,3,4), block_size=0.8:
        // position is (0.2, 0.3, 0.4), so value = 0.2 + 0.3 + 0.4 = 0.9
        assert!((result.value() - 0.9).abs() < 1e-4, "Value = {}", result.value());
    }

    #[test]
    fn should_interpolate_linear_sdf_at_cell_center() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        // At center (0.5, 0.5, 0.5) of cell (2,3,4):
        // The 8 corners are at cells (2,3,4) through (3,4,5)
        // World positions: (0.2,0.3,0.4) through (0.3,0.4,0.5)
        // Center interpolated value = (0.2+0.3)/2 + (0.3+0.4)/2 + (0.4+0.5)/2
        //                           = 0.25 + 0.35 + 0.45 = 1.05
        assert!(
            (result.value() - 1.05).abs() < 1e-4,
            "Value = {}",
            result.value()
        );
    }

    // -------------------------------------------------------------------------
    // Sphere SDF Properties
    // -------------------------------------------------------------------------

    #[test]
    fn should_return_negative_value_inside_sphere() {
        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        // Query at center
        let (block, cell, local) = decompose_point(provider.center, provider.cell_size, provider.grid_dim);
        let result = trilinear_interpolate_sdf(&provider, block, cell, local);

        if let Some(r) = result {
            assert!(r.value() < 0.0, "Center should be inside sphere, got {}", r.value());
        }
    }

    #[test]
    fn should_return_positive_value_outside_sphere() {
        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        let outside = Point3::new(2.0, 2.0, 2.0);
        let (block, cell, local) = decompose_point(outside, provider.cell_size, provider.grid_dim);
        let result = trilinear_interpolate_sdf(&provider, block, cell, local);

        if let Some(r) = result {
            assert!(r.value() > 0.0, "Point should be outside sphere, got {}", r.value());
        }
    }

    // -------------------------------------------------------------------------
    // Missing/Untrained Region Handling
    // -------------------------------------------------------------------------

    #[test]
    fn should_return_none_for_missing_block() {
        let provider = SparseProvider {
            grid_dim: 8,
            cell_size: 0.1,
            existing_blocks: vec![BlockCoord::new(0, 0, 0)],
        };

        // Query in non-existing block
        let block = BlockCoord::new(10, 10, 10);
        let cell = CellCoord::new(0, 0, 0);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local);
        assert!(result.is_none());
    }

    #[test]
    fn should_return_none_for_untrained_region() {
        let provider = UntrainedRegionProvider {
            grid_dim: 8,
            cell_size: 0.1,
            untrained_blocks: vec![BlockCoord::new(0, 0, 0)],
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local);
        assert!(result.is_none());
    }

    // -------------------------------------------------------------------------
    // Weight Correctness in Results
    // -------------------------------------------------------------------------

    #[test]
    fn should_include_correct_weights_in_result() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);
        let local = LocalCoord::new(0.25, 0.25, 0.25);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        // Verify weights match compute_trilinear_weights
        let expected_weights = compute_trilinear_weights(local);
        for i in 0..8 {
            assert!(
                (result.weights[i] - expected_weights[i]).abs() < 1e-6,
                "Weight {} mismatch",
                i
            );
        }
    }

    #[test]
    fn should_include_correct_coordinates_in_result() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(1, 2, 3);
        let cell = CellCoord::new(4, 5, 6);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        assert_eq!(result.block, block);
        assert_eq!(result.cell, cell);
    }
}

// =============================================================================
// SECTION 9: Gradient Tests
// =============================================================================

mod gradient_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Linear SDF Gradient (Known Analytical Solution)
    // -------------------------------------------------------------------------

    #[test]
    fn should_compute_gradient_for_linear_sdf() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let grad = trilinear_gradient_sdf(&provider, block, cell, local).unwrap();

        // For f(x,y,z) = x + y + z in world units, where x = cell_x * cell_size + local_u * cell_size,
        // the gradient in local coordinates is:
        // df/d(local_u) = d(cell_size * local_u)/d(local_u) = cell_size = 0.1
        //
        // This is the expected behavior: gradient is in "local units" which maps
        // to cell_size in world units.
        let expected = provider.cell_size;
        assert!((grad[0] - expected).abs() < 1e-4, "df/dx = {}", grad[0]);
        assert!((grad[1] - expected).abs() < 1e-4, "df/dy = {}", grad[1]);
        assert!((grad[2] - expected).abs() < 1e-4, "df/dz = {}", grad[2]);
    }

    #[test]
    fn should_compute_same_gradient_at_any_position_for_linear_sdf() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let test_positions = [
            LocalCoord::new(0.0, 0.0, 0.0),
            LocalCoord::new(1.0, 1.0, 1.0),
            LocalCoord::new(0.1, 0.9, 0.5),
            LocalCoord::new(0.5, 0.5, 0.5),
        ];

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        // For linear SDF, gradient should be constant everywhere (cell_size)
        let expected = provider.cell_size;

        for local in test_positions {
            let grad = trilinear_gradient_sdf(&provider, block, cell, local).unwrap();

            assert!(
                (grad[0] - expected).abs() < 1e-4,
                "df/dx = {} at {:?}",
                grad[0],
                local
            );
            assert!(
                (grad[1] - expected).abs() < 1e-4,
                "df/dy = {} at {:?}",
                grad[1],
                local
            );
            assert!(
                (grad[2] - expected).abs() < 1e-4,
                "df/dz = {} at {:?}",
                grad[2],
                local
            );
        }
    }

    // -------------------------------------------------------------------------
    // Numerical Gradient Verification
    // -------------------------------------------------------------------------

    #[test]
    fn should_match_numerical_gradient_for_sphere_sdf() {
        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.2,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(3, 3, 3);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        // Analytical gradient
        let analytical = trilinear_gradient_sdf(&provider, block, cell, local).unwrap();

        // Numerical gradient via finite differences
        let eps = 1e-4;
        let mut numerical = [0.0f32; 3];

        for axis in 0..3 {
            let mut local_plus = local;
            let mut local_minus = local;

            match axis {
                0 => {
                    local_plus.u += eps;
                    local_minus.u -= eps;
                }
                1 => {
                    local_plus.v += eps;
                    local_minus.v -= eps;
                }
                _ => {
                    local_plus.w += eps;
                    local_minus.w -= eps;
                }
            }

            let f_plus = trilinear_interpolate_sdf(&provider, block, cell, local_plus)
                .unwrap()
                .value();
            let f_minus = trilinear_interpolate_sdf(&provider, block, cell, local_minus)
                .unwrap()
                .value();
            numerical[axis] = (f_plus - f_minus) / (2.0 * eps);
        }

        for i in 0..3 {
            assert!(
                (analytical[i] - numerical[i]).abs() < 1e-2,
                "Gradient mismatch on axis {}: analytical={}, numerical={}",
                i,
                analytical[i],
                numerical[i]
            );
        }
    }

    // -------------------------------------------------------------------------
    // Combined Value and Gradient
    // -------------------------------------------------------------------------

    #[test]
    fn should_produce_same_result_as_separate_calls() {
        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.2,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(3, 3, 3);
        let local = LocalCoord::new(0.3, 0.6, 0.2);

        let (combined_result, combined_grad) =
            trilinear_with_gradient_sdf(&provider, block, cell, local).unwrap();
        let separate_result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();
        let separate_grad = trilinear_gradient_sdf(&provider, block, cell, local).unwrap();

        assert!(
            (combined_result.value() - separate_result.value()).abs() < 1e-6,
            "Value mismatch"
        );

        for i in 0..3 {
            assert!(
                (combined_grad[i] - separate_grad[i]).abs() < 1e-6,
                "Gradient mismatch on axis {}",
                i
            );
        }
    }

    // -------------------------------------------------------------------------
    // Sphere Gradient Direction
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_gradient_pointing_outward_at_surface() {
        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.05,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        // Query at a point on the surface in +x direction
        let surface_point = Point3::new(0.4 + 0.3, 0.4, 0.4);
        let (block, cell, local) = decompose_point(surface_point, provider.cell_size, provider.grid_dim);

        if let Some(grad) = trilinear_gradient_sdf(&provider, block, cell, local) {
            // Gradient should point in +x direction (outward)
            let grad_vec = Point3::new(grad[0], grad[1], grad[2]).normalize();

            assert!(
                grad_vec.x > 0.5,
                "Gradient should point outward (+x), got {:?}",
                grad_vec
            );
        }
    }
}

// =============================================================================
// SECTION 10: Backward Pass (Gradient Accumulation) Tests
// =============================================================================

mod backward_tests {
    use super::*;

    #[test]
    fn should_accumulate_gradients_to_all_eight_corners() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        let mut accumulator = TestGradientAccumulator::new();
        trilinear_backward(&mut accumulator, &result, [1.0]);

        assert_eq!(accumulator.accumulated.len(), 8);
    }

    #[test]
    fn should_distribute_gradient_according_to_weights() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        let mut accumulator = TestGradientAccumulator::new();
        let upstream_grad = 2.0;
        trilinear_backward(&mut accumulator, &result, [upstream_grad]);

        // At center, each weight is 0.125, so each gradient should be 0.125 * 2.0 = 0.25
        for (_, _, _, grad) in &accumulator.accumulated {
            assert!(
                (*grad - 0.25).abs() < 1e-6,
                "Expected gradient 0.25, got {}",
                grad
            );
        }
    }

    #[test]
    fn should_have_gradients_sum_to_upstream_gradient() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);
        let local = LocalCoord::new(0.3, 0.7, 0.2);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        let mut accumulator = TestGradientAccumulator::new();
        let upstream_grad = 3.5;
        trilinear_backward(&mut accumulator, &result, [upstream_grad]);

        let total = accumulator.total_gradient();
        assert!(
            (total - upstream_grad).abs() < 1e-5,
            "Total gradient {} should equal upstream {}",
            total,
            upstream_grad
        );
    }

    #[test]
    fn should_accumulate_to_corner_000_when_at_origin() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);
        let local = LocalCoord::new(0.0, 0.0, 0.0);

        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        let mut accumulator = TestGradientAccumulator::new();
        trilinear_backward(&mut accumulator, &result, [1.0]);

        // Only corner (0,0,0) should have gradient = 1.0
        let corner_000_grad = accumulator
            .accumulated
            .iter()
            .find(|(_, _, c, _)| *c == (0, 0, 0))
            .map(|(_, _, _, g)| *g);

        assert!(
            (corner_000_grad.unwrap() - 1.0).abs() < 1e-6,
            "Corner (0,0,0) should have gradient 1.0"
        );

        // All other corners should have gradient 0
        for (_, _, corner, grad) in &accumulator.accumulated {
            if *corner != (0, 0, 0) {
                assert!(
                    grad.abs() < 1e-6,
                    "Corner {:?} should have gradient 0, got {}",
                    corner,
                    grad
                );
            }
        }
    }
}

// =============================================================================
// SECTION 11: Hessian Tests
// =============================================================================

mod hessian_tests {
    use super::*;

    #[test]
    fn should_have_zero_mixed_partials_for_linear_sdf() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let hessian = trilinear_hessian_mixed(&provider, block, cell, local).unwrap();

        // For linear SDF f = x + y + z, all second derivatives are zero
        for i in 0..3 {
            assert!(
                hessian[i].abs() < 1e-5,
                "Mixed partial {} should be zero, got {}",
                i,
                hessian[i]
            );
        }
    }

    #[test]
    fn should_return_none_for_missing_data() {
        let provider = SparseProvider {
            grid_dim: 8,
            cell_size: 0.1,
            existing_blocks: vec![],
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_hessian_mixed(&provider, block, cell, local);
        assert!(result.is_none());
    }
}

// =============================================================================
// SECTION 12: Hash Function Tests
// =============================================================================

mod hash_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // FNV-1a Determinism
    // -------------------------------------------------------------------------

    #[test]
    fn should_produce_same_hash_for_same_input_64bit() {
        let coord = BlockCoord::new(42, -17, 1000);
        let h1 = fnv1a_64(coord);
        let h2 = fnv1a_64(coord);
        assert_eq!(h1, h2);
    }

    #[test]
    fn should_produce_same_hash_for_same_input_32bit() {
        let coord = BlockCoord::new(42, -17, 1000);
        let h1 = fnv1a_32(coord);
        let h2 = fnv1a_32(coord);
        assert_eq!(h1, h2);
    }

    // -------------------------------------------------------------------------
    // FNV-1a Differentiation
    // -------------------------------------------------------------------------

    #[test]
    fn should_produce_different_hashes_for_different_coords() {
        let coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 0, 0),
            BlockCoord::new(0, 1, 0),
            BlockCoord::new(0, 0, 1),
            BlockCoord::new(-1, 0, 0),
            BlockCoord::new(0, -1, 0),
        ];

        let hashes: Vec<u64> = coords.iter().map(|c| fnv1a_64(*c)).collect();

        // All should be unique
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(
                    hashes[i], hashes[j],
                    "Hash collision: {:?} and {:?}",
                    coords[i], coords[j]
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Hash Distribution
    // -------------------------------------------------------------------------

    #[test]
    fn should_distribute_across_buckets_reasonably() {
        let capacity = 256usize;
        let mut buckets = vec![0u32; capacity];

        // Hash many coordinates
        for x in -10..10 {
            for y in -10..10 {
                for z in -10..10 {
                    let coord = BlockCoord::new(x, y, z);
                    let idx = hash_to_index(coord, capacity);
                    buckets[idx] += 1;
                }
            }
        }

        // Check for reasonable distribution
        let total = 20 * 20 * 20; // 8000 coords
        let expected = total / capacity as u32;
        let max = *buckets.iter().max().unwrap();

        // Should not have any bucket more than 10x expected (very loose)
        assert!(
            max < expected * 10,
            "Poor distribution: max={}, expected={}",
            max,
            expected
        );
    }

    // -------------------------------------------------------------------------
    // Hash to Index Range
    // -------------------------------------------------------------------------

    #[test]
    fn should_always_return_index_in_valid_range() {
        let capacities = [1, 2, 7, 16, 100, 1024, 65536];

        for capacity in capacities {
            for x in -5..5 {
                for y in -5..5 {
                    for z in -5..5 {
                        let coord = BlockCoord::new(x, y, z);
                        let idx = hash_to_index(coord, capacity);
                        assert!(
                            idx < capacity,
                            "Index {} out of range for capacity {}",
                            idx,
                            capacity
                        );
                    }
                }
            }
        }
    }
}

// =============================================================================
// SECTION 13: Morton Encoding Tests
// =============================================================================

mod morton_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Bit Spread/Compact Roundtrip
    // -------------------------------------------------------------------------

    #[test]
    fn should_roundtrip_spread_compact_bits() {
        let test_values = [0u32, 1, 100, 1000, 0x1FFFFF];

        for x in test_values {
            let spread = spread_bits_3d(x);
            let back = compact_bits_3d(spread);
            assert_eq!(back, x & 0x1FFFFF, "Roundtrip failed for {}", x);
        }
    }

    #[test]
    fn should_mask_to_21_bits() {
        // Values larger than 21 bits should be masked
        let large = 0xFFFFFFFF_u32;
        let spread = spread_bits_3d(large);
        let back = compact_bits_3d(spread);
        assert_eq!(back, 0x1FFFFF); // Only lower 21 bits
    }

    // -------------------------------------------------------------------------
    // Morton Encode/Decode Unsigned Roundtrip
    // -------------------------------------------------------------------------

    #[test]
    fn should_roundtrip_morton_encode_decode_unsigned() {
        let test_cases = [
            (0u32, 0u32, 0u32),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
            (100, 200, 300),
            (0x1FFFFF, 0x1FFFFF, 0x1FFFFF), // Max 21-bit values
        ];

        for (x, y, z) in test_cases {
            let code = morton_encode_3d(x, y, z);
            let (dx, dy, dz) = morton_decode_3d(code);
            assert_eq!(
                (dx, dy, dz),
                (x, y, z),
                "Roundtrip failed for ({}, {}, {})",
                x,
                y,
                z
            );
        }
    }

    // -------------------------------------------------------------------------
    // Morton Encode/Decode Signed Roundtrip
    // -------------------------------------------------------------------------

    #[test]
    fn should_roundtrip_morton_encode_decode_signed() {
        let test_coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 2, 3),
            BlockCoord::new(-1, -2, -3),
            BlockCoord::new(1000, -500, 250),
            BlockCoord::new(-1048576, 0, 1048575), // Near limits
            BlockCoord::new(0, -1048576, 1048575),
        ];

        for coord in test_coords {
            let code = morton_encode_signed(coord);
            let back = morton_decode_signed(code);
            assert_eq!(back, coord, "Roundtrip failed for {:?}", coord);
        }
    }

    // -------------------------------------------------------------------------
    // Morton Code Ordering Properties
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_distinct_codes_for_distinct_coords() {
        let coords = [
            (0u32, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
        ];

        let codes: Vec<u64> = coords.iter().map(|(x, y, z)| morton_encode_3d(*x, *y, *z)).collect();

        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(
                    codes[i], codes[j],
                    "Morton collision: {:?} and {:?}",
                    coords[i], coords[j]
                );
            }
        }
    }

    #[test]
    fn should_have_origin_at_code_zero() {
        let code = morton_encode_3d(0, 0, 0);
        assert_eq!(code, 0);
    }

    #[test]
    fn should_interleave_bits_correctly() {
        // For (1, 0, 0): bit pattern should be ...001 (x bit in position 0)
        assert_eq!(morton_encode_3d(1, 0, 0), 0b001);

        // For (0, 1, 0): bit pattern should be ...010 (y bit in position 1)
        assert_eq!(morton_encode_3d(0, 1, 0), 0b010);

        // For (0, 0, 1): bit pattern should be ...100 (z bit in position 2)
        assert_eq!(morton_encode_3d(0, 0, 1), 0b100);

        // For (1, 1, 1): bit pattern should be ...111
        assert_eq!(morton_encode_3d(1, 1, 1), 0b111);
    }
}

// =============================================================================
// SECTION 14: Marching Cubes Table Tests
// =============================================================================

mod marching_cubes_table_tests {
    use super::*;
    use ash_core::marching_cubes::{CORNER_OFFSETS, EDGE_TABLE, EDGE_VERTICES, TRI_TABLE};

    // -------------------------------------------------------------------------
    // Table Size Invariants
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_256_edge_table_entries() {
        assert_eq!(EDGE_TABLE.len(), 256);
    }

    #[test]
    fn should_have_256_triangle_table_entries() {
        assert_eq!(TRI_TABLE.len(), 256);
    }

    #[test]
    fn should_have_16_elements_per_triangle_entry() {
        for entry in &TRI_TABLE {
            assert_eq!(entry.len(), 16);
        }
    }

    #[test]
    fn should_have_12_edge_vertex_pairs() {
        assert_eq!(EDGE_VERTICES.len(), 12);
    }

    #[test]
    fn should_have_8_corner_offsets() {
        assert_eq!(CORNER_OFFSETS.len(), 8);
    }

    // -------------------------------------------------------------------------
    // Edge Table Symmetry
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_no_edges_for_config_0_all_outside() {
        assert_eq!(EDGE_TABLE[0], 0);
    }

    #[test]
    fn should_have_no_edges_for_config_255_all_inside() {
        assert_eq!(EDGE_TABLE[255], 0);
    }

    #[test]
    fn should_have_symmetric_edge_masks_for_complement_configs() {
        for i in 0..128 {
            assert_eq!(
                EDGE_TABLE[i], EDGE_TABLE[255 - i],
                "Asymmetry at configs {} and {}",
                i,
                255 - i
            );
        }
    }

    // -------------------------------------------------------------------------
    // Triangle Table Termination
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_properly_terminated_triangle_entries() {
        for (i, entry) in TRI_TABLE.iter().enumerate() {
            let mut found_terminator = false;
            for &val in entry {
                if val == -1 {
                    found_terminator = true;
                } else if found_terminator {
                    panic!(
                        "Entry {} has values after -1 terminator at position",
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn should_have_no_triangles_for_config_0() {
        assert_eq!(TRI_TABLE[0][0], -1);
    }

    #[test]
    fn should_have_no_triangles_for_config_255() {
        assert_eq!(TRI_TABLE[255][0], -1);
    }

    // -------------------------------------------------------------------------
    // Single Corner Configurations
    // -------------------------------------------------------------------------

    #[test]
    fn should_produce_one_triangle_for_single_corner_inside() {
        // Config 1: only corner 0 inside
        // Config 2: only corner 1 inside
        // etc.
        let single_corner_configs = [1, 2, 4, 8, 16, 32, 64, 128];

        for config in single_corner_configs {
            let mut edge_count = 0;
            for &val in &TRI_TABLE[config] {
                if val != -1 {
                    edge_count += 1;
                }
            }
            assert_eq!(
                edge_count, 3,
                "Config {} (single corner) should produce 1 triangle (3 edges), got {}",
                config,
                edge_count / 3
            );
        }
    }

    // -------------------------------------------------------------------------
    // Edge Vertex Validity
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_valid_edge_endpoints() {
        for (edge_idx, (v0, v1)) in EDGE_VERTICES.iter().enumerate() {
            assert!(*v0 < 8, "Edge {} has invalid v0: {}", edge_idx, v0);
            assert!(*v1 < 8, "Edge {} has invalid v1: {}", edge_idx, v1);
            assert_ne!(v0, v1, "Edge {} has same endpoints", edge_idx);
        }
    }

    #[test]
    fn should_have_adjacent_corner_endpoints() {
        // Each edge should connect two corners that differ by exactly one coordinate
        for (edge_idx, (v0, v1)) in EDGE_VERTICES.iter().enumerate() {
            let c0 = CORNER_OFFSETS[*v0];
            let c1 = CORNER_OFFSETS[*v1];

            let dx = (c0.0 as i32 - c1.0 as i32).abs();
            let dy = (c0.1 as i32 - c1.1 as i32).abs();
            let dz = (c0.2 as i32 - c1.2 as i32).abs();

            assert_eq!(
                dx + dy + dz,
                1,
                "Edge {} connects non-adjacent corners: {:?} and {:?}",
                edge_idx,
                c0,
                c1
            );
        }
    }

    // -------------------------------------------------------------------------
    // Corner Offset Validity
    // -------------------------------------------------------------------------

    #[test]
    fn should_have_valid_corner_offsets() {
        for (corner_idx, (dx, dy, dz)) in CORNER_OFFSETS.iter().enumerate() {
            assert!(
                *dx <= 1 && *dy <= 1 && *dz <= 1,
                "Corner {} has invalid offset: ({}, {}, {})",
                corner_idx,
                dx,
                dy,
                dz
            );
        }
    }

    #[test]
    fn should_have_corner_offsets_matching_trait_function() {
        for i in 0..8 {
            let from_table = CORNER_OFFSETS[i];
            let from_trait = corner_from_index(i);
            assert_eq!(
                from_table, from_trait,
                "Corner {} offset mismatch",
                i
            );
        }
    }

    // -------------------------------------------------------------------------
    // Triangle Table Edge Reference Validity
    // -------------------------------------------------------------------------

    #[test]
    fn should_only_reference_valid_edges_in_triangle_table() {
        for (config_idx, entry) in TRI_TABLE.iter().enumerate() {
            for &val in entry {
                if val != -1 {
                    assert!(
                        val >= 0 && val < 12,
                        "Config {} references invalid edge {}",
                        config_idx,
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn should_only_reference_active_edges_in_triangle_table() {
        for config_idx in 0..256 {
            let edge_flags = EDGE_TABLE[config_idx];
            let entry = &TRI_TABLE[config_idx];

            for &val in entry {
                if val != -1 {
                    let edge_idx = val as usize;
                    let edge_active = (edge_flags & (1 << edge_idx)) != 0;
                    assert!(
                        edge_active,
                        "Config {} references inactive edge {}",
                        config_idx,
                        edge_idx
                    );
                }
            }
        }
    }
}

// =============================================================================
// SECTION 15: Marching Cubes Algorithm Tests
// =============================================================================

mod marching_cubes_algorithm_tests {
    use super::*;
    use ash_core::marching_cubes::{interpolate_vertex, process_cell_no_alloc};

    // -------------------------------------------------------------------------
    // Vertex Interpolation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_interpolate_to_midpoint_for_opposite_signs() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        let result = interpolate_vertex(p0, p1, -1.0, 1.0, 0.0);
        assert!((result.x - 0.5).abs() < 1e-6);
        assert!(result.y.abs() < 1e-6);
        assert!(result.z.abs() < 1e-6);
    }

    #[test]
    fn should_interpolate_to_first_point_when_value_equals_iso() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        let result = interpolate_vertex(p0, p1, 0.0, 1.0, 0.0);
        assert!((result.x - 0.0).abs() < 1e-6);
    }

    #[test]
    fn should_interpolate_to_second_point_when_value_equals_iso() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        let result = interpolate_vertex(p0, p1, -1.0, 0.0, 0.0);
        assert!((result.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_clamp_interpolation_to_edge() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        // Iso value outside the value range - should clamp
        let result = interpolate_vertex(p0, p1, 0.5, 1.0, 0.0);
        // t = (0 - 0.5) / (1.0 - 0.5) = -1.0, clamped to 0
        assert!((result.x - 0.0).abs() < 1e-6);
    }

    #[test]
    fn should_handle_equal_values_gracefully() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        // Both values equal - degenerate case
        let result = interpolate_vertex(p0, p1, 1.0, 1.0, 0.0);
        // Should return midpoint to avoid division by zero
        assert!((result.x - 0.5).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Process Cell Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_return_zero_triangles_for_uniform_positive_values() {
        let provider = ConstantProvider {
            grid_dim: 8,
            cell_size: 0.1,
            values: [1.0; 8], // All positive (outside)
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        let (_, count) = process_cell_no_alloc(&provider, block, cell, 0.0);
        assert_eq!(count, 0, "All outside should produce no triangles");
    }

    #[test]
    fn should_return_zero_triangles_for_uniform_negative_values() {
        let provider = ConstantProvider {
            grid_dim: 8,
            cell_size: 0.1,
            values: [-1.0; 8], // All negative (inside)
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        let (_, count) = process_cell_no_alloc(&provider, block, cell, 0.0);
        assert_eq!(count, 0, "All inside should produce no triangles");
    }

    #[test]
    fn should_return_one_triangle_for_single_corner_inside() {
        let mut values = [1.0f32; 8];
        values[0] = -1.0; // Only corner 0 inside

        let provider = ConstantProvider {
            grid_dim: 8,
            cell_size: 0.1,
            values,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        let (_, count) = process_cell_no_alloc(&provider, block, cell, 0.0);
        assert_eq!(count, 1, "Single corner inside should produce 1 triangle");
    }

    #[test]
    fn should_return_at_most_five_triangles() {
        // Create a complex configuration
        let values = [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0];

        let provider = ConstantProvider {
            grid_dim: 8,
            cell_size: 0.1,
            values,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        let (_, count) = process_cell_no_alloc(&provider, block, cell, 0.0);
        assert!(count <= 5, "Should produce at most 5 triangles, got {}", count);
    }

    #[test]
    fn should_produce_triangles_with_finite_vertices() {
        let values = [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let provider = ConstantProvider {
            grid_dim: 8,
            cell_size: 0.1,
            values,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        let (triangles, count) = process_cell_no_alloc(&provider, block, cell, 0.0);

        for i in 0..count {
            for vertex in &triangles[i] {
                assert!(vertex.x.is_finite(), "Non-finite x coordinate");
                assert!(vertex.y.is_finite(), "Non-finite y coordinate");
                assert!(vertex.z.is_finite(), "Non-finite z coordinate");
            }
        }
    }

    #[test]
    fn should_return_empty_for_missing_corner_data() {
        let provider = SparseProvider {
            grid_dim: 8,
            cell_size: 0.1,
            existing_blocks: vec![], // No blocks exist
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        let (_, count) = process_cell_no_alloc(&provider, block, cell, 0.0);
        assert_eq!(count, 0, "Missing data should produce no triangles");
    }

    #[test]
    fn should_return_empty_for_untrained_regions() {
        let provider = UntrainedRegionProvider {
            grid_dim: 8,
            cell_size: 0.1,
            untrained_blocks: vec![BlockCoord::new(0, 0, 0)],
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        let (_, count) = process_cell_no_alloc(&provider, block, cell, 0.0);
        assert_eq!(count, 0, "Untrained region should produce no triangles");
    }

    // -------------------------------------------------------------------------
    // Alloc vs No-Alloc Equivalence
    // -------------------------------------------------------------------------

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn should_produce_identical_results_with_and_without_alloc() {
        use ash_core::marching_cubes::process_cell;

        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        // Test multiple cells
        for cx in 0..8 {
            for cy in 0..8 {
                for cz in 0..8 {
                    let block = BlockCoord::new(0, 0, 0);
                    let cell = CellCoord::new(cx, cy, cz);

                    let alloc_result = process_cell(&provider, block, cell, 0.0);
                    let (no_alloc_result, no_alloc_count) =
                        process_cell_no_alloc(&provider, block, cell, 0.0);

                    assert_eq!(
                        alloc_result.len(),
                        no_alloc_count,
                        "Count mismatch at cell ({}, {}, {})",
                        cx,
                        cy,
                        cz
                    );

                    for i in 0..no_alloc_count {
                        for j in 0..3 {
                            let a = alloc_result[i][j];
                            let b = no_alloc_result[i][j];
                            assert!(
                                (a.x - b.x).abs() < 1e-6,
                                "X mismatch at cell ({}, {}, {}), tri {}, vert {}",
                                cx,
                                cy,
                                cz,
                                i,
                                j
                            );
                            assert!(
                                (a.y - b.y).abs() < 1e-6,
                                "Y mismatch at cell ({}, {}, {}), tri {}, vert {}",
                                cx,
                                cy,
                                cz,
                                i,
                                j
                            );
                            assert!(
                                (a.z - b.z).abs() < 1e-6,
                                "Z mismatch at cell ({}, {}, {}), tri {}, vert {}",
                                cx,
                                cy,
                                cz,
                                i,
                                j
                            );
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Sphere Mesh Generation
    // -------------------------------------------------------------------------

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn should_generate_triangles_for_sphere() {
        use ash_core::marching_cubes::process_cell;

        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.3,
        };

        let mut total_triangles = 0;

        // Scan cells around the sphere
        for bz in 0..=1 {
            for by in 0..=1 {
                for bx in 0..=1 {
                    let block = BlockCoord::new(bx, by, bz);

                    for cz in 0..8 {
                        for cy in 0..8 {
                            for cx in 0..8 {
                                let cell = CellCoord::new(cx, cy, cz);
                                let triangles = process_cell(&provider, block, cell, 0.0);
                                total_triangles += triangles.len();
                            }
                        }
                    }
                }
            }
        }

        assert!(
            total_triangles > 0,
            "Should generate some triangles for sphere"
        );
        assert!(
            total_triangles < 10000,
            "Should not generate excessive triangles: {}",
            total_triangles
        );
    }
}

// =============================================================================
// SECTION 16: Error Type Tests
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn should_have_display_impl() {
        let err = AshCoreError::MissingCornerValue { corner_index: 3 };
        let msg = format!("{}", err);
        assert!(msg.contains("corner"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn should_have_debug_impl() {
        let err = AshCoreError::UntrainedRegion;
        let debug = format!("{:?}", err);
        assert!(debug.contains("UntrainedRegion"));
    }

    #[test]
    fn should_implement_equality() {
        let err1 = AshCoreError::CellOutOfBounds { coord: 10, max: 7 };
        let err2 = AshCoreError::CellOutOfBounds { coord: 10, max: 7 };
        let err3 = AshCoreError::CellOutOfBounds { coord: 11, max: 7 };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn should_implement_copy() {
        let err = AshCoreError::UntrainedRegion;
        let copy = err;
        assert_eq!(err, copy);
    }

    #[test]
    fn should_implement_clone() {
        let err = AshCoreError::InsufficientCapacity {
            required: 10,
            provided: 5,
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}

// =============================================================================
// SECTION 17: UNTRAINED_SENTINEL Constant Tests
// =============================================================================

mod sentinel_tests {
    use super::*;

    #[test]
    fn should_be_large_positive_value() {
        assert!(UNTRAINED_SENTINEL > 1e8);
    }

    #[test]
    fn should_be_finite() {
        assert!(UNTRAINED_SENTINEL.is_finite());
    }

    #[test]
    fn should_be_distinguishable_from_valid_sdf_values() {
        // Valid SDF values are typically in range [-10, 10] or so
        // Sentinel should be clearly distinguishable
        assert!(UNTRAINED_SENTINEL > 1000.0);
    }
}

// =============================================================================
// SECTION 18: Property-Based Tests (Fuzz-Style)
// =============================================================================

mod property_tests {
    use super::*;

    /// Simple pseudo-random number generator for property tests
    fn simple_rand(seed: u64, index: usize) -> f32 {
        let mut x = seed.wrapping_add(index as u64);
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x = x.wrapping_mul(0x2545F4914F6CDD1D);
        (x as f64 / u64::MAX as f64) as f32
    }

    #[test]
    fn should_roundtrip_decompose_compose_for_many_points() {
        let cell_size = 0.1;
        let grid_dim = 8;

        for seed in 0..100 {
            let x = (simple_rand(seed, 0) - 0.5) * 200.0;
            let y = (simple_rand(seed, 1) - 0.5) * 200.0;
            let z = (simple_rand(seed, 2) - 0.5) * 200.0;
            let point = Point3::new(x, y, z);

            let (block, cell, local) = decompose_point(point, cell_size, grid_dim);
            let reconstructed = compose_point(block, cell, local, cell_size, grid_dim);

            let diff = (point - reconstructed).length();
            assert!(
                diff < 1e-3,
                "Roundtrip failed for seed {}: {:?} -> {:?}, diff={}",
                seed,
                point,
                reconstructed,
                diff
            );
        }
    }

    #[test]
    fn should_roundtrip_morton_for_many_coords() {
        for seed in 0..100 {
            let x = ((simple_rand(seed, 0) - 0.5) * 2_000_000.0) as i32;
            let y = ((simple_rand(seed, 1) - 0.5) * 2_000_000.0) as i32;
            let z = ((simple_rand(seed, 2) - 0.5) * 2_000_000.0) as i32;

            // Clamp to valid Morton range
            let x = x.clamp(-1048576, 1048575);
            let y = y.clamp(-1048576, 1048575);
            let z = z.clamp(-1048576, 1048575);

            let coord = BlockCoord::new(x, y, z);
            let code = morton_encode_signed(coord);
            let back = morton_decode_signed(code);

            assert_eq!(
                back, coord,
                "Morton roundtrip failed for seed {}: {:?}",
                seed, coord
            );
        }
    }

    #[test]
    fn should_have_weights_sum_to_one_for_many_positions() {
        for seed in 0..100 {
            let u = simple_rand(seed, 0);
            let v = simple_rand(seed, 1);
            let w = simple_rand(seed, 2);
            let local = LocalCoord::new(u, v, w);

            let weights = compute_trilinear_weights(local);
            let sum: f32 = weights.iter().sum();

            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Weight sum {} != 1.0 for seed {} at ({}, {}, {})",
                sum,
                seed,
                u,
                v,
                w
            );
        }
    }

    #[test]
    fn should_match_analytical_and_numerical_gradient_for_many_positions() {
        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.2,
        };

        let eps = 1e-4;

        for seed in 0..20 {
            // Random cell within first block
            let cx = ((simple_rand(seed, 0) * 7.0) as u32).min(6);
            let cy = ((simple_rand(seed, 1) * 7.0) as u32).min(6);
            let cz = ((simple_rand(seed, 2) * 7.0) as u32).min(6);

            // Random local position away from boundaries
            let u = 0.1 + simple_rand(seed, 3) * 0.8;
            let v = 0.1 + simple_rand(seed, 4) * 0.8;
            let w = 0.1 + simple_rand(seed, 5) * 0.8;

            let block = BlockCoord::new(0, 0, 0);
            let cell = CellCoord::new(cx, cy, cz);
            let local = LocalCoord::new(u, v, w);

            let analytical = match trilinear_gradient_sdf(&provider, block, cell, local) {
                Some(g) => g,
                None => continue,
            };

            let mut numerical = [0.0f32; 3];
            for axis in 0..3 {
                let mut local_plus = local;
                let mut local_minus = local;

                match axis {
                    0 => {
                        local_plus.u += eps;
                        local_minus.u -= eps;
                    }
                    1 => {
                        local_plus.v += eps;
                        local_minus.v -= eps;
                    }
                    _ => {
                        local_plus.w += eps;
                        local_minus.w -= eps;
                    }
                }

                let f_plus = match trilinear_interpolate_sdf(&provider, block, cell, local_plus) {
                    Some(r) => r.value(),
                    None => continue,
                };
                let f_minus = match trilinear_interpolate_sdf(&provider, block, cell, local_minus) {
                    Some(r) => r.value(),
                    None => continue,
                };
                numerical[axis] = (f_plus - f_minus) / (2.0 * eps);
            }

            for i in 0..3 {
                assert!(
                    (analytical[i] - numerical[i]).abs() < 5e-2,
                    "Gradient mismatch at seed {} axis {}: analytical={}, numerical={}",
                    seed,
                    i,
                    analytical[i],
                    numerical[i]
                );
            }
        }
    }
}

// =============================================================================
// SECTION 19: Block Size Calculation Tests
// =============================================================================

mod block_size_tests {
    use super::*;

    struct TestProvider {
        grid_dim: u32,
        cell_size: f32,
    }

    impl CellValueProvider<1> for TestProvider {
        fn get_corner_values(
            &self,
            _block: BlockCoord,
            _cell: CellCoord,
            _corner: (u32, u32, u32),
        ) -> Option<[f32; 1]> {
            Some([0.0])
        }

        fn grid_dim(&self) -> u32 {
            self.grid_dim
        }

        fn cell_size(&self) -> f32 {
            self.cell_size
        }
    }

    #[test]
    fn should_compute_block_size_as_grid_dim_times_cell_size() {
        let provider = TestProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let expected = 8.0 * 0.1;
        assert!((provider.block_size() - expected).abs() < 1e-6);
    }

    #[test]
    fn should_compute_block_size_for_different_dimensions() {
        let test_cases = [
            (4, 0.25, 1.0),
            (8, 0.1, 0.8),
            (16, 0.05, 0.8),
            (32, 0.025, 0.8),
        ];

        for (grid_dim, cell_size, expected) in test_cases {
            let provider = TestProvider {
                grid_dim,
                cell_size,
            };
            assert!(
                (provider.block_size() - expected).abs() < 1e-6,
                "block_size mismatch for grid_dim={}, cell_size={}",
                grid_dim,
                cell_size
            );
        }
    }
}

// =============================================================================
// SECTION 20: Integration Tests
// =============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn should_query_sdf_at_world_position_end_to_end() {
        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.5, 0.5, 0.5),
            radius: 0.2,
        };

        // Query at various world positions
        // Note: Due to discretization and interpolation, points very close to
        // the surface may have small numerical errors in their classification.
        let queries = [
            (Point3::new(0.5, 0.5, 0.5), true),  // Center (inside)
            (Point3::new(2.0, 2.0, 2.0), false), // Far (outside)
            (Point3::new(0.5, 0.5, 0.8), false), // Clearly outside (0.3 from center, radius 0.2)
        ];

        for (point, should_be_inside) in queries {
            let (block, cell, local) = decompose_point(point, provider.cell_size, provider.grid_dim);

            if let Some(result) = trilinear_interpolate_sdf(&provider, block, cell, local) {
                let is_inside = result.value() < 0.0;
                assert_eq!(
                    is_inside, should_be_inside,
                    "Point {:?} should be {} sphere, got SDF={}",
                    point,
                    if should_be_inside { "inside" } else { "outside" },
                    result.value()
                );
            }
        }
    }

    #[test]
    fn should_compute_gradient_for_navigation() {
        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.05,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.2,
        };

        // Query gradient at a point outside the sphere
        let query = Point3::new(0.7, 0.4, 0.4); // +x from center
        let (block, cell, local) = decompose_point(query, provider.cell_size, provider.grid_dim);

        if let Some(grad) = trilinear_gradient_sdf(&provider, block, cell, local) {
            let grad_vec = Point3::new(grad[0], grad[1], grad[2]).normalize();

            // Gradient should point away from center (approximately +x)
            assert!(
                grad_vec.x > 0.5,
                "Gradient should point outward, got {:?}",
                grad_vec
            );
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn should_extract_mesh_for_visualization() {
        use ash_core::marching_cubes::process_cell;

        let provider = SphereSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: Point3::new(0.4, 0.4, 0.4),
            radius: 0.2,
        };

        let mut mesh_vertices: Vec<Point3> = Vec::new();

        // Process all cells in block (0,0,0)
        let block = BlockCoord::new(0, 0, 0);
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let cell = CellCoord::new(x, y, z);
                    let triangles = process_cell(&provider, block, cell, 0.0);

                    for tri in triangles {
                        mesh_vertices.extend_from_slice(&tri);
                    }
                }
            }
        }

        // Should have extracted some mesh
        assert!(
            !mesh_vertices.is_empty(),
            "Should extract mesh vertices for sphere"
        );

        // All vertices should be near the sphere surface
        for vertex in &mesh_vertices {
            let dist = (*vertex - provider.center).length();
            assert!(
                (dist - provider.radius).abs() < provider.cell_size * 2.0,
                "Vertex {:?} is too far from surface: dist={}, radius={}",
                vertex,
                dist,
                provider.radius
            );
        }
    }

    #[test]
    fn should_perform_backward_pass_for_training() {
        let provider = LinearSdfProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let point = Point3::new(0.35, 0.45, 0.55);
        let (block, cell, local) = decompose_point(point, provider.cell_size, provider.grid_dim);

        // Forward pass
        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();

        // Backward pass
        let mut accumulator = TestGradientAccumulator::new();
        let upstream_grad = 1.0;
        trilinear_backward(&mut accumulator, &result, [upstream_grad]);

        // Verify gradients were accumulated
        assert_eq!(accumulator.accumulated.len(), 8);

        // Gradients should sum to upstream gradient
        let total = accumulator.total_gradient();
        assert!(
            (total - upstream_grad).abs() < 1e-5,
            "Total gradient {} should equal upstream {}",
            total,
            upstream_grad
        );
    }
}
