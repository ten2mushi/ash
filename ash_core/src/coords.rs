//! Coordinate mathematics for the sparse-dense SDF grid.
//!
//! Handles decomposition of world-space points into (block, cell, local) coordinates
//! and the inverse operation.

use crate::types::{BlockCoord, CellCoord, LocalCoord, Point3};

/// Decompose a world-space point into (BlockCoord, CellCoord, LocalCoord).
///
/// This function handles negative coordinates correctly using Euclidean division,
/// which always rounds toward negative infinity (unlike truncation which rounds toward zero).
///
/// # Arguments
/// * `point` - The world-space point to decompose
/// * `cell_size` - The size of each cell in world units
/// * `grid_dim` - The number of cells per axis in each block
///
/// # Returns
/// A tuple of (block, cell, local) coordinates where:
/// * `block` - The sparse block containing the point
/// * `cell` - The cell within the block (0 to grid_dim-1)
/// * `local` - The fractional position within the cell [0, 1)
///
/// # Example
/// ```
/// use ash_core::coords::decompose_point;
/// use ash_core::types::Point3;
///
/// let point = Point3::new(0.25, 0.5, 0.75);
/// let (block, cell, local) = decompose_point(point, 0.1, 8);
///
/// // Point (0.25, 0.5, 0.75) with cell_size=0.1, grid_dim=8:
/// // - Block size = 0.8
/// // - Block (0, 0, 0) covers [0, 0.8)
/// // - Cell (2, 5, 7) at position (0.2, 0.5, 0.7)
/// // - Local (0.5, 0.0, 0.5) within that cell
/// ```
#[inline]
pub fn decompose_point(
    point: Point3,
    cell_size: f32,
    grid_dim: u32,
) -> (BlockCoord, CellCoord, LocalCoord) {
    let block_size = cell_size * grid_dim as f32;
    let grid_dim_i = grid_dim as i32;

    // Convert to cell coordinates (can be negative)
    let cell_x = libm::floorf(point.x / cell_size);
    let cell_y = libm::floorf(point.y / cell_size);
    let cell_z = libm::floorf(point.z / cell_size);

    // Use Euclidean division to get block and cell-within-block
    // This handles negative coordinates correctly
    let block_x = euclidean_div(cell_x as i32, grid_dim_i);
    let block_y = euclidean_div(cell_y as i32, grid_dim_i);
    let block_z = euclidean_div(cell_z as i32, grid_dim_i);

    let local_cell_x = euclidean_rem(cell_x as i32, grid_dim_i) as u32;
    let local_cell_y = euclidean_rem(cell_y as i32, grid_dim_i) as u32;
    let local_cell_z = euclidean_rem(cell_z as i32, grid_dim_i) as u32;

    // Compute local coordinates within the cell [0, 1)
    let cell_origin_x = block_x as f32 * block_size + local_cell_x as f32 * cell_size;
    let cell_origin_y = block_y as f32 * block_size + local_cell_y as f32 * cell_size;
    let cell_origin_z = block_z as f32 * block_size + local_cell_z as f32 * cell_size;

    let local_u = (point.x - cell_origin_x) / cell_size;
    let local_v = (point.y - cell_origin_y) / cell_size;
    let local_w = (point.z - cell_origin_z) / cell_size;

    (
        BlockCoord::new(block_x, block_y, block_z),
        CellCoord::new(local_cell_x, local_cell_y, local_cell_z),
        LocalCoord::new(local_u, local_v, local_w),
    )
}

/// Compose a world-space point from (BlockCoord, CellCoord, LocalCoord).
///
/// This is the inverse of `decompose_point`.
///
/// # Arguments
/// * `block` - The sparse block coordinates
/// * `cell` - The cell coordinates within the block
/// * `local` - The fractional position within the cell [0, 1]
/// * `cell_size` - The size of each cell in world units
/// * `grid_dim` - The number of cells per axis in each block
///
/// # Returns
/// The world-space point corresponding to the given coordinates.
#[inline]
pub fn compose_point(
    block: BlockCoord,
    cell: CellCoord,
    local: LocalCoord,
    cell_size: f32,
    grid_dim: u32,
) -> Point3 {
    let block_size = cell_size * grid_dim as f32;

    let x = block.x as f32 * block_size + cell.x as f32 * cell_size + local.u * cell_size;
    let y = block.y as f32 * block_size + cell.y as f32 * cell_size + local.v * cell_size;
    let z = block.z as f32 * block_size + cell.z as f32 * cell_size + local.w * cell_size;

    Point3::new(x, y, z)
}

/// Resolve a corner that may cross block boundaries.
///
/// When accessing corners of a cell, corners at the edge of a block may actually
/// belong to the adjacent block. This function resolves the actual (block, cell)
/// for a given corner offset.
///
/// # Arguments
/// * `block` - The block containing the cell
/// * `cell` - The cell within the block
/// * `corner` - The corner offset (0 or 1 for each axis)
/// * `grid_dim` - The number of cells per axis in each block
///
/// # Returns
/// The resolved (block, cell) coordinates for the corner's vertex.
///
/// # Note
/// This function returns the cell that "owns" the corner vertex. For corners
/// at (1,1,1), this may be in an adjacent block.
#[inline]
pub fn resolve_corner(
    block: BlockCoord,
    cell: CellCoord,
    corner: (u32, u32, u32),
    grid_dim: u32,
) -> (BlockCoord, CellCoord) {
    let mut new_block = block;
    let mut new_cell_x = cell.x + corner.0;
    let mut new_cell_y = cell.y + corner.1;
    let mut new_cell_z = cell.z + corner.2;

    // Handle overflow to adjacent block
    if new_cell_x >= grid_dim {
        new_cell_x = 0;
        new_block.x += 1;
    }
    if new_cell_y >= grid_dim {
        new_cell_y = 0;
        new_block.y += 1;
    }
    if new_cell_z >= grid_dim {
        new_cell_z = 0;
        new_block.z += 1;
    }

    (new_block, CellCoord::new(new_cell_x, new_cell_y, new_cell_z))
}

/// Get all 27 neighboring blocks (including the block itself).
///
/// This is useful for marching cubes mesh extraction, where cells at block
/// boundaries need access to adjacent blocks.
///
/// # Arguments
/// * `block` - The center block
///
/// # Returns
/// An array of 27 BlockCoords representing the 3x3x3 neighborhood.
/// The center block is at index 13 (1 + 3*1 + 9*1).
#[inline]
pub fn get_neighbor_blocks(block: BlockCoord) -> [BlockCoord; 27] {
    let mut neighbors = [BlockCoord::new(0, 0, 0); 27];
    let mut i = 0;
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                neighbors[i] = BlockCoord::new(block.x + dx, block.y + dy, block.z + dz);
                i += 1;
            }
        }
    }
    neighbors
}

/// Get the world-space position of a cell's origin (corner 0).
///
/// # Arguments
/// * `block` - The block coordinates
/// * `cell` - The cell coordinates within the block
/// * `cell_size` - The size of each cell in world units
/// * `grid_dim` - The number of cells per axis in each block
///
/// # Returns
/// The world-space position of the cell's (0,0,0) corner.
#[inline]
pub fn cell_origin(
    block: BlockCoord,
    cell: CellCoord,
    cell_size: f32,
    grid_dim: u32,
) -> Point3 {
    let block_size = cell_size * grid_dim as f32;
    Point3::new(
        block.x as f32 * block_size + cell.x as f32 * cell_size,
        block.y as f32 * block_size + cell.y as f32 * cell_size,
        block.z as f32 * block_size + cell.z as f32 * cell_size,
    )
}

/// Get the world-space position of a specific corner of a cell.
///
/// # Arguments
/// * `block` - The block coordinates
/// * `cell` - The cell coordinates within the block
/// * `corner` - The corner offset (0 or 1 for each axis)
/// * `cell_size` - The size of each cell in world units
/// * `grid_dim` - The number of cells per axis in each block
///
/// # Returns
/// The world-space position of the specified corner.
#[inline]
pub fn corner_position(
    block: BlockCoord,
    cell: CellCoord,
    corner: (u32, u32, u32),
    cell_size: f32,
    grid_dim: u32,
) -> Point3 {
    let origin = cell_origin(block, cell, cell_size, grid_dim);
    Point3::new(
        origin.x + corner.0 as f32 * cell_size,
        origin.y + corner.1 as f32 * cell_size,
        origin.z + corner.2 as f32 * cell_size,
    )
}

/// Euclidean division that rounds toward negative infinity.
/// This is the mathematical definition of integer division.
#[inline]
const fn euclidean_div(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    // If remainder has opposite sign from divisor, adjust quotient
    // (r < 0 && b > 0) || (r > 0 && b < 0) means we need to subtract 1
    if (r < 0 && b > 0) || (r > 0 && b < 0) {
        q - 1
    } else {
        q
    }
}

/// Euclidean remainder (always non-negative when divisor is positive).
#[inline]
const fn euclidean_rem(a: i32, b: i32) -> i32 {
    let r = a % b;
    if r < 0 {
        r + b.abs()
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_div() {
        // Positive numbers
        assert_eq!(euclidean_div(7, 3), 2);
        assert_eq!(euclidean_div(6, 3), 2);

        // Negative dividend - key difference from truncation
        assert_eq!(euclidean_div(-1, 8), -1); // Not 0!
        assert_eq!(euclidean_div(-7, 8), -1);
        assert_eq!(euclidean_div(-8, 8), -1);
        assert_eq!(euclidean_div(-9, 8), -2);
    }

    #[test]
    fn test_euclidean_rem() {
        // Positive numbers
        assert_eq!(euclidean_rem(7, 3), 1);
        assert_eq!(euclidean_rem(6, 3), 0);

        // Negative dividend - should be non-negative
        assert_eq!(euclidean_rem(-1, 8), 7); // Not -1!
        assert_eq!(euclidean_rem(-7, 8), 1);
        assert_eq!(euclidean_rem(-8, 8), 0);
        assert_eq!(euclidean_rem(-9, 8), 7);
    }

    #[test]
    fn test_decompose_compose_roundtrip_positive() {
        let cell_size = 0.1;
        let grid_dim = 8;

        let test_points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.25, 0.5, 0.75),
            Point3::new(1.5, 2.3, 3.7),
            Point3::new(10.0, 10.0, 10.0),
        ];

        for point in &test_points {
            let (block, cell, local) = decompose_point(*point, cell_size, grid_dim);
            let reconstructed = compose_point(block, cell, local, cell_size, grid_dim);

            let diff = (*point - reconstructed).length();
            assert!(
                diff < 1e-5,
                "Roundtrip failed for {:?}: got {:?}, diff={}",
                point,
                reconstructed,
                diff
            );
        }
    }

    #[test]
    fn test_decompose_compose_roundtrip_negative() {
        let cell_size = 0.1;
        let grid_dim = 8;

        let test_points = [
            Point3::new(-0.1, 0.0, 0.0),
            Point3::new(-1.5, -2.3, -3.7),
            Point3::new(-0.05, -0.05, -0.05),
            Point3::new(-10.0, 10.0, -10.0),
        ];

        for point in &test_points {
            let (block, cell, local) = decompose_point(*point, cell_size, grid_dim);
            let reconstructed = compose_point(block, cell, local, cell_size, grid_dim);

            let diff = (*point - reconstructed).length();
            assert!(
                diff < 1e-5,
                "Roundtrip failed for {:?}: got {:?}, diff={}",
                point,
                reconstructed,
                diff
            );
        }
    }

    #[test]
    fn test_decompose_negative_coordinates() {
        let cell_size = 0.1;
        let grid_dim = 8;

        // Point just below origin
        let point = Point3::new(-0.05, -0.05, -0.05);
        let (block, cell, _local) = decompose_point(point, cell_size, grid_dim);

        // Should be in block (-1, -1, -1)
        assert_eq!(block.x, -1);
        assert_eq!(block.y, -1);
        assert_eq!(block.z, -1);

        // Should be in the last cell of that block (cell 7)
        assert_eq!(cell.x, 7);
        assert_eq!(cell.y, 7);
        assert_eq!(cell.z, 7);
    }

    #[test]
    fn test_resolve_corner_no_overflow() {
        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(3, 3, 3);
        let grid_dim = 8;

        // Corner (0,0,0) - no overflow
        let (b, c) = resolve_corner(block, cell, (0, 0, 0), grid_dim);
        assert_eq!(b, block);
        assert_eq!(c, cell);

        // Corner (1,0,0) - no overflow
        let (b, c) = resolve_corner(block, cell, (1, 0, 0), grid_dim);
        assert_eq!(b, block);
        assert_eq!(c, CellCoord::new(4, 3, 3));
    }

    #[test]
    fn test_resolve_corner_with_overflow() {
        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(7, 7, 7);
        let grid_dim = 8;

        // Corner (1,1,1) - all axes overflow
        let (b, c) = resolve_corner(block, cell, (1, 1, 1), grid_dim);
        assert_eq!(b, BlockCoord::new(1, 1, 1));
        assert_eq!(c, CellCoord::new(0, 0, 0));

        // Corner (1,0,0) - only x overflows
        let (b, c) = resolve_corner(block, cell, (1, 0, 0), grid_dim);
        assert_eq!(b, BlockCoord::new(1, 0, 0));
        assert_eq!(c, CellCoord::new(0, 7, 7));
    }

    #[test]
    fn test_get_neighbor_blocks() {
        let block = BlockCoord::new(5, 10, -3);
        let neighbors = get_neighbor_blocks(block);

        // Check size
        assert_eq!(neighbors.len(), 27);

        // Check center (index 13)
        assert_eq!(neighbors[13], block);

        // Check corners
        assert_eq!(neighbors[0], BlockCoord::new(4, 9, -4)); // (-1,-1,-1)
        assert_eq!(neighbors[26], BlockCoord::new(6, 11, -2)); // (1,1,1)
    }

    #[test]
    fn test_cell_origin() {
        let cell_size = 0.1;
        let grid_dim = 8;

        // Origin block, origin cell
        let origin = cell_origin(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(0, 0, 0),
            cell_size,
            grid_dim,
        );
        assert!((origin.x).abs() < 1e-6);
        assert!((origin.y).abs() < 1e-6);
        assert!((origin.z).abs() < 1e-6);

        // Cell (2, 3, 4) in block (0, 0, 0)
        let pos = cell_origin(
            BlockCoord::new(0, 0, 0),
            CellCoord::new(2, 3, 4),
            cell_size,
            grid_dim,
        );
        assert!((pos.x - 0.2).abs() < 1e-6);
        assert!((pos.y - 0.3).abs() < 1e-6);
        assert!((pos.z - 0.4).abs() < 1e-6);

        // Block (1, 0, 0), cell (0, 0, 0)
        let pos = cell_origin(
            BlockCoord::new(1, 0, 0),
            CellCoord::new(0, 0, 0),
            cell_size,
            grid_dim,
        );
        assert!((pos.x - 0.8).abs() < 1e-6); // block_size = 0.8
    }

    #[test]
    fn test_corner_position() {
        let cell_size = 0.1;
        let grid_dim = 8;

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(0, 0, 0);

        // Corner (0,0,0)
        let pos = corner_position(block, cell, (0, 0, 0), cell_size, grid_dim);
        assert!((pos.x).abs() < 1e-6);

        // Corner (1,1,1)
        let pos = corner_position(block, cell, (1, 1, 1), cell_size, grid_dim);
        assert!((pos.x - 0.1).abs() < 1e-6);
        assert!((pos.y - 0.1).abs() < 1e-6);
        assert!((pos.z - 0.1).abs() < 1e-6);
    }
}
