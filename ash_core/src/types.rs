//! Core types for ash_core SDF operations.
//!
//! Provides coordinate types, interpolation results, and constants used throughout the ecosystem.

use core::hash::{Hash, Hasher};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Sentinel value indicating an untrained/uninitialized SDF region.
/// This value is large enough to never occur in valid SDF data.
pub const UNTRAINED_SENTINEL: f32 = 1e9;

/// A 3D point with named fields for clarity.
///
/// Provides arithmetic operations and conversions to/from arrays.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point3 {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
}

impl Point3 {
    /// Create a new Point3.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Create a Point3 with all components set to the same value.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Convert to an array.
    #[inline]
    pub const fn as_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Linear interpolation between two points.
    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    /// Dot product with another point (treating both as vectors).
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product with another point (treating both as vectors).
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Squared length of the vector.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Length (magnitude) of the vector.
    #[inline]
    pub fn length(self) -> f32 {
        libm::sqrtf(self.length_squared())
    }

    /// Normalize the vector to unit length.
    /// Returns a zero vector if the length is zero.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            Self::splat(0.0)
        } else {
            self / len
        }
    }

    /// Component-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self {
            x: if self.x < other.x { self.x } else { other.x },
            y: if self.y < other.y { self.y } else { other.y },
            z: if self.z < other.z { self.z } else { other.z },
        }
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self {
            x: if self.x > other.x { self.x } else { other.x },
            y: if self.y > other.y { self.y } else { other.y },
            z: if self.z > other.z { self.z } else { other.z },
        }
    }

    /// Component-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            x: libm::fabsf(self.x),
            y: libm::fabsf(self.y),
            z: libm::fabsf(self.z),
        }
    }
}

impl From<[f32; 3]> for Point3 {
    #[inline]
    fn from(arr: [f32; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

impl From<Point3> for [f32; 3] {
    #[inline]
    fn from(p: Point3) -> Self {
        p.as_array()
    }
}

impl From<(f32, f32, f32)> for Point3 {
    #[inline]
    fn from((x, y, z): (f32, f32, f32)) -> Self {
        Self { x, y, z }
    }
}

impl Add for Point3 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Point3 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f32> for Point3 {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Mul<Point3> for f32 {
    type Output = Point3;

    #[inline]
    fn mul(self, point: Point3) -> Point3 {
        point * self
    }
}

impl Div<f32> for Point3 {
    type Output = Self;

    #[inline]
    fn div(self, scalar: f32) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl Neg for Point3 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Sparse block coordinates (signed for negative world regions).
///
/// Represents the position of a block in the sparse grid structure.
/// Each block contains a dense grid of cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockCoord {
    /// X coordinate in block space.
    pub x: i32,
    /// Y coordinate in block space.
    pub y: i32,
    /// Z coordinate in block space.
    pub z: i32,
}

impl BlockCoord {
    /// Create a new BlockCoord.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Convert to an array.
    #[inline]
    pub const fn as_array(&self) -> [i32; 3] {
        [self.x, self.y, self.z]
    }
}

impl From<[i32; 3]> for BlockCoord {
    #[inline]
    fn from(arr: [i32; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

impl From<BlockCoord> for [i32; 3] {
    #[inline]
    fn from(b: BlockCoord) -> Self {
        b.as_array()
    }
}

impl Hash for BlockCoord {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.z.hash(state);
    }
}

/// Dense cell coordinates within a block (unsigned, 0 to grid_dim-1).
///
/// Represents the position of a cell within its parent block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CellCoord {
    /// X coordinate within the block (0 to grid_dim-1).
    pub x: u32,
    /// Y coordinate within the block (0 to grid_dim-1).
    pub y: u32,
    /// Z coordinate within the block (0 to grid_dim-1).
    pub z: u32,
}

impl CellCoord {
    /// Create a new CellCoord.
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Convert to an array.
    #[inline]
    pub const fn as_array(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }

    /// Compute the flat index for a 3D array with the given grid dimension.
    /// Uses row-major ordering: index = x + y * dim + z * dim * dim
    #[inline]
    pub const fn flat_index(&self, grid_dim: u32) -> usize {
        (self.x + self.y * grid_dim + self.z * grid_dim * grid_dim) as usize
    }

    /// Create a CellCoord from a flat index and grid dimension.
    #[inline]
    pub const fn from_flat_index(index: usize, grid_dim: u32) -> Self {
        let index = index as u32;
        let dim_sq = grid_dim * grid_dim;
        Self {
            x: index % grid_dim,
            y: (index / grid_dim) % grid_dim,
            z: index / dim_sq,
        }
    }
}

impl From<[u32; 3]> for CellCoord {
    #[inline]
    fn from(arr: [u32; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

impl From<CellCoord> for [u32; 3] {
    #[inline]
    fn from(c: CellCoord) -> Self {
        c.as_array()
    }
}

/// Local interpolation coordinates within a cell, in the range [0, 1]³.
///
/// Represents the fractional position within a cell for trilinear interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LocalCoord {
    /// U coordinate (x-axis) in [0, 1].
    pub u: f32,
    /// V coordinate (y-axis) in [0, 1].
    pub v: f32,
    /// W coordinate (z-axis) in [0, 1].
    pub w: f32,
}

impl LocalCoord {
    /// Create a new LocalCoord.
    #[inline]
    pub const fn new(u: f32, v: f32, w: f32) -> Self {
        Self { u, v, w }
    }

    /// Convert to an array.
    #[inline]
    pub const fn as_array(&self) -> [f32; 3] {
        [self.u, self.v, self.w]
    }

    /// Clamp all components to the [0, 1] range.
    #[inline]
    pub fn clamped(&self) -> Self {
        Self {
            u: clamp(self.u, 0.0, 1.0),
            v: clamp(self.v, 0.0, 1.0),
            w: clamp(self.w, 0.0, 1.0),
        }
    }

    /// Check if all components are within the [0, 1] range.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.u >= 0.0
            && self.u <= 1.0
            && self.v >= 0.0
            && self.v <= 1.0
            && self.w >= 0.0
            && self.w <= 1.0
    }
}

impl From<[f32; 3]> for LocalCoord {
    #[inline]
    fn from(arr: [f32; 3]) -> Self {
        Self {
            u: arr[0],
            v: arr[1],
            w: arr[2],
        }
    }
}

impl From<LocalCoord> for [f32; 3] {
    #[inline]
    fn from(l: LocalCoord) -> Self {
        l.as_array()
    }
}

/// Result of trilinear interpolation with N features.
///
/// Contains the interpolated values, the weights for each corner (useful for backpropagation),
/// and the block/cell coordinates where the interpolation occurred.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpolationResult<const N: usize> {
    /// The interpolated feature values (N values).
    pub values: [f32; N],
    /// The interpolation weights for each of the 8 corners.
    /// weight[i] = ∂value/∂corner_value[i]
    pub weights: [f32; 8],
    /// The block containing the interpolation cell.
    pub block: BlockCoord,
    /// The cell within the block.
    pub cell: CellCoord,
}

impl<const N: usize> InterpolationResult<N> {
    /// Create a new InterpolationResult.
    #[inline]
    pub const fn new(
        values: [f32; N],
        weights: [f32; 8],
        block: BlockCoord,
        cell: CellCoord,
    ) -> Self {
        Self {
            values,
            weights,
            block,
            cell,
        }
    }
}

/// Type alias for SDF-only interpolation result (N=1).
pub type SdfInterpolationResult = InterpolationResult<1>;

impl InterpolationResult<1> {
    /// Get the SDF value (convenience method for N=1).
    #[inline]
    pub const fn value(&self) -> f32 {
        self.values[0]
    }
}

/// Clamp a value to a range (no_std compatible).
#[inline]
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point3_arithmetic() {
        let a = Point3::new(1.0, 2.0, 3.0);
        let b = Point3::new(4.0, 5.0, 6.0);

        let sum = a + b;
        assert_eq!(sum, Point3::new(5.0, 7.0, 9.0));

        let diff = b - a;
        assert_eq!(diff, Point3::new(3.0, 3.0, 3.0));

        let scaled = a * 2.0;
        assert_eq!(scaled, Point3::new(2.0, 4.0, 6.0));

        let div = b / 2.0;
        assert_eq!(div, Point3::new(2.0, 2.5, 3.0));

        let neg = -a;
        assert_eq!(neg, Point3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_point3_dot_cross() {
        let a = Point3::new(1.0, 0.0, 0.0);
        let b = Point3::new(0.0, 1.0, 0.0);

        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.dot(a), 1.0);

        let cross = a.cross(b);
        assert_eq!(cross, Point3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_point3_length() {
        let p = Point3::new(3.0, 4.0, 0.0);
        assert_eq!(p.length(), 5.0);

        let unit = p.normalize();
        assert!((unit.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_point3_lerp() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 10.0, 10.0);

        let mid = a.lerp(b, 0.5);
        assert_eq!(mid, Point3::new(5.0, 5.0, 5.0));

        let start = a.lerp(b, 0.0);
        assert_eq!(start, a);

        let end = a.lerp(b, 1.0);
        assert_eq!(end, b);
    }

    #[test]
    fn test_point3_conversions() {
        let arr = [1.0, 2.0, 3.0];
        let p: Point3 = arr.into();
        assert_eq!(p.as_array(), arr);

        let back: [f32; 3] = p.into();
        assert_eq!(back, arr);

        let tuple = (1.0f32, 2.0f32, 3.0f32);
        let p2: Point3 = tuple.into();
        assert_eq!(p2, p);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_block_coord_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::BuildHasher;
        use std::hash::BuildHasherDefault;

        let b1 = BlockCoord::new(1, 2, 3);
        let b2 = BlockCoord::new(1, 2, 3);
        let b3 = BlockCoord::new(1, 2, 4);

        // Simple hash equality check using default hasher
        fn hash_one<H: Hash>(h: &H) -> u64 {
            let hasher = BuildHasherDefault::<DefaultHasher>::default();
            hasher.hash_one(h)
        }

        assert_eq!(hash_one(&b1), hash_one(&b2));
        // Different coords should (usually) have different hashes
        assert_ne!(hash_one(&b1), hash_one(&b3));
    }

    #[test]
    fn test_cell_coord_flat_index() {
        let grid_dim = 8u32;

        // Origin cell
        let c0 = CellCoord::new(0, 0, 0);
        assert_eq!(c0.flat_index(grid_dim), 0);

        // Along x
        let c1 = CellCoord::new(1, 0, 0);
        assert_eq!(c1.flat_index(grid_dim), 1);

        // Along y
        let c2 = CellCoord::new(0, 1, 0);
        assert_eq!(c2.flat_index(grid_dim), 8);

        // Along z
        let c3 = CellCoord::new(0, 0, 1);
        assert_eq!(c3.flat_index(grid_dim), 64);

        // Round-trip test
        for z in 0..grid_dim {
            for y in 0..grid_dim {
                for x in 0..grid_dim {
                    let c = CellCoord::new(x, y, z);
                    let idx = c.flat_index(grid_dim);
                    let c2 = CellCoord::from_flat_index(idx, grid_dim);
                    assert_eq!(c, c2);
                }
            }
        }
    }

    #[test]
    fn test_local_coord_clamped() {
        let l = LocalCoord::new(-0.5, 0.5, 1.5);
        let clamped = l.clamped();
        assert_eq!(clamped.u, 0.0);
        assert_eq!(clamped.v, 0.5);
        assert_eq!(clamped.w, 1.0);
    }

    #[test]
    fn test_local_coord_is_valid() {
        let valid = LocalCoord::new(0.5, 0.5, 0.5);
        assert!(valid.is_valid());

        let invalid = LocalCoord::new(-0.1, 0.5, 0.5);
        assert!(!invalid.is_valid());

        let also_invalid = LocalCoord::new(0.5, 0.5, 1.1);
        assert!(!also_invalid.is_valid());
    }

    #[test]
    fn test_untrained_sentinel() {
        assert!(UNTRAINED_SENTINEL > 1e8);
        assert!(UNTRAINED_SENTINEL.is_finite());
    }
}
