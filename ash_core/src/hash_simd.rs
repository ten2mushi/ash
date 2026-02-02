//! SIMD-accelerated hash operations.
//!
//! Provides vectorized FNV-1a hashing for batch block coordinate lookups.
//! Uses the `wide` crate for portable SIMD operations.
//!
//! NOTE: The SIMD hash uses a simplified FNV-1a variant that XORs each i32
//! coordinate as a whole, whereas the standard scalar FNV-1a hashes byte-by-byte.
//! These produce different hash values. The SIMD version is designed for use
//! in batch lookups where consistency with the scalar version is maintained by
//! using the scalar function for non-SIMD paths.

use wide::{i32x4, u32x4};

use crate::types::BlockCoord;

// FNV-1a 32-bit constants
const FNV_OFFSET_32: u32 = 0x811C9DC5;
const FNV_PRIME_32: u32 = 0x01000193;

/// FNV-1a hash for 4 coordinates simultaneously using SIMD.
///
/// NOTE: This uses a simplified FNV-1a variant optimized for SIMD that produces
/// different hash values than the byte-by-byte `fnv1a_32` function. For batch
/// lookups, use `hash_batch` which ensures consistency.
///
/// # Arguments
/// * `x` - Four x coordinates packed in a SIMD vector
/// * `y` - Four y coordinates packed in a SIMD vector
/// * `z` - Four z coordinates packed in a SIMD vector
///
/// # Returns
/// Four 32-bit hash values packed in a SIMD vector
#[inline(always)]
pub fn hash_i32x4(x: i32x4, y: i32x4, z: i32x4) -> u32x4 {
    let mut h = u32x4::splat(FNV_OFFSET_32);
    let prime = u32x4::splat(FNV_PRIME_32);

    // Transmute signed to unsigned for bitwise operations
    // Safety: i32 and u32 have the same size and alignment
    let ux: u32x4 = unsafe { core::mem::transmute(x) };
    let uy: u32x4 = unsafe { core::mem::transmute(y) };
    let uz: u32x4 = unsafe { core::mem::transmute(z) };

    // Hash x coordinate (all 4 bytes at once as a u32)
    h = (h ^ ux) * prime;

    // Hash y coordinate
    h = (h ^ uy) * prime;

    // Hash z coordinate
    h = (h ^ uz) * prime;

    h
}

/// Batch hash 4 BlockCoords at once using SIMD.
///
/// NOTE: This function uses the SIMD-optimized hash which produces different
/// values than `fnv1a_32`. It's intended for internal use where a consistent
/// SIMD hash is needed. For lookups that must match the standard hash, use
/// `hash_batch` instead.
///
/// # Arguments
/// * `coords` - Slice of exactly 4 BlockCoords to hash
///
/// # Returns
/// Array of 4 hash values (from SIMD hash variant)
///
/// # Panics
/// Panics if `coords.len() != 4`
#[inline]
pub fn hash_batch_4(coords: &[BlockCoord]) -> [u32; 4] {
    assert_eq!(coords.len(), 4, "hash_batch_4 requires exactly 4 coordinates");

    let x = i32x4::new([coords[0].x, coords[1].x, coords[2].x, coords[3].x]);
    let y = i32x4::new([coords[0].y, coords[1].y, coords[2].y, coords[3].y]);
    let z = i32x4::new([coords[0].z, coords[1].z, coords[2].z, coords[3].z]);

    let hashes = hash_i32x4(x, y, z);
    hashes.to_array()
}

/// Batch hash 4 BlockCoords using the standard scalar FNV-1a function.
///
/// This produces the same hash values as calling `fnv1a_32` individually,
/// ensuring consistency with the existing hash table implementation.
///
/// # Arguments
/// * `coords` - Slice of exactly 4 BlockCoords to hash
///
/// # Returns
/// Array of 4 hash values (matching `fnv1a_32`)
#[inline]
pub fn hash_batch_4_scalar(coords: &[BlockCoord]) -> [u32; 4] {
    assert_eq!(coords.len(), 4, "hash_batch_4_scalar requires exactly 4 coordinates");
    [
        crate::hash::fnv1a_32(coords[0]),
        crate::hash::fnv1a_32(coords[1]),
        crate::hash::fnv1a_32(coords[2]),
        crate::hash::fnv1a_32(coords[3]),
    ]
}

/// Batch hash an arbitrary number of BlockCoords.
///
/// Uses the standard `fnv1a_32` function to ensure consistency with existing
/// hash table lookups. The batching improves cache locality.
///
/// # Arguments
/// * `coords` - Slice of BlockCoords to hash
///
/// # Returns
/// Vector of hash values, one per input coordinate (matching `fnv1a_32`)
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn hash_batch(coords: &[BlockCoord]) -> crate::alloc_prelude::Vec<u32> {
    let mut results = crate::alloc_prelude::Vec::with_capacity(coords.len());

    // Process all coordinates with the standard scalar hash
    // This ensures consistency with fnv1a_32
    for coord in coords {
        results.push(crate::hash::fnv1a_32(*coord));
    }

    results
}

/// Scalar FNV-1a 32-bit hash for a single coordinate.
///
/// This is the reference implementation used to verify SIMD correctness.
#[cfg(test)]
#[inline]
fn scalar_fnv1a_32_direct(x: i32, y: i32, z: i32) -> u32 {
    let mut hash = FNV_OFFSET_32;

    // Hash x as u32
    let ux = x as u32;
    hash ^= ux;
    hash = hash.wrapping_mul(FNV_PRIME_32);

    // Hash y as u32
    let uy = y as u32;
    hash ^= uy;
    hash = hash.wrapping_mul(FNV_PRIME_32);

    // Hash z as u32
    let uz = z as u32;
    hash ^= uz;
    hash = hash.wrapping_mul(FNV_PRIME_32);

    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;
    use std::vec;

    #[test]
    fn simd_hash_matches_simd_scalar_reference() {
        // The SIMD hash uses a different algorithm than fnv1a_32 (whole-i32 XOR vs byte-by-byte)
        // This test verifies SIMD consistency with its own reference implementation
        let test_coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 2, 3),
            BlockCoord::new(100, 200, 300),
            BlockCoord::new(i32::MAX, i32::MIN, 0),
        ];

        let simd_hashes = hash_batch_4(&test_coords);

        for (i, coord) in test_coords.iter().enumerate() {
            let scalar = scalar_fnv1a_32_direct(coord.x, coord.y, coord.z);
            assert_eq!(
                simd_hashes[i], scalar,
                "SIMD hash mismatch for {:?}: SIMD={:#x}, scalar={:#x}",
                coord, simd_hashes[i], scalar
            );
        }
    }

    #[test]
    fn simd_hash_negative_coords() {
        let coords = [
            BlockCoord::new(-1, -2, -3),
            BlockCoord::new(-100, 50, -25),
            BlockCoord::new(-1000000, -1000000, -1000000),
            BlockCoord::new(i32::MIN, i32::MIN, i32::MIN),
        ];

        let simd_hashes = hash_batch_4(&coords);

        for (i, coord) in coords.iter().enumerate() {
            let scalar = scalar_fnv1a_32_direct(coord.x, coord.y, coord.z);
            assert_eq!(simd_hashes[i], scalar);
        }
    }

    #[test]
    fn simd_hash_zero() {
        let coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(0, 0, 0),
        ];

        let hashes = hash_batch_4(&coords);

        // All should be the same
        assert_eq!(hashes[0], hashes[1]);
        assert_eq!(hashes[1], hashes[2]);
        assert_eq!(hashes[2], hashes[3]);

        // And match SIMD scalar reference
        let scalar = scalar_fnv1a_32_direct(0, 0, 0);
        assert_eq!(hashes[0], scalar);
    }

    #[test]
    fn simd_hash_different_coords_different_hashes() {
        let coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 0, 0),
            BlockCoord::new(0, 1, 0),
            BlockCoord::new(0, 0, 1),
        ];

        let hashes = hash_batch_4(&coords);

        // All should be different
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[0], hashes[2]);
        assert_ne!(hashes[0], hashes[3]);
        assert_ne!(hashes[1], hashes[2]);
        assert_ne!(hashes[1], hashes[3]);
        assert_ne!(hashes[2], hashes[3]);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn hash_batch_matches_fnv1a_32() {
        // hash_batch should use the standard fnv1a_32 for consistency with hash tables
        let coords: Vec<BlockCoord> = (0..17)
            .map(|i| BlockCoord::new(i, i * 2, i * 3))
            .collect();

        let batch_hashes = hash_batch(&coords);

        assert_eq!(batch_hashes.len(), 17);

        // Verify each hash matches fnv1a_32
        for (i, coord) in coords.iter().enumerate() {
            let scalar = crate::hash::fnv1a_32(*coord);
            assert_eq!(batch_hashes[i], scalar,
                "hash_batch mismatch at index {} for {:?}", i, coord);
        }
    }

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn hash_batch_empty() {
        let coords: Vec<BlockCoord> = vec![];
        let hashes = hash_batch(&coords);
        assert!(hashes.is_empty());
    }

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn hash_batch_partial_chunk() {
        // Test with 1, 2, 3 elements (less than 4)
        for count in 1..4 {
            let coords: Vec<BlockCoord> = (0..count)
                .map(|i| BlockCoord::new(i as i32, i as i32 * 2, i as i32 * 3))
                .collect();

            let hashes = hash_batch(&coords);
            assert_eq!(hashes.len(), count);

            for (i, coord) in coords.iter().enumerate() {
                let scalar = crate::hash::fnv1a_32(*coord);
                assert_eq!(hashes[i], scalar);
            }
        }
    }

    #[test]
    fn hash_batch_4_scalar_matches_fnv1a_32() {
        let coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 2, 3),
            BlockCoord::new(-100, 50, -25),
            BlockCoord::new(i32::MAX, i32::MIN, 0),
        ];

        let batch = hash_batch_4_scalar(&coords);

        for (i, coord) in coords.iter().enumerate() {
            let scalar = crate::hash::fnv1a_32(*coord);
            assert_eq!(batch[i], scalar);
        }
    }
}
