//! Spatial hashing primitives for sparse grid storage.
//!
//! Provides Morton encoding and FNV-1a hashing for efficient spatial lookups.
//! These are building blocks - downstream crates decide how to use them.

use crate::types::BlockCoord;

// FNV-1a constants
const FNV_OFFSET_64: u64 = 0xcbf29ce484222325;
const FNV_PRIME_64: u64 = 0x00000100000001b3;
const FNV_OFFSET_32: u32 = 0x811c9dc5;
const FNV_PRIME_32: u32 = 0x01000193;

/// FNV-1a 64-bit hash for a BlockCoord.
///
/// FNV-1a is simple, fast, and has reasonable distribution properties.
/// It's not cryptographic but works well for hash table lookups.
#[inline]
pub fn fnv1a_64(coord: BlockCoord) -> u64 {
    let mut hash = FNV_OFFSET_64;

    // Hash x coordinate (4 bytes)
    for byte in coord.x.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }

    // Hash y coordinate (4 bytes)
    for byte in coord.y.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }

    // Hash z coordinate (4 bytes)
    for byte in coord.z.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }

    hash
}

/// FNV-1a 32-bit hash for a BlockCoord.
///
/// Faster on 32-bit systems and when a smaller hash is sufficient.
#[inline]
pub fn fnv1a_32(coord: BlockCoord) -> u32 {
    let mut hash = FNV_OFFSET_32;

    // Hash x coordinate (4 bytes)
    for byte in coord.x.to_le_bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV_PRIME_32);
    }

    // Hash y coordinate (4 bytes)
    for byte in coord.y.to_le_bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV_PRIME_32);
    }

    // Hash z coordinate (4 bytes)
    for byte in coord.z.to_le_bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV_PRIME_32);
    }

    hash
}

/// Spread bits of a 21-bit value with 2-bit gaps for 3D Morton encoding.
///
/// Takes the lower 21 bits of input and spreads them so there are 2 zero bits
/// between each input bit. This is used to interleave three coordinates.
///
/// Input:  ....... ....... ...xxxxx xxxxxxxx xxxxxxxx (21 bits)
/// Output: ..x..x..x..x..x..x..x..x..x..x..x..x..x..x..x..x..x..x..x..x..x (63 bits)
#[inline]
pub fn spread_bits_3d(x: u32) -> u64 {
    // Use the portable bit-manipulation algorithm
    // This matches the reference implementation's fallback path
    let mut x = (x & 0x1FFFFF) as u64; // Mask to 21 bits
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8)) & 0x100F00F00F00F00F;
    x = (x | (x << 4)) & 0x10C30C30C30C30C3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

/// Compact bits from a spread 3D Morton value back to a 21-bit integer.
///
/// This is the inverse of `spread_bits_3d`.
#[inline]
pub fn compact_bits_3d(mut x: u64) -> u32 {
    x &= 0x1249249249249249;
    x = (x | (x >> 2)) & 0x10C30C30C30C30C3;
    x = (x | (x >> 4)) & 0x100F00F00F00F00F;
    x = (x | (x >> 8)) & 0x1F0000FF0000FF;
    x = (x | (x >> 16)) & 0x1F00000000FFFF;
    x = (x | (x >> 32)) & 0x1FFFFF;
    x as u32
}

/// Morton encode three unsigned 21-bit coordinates into a 63-bit code.
///
/// Morton encoding (Z-order curve) interleaves the bits of three coordinates
/// to produce a single integer that preserves spatial locality.
///
/// # Arguments
/// * `x`, `y`, `z` - Unsigned coordinates (only lower 21 bits used)
///
/// # Returns
/// A 63-bit Morton code where bits are interleaved as: z2y2x2z1y1x1z0y0x0...
#[inline]
pub fn morton_encode_3d(x: u32, y: u32, z: u32) -> u64 {
    spread_bits_3d(x) | (spread_bits_3d(y) << 1) | (spread_bits_3d(z) << 2)
}

/// Morton decode a 63-bit code to three unsigned 21-bit coordinates.
///
/// This is the inverse of `morton_encode_3d`.
#[inline]
pub fn morton_decode_3d(code: u64) -> (u32, u32, u32) {
    let x = compact_bits_3d(code);
    let y = compact_bits_3d(code >> 1);
    let z = compact_bits_3d(code >> 2);
    (x, y, z)
}

/// Morton encode a signed BlockCoord using offset encoding.
///
/// Signed coordinates are converted to unsigned by adding an offset (2^20)
/// to bring them into the range [0, 2^21) for standard Morton encoding.
///
/// This allows encoding coordinates in the range [-2^20, 2^20).
#[inline]
pub fn morton_encode_signed(coord: BlockCoord) -> u64 {
    const OFFSET: i32 = 1 << 20; // 2^20 = 1,048,576

    let x = (coord.x.wrapping_add(OFFSET)) as u32;
    let y = (coord.y.wrapping_add(OFFSET)) as u32;
    let z = (coord.z.wrapping_add(OFFSET)) as u32;

    morton_encode_3d(x, y, z)
}

/// Morton decode to a signed BlockCoord.
///
/// This is the inverse of `morton_encode_signed`.
#[inline]
pub fn morton_decode_signed(code: u64) -> BlockCoord {
    const OFFSET: i32 = 1 << 20;

    let (x, y, z) = morton_decode_3d(code);

    BlockCoord::new(
        (x as i32).wrapping_sub(OFFSET),
        (y as i32).wrapping_sub(OFFSET),
        (z as i32).wrapping_sub(OFFSET),
    )
}

/// Hash a BlockCoord to an index for hash table lookup.
///
/// Uses FNV-1a hashing and returns an index in [0, capacity).
#[inline]
pub fn hash_to_index(coord: BlockCoord, capacity: usize) -> usize {
    (fnv1a_64(coord) as usize) % capacity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_64_different_coords() {
        let c1 = BlockCoord::new(0, 0, 0);
        let c2 = BlockCoord::new(1, 0, 0);
        let c3 = BlockCoord::new(0, 1, 0);

        let h1 = fnv1a_64(c1);
        let h2 = fnv1a_64(c2);
        let h3 = fnv1a_64(c3);

        // Different coordinates should produce different hashes
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }

    #[test]
    fn test_fnv1a_64_consistent() {
        let coord = BlockCoord::new(42, -17, 1000);
        let h1 = fnv1a_64(coord);
        let h2 = fnv1a_64(coord);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_32_consistent() {
        let coord = BlockCoord::new(42, -17, 1000);
        let h1 = fnv1a_32(coord);
        let h2 = fnv1a_32(coord);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_spread_compact_roundtrip() {
        for x in [0, 1, 100, 1000, 0x1FFFFF] {
            let spread = spread_bits_3d(x);
            let back = compact_bits_3d(spread);
            assert_eq!(back, x & 0x1FFFFF, "Roundtrip failed for {}", x);
        }
    }

    #[test]
    fn test_morton_encode_decode_roundtrip() {
        let test_cases = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (100, 200, 300),
            (0x1FFFFF, 0x1FFFFF, 0x1FFFFF), // Max 21-bit values
        ];

        for (x, y, z) in test_cases {
            let code = morton_encode_3d(x, y, z);
            let (dx, dy, dz) = morton_decode_3d(code);
            assert_eq!((dx, dy, dz), (x, y, z), "Roundtrip failed for ({}, {}, {})", x, y, z);
        }
    }

    #[test]
    fn test_morton_preserves_locality() {
        // Adjacent coordinates should have similar Morton codes
        let c1 = morton_encode_3d(100, 100, 100);
        let c2 = morton_encode_3d(101, 100, 100);
        let c3 = morton_encode_3d(100, 101, 100);

        // Check that the codes differ only in the lower bits
        // (This is a weak test for locality, but validates basic behavior)
        assert_ne!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_morton_encode_decode_signed() {
        let test_coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 2, 3),
            BlockCoord::new(-1, -2, -3),
            BlockCoord::new(1000, -500, 250),
            BlockCoord::new(-1048576, 0, 1048575), // Near limits
        ];

        for coord in test_coords {
            let code = morton_encode_signed(coord);
            let back = morton_decode_signed(code);
            assert_eq!(back, coord, "Roundtrip failed for {:?}", coord);
        }
    }

    #[test]
    fn test_hash_to_index_in_range() {
        let capacity = 1024;
        let coords = [
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(100, -50, 25),
            BlockCoord::new(-1000, 500, -250),
        ];

        for coord in coords {
            let idx = hash_to_index(coord, capacity);
            assert!(idx < capacity, "Index {} out of range for capacity {}", idx, capacity);
        }
    }

    #[test]
    fn test_hash_distribution() {
        // Simple distribution test: hash many coords and check they spread across buckets
        let capacity = 256;
        let mut buckets = [0u32; 256];

        for x in -10..10 {
            for y in -10..10 {
                for z in -10..10 {
                    let coord = BlockCoord::new(x, y, z);
                    let idx = hash_to_index(coord, capacity);
                    buckets[idx] += 1;
                }
            }
        }

        // Check that no bucket is excessively full (crude test for distribution)
        let total = 20 * 20 * 20; // 8000 coords
        let expected_per_bucket = total / capacity as u32;
        let max_bucket = *buckets.iter().max().unwrap();

        // Allow up to 10x the expected average (very loose test)
        assert!(
            max_bucket < expected_per_bucket * 10,
            "Poor distribution: max bucket {} vs expected {}",
            max_bucket,
            expected_per_bucket
        );
    }
}
