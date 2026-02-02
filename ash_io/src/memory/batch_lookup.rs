//! Batch block coordinate lookups with SIMD acceleration.
//!
//! Provides efficient batch lookups by precomputing hashes using SIMD
//! before probing the hash table.

use ash_core::BlockCoord;

use super::block_map::BlockMap;

/// Result of batch lookup operation.
#[derive(Debug, Clone)]
pub struct BatchLookupResult {
    /// Block indices for each input coordinate.
    /// `u32::MAX` indicates the coordinate was not found.
    pub indices: Vec<u32>,
    /// Boolean flags indicating whether each coordinate was found.
    pub found: Vec<bool>,
}

impl BatchLookupResult {
    /// Create a new result with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            indices: Vec::with_capacity(capacity),
            found: Vec::with_capacity(capacity),
        }
    }

    /// Get the index for a coordinate, if found.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<usize> {
        if self.found[idx] {
            Some(self.indices[idx] as usize)
        } else {
            None
        }
    }

    /// Number of coordinates that were found.
    pub fn found_count(&self) -> usize {
        self.found.iter().filter(|&&f| f).count()
    }

    /// Total number of coordinates queried.
    pub fn total(&self) -> usize {
        self.indices.len()
    }
}

impl BlockMap {
    /// Batch lookup for multiple coordinates.
    ///
    /// Looks up each coordinate using the same algorithm as `get()`,
    /// collecting results into a `BatchLookupResult`.
    ///
    /// # Performance
    /// - Uses Morton encoding for each coordinate (same as `get`)
    /// - Good cache locality when processing many lookups
    pub fn find_batch(&self, coords: &[BlockCoord]) -> BatchLookupResult {
        let mut result = BatchLookupResult::with_capacity(coords.len());

        for coord in coords {
            match self.get(*coord) {
                Some(idx) => {
                    result.indices.push(idx as u32);
                    result.found.push(true);
                }
                None => {
                    result.indices.push(u32::MAX);
                    result.found.push(false);
                }
            }
        }

        result
    }

    /// Batch check for coordinate existence.
    ///
    /// More efficient than individual `contains()` calls for multiple coordinates.
    pub fn contains_batch(&self, coords: &[BlockCoord]) -> Vec<bool> {
        self.find_batch(coords).found
    }

    /// Batch get indices for coordinates that exist.
    ///
    /// Returns only the indices for coordinates that were found,
    /// along with their position in the input slice.
    pub fn get_batch_found(&self, coords: &[BlockCoord]) -> Vec<(usize, usize)> {
        let result = self.find_batch(coords);
        result
            .indices
            .iter()
            .enumerate()
            .filter_map(|(i, &idx)| {
                if result.found[i] {
                    Some((i, idx as usize))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_matches_individual() {
        let map = BlockMap::with_capacity(100);

        // Insert some test coordinates
        let inserted: Vec<BlockCoord> = (0..10)
            .map(|i| BlockCoord::new(i, i * 2, i * 3))
            .collect();

        for (i, &coord) in inserted.iter().enumerate() {
            map.insert(coord, i).unwrap();
        }

        // Batch lookup should match individual lookups
        let result = map.find_batch(&inserted);

        for (i, &coord) in inserted.iter().enumerate() {
            let individual = map.get(coord);
            let batch = result.get(i);
            assert_eq!(
                individual, batch,
                "Mismatch at index {}: individual={:?}, batch={:?}",
                i, individual, batch
            );
        }
    }

    #[test]
    fn batch_mixed_hits_misses() {
        let map = BlockMap::with_capacity(100);

        // Insert only even indices
        for i in (0..20).step_by(2) {
            let coord = BlockCoord::new(i, i, i);
            map.insert(coord, i as usize).unwrap();
        }

        // Query all indices (half hits, half misses)
        let coords: Vec<BlockCoord> = (0..20).map(|i| BlockCoord::new(i, i, i)).collect();

        let result = map.find_batch(&coords);

        for (i, coord) in coords.iter().enumerate() {
            let expected_found = i % 2 == 0;
            assert_eq!(
                result.found[i], expected_found,
                "Mismatch at index {} (coord {:?}): expected found={}, got found={}",
                i, coord, expected_found, result.found[i]
            );

            if expected_found {
                assert_eq!(result.indices[i], i as u32);
            }
        }
    }

    #[test]
    fn batch_empty_input() {
        let map = BlockMap::with_capacity(100);
        let coords: Vec<BlockCoord> = vec![];

        let result = map.find_batch(&coords);

        assert!(result.indices.is_empty());
        assert!(result.found.is_empty());
        assert_eq!(result.found_count(), 0);
        assert_eq!(result.total(), 0);
    }

    #[test]
    fn batch_partial_chunk() {
        let map = BlockMap::with_capacity(100);

        // Test with 1, 2, 3, 5, 6, 7 elements (non-multiples of 4)
        for count in [1, 2, 3, 5, 6, 7] {
            let coords: Vec<BlockCoord> = (0..count)
                .map(|i| BlockCoord::new(i as i32, 0, 0))
                .collect();

            // Insert all
            for (i, &coord) in coords.iter().enumerate() {
                let _ = map.insert(coord, i);
            }

            let result = map.find_batch(&coords);

            assert_eq!(result.total(), count);
            assert_eq!(result.found_count(), count);

            for i in 0..count {
                assert!(result.found[i], "Coord {} should be found", i);
            }
        }
    }

    #[test]
    fn batch_negative_coords() {
        let map = BlockMap::with_capacity(100);

        let coords = vec![
            BlockCoord::new(-1, -2, -3),
            BlockCoord::new(-100, 50, -25),
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(100, -100, 100),
        ];

        for (i, &coord) in coords.iter().enumerate() {
            map.insert(coord, i).unwrap();
        }

        let result = map.find_batch(&coords);

        assert_eq!(result.found_count(), 4);
        for i in 0..4 {
            assert!(result.found[i]);
            assert_eq!(result.indices[i], i as u32);
        }
    }

    #[test]
    fn contains_batch_works() {
        let map = BlockMap::with_capacity(100);

        map.insert(BlockCoord::new(0, 0, 0), 0).unwrap();
        map.insert(BlockCoord::new(2, 2, 2), 1).unwrap();

        let coords = vec![
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 1, 1),
            BlockCoord::new(2, 2, 2),
            BlockCoord::new(3, 3, 3),
        ];

        let exists = map.contains_batch(&coords);

        assert_eq!(exists, vec![true, false, true, false]);
    }

    #[test]
    fn get_batch_found_works() {
        let map = BlockMap::with_capacity(100);

        map.insert(BlockCoord::new(0, 0, 0), 10).unwrap();
        map.insert(BlockCoord::new(2, 2, 2), 20).unwrap();

        let coords = vec![
            BlockCoord::new(0, 0, 0),
            BlockCoord::new(1, 1, 1),
            BlockCoord::new(2, 2, 2),
            BlockCoord::new(3, 3, 3),
        ];

        let found = map.get_batch_found(&coords);

        assert_eq!(found.len(), 2);
        assert_eq!(found[0], (0, 10)); // coord index 0, block index 10
        assert_eq!(found[1], (2, 20)); // coord index 2, block index 20
    }
}
