//! Lock-free hash map for block coordinate lookups.
//!
//! Provides O(1) average-case lookup with lock-free concurrent read access.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use ash_core::{morton_encode_signed, BlockCoord};

use crate::error::{AshIoError, Result};

/// Entry state for the lock-free hash table.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryState {
    /// Slot is empty and available.
    Empty = 0,
    /// Slot is being inserted (transient state).
    Busy = 1,
    /// Slot contains a valid entry.
    Occupied = 2,
}

/// Packed atomic entry for the lock-free hash table.
///
/// Layout: `[state:2][index:30][morton_hash:32]` = 64 bits
///
/// - Bits 0-31: Morton hash (32 bits of the full 64-bit Morton code)
/// - Bits 32-61: Block index (30 bits, supports up to ~1 billion blocks)
/// - Bits 62-63: Entry state (2 bits)
#[repr(transparent)]
pub struct AtomicEntry(AtomicU64);

impl AtomicEntry {
    const HASH_MASK: u64 = 0xFFFF_FFFF;
    const INDEX_SHIFT: u32 = 32;
    const INDEX_MASK: u64 = 0x3FFF_FFFF; // 30 bits
    const STATE_SHIFT: u32 = 62;

    /// Create a new empty entry.
    #[inline]
    pub const fn empty() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Pack state, index, and hash into a u64.
    #[inline]
    fn pack(state: EntryState, index: u32, hash: u32) -> u64 {
        let state_bits = (state as u64) << Self::STATE_SHIFT;
        let index_bits = ((index as u64) & Self::INDEX_MASK) << Self::INDEX_SHIFT;
        let hash_bits = hash as u64;
        state_bits | index_bits | hash_bits
    }

    /// Extract state from a packed value.
    #[inline]
    fn unpack_state(value: u64) -> EntryState {
        match value >> Self::STATE_SHIFT {
            0 => EntryState::Empty,
            1 => EntryState::Busy,
            _ => EntryState::Occupied,
        }
    }

    /// Extract index from a packed value.
    #[inline]
    fn unpack_index(value: u64) -> u32 {
        ((value >> Self::INDEX_SHIFT) & Self::INDEX_MASK) as u32
    }

    /// Extract hash from a packed value.
    #[inline]
    fn unpack_hash(value: u64) -> u32 {
        (value & Self::HASH_MASK) as u32
    }

    /// Load the current value with Acquire ordering.
    #[inline]
    pub fn load(&self) -> (EntryState, u32, u32) {
        let value = self.0.load(Ordering::Acquire);
        (
            Self::unpack_state(value),
            Self::unpack_index(value),
            Self::unpack_hash(value),
        )
    }

    /// Attempt to transition from Empty to Busy (claim the slot).
    #[inline]
    pub fn try_claim(&self, index: u32, hash: u32) -> bool {
        let empty = Self::pack(EntryState::Empty, 0, 0);
        let busy = Self::pack(EntryState::Busy, index, hash);
        self.0
            .compare_exchange(empty, busy, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Commit the entry (transition from Busy to Occupied).
    #[inline]
    pub fn commit(&self, index: u32, hash: u32) {
        let occupied = Self::pack(EntryState::Occupied, index, hash);
        self.0.store(occupied, Ordering::Release);
    }
}

/// Lock-free hash map for block coordinate → index lookup.
///
/// Uses open-addressed hashing with linear probing.
/// Optimized for read-heavy workloads typical in SDF queries.
pub struct BlockMap {
    /// Hash table entries.
    entries: Box<[AtomicEntry]>,
    /// Table capacity (should be ~2x expected blocks for good performance).
    capacity: usize,
    /// Number of occupied entries.
    count: AtomicUsize,
}

impl BlockMap {
    /// Create a new block map with the given capacity.
    ///
    /// The actual table size will be the capacity to enable efficient probing.
    pub fn with_capacity(capacity: usize) -> Self {
        let entries = (0..capacity)
            .map(|_| AtomicEntry::empty())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            entries,
            capacity,
            count: AtomicUsize::new(0),
        }
    }

    /// Look up a block coordinate and return its index.
    ///
    /// # Performance
    /// - O(1) average case
    /// - O(capacity) worst case (full table)
    #[inline]
    pub fn get(&self, coord: BlockCoord) -> Option<usize> {
        let morton = morton_encode_signed(coord);
        let hash = morton as u32; // Use lower 32 bits
        self.probe_with_hash(hash, morton as usize)
    }

    /// Probe the hash table with a precomputed hash value.
    ///
    /// This is used for batch lookups where hashes are precomputed with SIMD.
    ///
    /// # Arguments
    /// * `hash` - The 32-bit hash value (lower bits of Morton code)
    /// * `start_idx` - The starting index for probing (typically morton % capacity)
    ///
    /// # Returns
    /// The block index if found, or None if not present
    #[inline]
    pub fn probe_with_hash(&self, hash: u32, start_idx: usize) -> Option<usize> {
        let mut idx = start_idx % self.capacity;

        for _ in 0..self.capacity {
            let (state, entry_index, entry_hash) = self.entries[idx].load();
            match state {
                EntryState::Empty => return None,
                EntryState::Occupied if entry_hash == hash => {
                    return Some(entry_index as usize);
                }
                _ => {
                    idx = (idx + 1) % self.capacity;
                }
            }
        }
        None
    }

    /// Insert a block coordinate with its index.
    ///
    /// # Returns
    /// - `Ok(())` on success
    /// - `Err(DuplicateBlock)` if the coordinate already exists
    /// - `Err(HashTableFull)` if the table is full
    pub fn insert(&self, coord: BlockCoord, index: usize) -> Result<()> {
        let morton = morton_encode_signed(coord);
        let hash = morton as u32;
        let mut idx = (morton as usize) % self.capacity;

        for _ in 0..self.capacity {
            let (state, _, entry_hash) = self.entries[idx].load();
            match state {
                EntryState::Empty => {
                    // Try to claim this slot
                    if self.entries[idx].try_claim(index as u32, hash) {
                        // Successfully claimed, commit the entry
                        self.entries[idx].commit(index as u32, hash);
                        self.count.fetch_add(1, Ordering::Relaxed);
                        return Ok(());
                    }
                    // CAS failed, someone else claimed it, retry at same slot
                }
                EntryState::Occupied if entry_hash == hash => {
                    return Err(AshIoError::DuplicateBlock {
                        x: coord.x,
                        y: coord.y,
                        z: coord.z,
                    });
                }
                _ => {
                    idx = (idx + 1) % self.capacity;
                }
            }
        }
        Err(AshIoError::HashTableFull)
    }

    /// Check if the map contains a coordinate.
    #[inline]
    pub fn contains(&self, coord: BlockCoord) -> bool {
        self.get(coord).is_some()
    }

    /// Get the number of entries in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the capacity of the map.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// Safety: BlockMap uses only atomic operations
unsafe impl Send for BlockMap {}
unsafe impl Sync for BlockMap {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_map_insert_and_get() {
        let map = BlockMap::with_capacity(100);

        let coord = BlockCoord::new(1, 2, 3);
        assert!(map.insert(coord, 0).is_ok());
        assert_eq!(map.get(coord), Some(0));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_block_map_duplicate() {
        let map = BlockMap::with_capacity(100);

        let coord = BlockCoord::new(1, 2, 3);
        assert!(map.insert(coord, 0).is_ok());
        assert!(matches!(
            map.insert(coord, 1),
            Err(AshIoError::DuplicateBlock { x: 1, y: 2, z: 3 })
        ));
    }

    #[test]
    fn test_block_map_not_found() {
        let map = BlockMap::with_capacity(100);

        let coord = BlockCoord::new(1, 2, 3);
        assert_eq!(map.get(coord), None);
    }

    #[test]
    fn test_block_map_many_entries() {
        let map = BlockMap::with_capacity(1000);

        // Insert many entries
        for i in 0..100 {
            let coord = BlockCoord::new(i, i * 2, i * 3);
            assert!(map.insert(coord, i as usize).is_ok());
        }

        // Verify all can be found
        for i in 0..100 {
            let coord = BlockCoord::new(i, i * 2, i * 3);
            assert_eq!(map.get(coord), Some(i as usize));
        }
    }

    #[test]
    fn test_block_map_negative_coords() {
        let map = BlockMap::with_capacity(100);

        let coords = [
            BlockCoord::new(-1, -2, -3),
            BlockCoord::new(-100, 50, -25),
            BlockCoord::new(0, 0, 0),
        ];

        for (i, &coord) in coords.iter().enumerate() {
            assert!(map.insert(coord, i).is_ok());
        }

        for (i, &coord) in coords.iter().enumerate() {
            assert_eq!(map.get(coord), Some(i));
        }
    }

    #[test]
    fn test_atomic_entry_pack_unpack() {
        let state = EntryState::Occupied;
        let index = 12345u32;
        let hash = 0xDEADBEEFu32;

        let packed = AtomicEntry::pack(state, index, hash);
        assert_eq!(AtomicEntry::unpack_state(packed), state);
        assert_eq!(AtomicEntry::unpack_index(packed), index);
        assert_eq!(AtomicEntry::unpack_hash(packed), hash);
    }
}
