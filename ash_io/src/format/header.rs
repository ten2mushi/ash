//! ASH file format header definition.

/// Magic bytes for .ash file format.
pub const ASH_MAGIC: [u8; 4] = *b"ASHG";

/// Header size in bytes.
pub const HEADER_SIZE: usize = 32;

/// ASH file header.
///
/// Layout (32 bytes total):
/// - Bytes 0-3: Magic "ASHG"
/// - Bytes 4-7: cell_size (f32 LE)
/// - Bytes 8-11: grid_dim (u32 LE)
/// - Bytes 12-15: num_blocks (u32 LE)
/// - Bytes 16-17: feature_dim (u16 LE)
/// - Bytes 18-19: flags (u16 LE)
/// - Bytes 20-31: reserved (12 bytes)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AshHeader {
    /// Magic bytes "ASHG".
    pub magic: [u8; 4],
    /// Cell size in world units.
    pub cell_size: f32,
    /// Number of cells per axis per block.
    pub grid_dim: u32,
    /// Number of blocks in the file.
    pub num_blocks: u32,
    /// Number of features per cell.
    pub feature_dim: u16,
    /// Flags (reserved for future use).
    pub flags: u16,
    /// Reserved bytes for future expansion.
    pub reserved: [u8; 12],
}

impl AshHeader {
    /// Create a new header with the given parameters.
    pub fn new(cell_size: f32, grid_dim: u32, num_blocks: u32, feature_dim: u16) -> Self {
        Self {
            magic: ASH_MAGIC,
            cell_size,
            grid_dim,
            num_blocks,
            feature_dim,
            flags: 0,
            reserved: [0; 12],
        }
    }

    /// Validate the header magic bytes.
    pub fn is_valid(&self) -> bool {
        self.magic == ASH_MAGIC
    }

    /// Serialize the header to a byte array.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];

        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.cell_size.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.grid_dim.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.num_blocks.to_le_bytes());
        bytes[16..18].copy_from_slice(&self.feature_dim.to_le_bytes());
        bytes[18..20].copy_from_slice(&self.flags.to_le_bytes());
        bytes[20..32].copy_from_slice(&self.reserved);

        bytes
    }

    /// Deserialize a header from a byte array.
    pub fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Self {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);

        let cell_size = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let grid_dim = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let num_blocks = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let feature_dim = u16::from_le_bytes([bytes[16], bytes[17]]);
        let flags = u16::from_le_bytes([bytes[18], bytes[19]]);

        let mut reserved = [0u8; 12];
        reserved.copy_from_slice(&bytes[20..32]);

        Self {
            magic,
            cell_size,
            grid_dim,
            num_blocks,
            feature_dim,
            flags,
            reserved,
        }
    }
}

/// Compute the expected file size for a grid.
///
/// Useful for progress reporting or pre-allocation.
pub fn compute_file_size(num_blocks: usize, grid_dim: u32, feature_dim: usize) -> usize {
    let cells_per_block = (grid_dim as usize).pow(3);
    // Header + coords + (feature_dim * blocks * cells_per_block * sizeof(f32))
    HEADER_SIZE + num_blocks * 12 + feature_dim * num_blocks * cells_per_block * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = AshHeader::new(0.1, 8, 100, 4);
        let bytes = header.to_bytes();
        let restored = AshHeader::from_bytes(&bytes);

        assert_eq!(header, restored);
    }

    #[test]
    fn test_header_magic() {
        let header = AshHeader::new(0.1, 8, 100, 1);
        assert!(header.is_valid());

        let mut bad_header = header;
        bad_header.magic = *b"BADM";
        assert!(!bad_header.is_valid());
    }

    #[test]
    fn test_compute_file_size_n1() {
        // N=1, 8 blocks, 8³=512 cells per block
        let size = compute_file_size(8, 8, 1);
        // Header: 32 bytes
        // Coords: 8 * 12 = 96 bytes
        // Values: 1 * 8 * 512 * 4 = 16384 bytes
        // Total: 16512 bytes
        assert_eq!(size, 32 + 96 + 16384);
    }

    #[test]
    fn test_compute_file_size_n4() {
        // N=4, 8 blocks, 8³=512 cells per block
        let size = compute_file_size(8, 8, 4);
        // Header: 32 bytes
        // Coords: 8 * 12 = 96 bytes
        // Values: 4 * 8 * 512 * 4 = 65536 bytes
        // Total: 65664 bytes
        assert_eq!(size, 32 + 96 + 65536);
    }
}
