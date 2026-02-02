//! Checkpoint save/load functionality for training state.
//!
//! Enables saving and resuming training sessions with full state preservation.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use burn::prelude::*;

use crate::config::TrainingConfig;
use crate::error::{NeuralAshError, Result};

/// Checkpoint metadata stored as JSON.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Current epoch.
    pub epoch: usize,
    /// Total training steps.
    pub total_steps: usize,
    /// Best loss achieved.
    pub best_loss: f32,
    /// Average loss.
    pub avg_loss: f32,
    /// Number of allocated blocks.
    pub num_blocks: usize,
    /// Grid dimension.
    pub grid_dim: u32,
    /// Cell size.
    pub cell_size: f32,
    /// Checkpoint version for compatibility.
    pub version: u32,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            epoch: 0,
            total_steps: 0,
            best_loss: f32::INFINITY,
            avg_loss: 0.0,
            num_blocks: 0,
            grid_dim: 8,
            cell_size: 0.05,
            version: 1,
        }
    }
}

impl CheckpointMetadata {
    /// Create metadata from training state.
    pub fn new(epoch: usize, total_steps: usize, best_loss: f32, avg_loss: f32) -> Self {
        Self {
            epoch,
            total_steps,
            best_loss,
            avg_loss,
            ..Default::default()
        }
    }

    /// Set grid info.
    pub fn with_grid_info(mut self, num_blocks: usize, grid_dim: u32, cell_size: f32) -> Self {
        self.num_blocks = num_blocks;
        self.grid_dim = grid_dim;
        self.cell_size = cell_size;
        self
    }

    /// Parse metadata from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        // Simple JSON parsing without serde
        let mut metadata = Self::default();

        for line in json.lines() {
            let line = line.trim();
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().trim_matches('"');
                let value = value.trim().trim_end_matches(',').trim_matches('"');

                match key {
                    "epoch" => {
                        metadata.epoch = value.parse().unwrap_or(0);
                    }
                    "total_steps" => {
                        metadata.total_steps = value.parse().unwrap_or(0);
                    }
                    "best_loss" => {
                        metadata.best_loss = value.parse().unwrap_or(f32::INFINITY);
                    }
                    "avg_loss" => {
                        metadata.avg_loss = value.parse().unwrap_or(0.0);
                    }
                    "num_blocks" => {
                        metadata.num_blocks = value.parse().unwrap_or(0);
                    }
                    "grid_dim" => {
                        metadata.grid_dim = value.parse().unwrap_or(8);
                    }
                    "cell_size" => {
                        metadata.cell_size = value.parse().unwrap_or(0.05);
                    }
                    "version" => {
                        metadata.version = value.parse().unwrap_or(1);
                    }
                    _ => {}
                }
            }
        }

        Ok(metadata)
    }

    /// Convert metadata to JSON string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{
  "version": {},
  "epoch": {},
  "total_steps": {},
  "best_loss": {},
  "avg_loss": {},
  "num_blocks": {},
  "grid_dim": {},
  "cell_size": {}
}}"#,
            self.version,
            self.epoch,
            self.total_steps,
            self.best_loss,
            self.avg_loss,
            self.num_blocks,
            self.grid_dim,
            self.cell_size
        )
    }
}

/// Save training checkpoint to a directory.
///
/// Creates the following files:
/// - `metadata.json`: Training progress and grid info
/// - `config.json`: Training configuration
/// - `embeddings.bin`: Raw embedding tensor data
/// - `blocks.bin`: Block coordinate data
///
/// # Arguments
/// * `dir` - Directory to save checkpoint to
/// * `embeddings` - Grid embeddings tensor
/// * `block_coords` - Block coordinates in allocation order
/// * `config` - Training configuration
/// * `metadata` - Checkpoint metadata
pub fn save_checkpoint<B: Backend>(
    dir: &Path,
    embeddings: &Tensor<B, 2>,
    block_coords: &[ash_core::BlockCoord],
    config: &TrainingConfig,
    metadata: &CheckpointMetadata,
) -> Result<()> {
    // Create directory if needed
    fs::create_dir_all(dir).map_err(|e| NeuralAshError::IoError(e.into()))?;

    // Save metadata
    let metadata_path = dir.join("metadata.json");
    let mut metadata_file =
        BufWriter::new(File::create(&metadata_path).map_err(|e| NeuralAshError::IoError(e.into()))?);
    metadata_file
        .write_all(metadata.to_json().as_bytes())
        .map_err(|e| NeuralAshError::IoError(e.into()))?;

    // Save config (simplified - just the essential fields)
    let config_path = dir.join("config.json");
    let config_json = format!(
        r#"{{
  "grid_dim": {},
  "cell_size": {},
  "capacity": {},
  "learning_rate": {},
  "surface_weight": {},
  "free_space_weight": {},
  "eikonal_weight": {}
}}"#,
        config.grid.grid_dim,
        config.grid.cell_size,
        config.grid.capacity,
        config.learning_rate,
        config.loss.surface_weight,
        config.loss.free_space_weight,
        config.loss.eikonal_weight
    );
    let mut config_file =
        BufWriter::new(File::create(&config_path).map_err(|e| NeuralAshError::IoError(e.into()))?);
    config_file
        .write_all(config_json.as_bytes())
        .map_err(|e| NeuralAshError::IoError(e.into()))?;

    // Save embeddings as raw f32 data
    let embeddings_path = dir.join("embeddings.bin");
    let embeddings_data = embeddings.to_data();
    let embeddings_vec: Vec<f32> = embeddings_data.to_vec().unwrap();
    let embeddings_bytes: Vec<u8> = embeddings_vec
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let mut embeddings_file =
        BufWriter::new(File::create(&embeddings_path).map_err(|e| NeuralAshError::IoError(e.into()))?);
    embeddings_file
        .write_all(&embeddings_bytes)
        .map_err(|e| NeuralAshError::IoError(e.into()))?;

    // Save block coordinates
    let blocks_path = dir.join("blocks.bin");
    let blocks_bytes: Vec<u8> = block_coords
        .iter()
        .flat_map(|b| {
            let mut bytes = Vec::with_capacity(12);
            bytes.extend_from_slice(&b.x.to_le_bytes());
            bytes.extend_from_slice(&b.y.to_le_bytes());
            bytes.extend_from_slice(&b.z.to_le_bytes());
            bytes
        })
        .collect();
    let mut blocks_file =
        BufWriter::new(File::create(&blocks_path).map_err(|e| NeuralAshError::IoError(e.into()))?);
    blocks_file
        .write_all(&blocks_bytes)
        .map_err(|e| NeuralAshError::IoError(e.into()))?;

    log::info!(
        "Saved checkpoint to {:?} (epoch {}, {} blocks)",
        dir,
        metadata.epoch,
        block_coords.len()
    );

    Ok(())
}

/// Load training checkpoint from a directory.
///
/// # Arguments
/// * `dir` - Directory containing checkpoint files
/// * `device` - Burn device for tensor creation
///
/// # Returns
/// (embeddings, block_coords, metadata)
pub fn load_checkpoint<B: Backend>(
    dir: &Path,
    device: &B::Device,
) -> Result<(Tensor<B, 2>, Vec<ash_core::BlockCoord>, CheckpointMetadata)> {
    // Load metadata
    let metadata_path = dir.join("metadata.json");
    let mut metadata_file =
        BufReader::new(File::open(&metadata_path).map_err(|e| NeuralAshError::IoError(e.into()))?);
    let mut metadata_str = String::new();
    metadata_file
        .read_to_string(&mut metadata_str)
        .map_err(|e| NeuralAshError::IoError(e.into()))?;
    let metadata = CheckpointMetadata::from_json(&metadata_str)?;

    // Load block coordinates
    let blocks_path = dir.join("blocks.bin");
    let mut blocks_file =
        BufReader::new(File::open(&blocks_path).map_err(|e| NeuralAshError::IoError(e.into()))?);
    let mut blocks_bytes = Vec::new();
    blocks_file
        .read_to_end(&mut blocks_bytes)
        .map_err(|e| NeuralAshError::IoError(e.into()))?;

    let block_coords: Vec<ash_core::BlockCoord> = blocks_bytes
        .chunks(12)
        .map(|chunk| {
            let x = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let y = i32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            let z = i32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
            ash_core::BlockCoord::new(x, y, z)
        })
        .collect();

    // Load embeddings
    let embeddings_path = dir.join("embeddings.bin");
    let mut embeddings_file =
        BufReader::new(File::open(&embeddings_path).map_err(|e| NeuralAshError::IoError(e.into()))?);
    let mut embeddings_bytes = Vec::new();
    embeddings_file
        .read_to_end(&mut embeddings_bytes)
        .map_err(|e| NeuralAshError::IoError(e.into()))?;

    let embeddings_vec: Vec<f32> = embeddings_bytes
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Compute dimensions
    let cells_per_block = (metadata.grid_dim * metadata.grid_dim * metadata.grid_dim) as usize;
    let total_cells = block_coords.len() * cells_per_block;
    let n_features = embeddings_vec.len() / total_cells;

    if n_features == 0 || embeddings_vec.len() != total_cells * n_features {
        return Err(NeuralAshError::InvalidData(format!(
            "Embedding size mismatch: {} values for {} cells",
            embeddings_vec.len(),
            total_cells
        )));
    }

    let embeddings =
        Tensor::from_data(TensorData::new(embeddings_vec, [total_cells, n_features]), device);

    log::info!(
        "Loaded checkpoint from {:?} (epoch {}, {} blocks)",
        dir,
        metadata.epoch,
        block_coords.len()
    );

    Ok((embeddings, block_coords, metadata))
}

/// Check if a valid checkpoint exists at the given path.
pub fn checkpoint_exists(dir: &Path) -> bool {
    dir.join("metadata.json").exists()
        && dir.join("embeddings.bin").exists()
        && dir.join("blocks.bin").exists()
}

/// Get the latest checkpoint from a series of numbered checkpoints.
///
/// Looks for directories named `checkpoint_N` where N is an epoch number.
pub fn find_latest_checkpoint(base_dir: &Path) -> Option<std::path::PathBuf> {
    let mut latest_epoch = 0;
    let mut latest_path = None;

    if let Ok(entries) = fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Some(epoch_str) = name.strip_prefix("checkpoint_") {
                        if let Ok(epoch) = epoch_str.parse::<usize>() {
                            if epoch > latest_epoch && checkpoint_exists(&path) {
                                latest_epoch = epoch;
                                latest_path = Some(path);
                            }
                        }
                    }
                }
            }
        }
    }

    latest_path
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use tempfile::TempDir;

    type TestBackend = NdArray;

    #[test]
    fn test_metadata_json_roundtrip() {
        let metadata = CheckpointMetadata::new(10, 1000, 0.05, 0.08)
            .with_grid_info(50, 8, 0.1);

        let json = metadata.to_json();
        let parsed = CheckpointMetadata::from_json(&json).unwrap();

        assert_eq!(parsed.epoch, 10);
        assert_eq!(parsed.total_steps, 1000);
        assert!((parsed.best_loss - 0.05).abs() < 1e-6);
        assert!((parsed.avg_loss - 0.08).abs() < 1e-6);
        assert_eq!(parsed.num_blocks, 50);
        assert_eq!(parsed.grid_dim, 8);
        assert!((parsed.cell_size - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("test_checkpoint");

        let device: <TestBackend as Backend>::Device = Default::default();

        // Create test data: 1 block with 2^3 = 8 cells, 1 feature each
        let embeddings = Tensor::<TestBackend, 2>::from_data(
            [[1.0f32], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]],
            &device,
        );
        let block_coords = vec![
            ash_core::BlockCoord::new(0, 0, 0),
        ];

        use crate::config::{DiffGridConfig, PointNetEncoderConfig, SdfDecoderConfig, SdfLossConfig};
        let config = crate::config::TrainingConfig::new(
            DiffGridConfig::new(2, 0.1).with_capacity(10), // 2^3 = 8 cells
            PointNetEncoderConfig::new(256),
            SdfDecoderConfig::new(256),
            SdfLossConfig::default(),
        );

        let metadata = CheckpointMetadata::new(5, 500, 0.1, 0.15)
            .with_grid_info(1, 2, 0.1);

        // Save
        save_checkpoint(&checkpoint_path, &embeddings, &block_coords, &config, &metadata).unwrap();

        // Verify files exist
        assert!(checkpoint_exists(&checkpoint_path));

        // Load
        let (loaded_embeddings, loaded_blocks, loaded_metadata) =
            load_checkpoint::<TestBackend>(&checkpoint_path, &device).unwrap();

        // Verify
        assert_eq!(loaded_blocks.len(), 1);
        assert_eq!(loaded_blocks[0], ash_core::BlockCoord::new(0, 0, 0));
        assert_eq!(loaded_metadata.epoch, 5);
        assert_eq!(loaded_metadata.total_steps, 500);

        let orig_data: Vec<f32> = embeddings.to_data().to_vec().unwrap();
        let loaded_data: Vec<f32> = loaded_embeddings.to_data().to_vec().unwrap();
        assert_eq!(orig_data.len(), loaded_data.len());
        for (a, b) in orig_data.iter().zip(loaded_data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_find_latest_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create some checkpoint directories
        for epoch in [5, 10, 3, 15] {
            let dir = base_path.join(format!("checkpoint_{}", epoch));
            fs::create_dir_all(&dir).unwrap();
            // Create minimal valid checkpoint files
            fs::write(dir.join("metadata.json"), "{}").unwrap();
            fs::write(dir.join("embeddings.bin"), "").unwrap();
            fs::write(dir.join("blocks.bin"), "").unwrap();
        }

        let latest = find_latest_checkpoint(base_path);
        assert!(latest.is_some());
        assert!(latest.unwrap().ends_with("checkpoint_15"));
    }
}
