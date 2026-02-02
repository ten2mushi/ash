//! End-to-end integration tests.

use burn::backend::{Autodiff, NdArray};

use neural_ash::{
    config::{DiffGridConfig, TrainingConfig, PointNetEncoderConfig, SdfDecoderConfig, SdfLossConfig},
    export::{discretize_grid, export_to_file},
    training::NeuralSdfTrainer,
    BlockCoord, Point3,
};

type TestBackend = Autodiff<NdArray>;

fn make_test_config(grid: DiffGridConfig) -> TrainingConfig {
    TrainingConfig::new(
        grid,
        PointNetEncoderConfig::new(256),
        SdfDecoderConfig::new(256),
        SdfLossConfig::default(),
    )
}

#[test]
fn test_train_export_roundtrip() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // Create a small trainer
    let config = make_test_config(DiffGridConfig::new(4, 0.1).with_capacity(100))
        .with_batch_size(64);

    let mut trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

    // Create a simple point cloud (sphere surface)
    let points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0 * std::f32::consts::PI * 2.0;
            Point3::new(t.cos() * 0.2, t.sin() * 0.2, 0.0)
        })
        .collect();

    // Initialize from points
    trainer.init_from_points(&points);
    assert!(trainer.grid.num_blocks() > 0);

    // Export the grid
    let grid = trainer.export_grid();
    assert!(grid.num_blocks() > 0);

    // Query should work
    let query_result = grid.query(Point3::new(0.0, 0.0, 0.0));
    // May or may not have a result depending on block allocation
    if let Some([sdf]) = query_result {
        assert!(sdf.is_finite());
    }
}

#[test]
fn test_multiple_training_steps() {
    use neural_ash::training::SdfBatch;
    use burn::prelude::*;

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let config = make_test_config(DiffGridConfig::new(4, 0.1).with_capacity(100))
        .with_batch_size(32);

    let mut trainer = NeuralSdfTrainer::<TestBackend>::new(config, &device);

    // Initialize
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.2, 0.2, 0.2),
    ];
    trainer.init_from_points(&points);

    // Create a batch
    let batch = SdfBatch::new(
        Tensor::zeros([16, 3], &device),
        Tensor::full([16, 3], 0.5, &device),
        Tensor::zeros([16, 3], &device),
    );

    // Run multiple training steps
    let mut prev_loss = f32::INFINITY;
    for i in 0..3 {
        let output = trainer.train_step(&batch);
        let loss = output.loss_value();

        assert!(loss.is_finite(), "Loss became non-finite at step {}", i);

        // Loss should generally decrease (though not guaranteed for every step)
        // Just check it's not exploding
        assert!(
            loss < prev_loss * 100.0,
            "Loss exploded at step {}: {} -> {}",
            i,
            prev_loss,
            loss
        );

        prev_loss = loss;
    }
}

#[test]
fn test_discretize_preserves_structure() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(4, 0.1).with_capacity(50);
    let mut grid = neural_ash::DiffSdfGrid::<TestBackend, 1>::new(config, &device);

    // Allocate specific blocks
    let blocks = vec![
        BlockCoord::new(0, 0, 0),
        BlockCoord::new(1, 0, 0),
        BlockCoord::new(0, 1, 0),
        BlockCoord::new(-1, -1, -1),
    ];

    for block in &blocks {
        grid.allocate_block(*block).unwrap();
    }

    // Discretize
    let memory_grid = discretize_grid(&grid);

    // Verify structure is preserved
    assert_eq!(memory_grid.num_blocks(), blocks.len());
    for block in &blocks {
        assert!(
            memory_grid.has_block(*block),
            "Block {:?} missing after discretization",
            block
        );
    }
}

#[test]
fn test_export_file_io() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(4, 0.1).with_capacity(10);
    let mut grid = neural_ash::DiffSdfGrid::<TestBackend, 1>::new(config, &device);

    grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();

    let memory_grid = discretize_grid(&grid);

    // Export to temp file
    let temp_path = std::env::temp_dir().join("neural_ash_test_integration.ash");
    let result = export_to_file(&memory_grid, &temp_path);
    assert!(result.is_ok(), "Export failed: {:?}", result.err());

    // Verify file exists
    assert!(temp_path.exists(), "File was not created");

    // Load it back using ash_io
    let loaded = ash_io::load_from_file::<1, _>(&temp_path);
    assert!(loaded.is_ok(), "Load failed: {:?}", loaded.err());

    let loaded_grid = loaded.unwrap();
    assert_eq!(loaded_grid.num_blocks(), 1);
    assert!(loaded_grid.has_block(BlockCoord::new(0, 0, 0)));

    // Clean up
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_neural_network_forward_pass() {
    use neural_ash::nn::{PointNetEncoder, SdfDecoder};
    use neural_ash::config::{PointNetEncoderConfig, SdfDecoderConfig};
    use burn::prelude::*;

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // Create encoder and decoder
    let encoder_config = PointNetEncoderConfig::new(64).with_hidden_dims(vec![32, 64]);
    let decoder_config = SdfDecoderConfig::new(64).with_hidden_dims(vec![64, 32]);

    let encoder = PointNetEncoder::<NdArray>::new(&encoder_config, &device);
    let decoder = SdfDecoder::<NdArray>::new(&decoder_config, &device);

    // Test encoder
    let point_cloud = Tensor::zeros([2, 50, 3], &device);
    let latent = encoder.forward(point_cloud);
    assert_eq!(latent.dims(), [2, 64]);

    // Test decoder
    let query_points = Tensor::zeros([2, 3], &device);
    let sdf = decoder.forward(latent, query_points);
    assert_eq!(sdf.dims(), [2, 1]);

    // Values should be finite
    let data = sdf.to_data();
    let values: Vec<f32> = data.to_vec().unwrap();
    for v in values {
        assert!(v.is_finite());
    }
}

#[test]
fn test_point_cloud_operations() {
    use neural_ash::data::PointCloud;

    let mut cloud = PointCloud::new(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    ]);

    // Test bounding box
    let (min, max) = cloud.bounding_box().unwrap();
    assert_eq!(min, Point3::new(0.0, 0.0, 0.0));
    assert_eq!(max, Point3::new(1.0, 1.0, 0.0));

    // Test centroid
    let centroid = cloud.centroid().unwrap();
    assert!((centroid.x - 0.5).abs() < 1e-6);
    assert!((centroid.y - 0.5).abs() < 1e-6);

    // Test centering
    cloud.center();
    let new_centroid = cloud.centroid().unwrap();
    assert!(new_centroid.length() < 1e-6);
}
