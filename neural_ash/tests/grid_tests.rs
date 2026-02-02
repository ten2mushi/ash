//! Integration tests for the differentiable grid.

use burn::backend::{Autodiff, NdArray};

use neural_ash::{
    config::DiffGridConfig,
    grid::DiffSdfGrid,
    BlockCoord, Point3,
};

type TestBackend = Autodiff<NdArray>;

#[test]
fn test_grid_basic_operations() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(8, 0.1).with_capacity(100);
    let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

    // Initially empty
    assert_eq!(grid.num_blocks(), 0);

    // Allocate blocks
    let idx0 = grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();
    let idx1 = grid.allocate_block(BlockCoord::new(1, 0, 0)).unwrap();
    let idx2 = grid.allocate_block(BlockCoord::new(0, 1, 0)).unwrap();

    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(idx2, 2);
    assert_eq!(grid.num_blocks(), 3);

    // Check block existence
    assert!(grid.has_block(BlockCoord::new(0, 0, 0)));
    assert!(grid.has_block(BlockCoord::new(1, 0, 0)));
    assert!(grid.has_block(BlockCoord::new(0, 1, 0)));
    assert!(!grid.has_block(BlockCoord::new(1, 1, 0)));
}

#[test]
fn test_grid_spatial_init() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(8, 0.1).with_capacity(1000);
    let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

    // Create points in a specific region
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.5, 0.5, 0.5),
        Point3::new(1.0, 1.0, 1.0),
    ];

    grid.spatial_init(&points, 0.2);

    // Should have allocated blocks around the points
    assert!(grid.num_blocks() > 0);

    // Origin block should exist
    assert!(grid.has_block(BlockCoord::new(0, 0, 0)));
}

#[test]
fn test_grid_query() {
    use burn::prelude::*;

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(4, 0.1).with_capacity(100);
    let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

    // Allocate a block
    grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();

    // Query within the block
    let query_points = Tensor::<TestBackend, 2>::from_data(
        [[0.1f32, 0.1, 0.1], [0.2, 0.2, 0.2]],
        &device,
    );

    let result = grid.query(query_points);
    assert_eq!(result.dims(), [2, 1]);

    // Results should be the sentinel value since embeddings are initialized to it
    let data = result.to_data();
    let values: Vec<f32> = data.to_vec().unwrap();
    for v in values {
        assert!(v.is_finite());
    }
}

#[test]
fn test_grid_to_memory_grid() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(4, 0.1).with_capacity(10);
    let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

    // Allocate some blocks
    grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();
    grid.allocate_block(BlockCoord::new(1, 0, 0)).unwrap();

    // Convert to InMemoryGrid
    let memory_grid = grid.to_memory_grid();

    assert_eq!(memory_grid.num_blocks(), 2);
    assert!(memory_grid.has_block(BlockCoord::new(0, 0, 0)));
    assert!(memory_grid.has_block(BlockCoord::new(1, 0, 0)));
}

#[test]
fn test_grid_negative_coords() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(4, 0.1).with_capacity(100);
    let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

    // Allocate blocks with negative coordinates
    grid.allocate_block(BlockCoord::new(-1, -1, -1)).unwrap();
    grid.allocate_block(BlockCoord::new(-2, 0, 1)).unwrap();

    assert_eq!(grid.num_blocks(), 2);
    assert!(grid.has_block(BlockCoord::new(-1, -1, -1)));
    assert!(grid.has_block(BlockCoord::new(-2, 0, 1)));
}

#[test]
fn test_grid_capacity_limit() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(4, 0.1).with_capacity(3);
    let mut grid = DiffSdfGrid::<TestBackend, 1>::new(config, &device);

    // Allocate up to capacity
    grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();
    grid.allocate_block(BlockCoord::new(1, 0, 0)).unwrap();
    grid.allocate_block(BlockCoord::new(2, 0, 0)).unwrap();

    // Should fail when exceeding capacity
    let result = grid.allocate_block(BlockCoord::new(3, 0, 0));
    assert!(result.is_err());
}

#[test]
fn test_multi_feature_grid() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = DiffGridConfig::new(4, 0.1).with_capacity(10);
    let mut grid = DiffSdfGrid::<TestBackend, 4>::new(config, &device);

    grid.allocate_block(BlockCoord::new(0, 0, 0)).unwrap();

    let memory_grid = grid.to_memory_grid();
    assert_eq!(memory_grid.num_blocks(), 1);
}
