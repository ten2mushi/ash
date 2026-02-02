//! Serialization round-trip tests.

#![cfg(feature = "serde")]

use ash_core::{BlockCoord, Point3};
use ash_rs::{GridBuilder, SparseDenseGrid};
use std::io::Cursor;

fn make_test_grid() -> SparseDenseGrid {
    let center = Point3::new(0.4, 0.4, 0.4);
    let radius = 0.3;

    let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);

    for bz in 0..2 {
        for by in 0..2 {
            for bx in 0..2 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| (pos - center).length() - radius);
            }
        }
    }

    builder.build().unwrap()
}

#[test]
fn test_roundtrip_properties() {
    let original = make_test_grid();

    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    let mut cursor = Cursor::new(buffer);
    let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

    // Check config matches
    assert_eq!(loaded.config().grid_dim, original.config().grid_dim);
    assert!((loaded.config().cell_size - original.config().cell_size).abs() < 1e-6);

    // Check block count matches
    assert_eq!(loaded.num_blocks(), original.num_blocks());
}

#[test]
fn test_roundtrip_query_values() {
    let original = make_test_grid();

    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    let mut cursor = Cursor::new(buffer);
    let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

    // Test many query points
    let test_points: Vec<Point3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Point3::new(t * 1.5, t * 1.5, t * 1.5)
        })
        .collect();

    for point in &test_points {
        let orig_val = original.query(*point);
        let loaded_val = loaded.query(*point);

        match (orig_val, loaded_val) {
            (Some(v1), Some(v2)) => {
                assert!(
                    (v1 - v2).abs() < 1e-6,
                    "Mismatch at {:?}: orig={}, loaded={}",
                    point,
                    v1,
                    v2
                );
            }
            (None, None) => {}
            _ => panic!("Validity mismatch at {:?}", point),
        }
    }
}

#[test]
fn test_roundtrip_block_coords() {
    let original = make_test_grid();

    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    let mut cursor = Cursor::new(buffer);
    let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

    // Check all blocks exist
    for coord in original.block_coords() {
        assert!(
            loaded.has_block(coord),
            "Block {:?} missing after load",
            coord
        );
    }
}

#[test]
fn test_roundtrip_gradients() {
    let original = make_test_grid();

    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    let mut cursor = Cursor::new(buffer);
    let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

    let test_points = [
        Point3::new(0.4, 0.4, 0.4),
        Point3::new(0.7, 0.4, 0.4),
        Point3::new(0.4, 0.7, 0.4),
    ];

    for point in &test_points {
        let orig = original.query_with_gradient(*point);
        let loaded_result = loaded.query_with_gradient(*point);

        match (orig, loaded_result) {
            (Some((v1, g1)), Some((v2, g2))) => {
                assert!((v1 - v2).abs() < 1e-6);
                for i in 0..3 {
                    assert!((g1[i] - g2[i]).abs() < 1e-5);
                }
            }
            (None, None) => {}
            _ => panic!("Validity mismatch at {:?}", point),
        }
    }
}

#[test]
fn test_empty_grid_roundtrip() {
    let original = GridBuilder::new(8, 0.1).with_capacity(10).build().unwrap();

    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    let mut cursor = Cursor::new(buffer);
    let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

    assert_eq!(loaded.num_blocks(), 0);
    assert_eq!(loaded.config().grid_dim, 8);
}

#[test]
fn test_negative_coords_roundtrip() {
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(10);

    let test_coords = [
        BlockCoord::new(-5, -10, 15),
        BlockCoord::new(100, -200, 300),
        BlockCoord::new(-1, -1, -1),
    ];

    for (i, &coord) in test_coords.iter().enumerate() {
        builder = builder.add_block_constant(coord, i as f32);
    }

    let original = builder.build().unwrap();

    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    let mut cursor = Cursor::new(buffer);
    let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

    for coord in &test_coords {
        assert!(loaded.has_block(*coord), "Missing block {:?}", coord);
    }
}

#[test]
fn test_large_grid_roundtrip() {
    // Create a moderately large grid
    let mut builder = GridBuilder::new(8, 0.1).with_capacity(200);

    for bz in 0..5 {
        for by in 0..5 {
            for bx in 0..5 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_constant(coord, (bx + by + bz) as f32);
            }
        }
    }

    let original = builder.build().unwrap();
    assert_eq!(original.num_blocks(), 125);

    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    // Check file size is reasonable
    let expected_size = ash_rs::compute_file_size(125, 8);
    assert_eq!(buffer.len(), expected_size);

    let mut cursor = Cursor::new(buffer);
    let loaded = SparseDenseGrid::load(&mut cursor).unwrap();

    assert_eq!(loaded.num_blocks(), 125);
}

#[test]
fn test_invalid_magic() {
    let data = b"BADM\x01\x00\x00\x00\x00\x00\x80\x3f\x08\x00\x00\x00\x00\x00\x00\x00";
    let mut cursor = Cursor::new(data.to_vec());
    let result = SparseDenseGrid::load(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_invalid_version() {
    let mut data = Vec::new();
    data.extend_from_slice(b"ASH1");
    data.extend_from_slice(&99u32.to_le_bytes()); // Bad version
    data.extend_from_slice(&0.1f32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let mut cursor = Cursor::new(data);
    let result = SparseDenseGrid::load(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_truncated_file() {
    let original = make_test_grid();

    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    // Truncate the file
    buffer.truncate(50);

    let mut cursor = Cursor::new(buffer);
    let result = SparseDenseGrid::load(&mut cursor);
    assert!(result.is_err());
}
