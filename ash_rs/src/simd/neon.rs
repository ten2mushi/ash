//! ARM NEON SIMD implementation for aarch64.
//!
//! Processes 4 points per iteration using 128-bit vectors.
//! Available on ARM64 devices (Apple Silicon, Raspberry Pi 4+, etc.).

#![cfg(all(target_arch = "aarch64", target_feature = "neon"))]

use core::arch::aarch64::*;

use ash_core::{decompose_point, trilinear_interpolate_sdf, CellValueProvider, Point3, SdfProvider};

use super::{scalar::query_batch_scalar, BatchResult};
use crate::SparseDenseGrid;

/// SIMD lane width for NEON (4 x f32).
const LANE_WIDTH: usize = 4;

/// Batch query using ARM NEON SIMD operations (simple version).
///
/// This version uses scalar interpolation but processes points in chunks
/// for better cache utilization.
#[allow(dead_code)]
pub fn query_batch_neon(grid: &SparseDenseGrid, points: &[Point3]) -> BatchResult {
    let n = points.len();
    if n == 0 {
        return BatchResult {
            values: Vec::new(),
            valid_mask: Vec::new(),
        };
    }

    // For small batches, scalar is more efficient
    if n < LANE_WIDTH * 2 {
        return query_batch_scalar(grid, points);
    }

    let mut values = vec![f32::NAN; n];
    let mut valid_mask = vec![false; n];

    let cell_size = grid.cell_size();
    let grid_dim = grid.grid_dim();

    // Process chunks of 4 points
    let chunks = n / LANE_WIDTH;

    for chunk_idx in 0..chunks {
        let base_idx = chunk_idx * LANE_WIDTH;

        // Process each point in the chunk using scalar trilinear interpolation
        // The chunking still helps with cache locality
        for lane in 0..LANE_WIDTH {
            let idx = base_idx + lane;
            let point = points[idx];

            let (block, cell, local) = decompose_point(point, cell_size, grid_dim);

            if let Some(result) = trilinear_interpolate_sdf(grid, block, cell, local) {
                values[idx] = result.value();
                valid_mask[idx] = true;
            }
        }
    }

    // Handle remainder with scalar fallback
    let remainder_start = chunks * LANE_WIDTH;
    for i in remainder_start..n {
        if let Some(v) = grid.query(points[i]) {
            values[i] = v;
            valid_mask[i] = true;
        }
    }

    BatchResult { values, valid_mask }
}

/// Optimized batch query with full SIMD trilinear interpolation.
///
/// This version gathers all corner values first, then performs
/// vectorized trilinear interpolation.
pub fn query_batch_neon_gather(grid: &SparseDenseGrid, points: &[Point3]) -> BatchResult {
    let n = points.len();
    if n == 0 {
        return BatchResult {
            values: Vec::new(),
            valid_mask: Vec::new(),
        };
    }

    if n < LANE_WIDTH * 2 {
        return query_batch_scalar(grid, points);
    }

    let mut values = vec![f32::NAN; n];
    let mut valid_mask = vec![false; n];

    let cell_size = grid.cell_size();
    let grid_dim = grid.grid_dim();

    // Process in chunks of 4
    let chunks = n / LANE_WIDTH;

    for chunk_idx in 0..chunks {
        let base_idx = chunk_idx * LANE_WIDTH;

        // Gather corner values for 4 points (8 corners each = 32 values)
        let mut corners: [[f32; 8]; 4] = [[f32::NAN; 8]; 4];
        let mut chunk_valid = [false; 4];

        for lane in 0..LANE_WIDTH {
            let point = points[base_idx + lane];
            let (block, cell, _local) = decompose_point(point, cell_size, grid_dim);

            // Try to gather all 8 corners
            let mut all_corners_valid = true;
            for corner_idx in 0..8 {
                let dx = (corner_idx & 1) as u32;
                let dy = ((corner_idx >> 1) & 1) as u32;
                let dz = ((corner_idx >> 2) & 1) as u32;

                if let Some(val) = SdfProvider::get_corner_value(grid, block, cell, (dx, dy, dz)) {
                    corners[lane][corner_idx] = val;
                } else {
                    all_corners_valid = false;
                    break;
                }
            }

            if all_corners_valid {
                chunk_valid[lane] = true;
            }
        }

        // Perform SIMD trilinear interpolation for valid points
        unsafe {
            // Load local coordinates
            let mut u_arr = [0.0f32; 4];
            let mut v_arr = [0.0f32; 4];
            let mut w_arr = [0.0f32; 4];

            for lane in 0..LANE_WIDTH {
                let point = points[base_idx + lane];
                let (_, _, local) = decompose_point(point, cell_size, grid_dim);
                u_arr[lane] = local.u;
                v_arr[lane] = local.v;
                w_arr[lane] = local.w;
            }

            let u = vld1q_f32(u_arr.as_ptr());
            let v = vld1q_f32(v_arr.as_ptr());
            let w = vld1q_f32(w_arr.as_ptr());

            // Compute trilinear weights
            let one = vdupq_n_f32(1.0);
            let nu = vsubq_f32(one, u);
            let nv = vsubq_f32(one, v);
            let nw = vsubq_f32(one, w);

            // Weight for each corner: w[i] = (1-u or u) * (1-v or v) * (1-w or w)
            let w0 = vmulq_f32(vmulq_f32(nu, nv), nw); // (1-u)(1-v)(1-w)
            let w1 = vmulq_f32(vmulq_f32(u, nv), nw); // u(1-v)(1-w)
            let w2 = vmulq_f32(vmulq_f32(nu, v), nw); // (1-u)v(1-w)
            let w3 = vmulq_f32(vmulq_f32(u, v), nw); // uv(1-w)
            let w4 = vmulq_f32(vmulq_f32(nu, nv), w); // (1-u)(1-v)w
            let w5 = vmulq_f32(vmulq_f32(u, nv), w); // u(1-v)w
            let w6 = vmulq_f32(vmulq_f32(nu, v), w); // (1-u)vw
            let w7 = vmulq_f32(vmulq_f32(u, v), w); // uvw

            // Load corner values and compute weighted sum
            let c0 = vld1q_f32([corners[0][0], corners[1][0], corners[2][0], corners[3][0]].as_ptr());
            let c1 = vld1q_f32([corners[0][1], corners[1][1], corners[2][1], corners[3][1]].as_ptr());
            let c2 = vld1q_f32([corners[0][2], corners[1][2], corners[2][2], corners[3][2]].as_ptr());
            let c3 = vld1q_f32([corners[0][3], corners[1][3], corners[2][3], corners[3][3]].as_ptr());
            let c4 = vld1q_f32([corners[0][4], corners[1][4], corners[2][4], corners[3][4]].as_ptr());
            let c5 = vld1q_f32([corners[0][5], corners[1][5], corners[2][5], corners[3][5]].as_ptr());
            let c6 = vld1q_f32([corners[0][6], corners[1][6], corners[2][6], corners[3][6]].as_ptr());
            let c7 = vld1q_f32([corners[0][7], corners[1][7], corners[2][7], corners[3][7]].as_ptr());

            // Weighted sum using FMA
            let mut result = vmulq_f32(w0, c0);
            result = vfmaq_f32(result, w1, c1);
            result = vfmaq_f32(result, w2, c2);
            result = vfmaq_f32(result, w3, c3);
            result = vfmaq_f32(result, w4, c4);
            result = vfmaq_f32(result, w5, c5);
            result = vfmaq_f32(result, w6, c6);
            result = vfmaq_f32(result, w7, c7);

            // Store results
            let result_arr: [f32; 4] = core::mem::transmute(result);
            for lane in 0..LANE_WIDTH {
                if chunk_valid[lane] {
                    values[base_idx + lane] = result_arr[lane];
                    valid_mask[base_idx + lane] = true;
                }
            }
        }
    }

    // Handle remainder
    let remainder_start = chunks * LANE_WIDTH;
    for i in remainder_start..n {
        if let Some(v) = grid.query(points[i]) {
            values[i] = v;
            valid_mask[i] = true;
        }
    }

    BatchResult { values, valid_mask }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar::query_batch_scalar;
    use crate::GridBuilder;
    use ash_core::BlockCoord;

    fn make_test_grid() -> SparseDenseGrid {
        let dim = 8;
        let cells_per_block = dim * dim * dim;

        let mut builder = GridBuilder::new(dim as u32, 0.1).with_capacity(8);

        for bz in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    let coord = BlockCoord::new(bx, by, bz);
                    let values = vec![0.5; cells_per_block];
                    builder = builder.add_block(coord, values).unwrap();
                }
            }
        }

        builder.build().unwrap()
    }

    fn make_sphere_grid() -> SparseDenseGrid {
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
    fn test_neon_matches_scalar() {
        let grid = make_test_grid();

        let points: Vec<Point3> = (0..100)
            .map(|i| {
                let f = i as f32 / 100.0;
                Point3::new(f * 1.4, f * 1.4, f * 1.4)
            })
            .collect();

        let neon_result = query_batch_neon(&grid, &points);
        let scalar_result = query_batch_scalar(&grid, &points);

        for i in 0..points.len() {
            assert_eq!(
                neon_result.valid_mask[i], scalar_result.valid_mask[i],
                "Validity mismatch at index {}",
                i
            );

            if neon_result.valid_mask[i] {
                let diff = (neon_result.values[i] - scalar_result.values[i]).abs();
                assert!(
                    diff < 1e-5,
                    "Value mismatch at index {}: {} vs {}",
                    i,
                    neon_result.values[i],
                    scalar_result.values[i]
                );
            }
        }
    }

    #[test]
    fn test_neon_gather_matches_scalar() {
        let grid = make_sphere_grid();

        let points: Vec<Point3> = (0..100)
            .map(|i| {
                let f = i as f32 / 100.0;
                Point3::new(f * 1.4, f * 1.4, f * 1.4)
            })
            .collect();

        let neon_result = query_batch_neon_gather(&grid, &points);
        let scalar_result = query_batch_scalar(&grid, &points);

        for i in 0..points.len() {
            assert_eq!(
                neon_result.valid_mask[i], scalar_result.valid_mask[i],
                "Validity mismatch at index {}",
                i
            );

            if neon_result.valid_mask[i] {
                let diff = (neon_result.values[i] - scalar_result.values[i]).abs();
                assert!(
                    diff < 1e-5,
                    "Value mismatch at index {}: {} vs {}",
                    i,
                    neon_result.values[i],
                    scalar_result.values[i]
                );
            }
        }
    }

    #[test]
    fn test_neon_empty_input() {
        let grid = make_test_grid();
        let points: Vec<Point3> = vec![];

        let result = query_batch_neon(&grid, &points);
        assert!(result.values.is_empty());
        assert!(result.valid_mask.is_empty());
    }

    #[test]
    fn test_neon_small_input() {
        let grid = make_test_grid();
        let points = vec![
            Point3::new(0.4, 0.4, 0.4),
            Point3::new(0.5, 0.5, 0.5),
        ];

        let neon_result = query_batch_neon(&grid, &points);
        let scalar_result = query_batch_scalar(&grid, &points);

        for i in 0..points.len() {
            assert_eq!(neon_result.valid_mask[i], scalar_result.valid_mask[i]);
            if neon_result.valid_mask[i] {
                assert!((neon_result.values[i] - scalar_result.values[i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_neon_boundary_points() {
        let grid = make_test_grid();

        // Test points at block boundaries
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.8, 0.0, 0.0),
            Point3::new(0.0, 0.8, 0.0),
            Point3::new(0.0, 0.0, 0.8),
            Point3::new(0.8, 0.8, 0.8),
            Point3::new(1.2, 1.2, 1.2),
            Point3::new(0.4, 0.4, 0.4),
            Point3::new(0.1, 0.1, 0.1),
        ];

        let neon_result = query_batch_neon(&grid, &points);
        let scalar_result = query_batch_scalar(&grid, &points);

        for i in 0..points.len() {
            assert_eq!(
                neon_result.valid_mask[i], scalar_result.valid_mask[i],
                "Validity mismatch at index {} for point {:?}",
                i, points[i]
            );
        }
    }
}
