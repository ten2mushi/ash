//! Trilinear interpolation with analytical gradients.
//!
//! Provides forward and backward passes for feature value interpolation,
//! enabling both efficient inference and analytical gradient training.

use crate::coords::resolve_corner;
use crate::traits::{corner_from_index, CellValueProvider, GradientAccumulator};
use crate::types::{BlockCoord, CellCoord, InterpolationResult, LocalCoord, UNTRAINED_SENTINEL};

/// Compute trilinear interpolation weights for a local coordinate.
///
/// Returns the 8 weights corresponding to corners 0-7 in the standard marching cubes order.
/// The weights sum to 1.0 and represent the contribution of each corner to the interpolated value.
///
/// For a local coordinate (u, v, w) in [0,1]³:
/// - weight[0] = (1-u)(1-v)(1-w)  at corner (0,0,0)
/// - weight[1] = u(1-v)(1-w)      at corner (1,0,0)
/// - weight[2] = uv(1-w)          at corner (1,1,0)
/// - weight[3] = (1-u)v(1-w)      at corner (0,1,0)
/// - weight[4] = (1-u)(1-v)w      at corner (0,0,1)
/// - weight[5] = u(1-v)w          at corner (1,0,1)
/// - weight[6] = uvw              at corner (1,1,1)
/// - weight[7] = (1-u)vw          at corner (0,1,1)
#[inline]
pub fn compute_trilinear_weights(local: LocalCoord) -> [f32; 8] {
    let u = local.u;
    let v = local.v;
    let w = local.w;

    let u0 = 1.0 - u;
    let v0 = 1.0 - v;
    let w0 = 1.0 - w;

    [
        u0 * v0 * w0, // corner 0: (0,0,0)
        u * v0 * w0,  // corner 1: (1,0,0)
        u * v * w0,   // corner 2: (1,1,0)
        u0 * v * w0,  // corner 3: (0,1,0)
        u0 * v0 * w,  // corner 4: (0,0,1)
        u * v0 * w,   // corner 5: (1,0,1)
        u * v * w,    // corner 6: (1,1,1)
        u0 * v * w,   // corner 7: (0,1,1)
    ]
}

/// Trilinear interpolation for N-dimensional features.
///
/// Interpolates feature values at a point within a cell and returns the weights
/// for each corner, which are useful for backpropagation.
///
/// # Arguments
/// * `provider` - Storage providing corner feature values
/// * `block` - The block containing the cell
/// * `cell` - The cell within the block
/// * `local` - The fractional position within the cell [0,1]³
///
/// # Returns
/// * `Some(result)` - The interpolation result with values, weights, and coordinates
/// * `None` - If any corner value is unavailable or contains the untrained sentinel
pub fn trilinear_interpolate<const N: usize, P: CellValueProvider<N>>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    local: LocalCoord,
) -> Option<InterpolationResult<N>> {
    let grid_dim = provider.grid_dim();
    let weights = compute_trilinear_weights(local);

    // Gather corner values
    let mut corner_values = [[0.0f32; N]; 8];
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        let corner = corner_from_index(i);
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, corner, grid_dim);
        let values = provider.get_corner_values(resolved_block, resolved_cell, (0, 0, 0))?;

        // Check for untrained regions (only check first feature for sentinel)
        if values[0] >= UNTRAINED_SENTINEL * 0.5 {
            return None;
        }

        corner_values[i] = values;
    }

    // Compute interpolated values for each feature dimension
    let mut result_values = [0.0f32; N];
    for j in 0..N {
        for i in 0..8 {
            result_values[j] += weights[i] * corner_values[i][j];
        }
    }

    Some(InterpolationResult::new(result_values, weights, block, cell))
}

/// Compute the gradient of features with respect to position.
///
/// Returns [[∂f₀/∂x, ∂f₀/∂y, ∂f₀/∂z], [∂f₁/∂x, ∂f₁/∂y, ∂f₁/∂z], ...]
/// - the rate of change of each feature value in each direction, normalized by cell_size.
///
/// # Arguments
/// * `provider` - Storage providing corner feature values
/// * `block` - The block containing the cell
/// * `cell` - The cell within the block
/// * `local` - The fractional position within the cell [0,1]³
///
/// # Returns
/// * `Some(gradients)` - The gradient for each feature in local cell coordinates
/// * `None` - If any corner value is unavailable
///
/// # Note
/// The returned gradients are in cell-space units. To get world-space gradients,
/// divide by cell_size.
pub fn trilinear_gradient<const N: usize, P: CellValueProvider<N>>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    local: LocalCoord,
) -> Option<[[f32; 3]; N]> {
    let grid_dim = provider.grid_dim();

    // Gather corner values
    let mut values = [[0.0f32; N]; 8];
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        let corner = corner_from_index(i);
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, corner, grid_dim);
        let corner_values = provider.get_corner_values(resolved_block, resolved_cell, (0, 0, 0))?;

        if corner_values[0] >= UNTRAINED_SENTINEL * 0.5 {
            return None;
        }

        values[i] = corner_values;
    }

    let u = local.u;
    let v = local.v;
    let w = local.w;

    let u0 = 1.0 - u;
    let v0 = 1.0 - v;
    let w0 = 1.0 - w;

    // Compute gradients for each feature dimension
    let mut gradients = [[0.0f32; 3]; N];
    for j in 0..N {
        // Partial derivatives of trilinear interpolation
        // ∂f/∂u = Σ (∂weight[i]/∂u) * value[i]
        let df_du = -v0 * w0 * values[0][j]
            + v0 * w0 * values[1][j]
            + v * w0 * values[2][j]
            - v * w0 * values[3][j]
            - v0 * w * values[4][j]
            + v0 * w * values[5][j]
            + v * w * values[6][j]
            - v * w * values[7][j];

        let df_dv = -u0 * w0 * values[0][j]
            - u * w0 * values[1][j]
            + u * w0 * values[2][j]
            + u0 * w0 * values[3][j]
            - u0 * w * values[4][j]
            - u * w * values[5][j]
            + u * w * values[6][j]
            + u0 * w * values[7][j];

        let df_dw = -u0 * v0 * values[0][j]
            - u * v0 * values[1][j]
            - u * v * values[2][j]
            - u0 * v * values[3][j]
            + u0 * v0 * values[4][j]
            + u * v0 * values[5][j]
            + u * v * values[6][j]
            + u0 * v * values[7][j];

        gradients[j] = [df_du, df_dv, df_dw];
    }

    Some(gradients)
}

/// Combined value and gradient computation for N features.
///
/// More efficient than calling `trilinear_interpolate` and `trilinear_gradient` separately
/// as it only gathers corner values once.
///
/// # Returns
/// * `Some((result, gradients))` - The interpolation result and per-feature gradients
/// * `None` - If any corner value is unavailable
pub fn trilinear_with_gradient<const N: usize, P: CellValueProvider<N>>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    local: LocalCoord,
) -> Option<(InterpolationResult<N>, [[f32; 3]; N])> {
    let grid_dim = provider.grid_dim();
    let weights = compute_trilinear_weights(local);

    // Gather corner values
    let mut corner_values = [[0.0f32; N]; 8];
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        let corner = corner_from_index(i);
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, corner, grid_dim);
        let values = provider.get_corner_values(resolved_block, resolved_cell, (0, 0, 0))?;

        if values[0] >= UNTRAINED_SENTINEL * 0.5 {
            return None;
        }

        corner_values[i] = values;
    }

    // Compute interpolated values
    let mut result_values = [0.0f32; N];
    for j in 0..N {
        for i in 0..8 {
            result_values[j] += weights[i] * corner_values[i][j];
        }
    }

    // Compute gradients
    let u = local.u;
    let v = local.v;
    let w = local.w;

    let u0 = 1.0 - u;
    let v0 = 1.0 - v;
    let w0 = 1.0 - w;

    let mut gradients = [[0.0f32; 3]; N];
    for j in 0..N {
        let df_du = -v0 * w0 * corner_values[0][j]
            + v0 * w0 * corner_values[1][j]
            + v * w0 * corner_values[2][j]
            - v * w0 * corner_values[3][j]
            - v0 * w * corner_values[4][j]
            + v0 * w * corner_values[5][j]
            + v * w * corner_values[6][j]
            - v * w * corner_values[7][j];

        let df_dv = -u0 * w0 * corner_values[0][j]
            - u * w0 * corner_values[1][j]
            + u * w0 * corner_values[2][j]
            + u0 * w0 * corner_values[3][j]
            - u0 * w * corner_values[4][j]
            - u * w * corner_values[5][j]
            + u * w * corner_values[6][j]
            + u0 * w * corner_values[7][j];

        let df_dw = -u0 * v0 * corner_values[0][j]
            - u * v0 * corner_values[1][j]
            - u * v * corner_values[2][j]
            - u0 * v * corner_values[3][j]
            + u0 * v0 * corner_values[4][j]
            + u * v0 * corner_values[5][j]
            + u * v * corner_values[6][j]
            + u0 * v * corner_values[7][j];

        gradients[j] = [df_du, df_dv, df_dw];
    }

    let result = InterpolationResult::new(result_values, weights, block, cell);
    Some((result, gradients))
}

/// Backward pass: accumulate gradients to corner embeddings.
///
/// Given an upstream gradient (∂L/∂value) and the interpolation result from
/// the forward pass, this computes and accumulates the gradients to each corner:
/// ∂L/∂corner[i] = weight[i] * upstream_grad
///
/// # Arguments
/// * `accumulator` - Gradient accumulator for the storage backend
/// * `result` - The interpolation result from the forward pass
/// * `upstream_grad` - The gradient from the loss (∂L/∂values) for each feature
pub fn trilinear_backward<const N: usize, A: GradientAccumulator<N>>(
    accumulator: &mut A,
    result: &InterpolationResult<N>,
    upstream_grad: [f32; N],
) {
    for i in 0..8 {
        let corner = corner_from_index(i);
        let weight = result.weights[i];
        accumulator.accumulate_gradient(result.block, result.cell, corner, weight, upstream_grad);
    }
}

/// Compute mixed partial derivatives (Hessian components) for feature 0 (SDF).
///
/// Returns [∂²f/∂x∂y, ∂²f/∂x∂z, ∂²f/∂y∂z] - the mixed second partial derivatives.
/// These are useful for computing the Eikonal loss and other regularizers.
///
/// # Arguments
/// * `provider` - Storage providing corner feature values
/// * `block` - The block containing the cell
/// * `cell` - The cell within the block
/// * `local` - The fractional position within the cell [0,1]³
///
/// # Returns
/// * `Some([dxy, dxz, dyz])` - The mixed partials in cell-space units
/// * `None` - If any corner value is unavailable
///
/// # Note
/// This only computes the Hessian for the first feature dimension (SDF).
pub fn trilinear_hessian_mixed<const N: usize, P: CellValueProvider<N>>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    local: LocalCoord,
) -> Option<[f32; 3]> {
    let grid_dim = provider.grid_dim();

    // Gather corner values (only need first feature for Hessian)
    let mut values = [0.0f32; 8];
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        let corner = corner_from_index(i);
        let (resolved_block, resolved_cell) = resolve_corner(block, cell, corner, grid_dim);
        let corner_values = provider.get_corner_values(resolved_block, resolved_cell, (0, 0, 0))?;

        if corner_values[0] >= UNTRAINED_SENTINEL * 0.5 {
            return None;
        }

        values[i] = corner_values[0];
    }

    let w = local.w;
    let w0 = 1.0 - w;
    let u = local.u;
    let u0 = 1.0 - u;
    let v = local.v;
    let v0 = 1.0 - v;

    // ∂²f/∂u∂v (mixed partial with respect to x and y)
    let d2f_dudv = w0 * (values[0] - values[1] + values[2] - values[3])
        + w * (-values[4] + values[5] - values[6] + values[7]);

    // ∂²f/∂u∂w (mixed partial with respect to x and z)
    let d2f_dudw = v0 * (values[0] - values[1] - values[4] + values[5])
        + v * (values[3] - values[2] - values[7] + values[6]);

    // ∂²f/∂v∂w (mixed partial with respect to y and z)
    let d2f_dvdw = u0 * (values[0] - values[3] - values[4] + values[7])
        + u * (values[1] - values[2] - values[5] + values[6]);

    Some([d2f_dudv, d2f_dudw, d2f_dvdw])
}

// ============================================================================
// Convenience functions for N=1 (SDF-only) grids
// ============================================================================

/// Trilinear interpolation for SDF-only grids (N=1).
///
/// This is a convenience wrapper that returns a simpler result type.
#[inline]
pub fn trilinear_interpolate_sdf<P: CellValueProvider<1>>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    local: LocalCoord,
) -> Option<InterpolationResult<1>> {
    trilinear_interpolate::<1, P>(provider, block, cell, local)
}

/// SDF gradient computation (N=1).
#[inline]
pub fn trilinear_gradient_sdf<P: CellValueProvider<1>>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    local: LocalCoord,
) -> Option<[f32; 3]> {
    trilinear_gradient::<1, P>(provider, block, cell, local).map(|g| g[0])
}

/// Combined SDF value and gradient (N=1).
#[inline]
pub fn trilinear_with_gradient_sdf<P: CellValueProvider<1>>(
    provider: &P,
    block: BlockCoord,
    cell: CellCoord,
    local: LocalCoord,
) -> Option<(InterpolationResult<1>, [f32; 3])> {
    trilinear_with_gradient::<1, P>(provider, block, cell, local).map(|(r, g)| (r, g[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock provider for testing with a simple linear SDF
    struct MockLinearProvider {
        grid_dim: u32,
        cell_size: f32,
    }

    impl CellValueProvider<1> for MockLinearProvider {
        fn get_corner_values(
            &self,
            _block: BlockCoord,
            cell: CellCoord,
            corner: (u32, u32, u32),
        ) -> Option<[f32; 1]> {
            // Simple linear SDF: f(x,y,z) = x + y + z
            let x = cell.x as f32 + corner.0 as f32;
            let y = cell.y as f32 + corner.1 as f32;
            let z = cell.z as f32 + corner.2 as f32;
            Some([x + y + z])
        }

        fn grid_dim(&self) -> u32 {
            self.grid_dim
        }

        fn cell_size(&self) -> f32 {
            self.cell_size
        }
    }

    /// Mock provider for a sphere SDF
    struct MockSphereProvider {
        grid_dim: u32,
        cell_size: f32,
        center: (f32, f32, f32),
        radius: f32,
    }

    impl CellValueProvider<1> for MockSphereProvider {
        fn get_corner_values(
            &self,
            block: BlockCoord,
            cell: CellCoord,
            corner: (u32, u32, u32),
        ) -> Option<[f32; 1]> {
            let block_size = self.grid_dim as f32 * self.cell_size;
            let x = block.x as f32 * block_size
                + cell.x as f32 * self.cell_size
                + corner.0 as f32 * self.cell_size;
            let y = block.y as f32 * block_size
                + cell.y as f32 * self.cell_size
                + corner.1 as f32 * self.cell_size;
            let z = block.z as f32 * block_size
                + cell.z as f32 * self.cell_size
                + corner.2 as f32 * self.cell_size;

            let dx = x - self.center.0;
            let dy = y - self.center.1;
            let dz = z - self.center.2;
            let dist = libm::sqrtf(dx * dx + dy * dy + dz * dz);
            Some([dist - self.radius])
        }

        fn grid_dim(&self) -> u32 {
            self.grid_dim
        }

        fn cell_size(&self) -> f32 {
            self.cell_size
        }
    }

    /// Mock provider for multi-feature testing
    struct MockMultiFeatureProvider {
        grid_dim: u32,
        cell_size: f32,
    }

    impl CellValueProvider<4> for MockMultiFeatureProvider {
        fn get_corner_values(
            &self,
            _block: BlockCoord,
            cell: CellCoord,
            corner: (u32, u32, u32),
        ) -> Option<[f32; 4]> {
            let x = cell.x as f32 + corner.0 as f32;
            let y = cell.y as f32 + corner.1 as f32;
            let z = cell.z as f32 + corner.2 as f32;
            // Different linear functions for each feature
            Some([x + y + z, x - y, y - z, x + z])
        }

        fn grid_dim(&self) -> u32 {
            self.grid_dim
        }

        fn cell_size(&self) -> f32 {
            self.cell_size
        }
    }

    #[test]
    fn test_weights_sum_to_one() {
        let test_locals = [
            LocalCoord::new(0.0, 0.0, 0.0),
            LocalCoord::new(1.0, 1.0, 1.0),
            LocalCoord::new(0.5, 0.5, 0.5),
            LocalCoord::new(0.25, 0.75, 0.33),
        ];

        for local in test_locals {
            let weights = compute_trilinear_weights(local);
            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Weights sum to {} for {:?}",
                sum,
                local
            );
        }
    }

    #[test]
    fn test_weights_at_corners() {
        // At corner (0,0,0), only weight[0] should be 1.0
        let w = compute_trilinear_weights(LocalCoord::new(0.0, 0.0, 0.0));
        assert!((w[0] - 1.0).abs() < 1e-6);
        for i in 1..8 {
            assert!(w[i].abs() < 1e-6);
        }

        // At corner (1,1,1), only weight[6] should be 1.0
        let w = compute_trilinear_weights(LocalCoord::new(1.0, 1.0, 1.0));
        assert!((w[6] - 1.0).abs() < 1e-6);
        for i in [0, 1, 2, 3, 4, 5, 7] {
            assert!(w[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_trilinear_interpolate_linear() {
        let provider = MockLinearProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);

        // At the corner (0,0,0) of cell (2,3,4), value = 2+3+4 = 9
        let local = LocalCoord::new(0.0, 0.0, 0.0);
        let result = trilinear_interpolate(&provider, block, cell, local).unwrap();
        assert!((result.values[0] - 9.0).abs() < 1e-5);

        // At (0.5, 0.5, 0.5), value should be (2+0.5) + (3+0.5) + (4+0.5) = 10.5
        let local = LocalCoord::new(0.5, 0.5, 0.5);
        let result = trilinear_interpolate(&provider, block, cell, local).unwrap();
        assert!((result.values[0] - 10.5).abs() < 1e-5);
    }

    #[test]
    fn test_trilinear_gradient_linear() {
        let provider = MockLinearProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);

        // For f(x,y,z) = x + y + z, gradient is (1, 1, 1) everywhere
        let local = LocalCoord::new(0.5, 0.5, 0.5);
        let grad = trilinear_gradient(&provider, block, cell, local).unwrap();

        // Gradient in cell units should be (1, 1, 1)
        assert!((grad[0][0] - 1.0).abs() < 1e-5, "df/dx = {}", grad[0][0]);
        assert!((grad[0][1] - 1.0).abs() < 1e-5, "df/dy = {}", grad[0][1]);
        assert!((grad[0][2] - 1.0).abs() < 1e-5, "df/dz = {}", grad[0][2]);
    }

    #[test]
    fn test_multi_feature_interpolation() {
        let provider = MockMultiFeatureProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_interpolate(&provider, block, cell, local).unwrap();

        // Check that we get 4 interpolated values
        assert_eq!(result.values.len(), 4);

        // First feature: x + y + z at (2.5, 3.5, 4.5) = 10.5
        assert!((result.values[0] - 10.5).abs() < 1e-5);
    }

    #[test]
    fn test_multi_feature_gradient() {
        let provider = MockMultiFeatureProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let grad = trilinear_gradient(&provider, block, cell, local).unwrap();

        // Feature 0: f = x + y + z, grad = (1, 1, 1)
        assert!((grad[0][0] - 1.0).abs() < 1e-5);
        assert!((grad[0][1] - 1.0).abs() < 1e-5);
        assert!((grad[0][2] - 1.0).abs() < 1e-5);

        // Feature 1: f = x - y, grad = (1, -1, 0)
        assert!((grad[1][0] - 1.0).abs() < 1e-5);
        assert!((grad[1][1] - (-1.0)).abs() < 1e-5);
        assert!((grad[1][2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_numerical_verification() {
        let provider = MockSphereProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: (0.4, 0.4, 0.4),
            radius: 0.2,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(3, 3, 3);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        // Compute analytical gradient
        let analytical = trilinear_gradient(&provider, block, cell, local).unwrap();

        // Compute numerical gradient using finite differences
        let eps = 1e-4;
        let mut numerical = [0.0f32; 3];

        for axis in 0..3 {
            let mut local_plus = local;
            let mut local_minus = local;

            match axis {
                0 => {
                    local_plus.u += eps;
                    local_minus.u -= eps;
                }
                1 => {
                    local_plus.v += eps;
                    local_minus.v -= eps;
                }
                _ => {
                    local_plus.w += eps;
                    local_minus.w -= eps;
                }
            }

            let f_plus = trilinear_interpolate(&provider, block, cell, local_plus)
                .unwrap()
                .values[0];
            let f_minus = trilinear_interpolate(&provider, block, cell, local_minus)
                .unwrap()
                .values[0];
            numerical[axis] = (f_plus - f_minus) / (2.0 * eps);
        }

        for i in 0..3 {
            assert!(
                (analytical[0][i] - numerical[i]).abs() < 1e-3,
                "Gradient mismatch on axis {}: analytical={}, numerical={}",
                i,
                analytical[0][i],
                numerical[i]
            );
        }
    }

    #[test]
    fn test_trilinear_with_gradient_matches_separate() {
        let provider = MockSphereProvider {
            grid_dim: 8,
            cell_size: 0.1,
            center: (0.4, 0.4, 0.4),
            radius: 0.2,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(3, 3, 3);
        let local = LocalCoord::new(0.3, 0.6, 0.2);

        let (combined_result, combined_grad) =
            trilinear_with_gradient(&provider, block, cell, local).unwrap();
        let separate_result = trilinear_interpolate(&provider, block, cell, local).unwrap();
        let separate_grad = trilinear_gradient(&provider, block, cell, local).unwrap();

        assert!((combined_result.values[0] - separate_result.values[0]).abs() < 1e-6);

        for i in 0..3 {
            assert!((combined_grad[0][i] - separate_grad[0][i]).abs() < 1e-6);
        }
    }

    /// Mock gradient accumulator for testing
    struct MockAccumulator<const N: usize> {
        gradients: [(BlockCoord, CellCoord, (u32, u32, u32), f32, [f32; N]); 8],
        count: usize,
    }

    impl<const N: usize> MockAccumulator<N> {
        fn new() -> Self {
            Self {
                gradients: [(
                    BlockCoord::new(0, 0, 0),
                    CellCoord::new(0, 0, 0),
                    (0, 0, 0),
                    0.0,
                    [0.0; N],
                ); 8],
                count: 0,
            }
        }
    }

    impl<const N: usize> GradientAccumulator<N> for MockAccumulator<N> {
        fn accumulate_gradient(
            &mut self,
            block: BlockCoord,
            cell: CellCoord,
            corner: (u32, u32, u32),
            weight: f32,
            upstream_grad: [f32; N],
        ) {
            if self.count < 8 {
                self.gradients[self.count] = (block, cell, corner, weight, upstream_grad);
                self.count += 1;
            }
        }
    }

    #[test]
    fn test_trilinear_backward() {
        let provider = MockLinearProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        let result = trilinear_interpolate(&provider, block, cell, local).unwrap();

        let mut accumulator = MockAccumulator::<1>::new();
        let upstream_grad = [1.0];
        trilinear_backward(&mut accumulator, &result, upstream_grad);

        assert_eq!(accumulator.count, 8);

        // At (0.5, 0.5, 0.5), all weights should be equal (0.125)
        for (_, _, _, weight, _) in &accumulator.gradients {
            assert!(
                (*weight - 0.125).abs() < 1e-6,
                "Expected weight 0.125, got {}",
                weight
            );
        }
    }

    #[test]
    fn test_hessian_mixed() {
        let provider = MockLinearProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        // For linear SDF f = x + y + z, all second derivatives are zero
        let hessian = trilinear_hessian_mixed(&provider, block, cell, local).unwrap();

        for i in 0..3 {
            assert!(
                hessian[i].abs() < 1e-5,
                "Expected zero mixed partial, got {}",
                hessian[i]
            );
        }
    }

    #[test]
    fn test_sdf_convenience_functions() {
        let provider = MockLinearProvider {
            grid_dim: 8,
            cell_size: 0.1,
        };

        let block = BlockCoord::new(0, 0, 0);
        let cell = CellCoord::new(2, 3, 4);
        let local = LocalCoord::new(0.5, 0.5, 0.5);

        // Test SDF-specific convenience functions
        let result = trilinear_interpolate_sdf(&provider, block, cell, local).unwrap();
        assert!((result.value() - 10.5).abs() < 1e-5);

        let grad = trilinear_gradient_sdf(&provider, block, cell, local).unwrap();
        assert!((grad[0] - 1.0).abs() < 1e-5);

        let (result2, grad2) = trilinear_with_gradient_sdf(&provider, block, cell, local).unwrap();
        assert!((result2.value() - 10.5).abs() < 1e-5);
        assert!((grad2[0] - 1.0).abs() < 1e-5);
    }
}
