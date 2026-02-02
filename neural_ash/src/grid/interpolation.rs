//! Interpolation modes for the differentiable grid.
//!
//! Provides linear and smoothstep interpolation options. Linear uses
//! ash_core's compute_trilinear_weights, while smoothstep provides
//! smoother derivatives at cell boundaries.

use ash_core::LocalCoord;

/// Interpolation mode for grid queries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Standard linear interpolation.
    /// Uses ash_core::compute_trilinear_weights.
    #[default]
    Linear,

    /// Smoothstep interpolation: 3t² - 2t³
    /// Provides C¹ continuity at cell boundaries.
    SmoothStep,

    /// Smootherstep interpolation: 6t⁵ - 15t⁴ + 10t³
    /// Provides C² continuity at cell boundaries.
    SmootherStep,
}

/// Smoothstep function: 3x² - 2x³
///
/// Maps [0, 1] to [0, 1] with zero derivative at endpoints.
#[inline]
pub fn smoothstep(x: f32) -> f32 {
    x * x * (3.0 - 2.0 * x)
}

/// Smootherstep function: 6x⁵ - 15x⁴ + 10x³
///
/// Maps [0, 1] to [0, 1] with zero first and second derivatives at endpoints.
#[inline]
pub fn smootherstep(x: f32) -> f32 {
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

/// Derivative of smoothstep: 6x - 6x²
#[inline]
pub fn smoothstep_derivative(x: f32) -> f32 {
    6.0 * x * (1.0 - x)
}

/// Derivative of smootherstep: 30x⁴ - 60x³ + 30x²
#[inline]
pub fn smootherstep_derivative(x: f32) -> f32 {
    30.0 * x * x * (x * (x - 2.0) + 1.0)
}

/// Compute trilinear interpolation weights with mode selection.
///
/// Returns the 8 weights corresponding to corners 0-7 in marching cubes order.
pub fn compute_weights_with_mode(local: LocalCoord, mode: InterpolationMode) -> [f32; 8] {
    let (u, v, w) = match mode {
        InterpolationMode::Linear => (local.u, local.v, local.w),
        InterpolationMode::SmoothStep => (
            smoothstep(local.u),
            smoothstep(local.v),
            smoothstep(local.w),
        ),
        InterpolationMode::SmootherStep => (
            smootherstep(local.u),
            smootherstep(local.v),
            smootherstep(local.w),
        ),
    };

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

/// Compute weight gradients with respect to local coordinates.
///
/// Returns [dw/du, dw/dv, dw/dw] for each of the 8 corners.
/// Useful for backpropagation through the interpolation.
pub fn compute_weight_gradients_with_mode(
    local: LocalCoord,
    mode: InterpolationMode,
) -> [[f32; 3]; 8] {
    let (u, v, w, du, dv, dw) = match mode {
        InterpolationMode::Linear => (local.u, local.v, local.w, 1.0, 1.0, 1.0),
        InterpolationMode::SmoothStep => (
            smoothstep(local.u),
            smoothstep(local.v),
            smoothstep(local.w),
            smoothstep_derivative(local.u),
            smoothstep_derivative(local.v),
            smoothstep_derivative(local.w),
        ),
        InterpolationMode::SmootherStep => (
            smootherstep(local.u),
            smootherstep(local.v),
            smootherstep(local.w),
            smootherstep_derivative(local.u),
            smootherstep_derivative(local.v),
            smootherstep_derivative(local.w),
        ),
    };

    let u0 = 1.0 - u;
    let v0 = 1.0 - v;
    let w0 = 1.0 - w;

    // Gradients for each corner: [dw/du, dw/dv, dw/dw]
    [
        // corner 0: (0,0,0) - weight = u0 * v0 * w0
        [-du * v0 * w0, -dv * u0 * w0, -dw * u0 * v0],
        // corner 1: (1,0,0) - weight = u * v0 * w0
        [du * v0 * w0, -dv * u * w0, -dw * u * v0],
        // corner 2: (1,1,0) - weight = u * v * w0
        [du * v * w0, dv * u * w0, -dw * u * v],
        // corner 3: (0,1,0) - weight = u0 * v * w0
        [-du * v * w0, dv * u0 * w0, -dw * u0 * v],
        // corner 4: (0,0,1) - weight = u0 * v0 * w
        [-du * v0 * w, -dv * u0 * w, dw * u0 * v0],
        // corner 5: (1,0,1) - weight = u * v0 * w
        [du * v0 * w, -dv * u * w, dw * u * v0],
        // corner 6: (1,1,1) - weight = u * v * w
        [du * v * w, dv * u * w, dw * u * v],
        // corner 7: (0,1,1) - weight = u0 * v * w
        [-du * v * w, dv * u0 * w, dw * u0 * v],
    ]
}

/// Interpolate values using the specified mode.
///
/// # Arguments
/// * `corner_values` - Values at the 8 corners in marching cubes order
/// * `local` - Local coordinates within the cell [0, 1]³
/// * `mode` - Interpolation mode
///
/// # Returns
/// Interpolated value at the local coordinate.
pub fn interpolate<const N: usize>(
    corner_values: &[[f32; N]; 8],
    local: LocalCoord,
    mode: InterpolationMode,
) -> [f32; N] {
    let weights = compute_weights_with_mode(local, mode);

    let mut result = [0.0f32; N];
    for j in 0..N {
        for i in 0..8 {
            result[j] += weights[i] * corner_values[i][j];
        }
    }

    result
}

/// Compute gradient of interpolated value with respect to position.
///
/// # Arguments
/// * `corner_values` - Values at the 8 corners
/// * `local` - Local coordinates within the cell [0, 1]³
/// * `mode` - Interpolation mode
///
/// # Returns
/// Gradient [dv/du, dv/dv, dv/dw] for each of the N features.
pub fn interpolate_gradient<const N: usize>(
    corner_values: &[[f32; N]; 8],
    local: LocalCoord,
    mode: InterpolationMode,
) -> [[f32; 3]; N] {
    let weight_grads = compute_weight_gradients_with_mode(local, mode);

    let mut result = [[0.0f32; 3]; N];
    for j in 0..N {
        for i in 0..8 {
            for k in 0..3 {
                result[j][k] += weight_grads[i][k] * corner_values[i][j];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothstep() {
        // Test endpoints
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-6);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-6);

        // Test midpoint
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-6);

        // Test monotonicity
        let mut prev = 0.0;
        for i in 1..=10 {
            let t = i as f32 / 10.0;
            let v = smoothstep(t);
            assert!(v >= prev, "smoothstep should be monotonic");
            prev = v;
        }
    }

    #[test]
    fn test_smoothstep_derivative() {
        // Derivative should be 0 at endpoints
        assert!(smoothstep_derivative(0.0).abs() < 1e-6);
        assert!(smoothstep_derivative(1.0).abs() < 1e-6);

        // Maximum at midpoint
        let mid_deriv = smoothstep_derivative(0.5);
        assert!((mid_deriv - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_smootherstep() {
        // Test endpoints
        assert!((smootherstep(0.0) - 0.0).abs() < 1e-6);
        assert!((smootherstep(1.0) - 1.0).abs() < 1e-6);

        // Test midpoint
        assert!((smootherstep(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let test_cases = [
            LocalCoord::new(0.0, 0.0, 0.0),
            LocalCoord::new(1.0, 1.0, 1.0),
            LocalCoord::new(0.5, 0.5, 0.5),
            LocalCoord::new(0.25, 0.75, 0.33),
        ];

        for mode in [
            InterpolationMode::Linear,
            InterpolationMode::SmoothStep,
            InterpolationMode::SmootherStep,
        ] {
            for local in test_cases {
                let weights = compute_weights_with_mode(local, mode);
                let sum: f32 = weights.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Weights sum to {} for {:?} mode {:?}",
                    sum,
                    local,
                    mode
                );
            }
        }
    }

    #[test]
    fn test_linear_matches_ash_core() {
        let local = LocalCoord::new(0.3, 0.7, 0.5);

        let our_weights = compute_weights_with_mode(local, InterpolationMode::Linear);
        let ash_core_weights = ash_core::compute_trilinear_weights(local);

        for i in 0..8 {
            assert!(
                (our_weights[i] - ash_core_weights[i]).abs() < 1e-6,
                "Weight {} mismatch: {} vs {}",
                i,
                our_weights[i],
                ash_core_weights[i]
            );
        }
    }

    #[test]
    fn test_interpolate() {
        // Create corner values with a simple linear function f(x,y,z) = x + y + z
        let corner_values: [[f32; 1]; 8] = [
            [0.0], // (0,0,0)
            [1.0], // (1,0,0)
            [2.0], // (1,1,0)
            [1.0], // (0,1,0)
            [1.0], // (0,0,1)
            [2.0], // (1,0,1)
            [3.0], // (1,1,1)
            [2.0], // (0,1,1)
        ];

        let local = LocalCoord::new(0.5, 0.5, 0.5);
        let result = interpolate(&corner_values, local, InterpolationMode::Linear);

        // For f(x,y,z) = x + y + z at (0.5, 0.5, 0.5), should get 1.5
        assert!((result[0] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_smoothstep_continuity() {
        // Test that smoothstep provides smoother transitions
        let corner_values: [[f32; 1]; 8] = [
            [0.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0], [0.0],
        ];

        // Sample along x-axis at y=0.5, z=0.5
        let mut linear_derivs = Vec::new();
        let mut smooth_derivs = Vec::new();

        for i in 0..10 {
            let u = i as f32 / 9.0;
            let local = LocalCoord::new(u, 0.5, 0.5);

            let linear_grad = interpolate_gradient(&corner_values, local, InterpolationMode::Linear);
            let smooth_grad =
                interpolate_gradient(&corner_values, local, InterpolationMode::SmoothStep);

            linear_derivs.push(linear_grad[0][0]);
            smooth_derivs.push(smooth_grad[0][0]);
        }

        // Smoothstep derivatives should be 0 at endpoints
        assert!(smooth_derivs[0].abs() < 1e-5, "Start derivative should be 0");
        assert!(smooth_derivs[9].abs() < 1e-5, "End derivative should be 0");
    }
}
