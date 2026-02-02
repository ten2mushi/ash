//! Integration tests for loss functions.

use burn::backend::NdArray;
use burn::prelude::*;

use neural_ash::{
    config::SdfLossConfig,
    loss::SdfLoss,
};

type TestBackend = NdArray;

#[test]
fn test_surface_loss_zero_input() {
    let device = Default::default();
    let loss_fn = SdfLoss::new(SdfLossConfig::default());

    // Zero SDF at surface should give zero loss
    let sdf = Tensor::<TestBackend, 2>::zeros([100, 1], &device);
    let loss = loss_fn.surface_loss(sdf);

    let value: f32 = loss.to_data().to_vec().unwrap()[0];
    assert!(value.abs() < 1e-6, "Expected zero loss, got {}", value);
}

#[test]
fn test_surface_loss_nonzero_input() {
    let device = Default::default();
    let loss_fn = SdfLoss::new(SdfLossConfig::default());

    // Non-zero SDF should give positive loss
    let sdf = Tensor::<TestBackend, 2>::full([100, 1], 0.5, &device);
    let loss = loss_fn.surface_loss(sdf);

    let value: f32 = loss.to_data().to_vec().unwrap()[0];
    assert!(value > 0.0, "Expected positive loss, got {}", value);
    // Loss should be mean(0.5^2) = 0.25
    assert!((value - 0.25).abs() < 1e-5, "Expected 0.25, got {}", value);
}

#[test]
fn test_free_space_loss_above_threshold() {
    let device = Default::default();
    let config = SdfLossConfig::new()
        .with_free_space_threshold(0.1);
    let loss_fn = SdfLoss::new(config);

    // SDF above threshold should give zero loss
    let sdf = Tensor::<TestBackend, 2>::full([100, 1], 0.5, &device);
    let loss = loss_fn.free_space_loss(sdf);

    let value: f32 = loss.to_data().to_vec().unwrap()[0];
    assert!(value.abs() < 1e-6, "Expected zero loss, got {}", value);
}

#[test]
fn test_free_space_loss_below_threshold() {
    let device = Default::default();
    let config = SdfLossConfig::new()
        .with_free_space_threshold(0.1);
    let loss_fn = SdfLoss::new(config);

    // SDF below threshold should give positive loss
    let sdf = Tensor::<TestBackend, 2>::full([100, 1], 0.05, &device);
    let loss = loss_fn.free_space_loss(sdf);

    let value: f32 = loss.to_data().to_vec().unwrap()[0];
    // Loss should be mean(max(0, 0.1 - 0.05)) = mean(0.05) = 0.05
    assert!((value - 0.05).abs() < 1e-5, "Expected 0.05, got {}", value);
}

#[test]
fn test_eikonal_loss_unit_gradient() {
    let device = Default::default();
    let loss_fn = SdfLoss::new(SdfLossConfig::default());

    // Unit gradients should give zero Eikonal loss
    let gradients = Tensor::<TestBackend, 2>::from_data(
        [
            [1.0f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.577, 0.577, 0.577], // Approximately unit length
        ],
        &device,
    );
    let loss = loss_fn.eikonal_loss(gradients);

    let value: f32 = loss.to_data().to_vec().unwrap()[0];
    assert!(value < 0.01, "Expected near-zero loss, got {}", value);
}

#[test]
fn test_eikonal_loss_non_unit_gradient() {
    let device = Default::default();
    let loss_fn = SdfLoss::new(SdfLossConfig::default());

    // Non-unit gradients should give positive Eikonal loss
    let gradients = Tensor::<TestBackend, 2>::from_data(
        [
            [2.0f32, 0.0, 0.0], // Magnitude 2
            [0.5, 0.0, 0.0],   // Magnitude 0.5
        ],
        &device,
    );
    let loss = loss_fn.eikonal_loss(gradients);

    let value: f32 = loss.to_data().to_vec().unwrap()[0];
    assert!(value > 0.0, "Expected positive loss, got {}", value);
}

#[test]
fn test_combined_loss() {
    let device = Default::default();
    let config = SdfLossConfig::new()
        .with_surface_weight(1.0)
        .with_free_space_weight(0.5)
        .with_eikonal_weight(0.1)
        .with_free_space_threshold(0.1);
    let loss_fn = SdfLoss::new(config);

    let surface_sdf = Tensor::<TestBackend, 2>::full([50, 1], 0.1, &device);
    let free_space_sdf = Tensor::<TestBackend, 2>::full([50, 1], 0.2, &device);
    // Create unit gradient vectors using repeat pattern
    // Start with a single unit vector and expand
    let unit_x = Tensor::<TestBackend, 2>::from_data([[1.0f32, 0.0, 0.0]], &device);
    let gradients = unit_x.repeat_dim(0, 50);

    let (total, surface, free, eikonal) =
        loss_fn.combined_loss(surface_sdf, free_space_sdf, gradients);

    // All losses should be finite
    assert!(total.to_data().to_vec::<f32>().unwrap()[0].is_finite());
    assert!(surface.to_data().to_vec::<f32>().unwrap()[0].is_finite());
    assert!(free.to_data().to_vec::<f32>().unwrap()[0].is_finite());
    assert!(eikonal.to_data().to_vec::<f32>().unwrap()[0].is_finite());

    // Total should be weighted sum of components
    let total_val: f32 = total.to_data().to_vec().unwrap()[0];
    let surface_val: f32 = surface.to_data().to_vec().unwrap()[0];
    let free_val: f32 = free.to_data().to_vec().unwrap()[0];
    let eikonal_val: f32 = eikonal.to_data().to_vec().unwrap()[0];

    let expected_total = 1.0 * surface_val + 0.5 * free_val + 0.1 * eikonal_val;
    assert!(
        (total_val - expected_total).abs() < 1e-5,
        "Total {} != weighted sum {}",
        total_val,
        expected_total
    );
}

#[test]
fn test_finite_difference_gradient() {
    let device = Default::default();
    let loss_fn = SdfLoss::new(SdfLossConfig::new()
        .with_gradient_epsilon(0.01));

    // Test with a simple linear function: f(x,y,z) = x
    // Gradient should be (1, 0, 0)
    let query_fn = |points: Tensor<TestBackend, 2>| -> Tensor<TestBackend, 2> {
        let n = points.dims()[0];
        points.slice([0..n, 0..1])
    };

    let points = Tensor::<TestBackend, 2>::from_data(
        [[0.5f32, 0.5, 0.5], [1.0, 1.0, 1.0]],
        &device,
    );

    let gradients = loss_fn.finite_difference_gradient(query_fn, points);

    let data = gradients.to_data();
    let values: Vec<f32> = data.to_vec().unwrap();

    // Should be approximately (1, 0, 0) for each point
    assert!((values[0] - 1.0).abs() < 0.01, "Expected grad_x = 1, got {}", values[0]);
    assert!(values[1].abs() < 0.01, "Expected grad_y = 0, got {}", values[1]);
    assert!(values[2].abs() < 0.01, "Expected grad_z = 0, got {}", values[2]);
}
