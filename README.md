# Adaptive Sparse Hierarchical SDF Grid System

library for representing, querying, and manipulating Signed Distance Fields (SDFs) (and any other features) using a sparse-dense hierarchical grid structure.


consists of 4 core crates for the moment:
- ash_core: algorithms & primitives
- ash_io: storage & I/O
- ash_rs: runtime queries (simd & parallel for embedded platforms with low memory cost of querying)
- neural_ash: learning applications on the grid (updates the grid for ash_rs queries)

ash_io owns data format for the environement:
can import/export .obj files to .ash files

ash_rs owns the real-time queries on the environement:
it either uses .ash files or in memory world representation

neural_ash owns the manipulation of features for each point in the worl (bare minimum is a sdf, but ash world format supports any number of features to create semantic map fo the world).
the intent is to have a system where neural nets can each process specific inputs and update specific subsets of features associated with points in the world. Those updates are either done on a .ash file, or directly in memory (ash_rs queries file/memory for the latest owrld representation)

## file formats

### .ash Binary Format

Compact binary format for SDF grids:

```
┌────────────────────────────────────────┐
│ Header (32 bytes)                      │
│   Magic: "ASH1"                        │
│   Version, grid_dim, cell_size         │
│   num_blocks, feature_dim              │
├────────────────────────────────────────┤
│ Block Coordinates                      │
│   [BlockCoord; num_blocks]             │
├────────────────────────────────────────┤
│ SDF Values                             │
│   [f32; num_blocks × cells_per_block]  │
└────────────────────────────────────────┘
```

### Wavefront OBJ

Standard OBJ format for mesh import/export. Supports:
- Vertex positions (`v x y z`)
- Triangle faces (`f i j k`)
- Vertex normals (computed automatically)


### basic usage

```rust
use ash_rs::{GridBuilder, SparseDenseGrid, Point3, BlockCoord};

// Build a grid with a sphere SDF
let grid = GridBuilder::new(8, 0.1)  // 8³ cells/block, 0.1m cells
    .with_capacity(100)
    .add_block_fn(BlockCoord::new(0, 0, 0), |pos| {
        let center = Point3::new(0.4, 0.4, 0.4);
        (pos - center).length() - 0.3  // Sphere radius 0.3
    })
    .build()
    .expect("Failed to build grid");

// Query SDF
if let Some(sdf) = grid.query(Point3::new(0.5, 0.5, 0.5)) {
    println!("Distance to surface: {:.3}m", sdf);
}

// Collision detection
if grid.in_collision(Point3::new(0.4, 0.4, 0.4), 0.05) {
    println!("Collision!");
}
```

## build

```bash
# Build all crates (release mode recommended for performance)
cd ash_workspace
cargo build --release

# Build with SIMD optimizations
cargo build --release --features simd

# Build with all features
cargo build --release --features "simd mesh rayon"
```

## tests

```bash
# Test all crates
cargo test --features simd

# Test individual crates
cargo test -p ash_core --features simd
cargo test -p ash_io --features "simd rayon"
cargo test -p ash_rs --features "simd mesh"

# Run with release optimizations (for performance tests)
cargo test --release --features simd
```

## Examples

The `examples` crate contains several demonstration binaries.

```bash
# OBJ Roundtrip (default paths)
cargo run --release -p ash_examples --features simd --bin obj_roundtrip

# OBJ Roundtrip (custom paths)
cargo run --release -p ash_examples --features simd --bin obj_roundtrip -- \
    input.obj output.obj intermediate.ash

# Mesh Extraction
cargo run --release -p ash_examples --features simd --bin mesh_extraction -- \
    input.obj output.obj

# Benchmark Suite (small)
cargo run --release -p ash_examples --features simd --bin benchmark_suite

# Benchmark Suite (large)
cargo run --release -p ash_examples --features simd --bin benchmark_suite -- --large

# Optimization Benchmarks
cargo run --release -p ash_examples --features simd --bin benchmark_optimizations
```

```bash
# Using default paths (input/landscape.obj → output/roundtrip.obj)
cargo run --release -p ash_examples --features simd --bin obj_roundtrip

# With custom paths
cargo run --release -p ash_examples --features simd --bin obj_roundtrip -- \
    path/to/input.obj \
    path/to/output.obj \
    path/to/intermediate.ash
```

there are also examples in the neural_ash crate:

```bash
cargo run --example denoising_tsdf --features examples
```

```bash
# Using default paths
cargo run --release -p ash_examples --features simd --bin mesh_extraction

# With custom paths
cargo run --release -p ash_examples --features simd --bin mesh_extraction -- \
    path/to/input.obj \
    path/to/output.obj
```

### benchmarks

Tests the new performance optimizations (BVH, narrow band, batch operations).

```bash
cargo run --release -p ash_examples --features simd --bin benchmark_optimizations
```

### tests

```bash
# All tests
cargo test --features simd

# Specific crate
cargo test -p ash_core --features simd
cargo test -p ash_io --features "simd rayon"
cargo test -p ash_rs --features "simd mesh"
cargo test -p neural_ash
```

## feature flags

### ash_core

| Feature | Description |
|---------|-------------|
| `std` | Standard library support (default) |
| `alloc` | Heap allocation without full std |
| `simd` | SIMD acceleration via `wide` crate |

### ash_io

| Feature | Description |
|---------|-------------|
| `std` | Standard library support (default) |
| `alloc` | Heap allocation without full std |
| `simd` | SIMD hash acceleration |
| `rayon` | Parallel iteration |

### ash_rs

| Feature | Description |
|---------|-------------|
| `std` | Standard library support (default) |
| `alloc` | Heap allocation without full std |
| `simd` | SIMD batch queries |
| `mesh` | Parallel mesh extraction (enables rayon) |
| `serde` | Binary serialization |


