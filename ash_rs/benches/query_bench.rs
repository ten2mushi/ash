//! Criterion benchmarks for ash_rs query performance.

use ash_core::{BlockCoord, Point3};
use ash_rs::{GridBuilder, SparseDenseGrid};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Create a test grid with sphere SDF
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

/// Create a larger grid for realistic benchmarks
fn make_large_grid() -> SparseDenseGrid {
    let center = Point3::new(2.0, 2.0, 2.0);
    let radius = 1.5;

    let mut builder = GridBuilder::new(8, 0.1).with_capacity(200);

    // 5x5x5 block grid
    for bz in 0..5 {
        for by in 0..5 {
            for bx in 0..5 {
                let coord = BlockCoord::new(bx, by, bz);
                builder = builder.add_block_fn(coord, |pos| (pos - center).length() - radius);
            }
        }
    }

    builder.build().unwrap()
}

fn bench_single_query(c: &mut Criterion) {
    let grid = make_sphere_grid();

    c.bench_function("single_query_warm", |b| {
        let point = Point3::new(0.4, 0.4, 0.4);
        b.iter(|| black_box(grid.query(black_box(point))))
    });

    c.bench_function("single_query_gradient", |b| {
        let point = Point3::new(0.4, 0.4, 0.4);
        b.iter(|| black_box(grid.query_with_gradient(black_box(point))))
    });

    c.bench_function("single_query_collision", |b| {
        let point = Point3::new(0.4, 0.4, 0.4);
        b.iter(|| black_box(grid.in_collision(black_box(point), 0.1)))
    });
}

fn bench_batch_query(c: &mut Criterion) {
    let grid = make_large_grid();

    let mut group = c.benchmark_group("batch_query");

    for size in [10, 100, 1000, 10000] {
        let points: Vec<Point3> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                Point3::new(t * 4.0, t * 4.0, t * 4.0)
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch", size), &points, |b, points| {
            b.iter(|| black_box(grid.query_batch(black_box(points))))
        });

        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &points,
            |b, points| {
                b.iter(|| {
                    let results: Vec<_> = points.iter().map(|p| grid.query(*p)).collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

fn bench_hash_lookup(c: &mut Criterion) {
    let grid = make_large_grid();

    c.bench_function("block_lookup_hit", |b| {
        let coord = BlockCoord::new(2, 2, 2);
        b.iter(|| black_box(grid.get_block_index(black_box(coord))))
    });

    c.bench_function("block_lookup_miss", |b| {
        let coord = BlockCoord::new(100, 100, 100);
        b.iter(|| black_box(grid.get_block_index(black_box(coord))))
    });
}

fn bench_builder(c: &mut Criterion) {
    c.bench_function("build_8_blocks", |b| {
        b.iter(|| {
            let mut builder = GridBuilder::new(8, 0.1).with_capacity(8);
            for bz in 0..2 {
                for by in 0..2 {
                    for bx in 0..2 {
                        builder = builder
                            .add_block_constant(BlockCoord::new(bx, by, bz), 0.5);
                    }
                }
            }
            black_box(builder.build().unwrap())
        })
    });

    c.bench_function("build_125_blocks", |b| {
        b.iter(|| {
            let mut builder = GridBuilder::new(8, 0.1).with_capacity(200);
            for bz in 0..5 {
                for by in 0..5 {
                    for bx in 0..5 {
                        builder = builder
                            .add_block_constant(BlockCoord::new(bx, by, bz), 0.5);
                    }
                }
            }
            black_box(builder.build().unwrap())
        })
    });
}

#[cfg(feature = "serde")]
fn bench_serialization(c: &mut Criterion) {
    use std::io::Cursor;

    let grid = make_large_grid();
    let mut buffer = Vec::new();
    grid.save(&mut buffer).unwrap();

    c.bench_function("save_125_blocks", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(buffer.len());
            grid.save(&mut buf).unwrap();
            black_box(buf)
        })
    });

    c.bench_function("load_125_blocks", |b| {
        b.iter(|| {
            let mut cursor = Cursor::new(&buffer);
            black_box(SparseDenseGrid::load(&mut cursor).unwrap())
        })
    });
}

#[cfg(any(feature = "std", feature = "alloc"))]
fn bench_mesh_extraction(c: &mut Criterion) {
    let grid = make_sphere_grid();

    c.bench_function("extract_single_block_mesh", |b| {
        b.iter(|| {
            black_box(grid.extract_block_mesh(BlockCoord::new(0, 0, 0), 0.0))
        })
    });

    c.bench_function("extract_mesh_callback", |b| {
        b.iter(|| {
            let mut count = 0;
            grid.extract_mesh_callback(0.0, |_| count += 1);
            black_box(count)
        })
    });
}

/// Benchmark shared memory interface (simulates neural_ash integration)
#[cfg(any(feature = "std", feature = "alloc"))]
fn bench_shared_memory(c: &mut Criterion) {
    use ash_rs::{compute_shared_size, SharedGridView, SharedHeader, SharedLayout, SHARED_MAGIC};
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    // Create a shared memory region
    let grid_dim = 8u32;
    let capacity = 200usize;
    let cells_per_block = (grid_dim as usize).pow(3);
    let size = compute_shared_size(grid_dim, capacity);
    let layout = SharedLayout::compute(grid_dim, capacity);

    // Allocate aligned memory (simulates mmap'd shared region)
    let mut buffer = vec![0u8; size + 64];
    let aligned_ptr = {
        let ptr = buffer.as_mut_ptr();
        let offset = ptr.align_offset(64);
        unsafe { ptr.add(offset) }
    };

    // Initialize header
    unsafe {
        let header = &mut *(aligned_ptr as *mut SharedHeader);
        header.magic = SHARED_MAGIC;
        header.grid_dim = grid_dim;
        header.cell_size = 0.1;
        header.capacity = capacity as u32;
        header.num_blocks = AtomicU32::new(125);
        header.global_version = AtomicU64::new(0);
    }

    // Initialize block map (simplified: just add blocks 0-124)
    unsafe {
        let map_ptr = aligned_ptr.add(layout.block_map_offset) as *mut AtomicU64;
        for i in 0..125 {
            let bx = i % 5;
            let by = (i / 5) % 5;
            let bz = i / 25;
            let coord = BlockCoord::new(bx as i32, by as i32, bz as i32);
            let morton = ash_core::morton_encode_signed(coord);
            let hash = morton as u32;
            let idx = (morton as usize) % (capacity * 2);

            // Pack: state=2 (occupied), index=i, hash
            let entry = (2u64 << 62) | ((i as u64) << 32) | (hash as u64);
            (*map_ptr.add(idx)).store(entry, Ordering::Release);
        }
    }

    // Initialize values (sphere SDF)
    let center = Point3::new(2.0, 2.0, 2.0);
    let radius = 1.5;
    unsafe {
        let values_ptr = aligned_ptr.add(layout.values_offset) as *mut f32;
        for block_idx in 0..125 {
            let bx = block_idx % 5;
            let by = (block_idx / 5) % 5;
            let bz = block_idx / 25;

            for z in 0..grid_dim {
                for y in 0..grid_dim {
                    for x in 0..grid_dim {
                        let cell_idx = (x + y * grid_dim + z * grid_dim * grid_dim) as usize;
                        let pos = Point3::new(
                            bx as f32 * 0.8 + x as f32 * 0.1,
                            by as f32 * 0.8 + y as f32 * 0.1,
                            bz as f32 * 0.8 + z as f32 * 0.1,
                        );
                        let sdf = (pos - center).length() - radius;
                        *values_ptr.add(block_idx * cells_per_block + cell_idx) = sdf;
                    }
                }
            }
        }
    }

    // Initialize versions (all even = not being written)
    unsafe {
        let versions_ptr = aligned_ptr.add(layout.versions_offset) as *mut AtomicU64;
        for i in 0..capacity {
            (*versions_ptr.add(i)).store(0, Ordering::Release);
        }
    }

    // Create view
    let view = unsafe { SharedGridView::from_ptr(aligned_ptr as *const u8).unwrap() };

    // Benchmark shared memory queries
    c.bench_function("shared_single_query", |b| {
        let point = Point3::new(2.0, 2.0, 2.0);
        b.iter(|| black_box(view.query(black_box(point))))
    });

    c.bench_function("shared_collision_check", |b| {
        let point = Point3::new(2.0, 2.0, 2.0);
        b.iter(|| black_box(view.in_collision(black_box(point), 0.1)))
    });

    // Benchmark to show warm cache benefit (query same region repeatedly)
    c.bench_function("shared_warm_cache_locality", |b| {
        // Query points in a small region - should benefit from cache
        let points: Vec<Point3> = (0..100)
            .map(|i| {
                let t = i as f32 / 100.0;
                Point3::new(2.0 + t * 0.1, 2.0 + t * 0.1, 2.0 + t * 0.1)
            })
            .collect();

        b.iter(|| {
            let mut sum = 0.0f32;
            for p in &points {
                if let Some(v) = view.query(*p) {
                    sum += v;
                }
            }
            black_box(sum)
        })
    });
}

fn bench_raycast(c: &mut Criterion) {
    let grid = make_sphere_grid();

    c.bench_function("raycast_hit", |b| {
        let origin = Point3::new(2.0, 0.4, 0.4);
        let direction = Point3::new(-1.0, 0.0, 0.0);
        b.iter(|| {
            black_box(grid.raycast(
                black_box(origin),
                black_box(direction),
                10.0,
                0.01,
            ))
        })
    });

    c.bench_function("raycast_miss", |b| {
        let origin = Point3::new(2.0, 2.0, 0.4);
        let direction = Point3::new(0.0, 1.0, 0.0);
        b.iter(|| {
            black_box(grid.raycast(
                black_box(origin),
                black_box(direction),
                10.0,
                0.01,
            ))
        })
    });
}

criterion_group!(
    benches,
    bench_single_query,
    bench_batch_query,
    bench_hash_lookup,
    bench_builder,
    bench_raycast,
);

#[cfg(feature = "serde")]
criterion_group!(serde_benches, bench_serialization);

#[cfg(any(feature = "std", feature = "alloc"))]
criterion_group!(mesh_benches, bench_mesh_extraction);

#[cfg(any(feature = "std", feature = "alloc"))]
criterion_group!(shared_benches, bench_shared_memory);

#[cfg(all(feature = "serde", any(feature = "std", feature = "alloc")))]
criterion_main!(benches, serde_benches, mesh_benches, shared_benches);

#[cfg(all(feature = "serde", not(any(feature = "std", feature = "alloc"))))]
criterion_main!(benches, serde_benches);

#[cfg(all(not(feature = "serde"), any(feature = "std", feature = "alloc")))]
criterion_main!(benches, mesh_benches, shared_benches);

#[cfg(not(any(feature = "serde", any(feature = "std", feature = "alloc"))))]
criterion_main!(benches);
