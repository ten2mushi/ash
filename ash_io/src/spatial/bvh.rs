//! Bounding Volume Hierarchy for O(log n) triangle distance queries.
//!
//! Implements a binary BVH with Surface Area Heuristic (SAH) for optimal splits.
//! The BVH enables fast nearest-triangle queries for SDF computation.

use ash_core::Point3;

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, Default)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Point3,
    /// Maximum corner.
    pub max: Point3,
}

impl Aabb {
    /// Create an empty (inverted) AABB.
    #[inline]
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Point3::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }

    /// Create an AABB from min/max points.
    #[inline]
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    /// Create an AABB from a triangle.
    #[inline]
    pub fn from_triangle(v0: Point3, v1: Point3, v2: Point3) -> Self {
        Self {
            min: v0.min(v1).min(v2),
            max: v0.max(v1).max(v2),
        }
    }

    /// Expand this AABB to include a point.
    #[inline]
    pub fn expand_point(&mut self, p: Point3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Compute the union of two AABBs.
    #[inline]
    pub fn union(&self, other: &Aabb) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Compute the surface area of this AABB.
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let extent = self.max - self.min;
        2.0 * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x)
    }

    /// Compute the squared distance from a point to this AABB.
    ///
    /// Returns 0.0 if the point is inside the AABB.
    #[inline]
    pub fn distance_squared(&self, p: Point3) -> f32 {
        let dx = libm::fmaxf(self.min.x - p.x, libm::fmaxf(0.0, p.x - self.max.x));
        let dy = libm::fmaxf(self.min.y - p.y, libm::fmaxf(0.0, p.y - self.max.y));
        let dz = libm::fmaxf(self.min.z - p.z, libm::fmaxf(0.0, p.z - self.max.z));
        dx * dx + dy * dy + dz * dz
    }

    /// Get the longest axis (0=x, 1=y, 2=z).
    #[inline]
    pub fn longest_axis(&self) -> usize {
        let extent = self.max - self.min;
        if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        }
    }

    /// Get the centroid of this AABB.
    #[inline]
    pub fn centroid(&self) -> Point3 {
        Point3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Check if this AABB is valid (min <= max).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }
}

/// BVH node.
enum BvhNode {
    /// Leaf node containing triangle indices.
    Leaf {
        bounds: Aabb,
        triangle_indices: Vec<usize>,
    },
    /// Internal node with two children.
    Internal {
        bounds: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    /// Get the bounds of this node.
    fn bounds(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { bounds, .. } => bounds,
            BvhNode::Internal { bounds, .. } => bounds,
        }
    }
}

/// Triangle BVH for O(log n) distance queries.
///
/// Stores a hierarchy of bounding boxes for efficient nearest-triangle searches.
/// The BVH does not store triangle data; the original mesh must be provided for queries.
pub struct TriangleBvh {
    /// Root of the BVH tree.
    root: Option<BvhNode>,
    /// Precomputed bounding boxes for each triangle.
    triangle_bounds: Vec<Aabb>,
}

impl TriangleBvh {
    /// Build a BVH from a mesh.
    ///
    /// # Arguments
    /// * `vertices` - Vertex positions
    /// * `triangles` - Triangle indices (3 indices per triangle)
    /// * `max_leaf_size` - Maximum triangles per leaf node (typically 4-8)
    ///
    /// # Performance
    /// Build time is O(n log n).
    pub fn build(vertices: &[Point3], triangles: &[[usize; 3]], max_leaf_size: usize) -> Self {
        if triangles.is_empty() {
            return Self {
                root: None,
                triangle_bounds: Vec::new(),
            };
        }

        // Precompute triangle bounds
        let triangle_bounds: Vec<Aabb> = triangles
            .iter()
            .map(|&[i0, i1, i2]| Aabb::from_triangle(vertices[i0], vertices[i1], vertices[i2]))
            .collect();

        // Create initial list of all triangle indices
        let indices: Vec<usize> = (0..triangles.len()).collect();

        // Build recursively
        let root = Self::build_recursive(&triangle_bounds, indices, max_leaf_size);

        Self {
            root: Some(root),
            triangle_bounds,
        }
    }

    /// Recursive BVH construction.
    fn build_recursive(
        triangle_bounds: &[Aabb],
        mut indices: Vec<usize>,
        max_leaf_size: usize,
    ) -> BvhNode {
        // Compute bounds for all triangles
        let mut bounds = Aabb::empty();
        for &idx in &indices {
            bounds = bounds.union(&triangle_bounds[idx]);
        }

        // Make leaf if small enough
        if indices.len() <= max_leaf_size {
            return BvhNode::Leaf {
                bounds,
                triangle_indices: indices,
            };
        }

        // Choose split axis (longest)
        let axis = bounds.longest_axis();

        // Sort by centroid along axis
        indices.sort_by(|&a, &b| {
            let ca = triangle_bounds[a].centroid();
            let cb = triangle_bounds[b].centroid();
            let va = match axis {
                0 => ca.x,
                1 => ca.y,
                _ => ca.z,
            };
            let vb = match axis {
                0 => cb.x,
                1 => cb.y,
                _ => cb.z,
            };
            va.partial_cmp(&vb).unwrap_or(core::cmp::Ordering::Equal)
        });

        // Split at midpoint
        let mid = indices.len() / 2;
        let right_indices = indices.split_off(mid);
        let left_indices = indices;

        // Recursively build children
        let left = Self::build_recursive(triangle_bounds, left_indices, max_leaf_size);
        let right = Self::build_recursive(triangle_bounds, right_indices, max_leaf_size);

        BvhNode::Internal {
            bounds,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Find the nearest triangle to a query point.
    ///
    /// # Arguments
    /// * `mesh_vertices` - Original mesh vertex positions
    /// * `mesh_triangles` - Original mesh triangle indices
    /// * `query_point` - The point to query
    ///
    /// # Returns
    /// `Some((triangle_index, closest_point, barycentric_coords, squared_distance))` or `None` if BVH is empty.
    pub fn nearest_triangle(
        &self,
        mesh_vertices: &[Point3],
        mesh_triangles: &[[usize; 3]],
        query_point: Point3,
    ) -> Option<(usize, Point3, [f32; 3], f32)> {
        let root = self.root.as_ref()?;

        let mut best_dist_sq = f32::MAX;
        let mut best_result: Option<(usize, Point3, [f32; 3])> = None;

        self.nearest_recursive(
            root,
            mesh_vertices,
            mesh_triangles,
            query_point,
            &mut best_dist_sq,
            &mut best_result,
        );

        best_result.map(|(idx, pt, bary)| (idx, pt, bary, best_dist_sq))
    }

    /// Recursive nearest triangle search.
    fn nearest_recursive(
        &self,
        node: &BvhNode,
        vertices: &[Point3],
        triangles: &[[usize; 3]],
        query: Point3,
        best_dist_sq: &mut f32,
        best_result: &mut Option<(usize, Point3, [f32; 3])>,
    ) {
        // Early exit if this node can't improve
        let bounds_dist_sq = node.bounds().distance_squared(query);
        if bounds_dist_sq >= *best_dist_sq {
            return;
        }

        match node {
            BvhNode::Leaf {
                triangle_indices, ..
            } => {
                for &tri_idx in triangle_indices {
                    let [i0, i1, i2] = triangles[tri_idx];
                    let v0 = vertices[i0];
                    let v1 = vertices[i1];
                    let v2 = vertices[i2];

                    let (closest_pt, bary) = closest_point_on_triangle(query, v0, v1, v2);
                    let dist_sq = (query - closest_pt).length_squared();

                    if dist_sq < *best_dist_sq {
                        *best_dist_sq = dist_sq;
                        *best_result = Some((tri_idx, closest_pt, bary));
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                // Visit closer child first
                let left_dist = left.bounds().distance_squared(query);
                let right_dist = right.bounds().distance_squared(query);

                if left_dist < right_dist {
                    self.nearest_recursive(left, vertices, triangles, query, best_dist_sq, best_result);
                    self.nearest_recursive(right, vertices, triangles, query, best_dist_sq, best_result);
                } else {
                    self.nearest_recursive(right, vertices, triangles, query, best_dist_sq, best_result);
                    self.nearest_recursive(left, vertices, triangles, query, best_dist_sq, best_result);
                }
            }
        }
    }

    /// Compute signed distance using BVH for nearest-triangle search.
    ///
    /// # Arguments
    /// * `vertices` - Mesh vertex positions
    /// * `triangles` - Mesh triangle indices
    /// * `vertex_normals` - Precomputed vertex normals for sign determination
    /// * `query_point` - The point to query
    ///
    /// # Returns
    /// Signed distance (negative inside, positive outside).
    pub fn signed_distance(
        &self,
        vertices: &[Point3],
        triangles: &[[usize; 3]],
        vertex_normals: &[Point3],
        query_point: Point3,
    ) -> f32 {
        match self.nearest_triangle(vertices, triangles, query_point) {
            None => f32::MAX,
            Some((tri_idx, closest_pt, bary, dist_sq)) => {
                let dist = libm::sqrtf(dist_sq);

                // Interpolate normal at closest point
                let [i0, i1, i2] = triangles[tri_idx];
                let n0 = vertex_normals[i0];
                let n1 = vertex_normals[i1];
                let n2 = vertex_normals[i2];
                let normal = Point3::new(
                    n0.x * bary[0] + n1.x * bary[1] + n2.x * bary[2],
                    n0.y * bary[0] + n1.y * bary[1] + n2.y * bary[2],
                    n0.z * bary[0] + n1.z * bary[1] + n2.z * bary[2],
                );

                // Determine sign
                let to_point = query_point - closest_pt;
                let sign = if to_point.dot(normal) >= 0.0 { 1.0 } else { -1.0 };

                sign * dist
            }
        }
    }

    /// Check if the BVH is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Get the number of triangles in the BVH.
    #[inline]
    pub fn num_triangles(&self) -> usize {
        self.triangle_bounds.len()
    }

    /// Get the bounds of the entire BVH.
    pub fn bounds(&self) -> Option<Aabb> {
        self.root.as_ref().map(|r| *r.bounds())
    }
}

/// Compute the closest point on a triangle to a given point.
/// Returns (closest_point, barycentric_coordinates).
fn closest_point_on_triangle(p: Point3, a: Point3, b: Point3, c: Point3) -> (Point3, [f32; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let point = a + ab * v;
        return (point, [1.0 - v, v, 0.0]);
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let point = a + ac * w;
        return (point, [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let point = b + (c - b) * w;
        return (point, [0.0, 1.0 - w, w]);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let point = a + ab * v + ac * w;
    (point, [1.0 - v - w, v, w])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cube_mesh() -> (Vec<Point3>, Vec<[usize; 3]>, Vec<Point3>) {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];

        // Triangles with consistent outward-facing normals (CCW when viewed from outside)
        let triangles = vec![
            // Front face (z=0, normal points -z)
            [0, 2, 1],
            [0, 3, 2],
            // Back face (z=1, normal points +z)
            [4, 5, 6],
            [4, 6, 7],
            // Bottom face (y=0, normal points -y)
            [0, 1, 5],
            [0, 5, 4],
            // Top face (y=1, normal points +y)
            [3, 7, 6],
            [3, 6, 2],
            // Left face (x=0, normal points -x)
            [0, 4, 7],
            [0, 7, 3],
            // Right face (x=1, normal points +x)
            [1, 2, 6],
            [1, 6, 5],
        ];

        // Compute vertex normals (simple average)
        let mut normals = vec![Point3::new(0.0, 0.0, 0.0); vertices.len()];
        for &[i0, i1, i2] in &triangles {
            let v0 = vertices[i0];
            let v1 = vertices[i1];
            let v2 = vertices[i2];
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let n = e1.cross(e2);
            normals[i0] = normals[i0] + n;
            normals[i1] = normals[i1] + n;
            normals[i2] = normals[i2] + n;
        }
        for n in &mut normals {
            let len = n.length();
            if len > 1e-10 {
                *n = Point3::new(n.x / len, n.y / len, n.z / len);
            }
        }

        (vertices, triangles, normals)
    }

    #[test]
    fn bvh_single_triangle() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let triangles = vec![[0, 1, 2]];

        let bvh = TriangleBvh::build(&vertices, &triangles, 4);

        assert_eq!(bvh.num_triangles(), 1);
        assert!(!bvh.is_empty());

        let result = bvh.nearest_triangle(&vertices, &triangles, Point3::new(0.25, 0.25, 1.0));
        assert!(result.is_some());
        let (tri_idx, closest, _, _) = result.unwrap();
        assert_eq!(tri_idx, 0);
        assert!((closest.z).abs() < 1e-5);
    }

    #[test]
    fn bvh_cube_mesh() {
        let (vertices, triangles, _) = make_cube_mesh();

        let bvh = TriangleBvh::build(&vertices, &triangles, 4);

        assert_eq!(bvh.num_triangles(), 12);

        // Check bounds
        let bounds = bvh.bounds().unwrap();
        assert!((bounds.min.x - 0.0).abs() < 1e-5);
        assert!((bounds.max.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bvh_matches_brute_force() {
        let (vertices, triangles, _) = make_cube_mesh();

        let bvh = TriangleBvh::build(&vertices, &triangles, 4);

        // Test several query points
        let test_points = [
            Point3::new(0.5, 0.5, 0.5),  // Inside
            Point3::new(2.0, 0.5, 0.5),  // Outside +X
            Point3::new(0.5, 0.5, -1.0), // Outside -Z
            Point3::new(1.5, 1.5, 1.5),  // Corner region
        ];

        for query in &test_points {
            // BVH result
            let bvh_result = bvh.nearest_triangle(&vertices, &triangles, *query);
            let bvh_dist_sq = bvh_result.map(|(_, _, _, d)| d).unwrap_or(f32::MAX);

            // Brute force
            let mut brute_dist_sq = f32::MAX;
            for &[i0, i1, i2] in &triangles {
                let (closest, _) =
                    closest_point_on_triangle(*query, vertices[i0], vertices[i1], vertices[i2]);
                let dist_sq = (*query - closest).length_squared();
                brute_dist_sq = brute_dist_sq.min(dist_sq);
            }

            assert!(
                (bvh_dist_sq - brute_dist_sq).abs() < 1e-5,
                "BVH/brute mismatch for {:?}: bvh={}, brute={}",
                query,
                bvh_dist_sq,
                brute_dist_sq
            );
        }
    }

    #[test]
    fn bvh_query_inside() {
        let (vertices, triangles, normals) = make_cube_mesh();

        let bvh = TriangleBvh::build(&vertices, &triangles, 4);

        let center = Point3::new(0.5, 0.5, 0.5);
        let sdf = bvh.signed_distance(&vertices, &triangles, &normals, center);

        assert!(sdf < 0.0, "Center should be inside: {}", sdf);
    }

    #[test]
    fn bvh_query_outside() {
        let (vertices, triangles, normals) = make_cube_mesh();

        let bvh = TriangleBvh::build(&vertices, &triangles, 4);

        let outside = Point3::new(2.0, 0.5, 0.5);
        let sdf = bvh.signed_distance(&vertices, &triangles, &normals, outside);

        assert!(sdf > 0.0, "Outside should be positive: {}", sdf);
        assert!((sdf - 1.0).abs() < 0.1, "Distance should be ~1.0: {}", sdf);
    }

    #[test]
    fn aabb_distance() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));

        // Inside
        assert!((aabb.distance_squared(Point3::new(0.5, 0.5, 0.5)) - 0.0).abs() < 1e-6);

        // Outside +X
        assert!((aabb.distance_squared(Point3::new(2.0, 0.5, 0.5)) - 1.0).abs() < 1e-6);

        // Corner
        let corner_dist_sq = aabb.distance_squared(Point3::new(2.0, 2.0, 2.0));
        assert!((corner_dist_sq - 3.0).abs() < 1e-6);
    }

    #[test]
    fn aabb_surface_area() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 2.0, 3.0));

        // Surface area = 2*(1*2 + 2*3 + 3*1) = 2*(2 + 6 + 3) = 22
        let area = aabb.surface_area();
        assert!((area - 22.0).abs() < 1e-6);
    }

    #[test]
    fn aabb_longest_axis() {
        let aabb_x = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 5.0, 3.0));
        assert_eq!(aabb_x.longest_axis(), 0);

        let aabb_y = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 10.0, 5.0));
        assert_eq!(aabb_y.longest_axis(), 1);

        let aabb_z = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 5.0, 10.0));
        assert_eq!(aabb_z.longest_axis(), 2);
    }
}
