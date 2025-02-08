use crate::util::indexing::*;
use crate::util::*;

/// Axis Aligned Bounding Box (AABB) for coordinate types.
/// Each instance is inclusive of both corners.
/// This class is responsible for alot of indexing operations,
/// where we map between a linear buffer and coordinates.
#[derive(Hash, Debug, Copy, Clone, Eq, PartialEq)]
pub struct AABB<const DIMENSION: usize> {
    pub bounds: Bounds<DIMENSION>,
}

impl<const GRID_DIMENSION: usize> std::fmt::Display for AABB<GRID_DIMENSION> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self.bounds)
    }
}

impl<const DIMENSION: usize> AABB<DIMENSION> {
    /// Create AABB from raw bounds.
    #[inline]
    pub fn new(bounds: Bounds<DIMENSION>) -> Self {
        AABB { bounds }
    }

    /// Create AABB from corners.
    pub fn from_mm(min: Coord<DIMENSION>, max: Coord<DIMENSION>) -> Self {
        let result = AABB {
            bounds: Bounds::from_columns(&[min, max]),
        };
        debug_assert!(result.check_validity());
        result
    }

    /// Moving min to the origin, returns the exclusie size in each direction
    /// i.e. [0, 9]  would have exclusive size of 10.
    pub fn exclusive_bounds(&self) -> Coord<DIMENSION> {
        (self.bounds.column(1) - self.bounds.column(0)).add_scalar(1)
    }

    /// Return the number of coordinates contained in the instance.
    #[inline]
    pub fn buffer_size(&self) -> usize {
        real_buffer_size(&self.exclusive_bounds())
    }

    /// Return the number of complex numbers needed for a FFTW buffer.
    #[inline]
    pub fn complex_buffer_size(&self) -> usize {
        complex_buffer_size(&self.exclusive_bounds())
    }

    /// Return the linear index for a coord in the instance
    pub fn coord_to_linear(&self, coord: &Coord<DIMENSION>) -> usize {
        coord_to_linear(&(coord - self.min()), &self.exclusive_bounds())
    }

    /// Return the coordinate in the instance for a given linear index.
    pub fn linear_to_coord(&self, index: usize) -> Coord<DIMENSION> {
        linear_to_coord(index, &self.exclusive_bounds()) + self.min()
    }

    /// Check whether the instance contains a coordinate.
    pub fn contains(&self, coord: &Coord<DIMENSION>) -> bool {
        for d in 0..DIMENSION {
            if coord[d] < self.bounds[(d, 0)] || coord[d] > self.bounds[(d, 1)]
            {
                return false;
            }
        }
        true
    }

    /// Check whether another AABB is contained in the instance.
    pub fn contains_aabb(&self, other: &Self) -> bool {
        for d in 0..DIMENSION {
            if other.bounds[(d, 0)] < self.bounds[(d, 0)]
                || other.bounds[(d, 1)] > self.bounds[(d, 1)]
            {
                return false;
            }
        }
        true
    }

    pub fn trim_to_aabb(&mut self, other: &Self) {
        for d in 0..DIMENSION {
            self.bounds[(d, 0)] = self.bounds[(d, 0)].max(other.bounds[(d, 0)]);
            self.bounds[(d, 1)] = self.bounds[(d, 1)].min(other.bounds[(d, 1)]);
        }
    }

    /// Element wise add the bounds diff.
    pub fn add_bounds_diff(&self, diff: Bounds<DIMENSION>) -> Self {
        Self::new(self.bounds + diff)
    }

    /// Find the coord within bound assuming periodic boundary conditions.
    /// Assumes that coords are no more than one box away!
    pub fn periodic_coord(&self, coord: &Coord<DIMENSION>) -> Coord<DIMENSION> {
        let mut result = Coord::zero();
        for d in 0..DIMENSION {
            let di_raw = coord[d];
            result[d] = if di_raw < self.bounds[(d, 0)] {
                (self.bounds[(d, 1)] + 1) - (self.bounds[(d, 0)] - di_raw)
            } else if di_raw > self.bounds[(d, 1)] {
                self.bounds[(d, 0)] + (di_raw - self.bounds[(d, 1)] - 1)
            } else {
                di_raw
            }
        }
        result
    }

    // TODO: can we return a view instead of allocating?
    /// Return min corner.
    pub fn min(&self) -> Coord<DIMENSION> {
        self.bounds.column(0).into()
    }

    // TODO: can we return a view instead of allocating?
    /// Return max corner
    pub fn max(&self) -> Coord<DIMENSION> {
        self.bounds.column(1).into()
    }

    /// Check that max >= min
    pub fn check_validity(&self) -> bool {
        for d in 0..DIMENSION {
            if self.bounds[(d, 0)] > self.bounds[(d, 1)] {
                return false;
            }
        }
        true
    }

    // TODO: file bug for this clippy issue
    /// Return iterator over contained coords
    /// in linear ordering.
    #[allow(clippy::needless_lifetimes)]
    pub fn coord_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = Coord<DIMENSION>> + use<'a, DIMENSION> {
        (0..self.buffer_size()).map(|i| self.linear_to_coord(i))
    }

    /// Given a bounding box within self,
    /// return decomposition of remaining coordinate space.
    /// Used for recursion during aperiodic algorithm.
    /// Until generic_const_exprs is stabilized,
    /// we need to return a nested array.
    pub fn decomposition(
        &self,
        center: &AABB<DIMENSION>,
    ) -> [[AABB<DIMENSION>; 2]; DIMENSION] {
        let mut result = [[AABB::new(Bounds::zero()); 2]; DIMENSION];
        let mut remaining_bounds = *self;
        for d in 0..DIMENSION {
            result[d][0] = remaining_bounds;
            result[d][0].bounds[(d, 1)] = center.bounds[(d, 0)] - 1;
            debug_assert!(result[d][0].check_validity());

            result[d][1] = remaining_bounds;
            result[d][1].bounds[(d, 0)] = center.bounds[(d, 1)] + 1;
            debug_assert!(result[d][1].check_validity());

            remaining_bounds.bounds[(d, 0)] = center.bounds[(d, 0)];
            remaining_bounds.bounds[(d, 1)] = center.bounds[(d, 1)];
        }

        result
    }

    /// Return the min of exclusive_bounds()
    pub fn min_size_len(&self) -> i32 {
        self.exclusive_bounds().min()
    }

    /// Using stencil slopes, find the number of steps and resulting AABB
    /// that approximatley shrinks the original bound by some ratio.
    /// This one is hard to explain, maybe a sign it needs to be
    /// broken up, or moved somewhere else?
    pub fn shrink(
        &self,
        ratio: f64,
        slopes: Bounds<DIMENSION>,
        max_steps: Option<usize>,
    ) -> (usize, AABB<DIMENSION>) {
        debug_assert!(ratio < 1.0);
        let inclusive_sides = self.bounds.column(1) - self.bounds.column(0);
        let (min_side_d, min_side_len) = inclusive_sides.argmin();
        let scaled_side_len = ((min_side_len as f64) * ratio) as i32;
        let diff = min_side_len - scaled_side_len;
        let min_side_slope = slopes[(min_side_d, 0)] + slopes[(min_side_d, 1)];

        // Round down?
        let mut steps = diff / min_side_slope;
        if let Some(max) = max_steps {
            if steps > max as i32 {
                steps = max as i32;
            }
        }

        // Okay now we can construct center
        let new_min = self.bounds.column(0) + steps * slopes.column(0);
        let new_max = self.bounds.column(1) - steps * slopes.column(1);

        (steps as usize, AABB::from_mm(new_min, new_max))
    }

    pub fn cell_bounds(&self) -> Self {
        let mut cell_bounds = *self;
        cell_bounds
            .bounds
            .set_column(1, &cell_bounds.bounds.column(1).add_scalar(-1));
        cell_bounds
    }

    pub fn coord_offset_to_linear<const NEIGHBORHOOD_SIZE: usize>(
        &self,
        coord_offsets: &[Coord<DIMENSION>; NEIGHBORHOOD_SIZE],
    ) -> [i32; NEIGHBORHOOD_SIZE] {
        // highest dimension goes the fastest
        let exclusive_bounds = self.exclusive_bounds();
        let mut linear_offsets = [0; NEIGHBORHOOD_SIZE];
        let mut accumulator = 1;
        for d in (0..DIMENSION).rev() {
            for o in 0..NEIGHBORHOOD_SIZE {
                linear_offsets[o] += coord_offsets[o][d] * accumulator;
            }
            accumulator *= exclusive_bounds[d];
        }

        linear_offsets
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use nalgebra::{matrix, vector};

    #[test]
    fn buffer_size_test() {
        {
            let a = AABB::new(matrix![0, 5]);
            let real_size = a.buffer_size();
            assert_eq!(real_size, 6);
            let complex_size = a.complex_buffer_size();
            assert_eq!(complex_size, (6 / 2) + 1);
        }

        {
            let dimensions = AABB::new(matrix![0, 5; 0, 7; 0, 9]);
            let real_size = dimensions.buffer_size();
            assert_eq!(real_size, 6 * 8 * 10);
            let complex_size = dimensions.complex_buffer_size();
            assert_eq!(complex_size, 6 * 8 * ((10 / 2) + 1));
        }

        {
            let dimensions = AABB::new(matrix![1, 6; 1, 8; 1, 10]);
            let real_size = dimensions.buffer_size();
            assert_eq!(real_size, 6 * 8 * 10);
            let complex_size = dimensions.complex_buffer_size();
            assert_eq!(complex_size, 6 * 8 * ((10 / 2) + 1));
        }
    }

    #[test]
    fn coord_to_linear_in_box_test() {
        let bb = AABB::new(matrix![0, 9; 0, 9; 0, 9]);
        let lin_1 = bb.coord_to_linear(&vector![5, 5, 5]);
        let lin_2 = coord_to_linear(&vector![5, 5, 5], &vector![10, 10, 10]);
        assert_eq!(lin_1, lin_2);
    }

    #[test]
    fn linear_to_coord_test() {
        {
            let bb = AABB::new(matrix![2, 8]);
            let c_1 = bb.linear_to_coord(5);
            assert_eq!(c_1, vector![7]);
        }

        {
            let a = AABB::new(matrix![1, 9]);
            assert_eq!(a.linear_to_coord(0), vector![1]);
        }
    }

    #[test]
    fn in_box_comp_test() {
        {
            let bound = AABB::new(matrix![0, 9]);
            let c = vector![8];
            let li = bound.coord_to_linear(&c);
            assert_eq!(c, bound.linear_to_coord(li));
        }

        {
            let bound = AABB::new(matrix![0, 9; 0, 9]);
            let c = vector![9, 8];
            let li = bound.coord_to_linear(&c);
            assert_eq!(c, bound.linear_to_coord(li));
        }
    }

    #[test]
    fn periodic_coord_test() {
        {
            let index = vector![0, 0];
            let bound = AABB::new(matrix![0, 10; 0, 10]);
            assert_eq!(bound.periodic_coord(&index), vector![0, 0]);
        }

        {
            let index = vector![10, 10];
            let bound = AABB::new(matrix![0, 10; 0, 10]);
            assert_eq!(bound.periodic_coord(&index), vector![10, 10]);
        }

        {
            let index = vector![-1, 0];
            let bound = AABB::new(matrix![0, 10; 0, 10]);
            assert_eq!(bound.periodic_coord(&index), vector![10, 0]);
        }

        {
            let index = vector![0, -1];
            let bound = AABB::new(matrix![0, 10; 0, 10]);
            assert_eq!(bound.periodic_coord(&index), vector![0, 10]);
        }

        {
            let index = vector![0, -1, -4, -19, 134];
            let bound =
                AABB::new(matrix![0, 100; 0, 100;0, 100; 0, 100;0, 100]);
            assert_eq!(
                bound.periodic_coord(&index),
                vector![0, 100, 97, 82, 33]
            );
        }

        // Mimic use in convolution op for non-origin AABB
        {
            let aabb = AABB::new(matrix![13, 96; 0, 40]);
            let offsets = [
                vector![0, 0],
                vector![1, 0],
                vector![-1, 0],
                vector![0, 1],
                vector![0, -1],
                vector![1, 1],
                vector![-1, 1],
                vector![1, -1],
                vector![-1, -1],
            ];
            let expected = [
                vector![13, 0],
                vector![14, 0],
                vector![96, 0],
                vector![13, 1],
                vector![13, 40],
                vector![14, 1],
                vector![96, 1],
                vector![14, 40],
                vector![96, 40],
            ];
            debug_assert_eq!(offsets.len(), expected.len());
            for (offset, expected) in offsets.iter().zip(expected.iter()) {
                let c: Coord<2> = aabb.min() + offset;
                let pc = aabb.periodic_coord(&c);
                debug_assert_eq!(pc, *expected);
            }
        }
    }

    #[test]
    fn contains_aabb_test() {
        {
            let a = AABB::new(matrix![0, 9]);
            let b = AABB::new(matrix![0, 9]);
            assert!(a.contains_aabb(&b));
        }
    }

    #[test]
    fn check_validity_test() {
        {
            let a = AABB::new(matrix![0, 9]);
            assert!(a.check_validity());
        }

        {
            let a = AABB::new(matrix![9, 0]);
            assert!(!a.check_validity());
        }

        {
            let a = AABB::new(matrix![0, 0]);
            assert!(a.check_validity());
        }
    }

    // Test that decomp + center = bounds exactly
    // by checking every coordinate appears once.
    fn test_decomp<const DIMENSION: usize>(
        bounds: &AABB<DIMENSION>,
        center: &AABB<DIMENSION>,
    ) {
        let mut coord_set = std::collections::HashSet::new();
        coord_set.extend(center.coord_iter());

        let d = bounds.decomposition(center);
        for [b1, b2] in d {
            for c in b1.coord_iter().chain(b2.coord_iter()) {
                assert!(!coord_set.contains(&c));
                coord_set.insert(c);
            }
        }

        for c in bounds.coord_iter() {
            assert!(coord_set.contains(&c));
        }

        assert_eq!(bounds.buffer_size(), coord_set.len());
    }

    #[test]
    fn decomp_test() {
        {
            let outer = AABB::new(matrix![0, 9]);
            let center = AABB::new(matrix![4, 6]);
            test_decomp(&outer, &center);
        }

        {
            let outer = AABB::new(matrix![0, 9; 0, 9]);
            let center = AABB::new(matrix![4, 6; 4, 6]);
            test_decomp(&outer, &center);
        }

        {
            let outer = AABB::new(matrix![2, 20; 5, 19; 40, 60]);
            let center = AABB::new(matrix![6, 14; 10, 13; 47, 53]);
            test_decomp(&outer, &center);
        }

        {
            let outer = AABB::new(matrix![0, 999; 0, 999]);
            let center = AABB::new(matrix![40, 959; 40, 959]);
            test_decomp(&outer, &center);
        }
    }

    #[test]
    fn min_size_len() {
        {
            let b = AABB::new(matrix![0, 5]);
            assert_eq!(b.min_size_len(), 6);
        }

        {
            let b = AABB::new(matrix![0, 5; 1, 3]);
            assert_eq!(b.min_size_len(), 3);
        }

        {
            let b = AABB::new(matrix![23, 45; 63, 1234; 123, 543; 22, 3454;]);
            assert_eq!(b.min_size_len(), 23);
        }
    }

    #[test]
    fn shrink_test() {
        {
            let b = AABB::new(matrix![0, 9]);
            let slopes = matrix![1, 1];
            let ratio = 0.5;
            let c = b.shrink(ratio, slopes, None);
            assert_eq!(c, (2, AABB::new(matrix![2, 7])));
        }
        {
            let b = AABB::new(matrix![0, 100; 0, 100; 0, 40;]);
            let slopes = matrix![6, 5; 4, 3; 2, 1];
            let ratio = 0.5;
            let c = b.shrink(ratio, slopes, None);
            assert_eq!(c, (6, AABB::new(matrix![36, 70; 24, 82; 12, 34])));
        }
        {
            let b = AABB::new(matrix![0, 100; 0, 100; 0, 40;]);
            let slopes = matrix![6, 5; 4, 3; 2, 1];
            let ratio = 0.5;
            let max_steps = 4;
            let c = b.shrink(ratio, slopes, Some(max_steps));
            assert_eq!(
                c,
                (max_steps, AABB::new(matrix![24, 80; 16, 88; 8, 36]))
            );
        }
    }

    #[test]
    fn offset_conversion_test() {
        {
            let aabb = AABB::new(matrix![0, 9]);
            let coord_offsets = [vector![-1], vector![0], vector![1]];
            let linear_offsets = aabb.coord_offset_to_linear(&coord_offsets);
            println!("1D: {:?}", linear_offsets);
        }

        {
            let aabb = AABB::new(matrix![0, 9; 0, 9]);
            let coord_offsets = [
                vector![0, 0],
                vector![-1, 0],
                vector![1, 0],
                vector![0, -1],
                vector![0, 1],
            ];
            let linear_offsets = aabb.coord_offset_to_linear(&coord_offsets);
            println!("2D: {:?}", linear_offsets);
        }

        {
            let aabb = AABB::new(matrix![0, 9; 0, 9; 0,9]);
            let coord_offsets = [
                vector![0, 0, 0],
                vector![-1, 0, 0],
                vector![1, 0, 0],
                vector![0, -1, 0],
                vector![0, 1, 0],
                vector![0, 0, -1],
                vector![0, 0, 1],
            ];
            let linear_offsets = aabb.coord_offset_to_linear(&coord_offsets);
            println!("3D: {:?}", linear_offsets);
        }
    }
}
