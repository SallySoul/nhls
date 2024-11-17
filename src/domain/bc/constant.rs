use crate::domain::bc::BCCheck;
use crate::util::*;

pub struct ConstantCheck<const GRID_DIMENSION: usize> {
    value: f32,
    bound: Box<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> ConstantCheck<GRID_DIMENSION> {
    pub fn new(value: f32, bound: Box<GRID_DIMENSION>) -> Self {
        ConstantCheck { value, bound }
    }
}

impl<const GRID_DIMENSION: usize> BCCheck<GRID_DIMENSION> for ConstantCheck<GRID_DIMENSION> {
    fn check(&self, coord: &Coord<GRID_DIMENSION>) -> Option<f32> {
        for d in 0..GRID_DIMENSION {
            let c = coord[d];
            if c < self.bound[(d, 0)] || c > self.bound[(d, 1)] {
                return Some(self.value);
            }
        }
        None
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::{matrix, vector};

    #[test]
    fn constant_check_test() {
        let bound = matrix![0, 10];
        let n_r = box_buffer_size(&bound);
        let mut buffer = fftw::array::AlignedVec::new(n_r);
        for i in 0..n_r {
            buffer.as_slice_mut()[i] = i as f32;
        }
        let bc = ConstantCheck::new(-1.0, bound);
        for i in 0..n_r {
            let v = bc.check(&vector![i as i32]);
            assert_eq!(v, None);
        }

        {
            let v = bc.check(&vector![-1]);
            assert!(v.is_some());
            assert_approx_eq!(f32, v.unwrap(), -1.0);
        }

        {
            let v = bc.check(&vector![11]);
            assert!(v.is_some());
            assert_approx_eq!(f32, v.unwrap(), -1.0);
        }
    }
}