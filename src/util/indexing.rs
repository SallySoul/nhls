use crate::util::*;

pub fn real_buffer_size<const DIMENSION: usize>(
    exclusive_bound: &Coord<DIMENSION>,
) -> usize {
    let mut accumulator = 1;
    for d in exclusive_bound {
        accumulator *= *d as usize;
    }
    accumulator
}

pub fn complex_buffer_size<const DIMENSION: usize>(
    exclusive_bound: &Coord<DIMENSION>,
) -> usize {
    let mut accumulator = 1;
    let mut size_iter = exclusive_bound.iter().rev();
    accumulator *= *size_iter.next().unwrap() as usize / 2 + 1;
    for d in size_iter {
        accumulator *= *d as usize;
    }
    accumulator
}

pub fn coord_to_linear<const GRID_DIMENSION: usize>(
    coord: &Coord<GRID_DIMENSION>,
    exclusive_bounds: &Coord<GRID_DIMENSION>,
) -> usize {
    // TODO this could be better
    let mut accumulator = 0;
    for d in 0..GRID_DIMENSION {
        debug_assert!(coord[d] >= 0);
        let mut dim_accumulator = coord[d] as usize;
        for dn in (d + 1)..GRID_DIMENSION {
            dim_accumulator *= exclusive_bounds[dn] as usize;
        }
        accumulator += dim_accumulator;
    }
    accumulator
}

pub fn linear_to_coord<const GRID_DIMENSION: usize>(
    linear_index: usize,
    exclusive_bounds: &Coord<GRID_DIMENSION>,
) -> Coord<GRID_DIMENSION> {
    let mut result = Coord::zero();
    let mut index_accumulator = linear_index;

    for d in 0..GRID_DIMENSION - 1 {
        let mut dim_accumulator = 1;
        for dn in (d + 1)..GRID_DIMENSION {
            dim_accumulator *= exclusive_bounds[dn] as usize;
        }

        result[d] = (index_accumulator / dim_accumulator) as i32;
        index_accumulator %= dim_accumulator;
    }
    result[GRID_DIMENSION - 1] = index_accumulator as i32;
    result
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use nalgebra::vector;

    #[test]
    fn buffer_size_test() {
        {
            let dimensions = vector![5];
            let real_size = real_buffer_size(&dimensions);
            assert_eq!(real_size, 5);
            let complex_size = complex_buffer_size(&dimensions);
            assert_eq!(complex_size, (5 / 2) + 1);
        }

        {
            let dimensions = vector![5, 7, 9];
            let real_size = real_buffer_size(&dimensions);
            assert_eq!(real_size, 5 * 7 * 9);
            let complex_size = complex_buffer_size(&dimensions);
            assert_eq!(complex_size, 5 * 7 * ((9 / 2) + 1));
        }
    }

    #[test]
    fn coord_to_linear_index_test() {
        {
            let index = vector![5, 7, 11];
            let bound = vector![20, 20, 20];
            assert_eq!(
                coord_to_linear(&index, &bound),
                5 * 20 * 20 + 7 * 20 + 11
            );
        }

        {
            let index = vector![5, 7];
            let bound = vector![20, 20];
            assert_eq!(coord_to_linear(&index, &bound), 5 * 20 + 7);
        }

        {
            let index = vector![5];
            let bound = vector![20];
            assert_eq!(coord_to_linear(&index, &bound), 5);
        }
    }

    #[test]
    fn linear_to_coord_test() {
        {
            let index = 67;
            let bound = vector![10, 10];
            assert_eq!(linear_to_coord(index, &bound), vector![6, 7]);
        }

        {
            let index = 67;
            let bound = vector![100];
            assert_eq!(linear_to_coord(index, &bound), vector![67]);
        }

        {
            let index = 0;
            let bound = vector![10, 10, 8, 10];
            assert_eq!(linear_to_coord(index, &bound), vector![0, 0, 0, 0]);
        }
    }
}
