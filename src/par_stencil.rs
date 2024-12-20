use crate::domain::*;
use crate::stencil::*;
use crate::util::*;
use rayon::prelude::*;

pub fn apply<
    BC,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    bc: &BC,
    stencil: &StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input: &Domain<GRID_DIMENSION>,
    output: &mut Domain<GRID_DIMENSION>,
    chunk_size: usize,
) where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    debug_assert!(input.aabb().contains_aabb(output.aabb()));
    output.par_modify_access(chunk_size).for_each(
        |mut d: DomainChunk<'_, GRID_DIMENSION>| {
            d.coord_iter_mut().for_each(
                |(world_coord, value_mut): (
                    Coord<GRID_DIMENSION>,
                    &mut f64,
                )| {
                    let args = gather_args(stencil, bc, input, &world_coord);
                    let result = stencil.apply(&args);
                    *value_mut = result;
                },
            )
        },
    )
}

#[cfg(test)]
mod unit_test {
    use super::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::matrix;

    #[test]
    fn par_stencil_test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f64; 1]| args[0]);
        let bound = AABB::new(matrix![0, 99]);
        let n_r = bound.buffer_size();
        {
            let mut input_buffer = vec![1.0; n_r];
            let input_domain = Domain::new(bound, &mut input_buffer);

            let mut output_buffer = vec![2.0; n_r];
            let mut output_domain = Domain::new(bound, &mut output_buffer);

            let bc = PeriodicCheck::new(&input_domain);
            apply(&bc, &stencil, &input_domain, &mut output_domain, 1);
            for x in &output_buffer {
                assert_approx_eq!(f64, *x, 1.0);
            }
        }

        {
            let mut input_buffer = vec![2.0; n_r];
            let input_domain = Domain::new(bound, &mut input_buffer);

            let mut output_buffer = vec![1.0; n_r];
            let mut output_domain = Domain::new(bound, &mut output_buffer);

            let bc = PeriodicCheck::new(&input_domain);
            apply(&bc, &stencil, &input_domain, &mut output_domain, 1);
            for x in &output_buffer {
                assert_approx_eq!(f64, *x, 2.0);
            }
        }
    }

    // Throw an error if we hit boundary
    struct ErrorCheck {
        bounds: AABB<1>,
    }
    impl BCCheck<1> for ErrorCheck {
        fn check(&self, c: &Coord<1>) -> Option<f64> {
            assert!(self.bounds.contains(c));
            None
        }
    }

    #[test]
    fn par_stencil_trapezoid_test_1d_simple() {
        let stencil = Stencil::new([[-1], [0], [1]], |args| {
            let mut r = 0.0;
            for a in args {
                r += a / 3.0;
            }
            r
        });

        let input_bound = AABB::new(matrix![0, 10]);
        let output_bound = AABB::new(matrix![1, 9]);

        let mut input_buffer = vec![1.0; input_bound.buffer_size()];
        let mut output_buffer = vec![0.0; output_bound.buffer_size()];

        let input_domain = Domain::new(input_bound, &mut input_buffer);
        let mut output_domain = Domain::new(output_bound, &mut output_buffer);

        let bc = ErrorCheck {
            bounds: input_bound,
        };

        apply(&bc, &stencil, &input_domain, &mut output_domain, 2);

        for i in output_buffer {
            assert_approx_eq!(f64, i, 1.0);
        }
    }
}
