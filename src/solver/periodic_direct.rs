use crate::domain::*;
use crate::par_stencil;
use crate::stencil::*;

pub fn direct_periodic_apply<
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    DomainType: DomainView<GRID_DIMENSION>,
>(
    stencil: &StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input: &mut DomainType,
    output: &mut DomainType,
    steps: usize,
    chunk_size: usize,
) where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    debug_assert_eq!(input.aabb(), output.aabb());
    for _ in 0..steps - 1 {
        {
            let bc = PeriodicCheck::new(input);
            par_stencil::apply(&bc, stencil, input, output, chunk_size);
        }
        std::mem::swap(input, output);
    }

    let bc = PeriodicCheck::new(input);
    par_stencil::apply(&bc, stencil, input, output, chunk_size);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::util::*;
    use float_cmp::assert_approx_eq;
    use nalgebra::matrix;

    fn test_unit_stencil<
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        bound: &AABB<GRID_DIMENSION>,
        steps: usize,
    ) where
        Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    {
        let chunk_size = 3;
        assert_approx_eq!(f64, stencil.apply(&[1.0; NEIGHBORHOOD_SIZE]), 1.0);

        let mut input_domain = OwnedDomain::new(*bound);
        let mut output_domain = OwnedDomain::new(*bound);
        input_domain.par_set_values(|_| 1.0, chunk_size);
        direct_periodic_apply(
            stencil,
            &mut input_domain,
            &mut output_domain,
            steps,
            chunk_size,
        );

        for x in output_domain.buffer() {
            assert_approx_eq!(f64, *x, 1.0);
        }
    }

    #[test]
    fn test_1d_simple() {
        let stencil = Stencil::new([[0]], |args: &[f64; 1]| args[0]);
        let bound = AABB::new(matrix![0, 100]);
        test_unit_stencil(&stencil, &bound, 100);
    }

    #[test]
    fn test_2d_simple() {
        let stencil = Stencil::new([[0, 0]], |args: &[f64; 1]| args[0]);
        let bound = AABB::new(matrix![0, 50; 0, 50]);
        test_unit_stencil(&stencil, &bound, 9);
    }

    #[test]
    fn test_2d_less_simple() {
        let stencil = Stencil::new(
            [[0, -1], [0, 1], [1, 0], [-1, 0], [0, 0]],
            |args: &[f64; 5]| {
                debug_assert_eq!(args.len(), 5);
                let mut r = 0.0;
                for a in args {
                    r += a / 5.0;
                }
                r
            },
        );
        let bound = AABB::new(matrix![0, 50;0,  50]);
        test_unit_stencil(&stencil, &bound, 10);
    }

    #[test]
    fn test_1d_less_simple() {
        let stencil = Stencil::new([[-1], [1], [0]], |args: &[f64; 3]| {
            debug_assert_eq!(args.len(), 3);
            let mut r = 0.0;
            for a in args {
                r += a / 3.0
            }
            r
        });
        let bound = AABB::new(matrix![0, 100]);
        test_unit_stencil(&stencil, &bound, 10);
    }

    #[test]
    fn test_3d() {
        let stencil = Stencil::new(
            [
                [0, 0, -2],
                [4, 5, 3],
                [0, -1, 0],
                [0, 1, 0],
                [1, 0, 0],
                [-1, 0, 4],
                [0, 0, 0],
            ],
            |args: &[f64; 7]| {
                let mut r = 0.0;
                for a in args {
                    r += a / 7.0;
                }
                r
            },
        );
        let bound = AABB::new(matrix![0, 20;0,  20;0,  20]);
        test_unit_stencil(&stencil, &bound, 5);
    }

    #[test]
    fn shifter() {
        let chunk_size = 1;
        let stencil = Stencil::new([[-1]], |args: &[f64; 1]| args[0]);
        let bound = AABB::new(matrix![0, 9]);

        let mut input_domain = OwnedDomain::new(bound);
        let mut output_domain = OwnedDomain::new(bound);
        input_domain
            .par_set_values(|coord: Coord<1>| coord[0] as f64, chunk_size);

        let n = 1;
        direct_periodic_apply(
            &stencil,
            &mut input_domain,
            &mut output_domain,
            n,
            chunk_size,
        );
        for i in 0..10 {
            assert_approx_eq!(
                f64,
                output_domain.buffer()[(i + n) % 10],
                i as f64
            );
        }
    }
}
