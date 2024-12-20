use crate::domain::*;
use crate::par_stencil;
use crate::stencil::*;
use crate::util::*;

// Calculate the input region for a trapezoidal solve
// based on output region size and other parameters.
pub fn trapezoid_input_region<const GRID_DIMENSION: usize>(
    steps: usize,
    output_box: &AABB<GRID_DIMENSION>,
    sloped_sides: &Bounds<GRID_DIMENSION>,
    stencil_slopes: &Bounds<GRID_DIMENSION>,
) -> AABB<GRID_DIMENSION> {
    let mut trapezoid_slopes = stencil_slopes.component_mul(sloped_sides);
    let negative_slopes = -1 * trapezoid_slopes.column(0);
    trapezoid_slopes.set_column(0, &negative_slopes);
    output_box.add_bounds_diff(steps as i32 * trapezoid_slopes)
}

pub fn trapezoid_apply<
    'a,
    BC,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    bc: &BC,
    stencil: &StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    input: &mut Domain<'a, GRID_DIMENSION>,
    output: &mut Domain<'a, GRID_DIMENSION>,
    sloped_sides: &Bounds<GRID_DIMENSION>,
    stencil_slopes: &Bounds<GRID_DIMENSION>,
    steps: usize,
    chunk_size: usize,
) where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    assert_eq!(input.aabb(), output.aabb());

    let mut trapezoid_slopes = stencil_slopes.component_mul(sloped_sides);
    let negative_slopes = -1 * trapezoid_slopes.column(1);
    trapezoid_slopes.set_column(1, &negative_slopes);

    let mut output_box = *input.aabb();
    for t in 0..steps {
        println!("trapezoid t: {}", t);
        output_box = output_box.add_bounds_diff(trapezoid_slopes);
        debug_assert!(input.aabb().buffer_size() >= output_box.buffer_size());
        output.set_aabb(output_box);
        println!("  output_box: {:?}", output_box);

        par_stencil::apply(bc, stencil, input, output, chunk_size);
        println!("  done with apply");

        std::mem::swap(input, output);
    }
    std::mem::swap(input, output);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    //use float_cmp::assert_approx_eq;
    use nalgebra::matrix;

    #[test]
    fn input_region_test() {
        {
            let steps = 5;
            let output_box = AABB::new(matrix![10, 20]);
            let sloped_sides = matrix![1, 1];
            let stencil_slopes = matrix![1, 1];
            assert_eq!(
                trapezoid_input_region(
                    steps,
                    &output_box,
                    &sloped_sides,
                    &stencil_slopes
                )
                .bounds,
                matrix![5, 25]
            );
        }

        {
            let steps = 5;
            let output_box = AABB::new(matrix![10, 20]);
            let sloped_sides = matrix![0, 1];
            let stencil_slopes = matrix![1, 1];
            assert_eq!(
                trapezoid_input_region(
                    steps,
                    &output_box,
                    &sloped_sides,
                    &stencil_slopes
                )
                .bounds,
                matrix![10, 25]
            );
        }

        {
            let steps = 5;
            let output_box = AABB::new(matrix![10, 20]);
            let sloped_sides = matrix![1, 0];
            let stencil_slopes = matrix![1, 1];
            assert_eq!(
                trapezoid_input_region(
                    steps,
                    &output_box,
                    &sloped_sides,
                    &stencil_slopes
                )
                .bounds,
                matrix![5, 20]
            );
        }

        {
            let steps = 5;
            let output_box = AABB::new(matrix![10, 20]);
            let sloped_sides = matrix![1, 1];
            let stencil_slopes = matrix![1, 2];
            assert_eq!(
                trapezoid_input_region(
                    steps,
                    &output_box,
                    &sloped_sides,
                    &stencil_slopes
                )
                .bounds,
                matrix![5, 30]
            );
        }

        {
            let steps = 5;
            let output_box = AABB::new(matrix![10, 20; 10, 20; 10, 20]);
            let sloped_sides = matrix![1, 1; 1, 1; 1, 1];
            let stencil_slopes = matrix![1, 2; 2, 1; 2, 3];
            assert_eq!(
                trapezoid_input_region(
                    steps,
                    &output_box,
                    &sloped_sides,
                    &stencil_slopes
                )
                .bounds,
                matrix![5, 30; 0, 25; 0, 35]
            );
        }
    }

    #[test]
    fn simple_1d_trapezoid_test() {
        let steps = 5;
        let chunk_size = 10;
        let stencil = Stencil::new([[-1], [0], [1]], |args| {
            let mut r = 0.0;
            for a in args {
                r += a / 3.0;
            }
            r
        });
        let stencil_slopes = stencil.slopes();

        {
            let sloped_sides = matrix![1, 1];
            let input_bound = AABB::new(matrix![10, 40]);
            let mut input_buffer = vec![1.0; input_bound.buffer_size()];
            let mut output_buffer = vec![1.0; input_bound.buffer_size()];
            let mut input_domain = Domain::new(input_bound, &mut input_buffer);
            let mut output_domain =
                Domain::new(input_bound, &mut output_buffer);
            let bc = ConstantCheck::new(1.0, input_bound);
            trapezoid_apply(
                &bc,
                &stencil,
                &mut input_domain,
                &mut output_domain,
                &sloped_sides,
                &stencil_slopes,
                steps,
                chunk_size,
            );
            assert_eq!(*output_domain.aabb(), AABB::new(matrix![15, 35]));
        }
    }
}
