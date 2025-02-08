use crate::domain::*;
use crate::stencil::*;
use crate::util::*;
use rayon::prelude::*;

pub fn apply<
    BC,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
    DomainType: DomainView<GRID_DIMENSION>,
>(
    bc: &BC,
    stencil: &Stencil<GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    stencil_slopes: &Bounds<GRID_DIMENSION>,
    sloped_sides: &Bounds<GRID_DIMENSION>,
    input: &DomainType,
    output: &mut DomainType,
    global_time: usize,
    chunk_size: usize,
) where
    BC: BCCheck<GRID_DIMENSION>,
{
    // Iterator for "central points"
    //  - Can't use par_modify_access
    //  - Can just create

    // New gather args implementation

    // Boundary iterators include world coords

    // Create central region
    let central_mod = output.aabb().add_bounds_diff(
        flip_sloped(stencil_slopes).component_mul(sloped_sides),
    );

    // how to get offsets into linear index offset

    debug_assert!(input.aabb().contains_aabb(output.aabb()));
    output.par_modify_access(chunk_size).for_each(
        |mut d: DomainChunk<'_, GRID_DIMENSION>| {
            d.coord_iter_mut().for_each(
                |(world_coord, value_mut): (
                    Coord<GRID_DIMENSION>,
                    &mut f64,
                )| {
                    let args = gather_args(
                        stencil,
                        bc,
                        input,
                        &world_coord,
                        global_time,
                    );
                    let result = stencil.apply(&args);
                    *value_mut = result;
                },
            )
        },
    )
}
