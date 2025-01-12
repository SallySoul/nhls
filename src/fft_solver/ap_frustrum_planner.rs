use crate::domain::*;
use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;

pub fn plan_ap_frustrum<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
>(
    input_aabb: &AABB<GRID_DIMENSION>,
    stencil_slopes: &Bounds<GRID_DIMENSION>,
    frustum_slopes: &Bounds<GRID_DIMENSION>,
    convolution_gen: &ConvolutionGenerator<
        'a,
        Operation,
        GRID_DIMENSION,
        NEIGHBORHOOD_SIZE,
    >,
    steps: usize,
) where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
}
