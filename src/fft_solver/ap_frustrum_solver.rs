use crate::domain::*;
use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;

pub struct APFrustrumSolver<
    'a,
    BC,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    bc: &'a BC,
    stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    sloped_sides: Bounds<GRID_DIMENSION>,
    steps: usize,
    plan_type: PlanType,
    chunk_size: usize,
}

impl<
        'a,
        BC,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APFrustrumSolver<'a, BC, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn create<DomainType: DomainView<GRID_DIMENSION>>(
        bc: &'a BC,
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        input_domain: &mut DomainType,
        sloped_sides: Bounds<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        chunk_size: usize,
    ) -> Self {
        APFrustrumSolver {
            bc,
            stencil,
            sloped_sides,
            steps,
            plan_type,
            chunk_size,
        }
    }
}
