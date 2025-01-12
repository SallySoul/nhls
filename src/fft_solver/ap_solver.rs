use crate::domain::*;
use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;

struct APSolverContext<'a, const GRID_DIMENSION: usize> {
    stencil_slopes: &'a Bounds<GRID_DIMENSION>,
    convolution_store: &'a ConvolutionStore,
}

pub struct APSolver<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    pub stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
    pub aabb: AABB<GRID_DIMENSION>,
    pub steps: usize,
    pub convolution_store: ConvolutionStore,
    pub plan: APPlan<GRID_DIMENSION>,
    pub chunk_size: usize,
}

impl<
        'a,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APSolver<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    pub fn new(
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        cutoff: i32,
        ratio: f64,
        chunk_size: usize,
    ) -> Self {
        let planner = APPlanner::new(
            stencil, aabb, steps, plan_type, cutoff, ratio, chunk_size,
        );
        let planner_result = planner.finish();

        APSolver {
            stencil,
            stencil_slopes: planner_result.stencil_slopes,
            aabb,
            steps,
            convolution_store: planner_result.convolution_store,
            plan: planner_result.plan,
            chunk_size,
        }
    }

    pub fn apply<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        _input: &mut DomainType,
        _output: &mut DomainType,
    ) {
    }
}
