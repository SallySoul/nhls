use crate::domain::*;
use crate::fft_solver::*;
use crate::util::*;

struct StackFrame {
    op: NodeId,
}

pub struct APFrustrumPlanRunner<'a, const GRID_DIMENSION: usize> {
    complex_buffer: AlignedVec<c64>,
    real_buffer: AlignedVec<f64>,
    chunk_size: usize,
    plan: ap_frustrum_plan::APFrustrumPlan<GRID_DIMENSION>,
    op_store: &'a ConvolutionStore,
    stack: Vec<StackFrame>,
}

impl<'a, const GRID_DIMENSION: usize> APFrustrumPlanRunner<'a, GRID_DIMENSION> {
    pub fn apply<DomainType: DomainView<GRID_DIMENSION>>(
        solve_input_domain: &DomainType,
        solve_output_domain: &mut DomainType,
    ) {
    }
}
