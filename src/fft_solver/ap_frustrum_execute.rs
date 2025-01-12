use crate::domain::*;
use crate::fft_solver::*;
use crate::util::*;

struct StackFrame<'a, const GRID_DIMENSION: usize> {
    op: NodeId,
    domain_id: DomainId,
    domain: &'a SliceDomain<'a, GRID_DIMENSION>,
}

pub struct APFrustrumPlanRunner<'a, const GRID_DIMENSION: usize> {
    complex_buffer: AlignedVec<c64>,
    chunk_size: usize,
    plan: ap_frustrum_plan::APFrustrumPlan<GRID_DIMENSION>,
    op_store: &'a ConvolutionStore,
    domain_stack: &'a DomainStack<'a, GRID_DIMENSION>,
    stack: Vec<StackFrame<'a, GRID_DIMENSION>>,
}

impl<'a, const GRID_DIMENSION: usize> APFrustrumPlanRunner<'a, GRID_DIMENSION> {
    pub fn apply<DomainType: DomainView<GRID_DIMENSION>>(
        solve_input_domain: &DomainType,
        solve_output_domain: &mut DomainType,
    ) {
        loop {}
    }
}
