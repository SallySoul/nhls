use crate::domain::*;
use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;

pub struct APSolver<
    'a,
    BC: BCCheck<GRID_DIMENSION>,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub direct_frustrum_solver: DirectFrustrumSolver<
        'a,
        BC,
        Operation,
        GRID_DIMENSION,
        NEIGHBORHOOD_SIZE,
    >,
    pub convolution_store: ConvolutionStore,
    pub plan: APPlan<GRID_DIMENSION>,
    pub node_scratch_descriptors: Vec<ScratchDescriptor>,
    pub scratch_space: ScratchSpace,
    pub chunk_size: usize,
}

impl<
        'a,
        BC,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APSolver<'a, BC, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    BC: BCCheck<GRID_DIMENSION>,
{
    pub fn new(
        bc: &'a BC,
        stencil: &'a StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        aabb: AABB<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        cutoff: i32,
        ratio: f64,
        chunk_size: usize,
    ) -> Self {
        // Create our plan and convolution_store
        let planner = APPlanner::new(
            stencil, aabb, steps, plan_type, cutoff, ratio, chunk_size,
        );
        let planner_result = planner.finish();
        let plan = planner_result.plan;
        let convolution_store = planner_result.convolution_store;
        let stencil_slopes = planner_result.stencil_slopes;

        let (node_scratch_descriptors, scratch_space) =
            APScratchBuilder::build(&plan);

        let direct_frustrum_solver = DirectFrustrumSolver {
            bc,
            stencil,
            stencil_slopes,
            chunk_size,
        };

        APSolver {
            direct_frustrum_solver,
            convolution_store,
            plan,
            node_scratch_descriptors,
            scratch_space,
            chunk_size,
        }
    }

    pub fn apply(
        &self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
    ) {
        self.solve_root(input_domain, output_domain);
    }

    pub fn to_dot_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        self.plan.to_dot_file(path);
    }

    fn get_input_output(
        &self,
        node_id: usize,
        aabb: &AABB<GRID_DIMENSION>,
    ) -> (SliceDomain<GRID_DIMENSION>, SliceDomain<GRID_DIMENSION>) {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        let input_buffer = self.scratch_space.unsafe_get_buffer(
            scratch_descriptor.input_offset,
            scratch_descriptor.real_buffer_size,
        );
        let output_buffer = self.scratch_space.unsafe_get_buffer(
            scratch_descriptor.output_offset,
            scratch_descriptor.real_buffer_size,
        );
        /*
        println!(
            "input_output for node_id: {}, input_buffer len: {}, aabb: {}",
            node_id,
            input_buffer.len(),
            aabb.buffer_size()
        );
        println!("  - {:?}", self.plan.get_node(node_id));
        */
        debug_assert!(input_buffer.len() >= aabb.buffer_size());
        debug_assert!(output_buffer.len() >= aabb.buffer_size());

        let input_domain = SliceDomain::new(*aabb, input_buffer);
        let output_domain = SliceDomain::new(*aabb, output_buffer);
        (input_domain, output_domain)
    }

    fn get_complex(&self, node_id: usize) -> &mut [c64] {
        let scratch_descriptor = &self.node_scratch_descriptors[node_id];
        self.scratch_space.unsafe_get_buffer(
            scratch_descriptor.complex_offset,
            scratch_descriptor.complex_buffer_size,
        )
    }

    pub fn solve_root(
        &self,
        input_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'a, GRID_DIMENSION>,
    ) {
        let repeat_solve = self.plan.unwrap_repeat_node(self.plan.root);
        /*
        println!("- Solve Root, n_id: {}, {:?}", self.plan.root, repeat_solve);
        */
        for _ in 0..repeat_solve.n {
            self.periodic_solve_preallocated_io(
                repeat_solve.node,
                false,
                input_domain,
                output_domain,
            );
            std::mem::swap(input_domain, output_domain);
        }
        if let Some(next) = repeat_solve.next {
            self.periodic_solve_preallocated_io(
                next,
                false,
                input_domain,
                output_domain,
            )
        } else {
            std::mem::swap(input_domain, output_domain);
        }
    }

    pub fn unknown_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.direct_solve_allocate_io(node_id, input, output);
            }
            PlanNode::AOBDirectSolve(_) => {
                self.aob_direct_solve_allocate_io(node_id, input, output);
            }
            PlanNode::PeriodicSolve(_) => {
                self.periodic_solve_allocate_io(node_id, input, output);
            }
            PlanNode::Repeat(_) => {
                panic!("ERROR: Not expecting repeat node");
            }
        }
    }

    pub fn unknown_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input: &mut SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        match self.plan.get_node(node_id) {
            PlanNode::DirectSolve(_) => {
                self.direct_solve_preallocated_io(node_id, input, output);
            }
            PlanNode::AOBDirectSolve(_) => {
                self.aob_direct_solve_preallocated_io(node_id, input, output);
            }
            PlanNode::PeriodicSolve(_) => {
                self.periodic_solve_preallocated_io(
                    node_id, true, input, output,
                );
            }
            PlanNode::Repeat(_) => {
                panic!("ERROR: Not expecting repeat node");
            }
        }
    }

    pub fn periodic_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        resize: bool,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);
        /*
        println!(
            "- Solve Periodic PreAlloc, n_id: {}, {:?}",
            node_id, periodic_solve
        );
        */

        // Likely the input domain will be larger than needed?
        std::mem::swap(input_domain, output_domain);
        input_domain.set_aabb(periodic_solve.input_aabb);
        input_domain.par_from_superset(output_domain, self.chunk_size);
        output_domain.set_aabb(periodic_solve.input_aabb);

        // Apply convolution
        {
            let convolution_op =
                self.convolution_store.get(periodic_solve.convolution_id);
            convolution_op.apply(
                input_domain,
                output_domain,
                self.get_complex(node_id),
                self.chunk_size,
            );
        }

        // Boundary
        // In a rayon scope, we fork for each of the boundary solves,
        // each of which will fill in their part of of output_domain
        {
            let input_domain_const: &SliceDomain<'b, GRID_DIMENSION> =
                input_domain;
            rayon::scope(|s| {
                for node_id in periodic_solve.boundary_nodes.clone() {
                    // Our plan should provide the guarantee that
                    // that boundary nodes have mutually exclusive
                    // access to the output_domain
                    let mut node_output = output_domain.unsafe_mut_access();

                    // Each boundary solve will need
                    // new input / output domains from the scratch space
                    s.spawn(move |_| {
                        self.unknown_solve_allocate_io(
                            node_id,
                            input_domain_const,
                            &mut node_output,
                        );
                    });
                }
            });
        }

        if resize {
            std::mem::swap(input_domain, output_domain);
            output_domain.set_aabb(periodic_solve.output_aabb);
            output_domain.par_from_superset(input_domain, self.chunk_size);
            input_domain.set_aabb(periodic_solve.output_aabb);
        }

        // call time cut if needed
        if let Some(next_id) = periodic_solve.time_cut {
            std::mem::swap(input_domain, output_domain);
            self.unknown_solve_preallocated_io(
                next_id,
                input_domain,
                output_domain,
            );
        }
    }

    pub fn periodic_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let periodic_solve = self.plan.unwrap_periodic_node(node_id);

        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &periodic_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        self.periodic_solve_preallocated_io(
            node_id,
            true,
            &mut input_domain,
            &mut output_domain,
        );

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn direct_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &direct_solve.input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        self.direct_solve_preallocated_io(
            node_id,
            &mut input_domain,
            &mut output_domain,
        );
        debug_assert_eq!(*output_domain.aabb(), direct_solve.output_aabb);

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn direct_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let direct_solve = self.plan.unwrap_direct_node(node_id);

        debug_assert!(input_domain
            .aabb()
            .contains_aabb(&direct_solve.input_aabb));

        // For time-cuts, the provided domains
        // will not have the expected sizes.
        // All we know is that the provided input domain contains
        // the expected input domain
        std::mem::swap(input_domain, output_domain);
        input_domain.set_aabb(direct_solve.input_aabb);
        input_domain.par_from_superset(output_domain, self.chunk_size);
        output_domain.set_aabb(direct_solve.input_aabb);
        debug_assert_eq!(*input_domain.aabb(), direct_solve.input_aabb);

        // invoke direct solver
        self.direct_frustrum_solver.apply(
            input_domain,
            output_domain,
            &direct_solve.sloped_sides,
            direct_solve.steps,
        );
        debug_assert_eq!(direct_solve.output_aabb, *output_domain.aabb());
    }

    pub fn aob_direct_solve_allocate_io<'b>(
        &self,
        node_id: NodeId,
        input: &SliceDomain<'b, GRID_DIMENSION>,
        output: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let aob_direct_solve = self.plan.unwrap_aob_direct_node(node_id);

        let (mut input_domain, mut output_domain) =
            self.get_input_output(node_id, &aob_direct_solve.init_input_aabb);

        // copy input
        input_domain.par_from_superset(input, self.chunk_size);

        self.aob_direct_solve_preallocated_io(
            node_id,
            &mut input_domain,
            &mut output_domain,
        );

        // copy output to output
        output.par_set_subdomain(&output_domain, self.chunk_size);
    }

    pub fn aob_direct_solve_preallocated_io<'b>(
        &self,
        node_id: NodeId,
        input_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
        output_domain: &mut SliceDomain<'b, GRID_DIMENSION>,
    ) {
        let aob_direct_solve = self.plan.unwrap_aob_direct_node(node_id);

        debug_assert!(input_domain
            .aabb()
            .contains_aabb(&aob_direct_solve.init_input_aabb));

        // For time-cuts, the provided domains
        // will not have the expected sizes.
        // All we know is that the provided input domain contains
        // the expected input domain
        // TODO: We don't need this in the case we allocate
        std::mem::swap(input_domain, output_domain);
        input_domain.set_aabb(aob_direct_solve.init_input_aabb);
        input_domain.par_from_superset(output_domain, self.chunk_size);
        output_domain.set_aabb(aob_direct_solve.init_input_aabb);

        // invoke direct solver
        self.direct_frustrum_solver.aob_apply(
            input_domain,
            output_domain,
            &aob_direct_solve.input_aabb,
            &aob_direct_solve.sloped_sides,
            aob_direct_solve.steps,
        );

        debug_assert_eq!(aob_direct_solve.output_aabb, *output_domain.aabb());
    }
}
