use crate::fft_solver::*;
use crate::stencil::*;
use crate::util::*;

pub struct PlannerResult<const GRID_DIMENSION: usize> {
    pub plan: APPlan<GRID_DIMENSION>,
    pub convolution_store: ConvolutionStore,
    pub stencil_slopes: Bounds<GRID_DIMENSION>,
}

// generic over stencil
pub struct APPlanner<
    'a,
    Operation,
    const GRID_DIMENSION: usize,
    const NEIGHBORHOOD_SIZE: usize,
> where
    Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
{
    stencil_slopes: Bounds<GRID_DIMENSION>,
    aabb: AABB<GRID_DIMENSION>,
    steps: usize,
    cutoff: i32,
    ratio: f64,
    convolution_gen:
        ConvolutionGenerator<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
    pub nodes: Vec<PlanNode<GRID_DIMENSION>>,
}

impl<
        'a,
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    > APPlanner<'a, Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>
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
        let stencil_slopes = stencil.slopes();
        let convolution_gen =
            ConvolutionGenerator::new(&aabb, stencil, plan_type, chunk_size);
        let nodes = Vec::new();
        APPlanner {
            stencil_slopes,
            aabb,
            steps,
            cutoff,
            ratio,
            convolution_gen,
            nodes,
        }
    }

    pub fn add_node(&mut self, node: PlanNode<GRID_DIMENSION>) -> NodeId {
        let result = self.nodes.len();
        self.nodes.push(node);
        result
    }
    pub fn generate_frustrum(
        &mut self,
        mut frustrum: APFrustrum<GRID_DIMENSION>,
    ) -> PlanNode<GRID_DIMENSION> {
        let solve_params = PeriodicSolveParams {
            stencil_slopes: self.stencil_slopes,
            cutoff: self.cutoff,
            ratio: self.ratio,
            max_steps: Some(frustrum.steps),
        };
        let input_aabb = frustrum.input_aabb(&self.stencil_slopes);

        // Can we do a periodic solve or do we direct solve?
        let maybe_periodic_solve =
            find_periodic_solve(&input_aabb, &solve_params);
        if maybe_periodic_solve.is_none() {
            return frustrum.generate_direct_node(&self.stencil_slopes);
        }
        let periodic_solve = maybe_periodic_solve.unwrap();
        let convolution_id = self
            .convolution_gen
            .get_op(&input_aabb, periodic_solve.steps);

        // Do we need a time cut.
        // If so that will "trim" frustrum
        let mut time_cut = None;
        let maybe_next_frustrum =
            frustrum.time_cut(periodic_solve.steps, &self.stencil_slopes);
        if let Some(next_frustrum) = maybe_next_frustrum {
            let next_node = self.generate_frustrum(next_frustrum);
            time_cut = Some(self.add_node(next_node));
        }

        debug_assert!(frustrum
            .output_aabb
            .contains_aabb(&periodic_solve.output_aabb));
        let boundary_frustrums = frustrum.decompose();

        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
        for bf in boundary_frustrums {
            sub_nodes.push(self.generate_frustrum(bf));
        }
        // add the nodes, find the range
        let first_node = self.nodes.len();
        let last_node = first_node + sub_nodes.len();
        self.nodes.extend(&mut sub_nodes.drain(..));

        let periodic_node = PeriodicSolveNode {
            input_aabb,
            output_aabb: frustrum.output_aabb,
            convolution_id,
            steps: periodic_solve.steps,
            remainder: first_node..last_node,
            time_cut,
        };

        PlanNode::PeriodicSolve(periodic_node)
    }

    pub fn generate_central(&mut self, max_steps: usize) -> (NodeId, usize) {
        let solve_params = PeriodicSolveParams {
            stencil_slopes: self.stencil_slopes,
            cutoff: self.cutoff,
            ratio: self.ratio,
            max_steps: Some(max_steps),
        };

        let periodic_solve =
            find_periodic_solve(&self.aabb, &solve_params).unwrap();

        let convolution_id = self
            .convolution_gen
            .get_op(&self.aabb, periodic_solve.steps);

        println!("aabb: {:?}", self.aabb);
        println!("output: {:?}", periodic_solve.output_aabb);

        let decomposition =
            self.aabb.decomposition(&periodic_solve.output_aabb);
        let mut sub_nodes = Vec::with_capacity(2 * GRID_DIMENSION);
        for d in 0..GRID_DIMENSION {
            for side in [Side::Min, Side::Max] {
                sub_nodes.push(self.generate_frustrum(APFrustrum::new(
                    decomposition[d][side.outer_index()],
                    d,
                    side,
                    periodic_solve.steps,
                )));
            }
        }

        // add the nodes, find the range
        let first_node = self.nodes.len();
        let last_node = first_node + sub_nodes.len();
        self.nodes.extend(&mut sub_nodes.drain(..));

        let periodic_solve_node = PeriodicSolveNode {
            input_aabb: self.aabb,
            output_aabb: periodic_solve.output_aabb,
            convolution_id,
            steps: periodic_solve.steps,
            remainder: first_node..last_node,
            time_cut: None,
        };

        let root_node =
            self.add_node(PlanNode::PeriodicSolve(periodic_solve_node));
        (root_node, periodic_solve.steps)
    }

    pub fn generate(&mut self) -> NodeId {
        // generate central once,
        let (central_solve_node, central_solve_steps) =
            self.generate_central(self.steps);

        let n = self.steps / central_solve_steps;
        let remainder = self.steps % central_solve_steps;
        println!("n: {}, remainder: {}", n, remainder);
        let mut next = None;
        if remainder != 0 {
            let (remainder_solve_node, remainder_solve_steps) =
                self.generate_central(remainder);
            next = Some(remainder_solve_node);
            debug_assert_eq!(remainder_solve_steps, remainder);
        }

        let repeat_node = RepeatNode {
            n,
            node: central_solve_node,
            next,
        };

        self.add_node(PlanNode::Repeat(repeat_node))
    }

    pub fn finish(mut self) -> PlannerResult<GRID_DIMENSION> {
        let root = self.generate();
        let convolution_store = self.convolution_gen.finish();
        let stencil_slopes = self.stencil_slopes;
        let plan = APPlan {
            nodes: self.nodes,
            root,
        };

        PlannerResult {
            plan,
            convolution_store,
            stencil_slopes,
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::standard_stencils::*;

    #[test]
    fn test_gen_1d() {
        let stencil = heat_1d(1.0, 1.0, 0.5);
        let aabb = AABB::new(matrix![0, 100]);
        let cutoff = 20;
        let ratio = 0.5;
        let steps = 110;
        let plan_type = PlanType::Estimate;
        let chunk_size = 10;

        let planner = APPlanner::new(
            &stencil, aabb, steps, plan_type, cutoff, ratio, chunk_size,
        );

        let result = planner.finish();
        result.plan.to_dot_file(&"test.dot");

        for (i, n) in result.plan.nodes.iter().enumerate() {
            println!("i: {}, n: {:?}", i, n);
        }

        let mut p_n = 0;
        let mut d_n = 0;
        let mut r_n = 0;
        for node in &result.plan.nodes {
            match node {
                PlanNode::PeriodicSolve(_) => p_n += 1,
                PlanNode::DirectSolve(_) => d_n += 1,
                PlanNode::Repeat(_) => r_n += 1,
            }
        }
        println!("n: {}", result.plan.nodes.len());
        println!("p_n: {}", p_n);
        println!("d_n: {}", d_n);
        println!("r_n: {}", r_n);
    }

    #[test]
    fn test_gen_2d() {
        let stencil = heat_2d(1.0, 1.0, 1.0, 1.0, 0.5);
        let aabb = AABB::new(matrix![0, 100; 0, 100]);
        let cutoff = 20;
        let ratio = 0.5;
        let steps = 100;
        let plan_type = PlanType::Estimate;
        let chunk_size = 10;

        let planner = APPlanner::new(
            &stencil, aabb, steps, plan_type, cutoff, ratio, chunk_size,
        );
        let result = planner.finish();
        result.plan.to_dot_file(&"test.dot");

        for (i, n) in result.plan.nodes.iter().enumerate() {
            println!("i: {}, n: {:?}", i, n);
        }

        let mut p_n = 0;
        let mut d_n = 0;
        let mut r_n = 0;
        for node in &result.plan.nodes {
            match node {
                PlanNode::PeriodicSolve(_) => p_n += 1,
                PlanNode::DirectSolve(_) => d_n += 1,
                PlanNode::Repeat(_) => r_n += 1,
            }
        }
        println!("n: {}", result.plan.nodes.len());
        println!("p_n: {}", p_n);
        println!("d_n: {}", d_n);
        println!("r_n: {}", r_n);
    }
}
