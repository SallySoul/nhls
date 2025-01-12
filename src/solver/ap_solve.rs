use crate::decomposition::*;
use crate::domain::*;
use crate::fft_solver::PlanType;
use crate::solver::*;
use crate::stencil::*;
use crate::util::*;

pub struct APSolver<
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
    params: FFTSolveParams<GRID_DIMENSION>,
    chunk_size: usize,
    slopes: Bounds<GRID_DIMENSION>,
    plan_type: PlanType,
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
        cutoff: i32,
        ratio: f64,
        plan_type: PlanType,
        chunk_size: usize,
    ) -> Self {
        let params = FFTSolveParams {
            slopes: stencil.slopes(),
            cutoff,
            ratio,
        };

        let slopes = stencil.slopes();

        APSolver {
            bc,
            stencil,
            params,
            chunk_size,
            slopes,
            plan_type,
        }
    }

    /// Performs outermost loop of AP algorithm
    /// Here we find the next root FFT solve,
    /// which may not get us to the desired number of steps,
    /// we we have to keep finding root FFT Solves and applying
    /// the recursion until the desired steps are reached.
    ///
    /// NOTE: incomplete implementation
    pub fn loop_solve<DomainType: DomainView<GRID_DIMENSION> + Sync>(
        &mut self,
        input: &mut DomainType,
        output: &mut DomainType,
        steps: usize,
    ) {
        // TODO: we just frustrum solve each of the reguins, no recusion
        let mut remaining_steps = steps;
        while remaining_steps != 0 {
            let mut sub_buf_tot = 0;
            let mut sub_com_buf_tot = 0;
            println!(
                "big box buffer_size: {}, complex: {}",
                input.aabb().buffer_size(),
                input.aabb().complex_buffer_size()
            );
            let maybe_fft_solve =
                try_fftsolve(input.aabb(), &self.params, Some(remaining_steps));
            if maybe_fft_solve.is_none() {
                box_apply(
                    self.bc,
                    self.stencil,
                    input,
                    output,
                    remaining_steps,
                    self.chunk_size,
                );
                return;
            }

            let fft_solve = maybe_fft_solve.unwrap();
            let iter_steps = fft_solve.steps;

            // Output domain now has FFT solve
            self.periodic_lib
                .apply(input, output, iter_steps, self.chunk_size);

            // For each degree make domain
            let sub_domain_bounds =
                input.aabb().decomposition(&fft_solve.solve_region);
            let sub_domain_sloped_sides =
                decomposition_slopes::<GRID_DIMENSION>();

            for d in 0..GRID_DIMENSION {
                for r in 0..2 {
                    // get bounds
                    let output_aabb = sub_domain_bounds[d][r];
                    let sloped_sides = sub_domain_sloped_sides[d][r];
                    let input_aabb = trapezoid_input_region(
                        iter_steps,
                        &output_aabb,
                        &sloped_sides,
                        &self.slopes,
                    );

                    println!(
                        "domain, d: {}, r: {}, bs: {}, cs: {}, {:?}",
                        d,
                        r,
                        input_aabb.buffer_size(),
                        input_aabb.complex_buffer_size(),
                        input_aabb
                    );
                    sub_buf_tot += input_aabb.buffer_size();
                    sub_com_buf_tot += input_aabb.complex_buffer_size();

                    // Make sub domain
                    let mut input_domain = OwnedDomain::new(input_aabb);
                    let mut output_domain = OwnedDomain::new(input_aabb);

                    // copy input
                    input_domain.par_from_superset(input, self.chunk_size);

                    trapezoid_apply(
                        self.bc,
                        self.stencil,
                        &mut input_domain,
                        &mut output_domain,
                        &sloped_sides,
                        &self.slopes,
                        iter_steps,
                        self.chunk_size,
                    );

                    // copy output
                    debug_assert_eq!(output_domain.aabb(), &output_aabb);
                    output.par_set_subdomain(&output_domain, self.chunk_size);
                }
            }
            println!("Total: {}, ctotal: {}", sub_buf_tot, sub_com_buf_tot);
            remaining_steps -= iter_steps;
            std::mem::swap(input, output);
        }
        std::mem::swap(input, output);
    }
}
