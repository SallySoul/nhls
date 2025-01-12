use crate::fft_solver::*;
use crate::util::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    Min,
    Max,
}

impl Side {
    #[inline]
    pub fn inner_index(&self) -> usize {
        match self {
            Side::Min => 1,
            Side::Max => 0,
        }
    }

    #[inline]
    pub fn outer_index(&self) -> usize {
        match self {
            Side::Min => 0,
            Side::Max => 1,
        }
    }

    #[inline]
    fn inner_coef(&self) -> i32 {
        match self {
            Side::Min => -1,
            Side::Max => 1,
        }
    }

    #[inline]
    fn outer_coef(&self) -> i32 {
        match self {
            Side::Min => 1,
            Side::Max => -1,
        }
    }
}

// Given some output aabb
#[derive(Debug, PartialEq, Eq)]
pub struct APFrustrum<const GRID_DIMENSION: usize> {
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub recursion_dimension: usize,
    pub side: Side,
    pub steps: usize,
}

impl<const GRID_DIMENSION: usize> APFrustrum<GRID_DIMENSION> {
    pub fn new(
        output_aabb: AABB<GRID_DIMENSION>,
        recursion_dimension: usize,
        side: Side,
        steps: usize,
    ) -> Self {
        APFrustrum {
            output_aabb,
            recursion_dimension,
            side,
            steps,
        }
    }

    pub fn sloped_sides(&self) -> Bounds<GRID_DIMENSION> {
        let mut result = Bounds::from_element(1);
        result[(self.recursion_dimension, self.side.outer_index())] = 0;
        for d in self.recursion_dimension + 1..GRID_DIMENSION {
            result[(d, 0)] = 0;
            result[(d, 1)] = 0;
        }
        result
    }

    pub fn input_aabb(
        &self,
        stencil_slopes: &Bounds<GRID_DIMENSION>,
    ) -> AABB<GRID_DIMENSION> {
        let sloped_sides = self.sloped_sides();
        frustrum_input_aabb(
            self.steps,
            &self.output_aabb,
            &sloped_sides,
            stencil_slopes,
        )
    }

    pub fn generate_direct_node(
        &self,
        stencil_slopes: &Bounds<GRID_DIMENSION>,
    ) -> PlanNode<GRID_DIMENSION> {
        let direct_node = DirectSolveNode {
            input_aabb: self.input_aabb(stencil_slopes),
            output_aabb: self.output_aabb,
            steps: self.steps,
        };
        PlanNode::DirectSolve(direct_node)
    }

    // Steps from input face,
    // slice off the rest, return it
    pub fn time_cut(
        &mut self,
        cut_steps: usize,
        stencil_slopes: &Bounds<GRID_DIMENSION>,
    ) -> Option<APFrustrum<GRID_DIMENSION>> {
        // The input_aabb for this frustrum doesn't change...
        // but we define frustrum by output_aabb
        // so we need to calculate that
        // the output
        debug_assert!(cut_steps <= self.steps);
        // No time cut
        if cut_steps == self.steps {
            return None;
        }

        let remaining_steps = self.steps - cut_steps;
        let next_frustrum = APFrustrum::new(
            self.output_aabb,
            self.recursion_dimension,
            self.side,
            remaining_steps,
        );
        self.output_aabb = next_frustrum.input_aabb(stencil_slopes);
        self.steps = cut_steps;
        println!("timecut: {}", cut_steps);
        Some(next_frustrum)
    }

    // TODO: add tests
    // in particular,
    pub fn decompose(&self) -> Vec<APFrustrum<GRID_DIMENSION>> {
        // Cause FFT goes steps in, so sub one, shrug
        let i_steps = self.steps as i32 - 1;
        let mut result = Vec::new();

        // 1 for this dimension,
        let mut outer_aabb = self.output_aabb;
        let outer_bound = self.output_aabb.bounds
            [(self.recursion_dimension, self.side.outer_index())];
        outer_aabb.bounds
            // TODO: Stencil slope here!
            [(self.recursion_dimension, self.side.inner_index())] =
            outer_bound + self.side.outer_coef() * i_steps;
        result.push(APFrustrum::new(
            outer_aabb,
            self.recursion_dimension,
            self.side,
            self.steps,
        ));

        let mut remainder = self.output_aabb;
        remainder.bounds
            [(self.recursion_dimension, self.side.outer_index())] +=
            self.side.outer_coef() * i_steps;
        //println!(" -- remainder: {:?}", remainder);

        // 2 for each lower dimension
        for d in self.recursion_dimension + 1..GRID_DIMENSION {
            let mut min_aabb = remainder;
            let min_bound = min_aabb.bounds[(d, 0)];
            min_aabb.bounds[(d, 1)] = min_bound + i_steps;
            result.push(APFrustrum::new(min_aabb, d, Side::Min, self.steps));

            let mut max_aabb = remainder;
            let max_bound = max_aabb.bounds[(d, 1)];
            max_aabb.bounds[(d, Side::Max.inner_index())] = max_bound - i_steps;
            result.push(APFrustrum::new(max_aabb, d, Side::Max, self.steps));
        }

        result
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn decompose() {
        // 1D
        {
            let aabb = AABB::new(matrix![0, 10]);
            println!("aabb: {:?}", aabb);
            let f1 = APFrustrum::new(aabb, 0, Side::Min, 2);
            let d1 = f1.decompose();
            assert_eq!(d1.len(), 1);
            assert_eq!(
                d1[0],
                APFrustrum::new(AABB::new(matrix![0, 1]), 0, Side::Min, 2)
            );

            let f2 = APFrustrum::new(aabb, 0, Side::Max, 2);
            let d2 = f2.decompose();
            assert_eq!(d2.len(), 1);
            assert_eq!(
                d2[0],
                APFrustrum::new(AABB::new(matrix![9, 10]), 0, Side::Max, 2)
            );
        }

        // 2D d0
        {
            let steps = 20;
            let aabb = AABB::new(matrix![0, 50; 0, 200]);
            let f1 = APFrustrum::new(aabb, 0, Side::Min, steps);
            let d1 = f1.decompose();
            assert_eq!(d1.len(), 3);
            assert_eq!(
                d1[0],
                APFrustrum::new(
                    AABB::new(matrix![0, 19; 0, 200]),
                    0,
                    Side::Min,
                    steps,
                )
            );
            assert_eq!(
                d1[1],
                APFrustrum::new(
                    AABB::new(matrix![19, 50; 0, 19]),
                    1,
                    Side::Min,
                    steps,
                )
            );
            assert_eq!(
                d1[2],
                APFrustrum::new(
                    AABB::new(matrix![19, 50; 181, 200]),
                    1,
                    Side::Max,
                    steps,
                )
            );

            let f2 = APFrustrum::new(aabb, 0, Side::Max, steps);
            let d2 = f2.decompose();
            assert_eq!(d2.len(), 3);
            assert_eq!(
                d2[0],
                APFrustrum::new(
                    AABB::new(matrix![31, 50; 0, 200]),
                    0,
                    Side::Max,
                    steps,
                )
            );
            assert_eq!(
                d2[1],
                APFrustrum::new(
                    AABB::new(matrix![0, 31; 0, 19]),
                    1,
                    Side::Min,
                    steps
                )
            );
            assert_eq!(
                d2[2],
                APFrustrum::new(
                    AABB::new(matrix![0, 31; 181, 200]),
                    1,
                    Side::Max,
                    steps,
                )
            );
        }

        // 2D d1
        {
            let steps = 20;
            let aabb = AABB::new(matrix![0, 200; 0, 50]);
            let f1 = APFrustrum::new(aabb, 1, Side::Min, steps);
            let d1 = f1.decompose();
            assert_eq!(d1.len(), 1);
            assert_eq!(
                d1[0],
                APFrustrum::new(
                    AABB::new(matrix![0, 200; 0, 19]),
                    1,
                    Side::Min,
                    steps,
                )
            );

            let f2 = APFrustrum::new(aabb, 1, Side::Max, steps);
            let d2 = f2.decompose();
            assert_eq!(d2.len(), 1);
            assert_eq!(
                d2[0],
                APFrustrum::new(
                    AABB::new(matrix![0, 200; 31, 50]),
                    1,
                    Side::Max,
                    steps,
                )
            );
        }
        /*
        let aabb = AABB::new(matrix![0, 50; 0, 200; 0, 200]);
        println!("aabb: {:?}", aabb);
        let f1 = APFrustrum::new(aabb, 0, Side::Min);
        println!("Min: {:?}", f1);
        let d1 = f1.decompose(20);
        for (i, d) in d1.iter().enumerate() {
            println!(" -- rec {}: {:?}", i, d);
        }


        let f2 = APFrustrum::new(aabb, 0, Side::Max);
        println!("Max: {:?}", f2);
        let d2 = f2.decompose(20);
        for (i, d) in d2.iter().enumerate() {
            println!(" -- rec {}: {:?}", i, d);
        }
        */
    }

    #[test]
    fn memory() {
        let s = crate::standard_stencils::heat_3d(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
        );
        let ss = s.slopes();
        let f = APFrustrum::new(
            AABB::new(matrix![0, 51; 0, 100; 0, 100]),
            0,
            Side::Min,
            12,
        );
        println!("f_o: {}", f.input_aabb(&ss).buffer_size());
        let ds = f.decompose();
        let mut sum = 0;
        for (_i, d) in ds.iter().enumerate() {
            let input = d.input_aabb(&ss);
            sum += input.buffer_size();
        }
        println!("sum: {}", sum);
    }
}
