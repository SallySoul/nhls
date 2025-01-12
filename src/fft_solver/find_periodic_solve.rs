use crate::util::*;

pub struct PeriodicSolveParams<const DIMENSION: usize> {
    pub stencil_slopes: Bounds<DIMENSION>,
    pub cutoff: i32,
    pub ratio: f64,
    pub max_steps: Option<usize>,
}

pub struct PeriodicSolve<const DIMENSION: usize> {
    pub output_aabb: AABB<DIMENSION>,
    pub steps: usize,
}

pub fn find_periodic_solve<const DIMENSION: usize>(
    input_aabb: &AABB<DIMENSION>,
    params: &PeriodicSolveParams<DIMENSION>,
) -> Option<PeriodicSolve<DIMENSION>> {
    if input_aabb.min_size_len() <= params.cutoff {
        return None;
    }

    let (steps, output_aabb) = input_aabb.shrink(
        params.ratio,
        params.stencil_slopes,
        params.max_steps,
    );

    println!(
        "Found fft solve, steps: {}, region: {:?}",
        steps, output_aabb
    );

    Some(PeriodicSolve { output_aabb, steps })
}
