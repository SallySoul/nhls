// Recursive domain decomposition.
//
// We need boundary condition functions
//
// What is value out of bounds

//
// Face vectors

/*
#[derive(Debug)]
pub enum Boundary {
    SPACE,
    PERIODIC,
    FIXED,
}

#[derive(Debug)]
pub struct FFTParams {
    sigma: i32,
    cutoff: i32,
    ratio: f64,
}

#[derive(Debug)]
pub struct FFTSolveArgs {
    boundaries: [Boundary; 2],
    t0: usize,
    t_end: usize,
    x0: usize,
    x_end: usize,
}

pub fn recursive_solve(args: FFTSolveArgs, p: FFTParams, level: usize) {
    println!("RS: args: {:?}, p: {:?}", args, p);
}
*/
use crate::util::*;

pub struct FFTSolveParams<const DIMENSION: usize> {
    pub slopes: Bounds<DIMENSION>,
    pub cutoff: i32,
    pub ratio: f64,
}

pub struct FFTSolve<const DIMENSION: usize> {
    pub solve_region: AABB<DIMENSION>,
    pub steps: i32,
}

pub fn try_fftsolve<const DIMENSION: usize>(
    bounds: AABB<DIMENSION>,
    params: FFTSolveParams<DIMENSION>,
) -> Option<FFTSolve<DIMENSION>> {
    if bounds.min_size_len() <= params.cutoff {
        return None;
    }

    let (steps, solve_region) = bounds.shrink(params.ratio, params.slopes);

    Some(FFTSolve {
        solve_region,
        steps,
    })
}
