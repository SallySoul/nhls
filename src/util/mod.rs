pub mod indexing;

mod aabb;
pub use aabb::*;
pub use fftw::array::AlignedVec;
pub use nalgebra::{matrix, vector};

pub use num_traits::{Num, One, Zero};

pub use fftw::array::*;
pub use fftw::types::c64;

pub trait NumTrait = Num + Copy + Send + Sync;

pub type Coord<const GRID_DIMENSION: usize> =
    nalgebra::SVector<i32, { GRID_DIMENSION }>;
