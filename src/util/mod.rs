pub use num_traits::{Num, One, Zero};

pub trait NumTrait = Num + Copy + Send + Sync;

pub use fftw::types::c64;

mod aabb;
pub mod indexing;
pub use aabb::*;
pub use fftw::array::AlignedVec;
pub use nalgebra::{matrix, vector};

pub type Coord<const GRID_DIMENSION: usize> =
    nalgebra::SVector<i32, { GRID_DIMENSION }>;
