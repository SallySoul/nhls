// Used for Stencil traits
#![feature(trait_alias)]
// We use these alot with const ranges,
// don't like this warning for this codebase.
#![allow(clippy::needless_range_loop)]

pub mod domain;
pub mod fft_solver;
pub mod image;
pub mod image_1d_example;
pub mod image_2d_example;
pub mod image_3d_example;
pub mod init;
pub mod mem_fmt;
pub mod par_slice;
pub mod par_stencil;
pub mod par_stencil_2;
pub mod solver;
pub mod standard_stencils;
pub mod stencil;
pub mod util;
pub mod vtk;
