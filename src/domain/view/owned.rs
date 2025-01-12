use super::*;
use crate::util::*;
use fftw::array::*;

pub struct OwnedDomain<const GRID_DIMENSION: usize> {
    aabb: AABB<GRID_DIMENSION>,
    buffer: AlignedVec<f64>,
}

impl<const GRID_DIMENSION: usize> OwnedDomain<GRID_DIMENSION> {
    pub fn new(aabb: AABB<GRID_DIMENSION>) -> Self {
        let buffer = AlignedVec::new(aabb.buffer_size());
        OwnedDomain { aabb, buffer }
    }
}

impl<const GRID_DIMENSION: usize> DomainView<GRID_DIMENSION>
    for OwnedDomain<GRID_DIMENSION>
{
    fn aabb(&self) -> &AABB<GRID_DIMENSION> {
        &self.aabb
    }

    fn set_aabb(&mut self, aabb: AABB<GRID_DIMENSION>) {
        debug_assert!(aabb.buffer_size() <= self.buffer.len());
        // TODO: should we re-slice here?
        self.aabb = aabb;
    }

    fn buffer(&self) -> &[f64] {
        &self.buffer
    }

    fn buffer_mut(&mut self) -> &mut [f64] {
        &mut self.buffer
    }

    fn aabb_buffer_mut(&mut self) -> (&AABB<GRID_DIMENSION>, &mut [f64]) {
        (&self.aabb, &mut self.buffer)
    }

    fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> f64 {
        debug_assert!(self.aabb.contains(world_coord));
        let index = self.aabb.coord_to_linear(world_coord);
        self.buffer[index]
    }
}
