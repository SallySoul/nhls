use super::*;
use crate::util::*;

pub struct SliceDomain<'a, const GRID_DIMENSION: usize> {
    aabb: AABB<GRID_DIMENSION>,
    buffer: &'a mut [f64],
}

impl<'a, const GRID_DIMENSION: usize> SliceDomain<'a, GRID_DIMENSION> {
    pub fn new(aabb: AABB<GRID_DIMENSION>, buffer: &'a mut [f64]) -> Self {
        debug_assert_eq!(buffer.len(), aabb.buffer_size());
        SliceDomain { aabb, buffer }
    }
}

impl<'a, const GRID_DIMENSION: usize> DomainView<GRID_DIMENSION>
    for SliceDomain<'a, GRID_DIMENSION>
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
        self.buffer
    }

    fn buffer_mut(&mut self) -> &mut [f64] {
        self.buffer
    }

    fn aabb_buffer_mut(&mut self) -> (&AABB<GRID_DIMENSION>, &mut [f64]) {
        (&self.aabb, self.buffer)
    }

    fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> f64 {
        debug_assert!(self.aabb.contains(world_coord));
        let index = self.aabb.coord_to_linear(world_coord);
        self.buffer[index]
    }
}
