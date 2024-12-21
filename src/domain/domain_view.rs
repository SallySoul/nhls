use crate::util::*;
use rayon::prelude::*;

pub struct Domain<'a, const GRID_DIMENSION: usize> {
    aabb: AABB<GRID_DIMENSION>,
    buffer: &'a mut [f64],
}

impl<'a, const GRID_DIMENSION: usize> Domain<'a, GRID_DIMENSION> {
    pub fn aabb(&self) -> &AABB<GRID_DIMENSION> {
        &self.aabb
    }

    pub fn set_aabb(&mut self, aabb: AABB<GRID_DIMENSION>) {
        debug_assert!(aabb.buffer_size() <= self.buffer.len());
        // TODO: should we re-slice here?
        self.aabb = aabb;
    }

    pub fn buffer(&self) -> &[f64] {
        self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [f64] {
        self.buffer
    }

    pub fn new(aabb: AABB<GRID_DIMENSION>, buffer: &'a mut [f64]) -> Self {
        debug_assert_eq!(buffer.len(), aabb.buffer_size());
        Domain { aabb, buffer }
    }

    pub fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> f64 {
        debug_assert!(self.aabb.contains(world_coord));
        let index = self.aabb.coord_to_linear(world_coord);
        self.buffer[index]
    }

    pub fn modify(&mut self, world_coord: &Coord<GRID_DIMENSION>, value: f64) {
        debug_assert!(self.aabb.contains(world_coord));
        let index = self.aabb.coord_to_linear(world_coord);
        self.buffer[index] = value;
    }

    pub fn par_modify_access(
        &mut self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = DomainChunk<'_, GRID_DIMENSION>> {
        par_modify_access_impl(self.buffer, &self.aabb, chunk_size)
    }

    pub fn par_set_values<
        F: FnOnce(Coord<GRID_DIMENSION>) -> f64 + Send + Sync + Copy,
    >(
        &mut self,
        f: F,
        chunk_size: usize,
    ) {
        self.par_modify_access(chunk_size).for_each(
            |mut d: DomainChunk<'_, GRID_DIMENSION>| {
                d.coord_iter_mut().for_each(
                    |(world_coord, value_mut): (
                        Coord<GRID_DIMENSION>,
                        &mut f64,
                    )| {
                        *value_mut = f(world_coord);
                    },
                )
            },
        );
    }

    /// Copy other domain into self
    pub fn par_set_subdomain(
        &mut self,
        other: &Domain<GRID_DIMENSION>,
        chunk_size: usize,
    ) {
        let const_self_ref: &Domain<GRID_DIMENSION> = self;
        other.buffer[0..other.aabb.buffer_size()]
            .par_chunks(chunk_size)
            .enumerate()
            .for_each(move |(i, buffer_chunk): (usize, &[f64])| {
                let self_ptr = const_self_ref as *const Domain<GRID_DIMENSION>;
                let mut_self_ref: &mut Domain<GRID_DIMENSION> = unsafe {
                    &mut *(self_ptr as *mut Domain<GRID_DIMENSION>)
                        as &mut Domain<GRID_DIMENSION>
                };
                let offset = i * chunk_size;
                for i in 0..buffer_chunk.len() {
                    let other_linear_index = i + offset;
                    let world_coord =
                        other.aabb.linear_to_coord(other_linear_index);
                    let self_linear_index =
                        mut_self_ref.aabb.coord_to_linear(&world_coord);
                    mut_self_ref.buffer[self_linear_index] =
                        other.buffer[other_linear_index];
                }
            });
    }

    /// Copy self coords from other into self
    pub fn par_from_superset(
        &mut self,
        other: &Domain<GRID_DIMENSION>,
        chunk_size: usize,
    ) {
        self.par_set_values(|world_coord| other.view(&world_coord), chunk_size);
    }
}

/// Why not just put this into Domain::par_modify_access?
/// Rust compiler can't figure out how to borrow aabb and buffer
/// at the same time in this way.
/// By putting their borrows into one function call first we work around it.
fn par_modify_access_impl<'a, const GRID_DIMENSION: usize>(
    buffer: &'a mut [f64],
    aabb: &'a AABB<GRID_DIMENSION>,
    chunk_size: usize,
) -> impl ParallelIterator<Item = DomainChunk<'a, GRID_DIMENSION>> + 'a {
    buffer[0..aabb.buffer_size()]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(move |(i, buffer_chunk): (usize, &mut [f64])| {
            let offset = i * chunk_size;
            DomainChunk::new(offset, aabb, buffer_chunk)
        })
}

pub struct DomainChunk<'a, const GRID_DIMENSION: usize> {
    offset: usize,
    aabb: &'a AABB<GRID_DIMENSION>,
    buffer: &'a mut [f64],
}

impl<'a, const GRID_DIMENSION: usize> DomainChunk<'a, GRID_DIMENSION> {
    pub fn new(
        offset: usize,
        aabb: &'a AABB<GRID_DIMENSION>,
        buffer: &'a mut [f64],
    ) -> Self {
        DomainChunk {
            offset,
            aabb,
            buffer,
        }
    }

    pub fn coord_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (Coord<GRID_DIMENSION>, &mut f64)> {
        self.buffer
            .iter_mut()
            .enumerate()
            .map(|(i, v): (usize, &mut f64)| {
                let linear_index = self.offset + i;
                let coord = self.aabb.linear_to_coord(linear_index);
                (coord, v)
            })
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn par_set_subdomain_test() {
        {
            let mut buffer = vec![0.0; 100];
            let bounds = AABB::new(matrix![0, 9; 0, 9;]);
            let mut domain = Domain::new(bounds, &mut buffer);

            let mut i_buffer = vec![1.0; 25];
            let i_bounds = AABB::new(matrix![3, 7; 3, 7]);
            let i_domain = Domain::new(i_bounds, &mut i_buffer);

            domain.par_set_subdomain(&i_domain, 2);

            for c in domain.aabb.coord_iter() {
                if i_bounds.contains(&c) {
                    assert_eq!(domain.view(&c), 1.0);
                } else {
                    assert_eq!(domain.view(&c), 0.0);
                }
            }
        }
    }

    #[test]
    fn par_from_superset_test() {
        {
            let mut buffer = vec![0.0; 100];
            let bounds = AABB::new(matrix![0, 9; 0, 9;]);
            let domain = Domain::new(bounds, &mut buffer);

            let mut i_buffer = vec![1.0; 25];
            let i_bounds = AABB::new(matrix![3, 7; 3, 7]);
            let mut i_domain = Domain::new(i_bounds, &mut i_buffer);

            i_domain.par_from_superset(&domain, 3);
            for c in i_domain.aabb.coord_iter() {
                assert_eq!(domain.view(&c), 0.0);
            }
        }
    }
}
