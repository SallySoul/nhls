mod chunk;
mod owned;
mod slice;

pub use chunk::*;
pub use owned::*;
pub use slice::*;

use crate::util::*;
use rayon::prelude::*;

pub trait DomainView<const GRID_DIMENSION: usize>: Sync {
    fn aabb(&self) -> &AABB<GRID_DIMENSION>;

    fn set_aabb(&mut self, aabb: AABB<GRID_DIMENSION>);

    fn buffer(&self) -> &[f64];

    fn buffer_mut(&mut self) -> &mut [f64];

    fn aabb_buffer_mut(&mut self) -> (&AABB<GRID_DIMENSION>, &mut [f64]);

    fn view(&self, world_coord: &Coord<GRID_DIMENSION>) -> f64;

    fn par_modify_access<'a>(
        &'a mut self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = DomainChunk<'a, GRID_DIMENSION>> {
        let (aabb, buffer) = self.aabb_buffer_mut();
        par_modify_access_impl(buffer, aabb, chunk_size)
    }

    fn par_set_values<
        F: FnOnce(Coord<GRID_DIMENSION>) -> f64 + Send + Sync + Copy,
    >(
        &mut self,
        f: F,
        chunk_size: usize,
    ) {
        self.par_modify_access(chunk_size).for_each(
            |mut d: DomainChunk<'_, GRID_DIMENSION>| {
                d.coord_iter_mut().for_each(|(world_coord, value_mut)| {
                    *value_mut = f(world_coord);
                })
            },
        );
    }

    /// Copy other domain into self
    fn par_set_subdomain<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        other: &DomainType,
        chunk_size: usize,
    ) {
        let const_self_ref: &Self = self;
        other.buffer()[0..other.aabb().buffer_size()]
            .par_chunks(chunk_size)
            .enumerate()
            .for_each(move |(i, buffer_chunk): (usize, &[f64])| {
                let self_ptr = const_self_ref as *const Self;
                let mut_self_ref: &mut Self =
                    unsafe { &mut *(self_ptr as *mut Self) as &mut Self };
                let offset = i * chunk_size;
                for i in 0..buffer_chunk.len() {
                    let other_linear_index = i + offset;
                    let world_coord =
                        other.aabb().linear_to_coord(other_linear_index);
                    let self_linear_index =
                        mut_self_ref.aabb().coord_to_linear(&world_coord);
                    mut_self_ref.buffer_mut()[self_linear_index] =
                        other.buffer()[other_linear_index];
                }
            });
    }

    /// Copy self coords from other into self
    fn par_from_superset<DomainType: DomainView<GRID_DIMENSION>>(
        &mut self,
        other: &DomainType,
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

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn par_set_subdomain_test() {
        {
            let chunk_size = 1;
            let bounds = AABB::new(matrix![0, 9; 0, 9;]);
            let mut domain = OwnedDomain::new(bounds);

            let i_bounds = AABB::new(matrix![3, 7; 3, 7]);
            let mut i_domain = OwnedDomain::new(i_bounds);
            i_domain.par_set_values(|_| 1.0, chunk_size);

            domain.par_set_subdomain(&i_domain, 2);

            for c in domain.aabb().coord_iter() {
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
            let chunk_size = 1;
            let bounds = AABB::new(matrix![0, 9; 0, 9;]);
            let domain = OwnedDomain::new(bounds);

            let i_bounds = AABB::new(matrix![3, 7; 3, 7]);
            let mut i_domain = OwnedDomain::new(i_bounds);
            i_domain.par_set_values(|_| 1.0, chunk_size);

            i_domain.par_from_superset(&domain, 3);
            for c in i_domain.aabb().coord_iter() {
                assert_eq!(domain.view(&c), 0.0);
            }
        }
    }
}
