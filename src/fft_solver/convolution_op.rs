use crate::domain::*;
use crate::fft_solver::*;
use crate::par_slice;
use crate::stencil::*;
use crate::util::*;
use fftw::array::*;
use fftw::plan::*;

pub struct ConvolutionOperation {
    pub forward_plan: fftw::plan::Plan<f64, c64, fftw::plan::Plan64>,
    pub backward_plan: fftw::plan::Plan<c64, f64, fftw::plan::Plan64>,
    pub convolution: AlignedVec<c64>,
}

impl ConvolutionOperation {
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn create<
        Operation,
        const GRID_DIMENSION: usize,
        const NEIGHBORHOOD_SIZE: usize,
    >(
        stencil: &StencilF64<Operation, GRID_DIMENSION, NEIGHBORHOOD_SIZE>,
        stencil_weights: &[f64; NEIGHBORHOOD_SIZE],
        real_buffer: &mut [f64],
        convolution_buffer: &mut [c64],
        aabb: &AABB<GRID_DIMENSION>,
        steps: usize,
        plan_type: PlanType,
        chunk_size: usize,
    ) -> Self
    where
        Operation: StencilOperation<f64, NEIGHBORHOOD_SIZE>,
    {
        let size = aabb.exclusive_bounds();
        let plan_size = size.try_cast::<usize>().unwrap();
        let forward_plan = fftw::plan::R2CPlan64::aligned(
            plan_size.as_slice(),
            plan_type.to_fftw3_flag(),
        )
        .unwrap();
        let backward_plan = fftw::plan::C2RPlan64::aligned(
            plan_size.as_slice(),
            plan_type.to_fftw3_flag(),
        )
        .unwrap();

        // Place offsets in real buffer
        let offsets = stencil.offsets();
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let rn_i: Coord<GRID_DIMENSION> = offsets[n_i] * -1;
            let index = &aabb.periodic_coord(&rn_i);
            let l = aabb.coord_to_linear(index);
            real_buffer[l] = stencil_weights[n_i];
        }

        // Calculate convolution of stencil
        let n_r = aabb.buffer_size();
        let n_c = aabb.complex_buffer_size();
        forward_plan
            .r2c(&mut real_buffer[0..n_r], &mut convolution_buffer[0..n_c])
            .unwrap();

        // clean up real buffer
        for n_i in 0..NEIGHBORHOOD_SIZE {
            let rn_i: Coord<GRID_DIMENSION> = offsets[n_i] * -1;
            let index = &aabb.periodic_coord(&rn_i);
            let l = aabb.coord_to_linear(index);
            real_buffer[l] = 0.0;
        }

        // Apply power calculation to convolution
        let mut result_buffer = fftw::array::AlignedVec::new(n_c);
        par_slice::power(
            steps,
            &mut convolution_buffer[0..n_c],
            &mut result_buffer[0..n_c],
            chunk_size,
        );

        // Clear convoluton_buffer
        par_slice::set_value(
            &mut convolution_buffer[0..n_c],
            c64::zero(),
            chunk_size,
        );

        ConvolutionOperation {
            forward_plan,
            backward_plan,
            convolution: result_buffer,
        }
    }

    #[inline]
    pub fn apply<
        const GRID_DIMENSION: usize,
        DomainType: DomainView<GRID_DIMENSION>,
    >(
        &self,
        input: &mut DomainType,
        output: &mut DomainType,
        complex_buffer: &mut [c64],
        chunk_size: usize,
    ) {
        // fftw bindings expect slices of specific size
        let n_r = input.aabb().buffer_size();
        let n_c = input.aabb().complex_buffer_size();
        self.forward_plan
            .r2c(input.buffer_mut(), &mut complex_buffer[0..n_c])
            .unwrap();
        par_slice::multiply_by(
            &mut complex_buffer[0..n_c],
            self.convolution.as_slice(),
            chunk_size,
        );
        self.backward_plan
            .c2r(&mut complex_buffer[0..n_c], output.buffer_mut())
            .unwrap();
        par_slice::div(output.buffer_mut(), n_r as f64, chunk_size);
    }
}
