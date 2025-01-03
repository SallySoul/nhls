use fftw::plan::*;
use fftw::types::*;

use crate::util::*;

use clap::ValueEnum;

/// FFTW3 Provides several strategies for plan creation,
/// we expose three of them.
#[derive(Copy, Clone, Debug, ValueEnum, Default)]
pub enum PlanType {
    /// Create optimziated plan
    #[default]
    Measure,

    /// Create optimized plan with more exhaustive search than Measaure
    Patient,

    /// Create an un-optimal plan quickly
    Estimate,

    /// Create plan only based on loaded wisdom
    WisdomOnly,
}

impl PlanType {
    pub fn to_fftw3_flag(&self) -> Flag {
        match self {
            PlanType::Measure => Flag::MEASURE,
            PlanType::Patient => Flag::PATIENT,
            PlanType::Estimate => Flag::ESTIMATE,
            PlanType::WisdomOnly => Flag::WISDOWMONLY,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct FFTPlanDescriptor<const GRID_DIMENSION: usize> {
    pub bound: AABB<GRID_DIMENSION>,
}

impl<const GRID_DIMENSION: usize> FFTPlanDescriptor<GRID_DIMENSION> {
    pub fn new(bound: AABB<GRID_DIMENSION>) -> Self {
        FFTPlanDescriptor { bound }
    }
}

pub struct FFTPlan {
    pub forward_plan: fftw::plan::Plan<f64, c64, fftw::plan::Plan64>,
    pub backward_plan: fftw::plan::Plan<c64, f64, fftw::plan::Plan64>,
}

impl FFTPlan {
    pub fn new<const GRID_DIMENSION: usize>(
        bound: AABB<GRID_DIMENSION>,
        plan_type: PlanType,
    ) -> Self {
        let size = bound.exclusive_bounds();
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
        FFTPlan {
            forward_plan,
            backward_plan,
        }
    }
}

// We need storage for plans
pub struct FFTPlanLibrary<const GRID_DIMENSION: usize> {
    pub plan_map:
        std::collections::HashMap<FFTPlanDescriptor<GRID_DIMENSION>, FFTPlan>,
    pub plan_type: PlanType,
}

impl<const GRID_DIMENSION: usize> FFTPlanLibrary<GRID_DIMENSION> {
    pub fn new(plan_type: PlanType) -> Self {
        FFTPlanLibrary {
            plan_map: std::collections::HashMap::new(),
            plan_type,
        }
    }

    pub fn get_plan(&mut self, bound: AABB<GRID_DIMENSION>) -> &mut FFTPlan {
        let key = FFTPlanDescriptor::new(bound);
        self.plan_map
            .entry(key)
            .or_insert(FFTPlan::new(bound, self.plan_type))
    }
}
