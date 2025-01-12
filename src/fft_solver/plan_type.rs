use clap::ValueEnum;
use fftw::types::Flag;

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
