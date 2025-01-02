use crate::solver::fft_plan::PlanType;
use crate::util::*;
use clap::Parser;
use std::path::PathBuf;

/// nhls 1D stencil executable
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Directory for output files, will be created.
    /// WARNING, if this Directory
    /// already exists, current contents will be removed.
    #[arg(short, long)]
    pub output_dir: std::path::PathBuf,

    /// Chunk size to use for parallelism.
    #[arg(short, long, default_value = "1000")]
    pub chunk_size: usize,

    /// How many lines the output image should have.
    #[arg(short, long, default_value = "1000")]
    pub lines: usize,

    /// How many steps to take per line.
    #[arg(short, long, default_value = "16")]
    pub steps_per_line: usize,

    /// Domain size
    #[arg(short, long, default_value = "1000")]
    pub domain_size: usize,

    /// Write out image, WARNING: we do not check image size, so be reasonable.
    #[arg(short, long)]
    pub write_image: bool,

    /// The number of threads to use.
    #[arg(short, long, default_value = "8")]
    pub threads: usize,

    /// FFTW3 plan creation strategy.
    #[arg(short, long, default_value = "estimate")]
    pub plan_type: PlanType,

    /// File to load and save FFTW3 wisdom.
    #[arg(short, long)]
    pub wisdom_file: Option<PathBuf>,

    /// Fill with random values matching 2023 implementation
    #[arg(short, long)]
    pub rand_init: bool,
}

impl Args {
    pub fn cli_parse(name: &str) -> (Self, PathBuf) {
        println!("EXAMPLE: {}", name);
        println!("GIT DESCRIBE: {}", env!("GIT_DESCRIBE"));
        println!("GIT HASH: {}", env!("GIT_HASH"));
        let args = Args::parse();

        let output_dir = args.output_dir.to_str().unwrap();
        let _ = std::fs::remove_dir_all(output_dir);
        std::fs::create_dir(output_dir).unwrap();

        let mut output_image_path = args.output_dir.clone();
        output_image_path.push(format!("{}.png", name));

        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
        fftw::threading::init_threads_f64().unwrap();
        fftw::threading::plan_with_nthreads_f64(args.threads);

        if let Some(ref wisdom_path) = args.wisdom_file {
            if wisdom_path.exists() {
                fftw::wisdom::import_wisdom_file_f64(&wisdom_path).unwrap();
            }
        }

        (args, output_image_path)
    }

    pub fn grid_bounds(&self) -> AABB<1> {
        AABB::new(matrix![0, self.domain_size as i32 - 1])
    }

    pub fn save_wisdom(&self) {
        if let Some(ref wisdom_path) = self.wisdom_file {
            fftw::wisdom::export_wisdom_file_f64(&wisdom_path).unwrap();
        }
    }
}
