use nhls::domain::*;
use nhls::fft_solver::PeriodicSolver;
use nhls::image_1d_example::*;
use nhls::init;

fn main() {
    let (args, output_image_path) = Args::cli_parse("heat_1d_p_fft");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    if args.rand_init {
        init::rand(&mut input_domain, 1024, args.chunk_size);
    } else {
        init::normal_ic_1d(&mut input_domain, args.chunk_size);
    }

    let mut img = None;
    if args.write_image {
        let mut i = nhls::image::Image1D::new(grid_bound, args.lines as u32);
        i.add_line(0, input_domain.buffer());
        img = Some(i);
    }

    let mut periodic_solver = PeriodicSolver::create(
        &stencil,
        output_domain.buffer_mut(),
        &grid_bound,
        args.steps_per_line,
        args.plan_type,
        args.chunk_size,
    );
    for t in 1..args.lines as u32 {
        periodic_solver.apply(&mut input_domain, &mut output_domain);
        std::mem::swap(&mut input_domain, &mut output_domain);
        if let Some(i) = img.as_mut() {
            i.add_line(t, input_domain.buffer());
        }
    }
    if let Some(i) = img {
        i.write(&output_image_path);
    }

    args.save_wisdom();
}
