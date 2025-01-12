use nhls::domain::*;
use nhls::fft_solver::PeriodicSolver;
use nhls::image_3d_example::*;
use nhls::init;
use nhls::vtk::*;

fn main() {
    let args = Args::cli_parse("heat_3d_p_fft");

    let stencil =
        nhls::standard_stencils::heat_3d(1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);

    if args.rand_init {
        init::rand(&mut input_domain, 1024, args.chunk_size);
    } else {
        init::normal_ic_3d(&mut input_domain, args.chunk_size);
    }
    if args.write_images {
        write_vtk3d(&input_domain, &args.frame_name(0));
    }

    // Apply periodic solver
    let mut periodic_solver = PeriodicSolver::create(
        &stencil,
        output_domain.buffer_mut(),
        &grid_bound,
        args.steps_per_image,
        args.plan_type,
        args.chunk_size,
    );
    for t in 1..args.images {
        periodic_solver.apply(&mut input_domain, &mut output_domain);
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            write_vtk3d(&input_domain, &args.frame_name(t));
        }
    }

    args.save_wisdom();
}
