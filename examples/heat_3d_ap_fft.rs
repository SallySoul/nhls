use nhls::domain::*;
use nhls::fft_solver::*;
use nhls::image_3d_example::*;
use nhls::vtk::*;

fn main() {
    let args = Args::cli_parse("heat_3d_ap_fft");

    let stencil =
        nhls::standard_stencils::heat_3d(1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1);

    let grid_bound = args.grid_bounds();

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Create AP Solver
    let cutoff = 40;
    let ratio = 0.5;
    let solver = APSolver::new(
        &bc,
        &stencil,
        grid_bound,
        args.steps_per_image,
        args.plan_type,
        cutoff,
        ratio,
        args.chunk_size,
    );
    if args.write_dot {
        println!("WRITING DOT FILE");
        let mut dot_path = args.output_dir.clone();
        dot_path.push("plan.dot");
        solver.to_dot_file(&dot_path);

        let mut d_path = args.output_dir.clone();
        d_path.push("scratch.txt");
        solver.scratch_descriptor_file(&d_path);
        //std::process::exit(0);
    }

    // Create domains
    // Create domains
    let mut buffer_1 = OwnedDomain::new(grid_bound);
    let mut buffer_2 = OwnedDomain::new(grid_bound);
    let mut input_domain = buffer_1.as_slice_domain();
    let mut output_domain = buffer_2.as_slice_domain();
    if args.write_images {
        write_vtk3d(&input_domain, &args.frame_name(0));
    }

    for t in 1..args.images {
        solver.apply(&mut input_domain, &mut output_domain);
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            write_vtk3d(&input_domain, &args.frame_name(t));
        }
    }

    args.save_wisdom();
}
