/*
use nhls::domain::*;
use nhls::image::*;
use nhls::image_2d_example::*;
use nhls::solver::*;
*/

fn main() {
    /*
    let args = Args::cli_parse("heat_2d_ap_fft");

    let stencil = nhls::standard_stencils::heat_2d(1.0, 1.0, 1.0, 0.2, 0.2);

    // Create domains
    let grid_bound = args.grid_bounds();
    let mut input_domain = OwnedDomain::new(grid_bound);
    let mut output_domain = OwnedDomain::new(grid_bound);
    if args.write_images {
        image2d(&input_domain, &args.frame_name(0));
    }

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Create AP Solver
    let cutoff = 40;
    let ratio = 0.5;
    let mut solver = APSolver::new(
        &bc,
        &stencil,
        cutoff,
        ratio,
        &grid_bound,
        args.plan_type,
        args.chunk_size,
    );
    for t in 1..args.images {
        solver.loop_solve(
            &mut input_domain,
            &mut output_domain,
            args.steps_per_image,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        if args.write_images {
            image2d(&input_domain, &args.frame_name(t));
        }
    }

    args.save_wisdom();
    */
}
