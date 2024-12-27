use nhls::domain::*;
use nhls::image_1d_example::*;
use nhls::solver::*;

fn main() {
    let (args, output_image_path) = Args::cli_parse("heat_1d_ap_direct");

    let stencil = nhls::standard_stencils::heat_1d(1.0, 1.0, 0.5);

    // Create domains
    let grid_bound = args.grid_bounds();
    let buffer_size = grid_bound.buffer_size();
    let mut grid_input = vec![0.0; buffer_size];
    let mut input_domain = Domain::new(grid_bound, &mut grid_input);

    let mut grid_output = vec![0.0; buffer_size];
    let mut output_domain = Domain::new(grid_bound, &mut grid_output);

    // Create BC
    let bc = ConstantCheck::new(1.0, grid_bound);

    // Make image
    let mut img = nhls::image::Image1D::new(grid_bound, args.lines as u32);
    img.add_line(0, input_domain.buffer());
    for t in 1..args.lines as u32 {
        box_apply(
            &bc,
            &stencil,
            &mut input_domain,
            &mut output_domain,
            args.steps_per_line,
            args.chunk_size,
        );
        std::mem::swap(&mut input_domain, &mut output_domain);
        img.add_line(t, input_domain.buffer());
    }

    img.write(&output_image_path);
}
