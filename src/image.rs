use crate::domain::*;
use crate::util::*;

pub struct Image1D {
    img_buffer: image::RgbImage,
}

impl Image1D {
    pub fn new(bound: AABB<1>, lines: u32) -> Self {
        debug_assert_eq!(bound.bounds[(0, 0)], 0);
        let exclusive_bound = bound.exclusive_bounds();
        Image1D {
            img_buffer: image::RgbImage::new(exclusive_bound[0] as u32, lines),
        }
    }

    pub fn add_line(&mut self, l: u32, v: &[f64]) {
        debug_assert!(l < self.img_buffer.height());
        debug_assert_eq!(v.len(), self.img_buffer.width() as usize);
        let gradient = colorous::TURBO;
        for x in 0..self.img_buffer.width() {
            let r = v[x as usize];
            let c = gradient.eval_continuous(r);
            self.img_buffer.put_pixel(x, l, image::Rgb(c.as_array()));
        }
    }

    pub fn write<F: AsRef<std::path::Path>>(self, s: &F) {
        self.img_buffer.save(s).expect("Couldn't save image");
    }
}

pub fn image2d<F: AsRef<std::path::Path>, DomainType: DomainView<2>>(
    domain: &DomainType,
    s: &F,
) {
    println!("Writing png: {:?}", s.as_ref());
    let aabb = domain.aabb();
    let exclusive_bounds = aabb.exclusive_bounds();
    let gradient = colorous::TURBO;
    let mut img = image::RgbImage::new(
        exclusive_bounds[0] as u32,
        exclusive_bounds[1] as u32,
    );
    for l in 0..exclusive_bounds[0] * exclusive_bounds[1] {
        let coord = domain.aabb().linear_to_coord(l as usize);
        let r = domain.view(&coord);
        let c = gradient.eval_continuous(r);
        img.put_pixel(
            coord[0] as u32,
            coord[1] as u32,
            image::Rgb(c.as_array()),
        );
    }
    img.save(s).expect("Couldn't save image");
}
