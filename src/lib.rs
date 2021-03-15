
extern crate euclid;
extern crate image;
#[macro_use]
extern crate lazy_static;
extern crate nalgebra;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
#[cfg(test)]
extern crate quickcheck;

use image::{ImageBuffer, Luma, FilterType};
use image::imageops::resize;

pub mod decode;
pub mod localize;
pub mod segmentation;
pub mod debug;

fn add_border(input: &mut ImageBuffer<Luma<u8>, Vec<u8>>) {
    let (x, y) = input.dimensions();
    for i in 0..x {
        input.put_pixel(i, 0, Luma([0]));
        input.put_pixel(i, y - 1, Luma([0]));
    }

    for i in 0..y {
        input.put_pixel(0, i, Luma([0]));
        input.put_pixel(x - 1, i, Luma([0]));
    }
}

fn create_threshold(input: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
    let dim = input.dimensions();
    let threshold_map_res_div: u32 = 8;

    if dim.0 % threshold_map_res_div != 0 {
        return None
    }
    if dim.1 % threshold_map_res_div != 0 {
        return None
    }

    assert_eq!(dim.0 % threshold_map_res_div, 0);
    assert_eq!(dim.1 % threshold_map_res_div, 0);
    let thresold_map_dim = (
        (dim.0 / threshold_map_res_div) as usize,
        (dim.1 / threshold_map_res_div) as usize,
    );
    let mut threshold_map: Vec<Vec<u32>> = vec![vec![0; thresold_map_dim.0]; thresold_map_dim.1];
    for i in input.enumerate_pixels() {
        threshold_map[(i.1 / threshold_map_res_div) as usize]
            [(i.0 / threshold_map_res_div) as usize] += i.2[0] as u32;
    }
    let mut output = ImageBuffer::new(dim.0, dim.1);
    for i in input.enumerate_pixels() {
        output.put_pixel(
            i.0,
            i.1,
            Luma([{
                    if i.0 >= (dim.0 - threshold_map_res_div)
                        || i.1 >= (dim.1 - threshold_map_res_div)
                    {
                        ((threshold_map[(i.1 / threshold_map_res_div) as usize]
                            [(i.0 / threshold_map_res_div) as usize])
                            / (threshold_map_res_div * threshold_map_res_div))
                            as u8
                    } else {
                        let p1 = ((threshold_map[(i.1 / threshold_map_res_div) as usize]
                            [(i.0 / threshold_map_res_div) as usize])
                            / (threshold_map_res_div * threshold_map_res_div))
                            as u32;
                        let p2 = ((threshold_map[(i.1 / threshold_map_res_div) as usize + 1]
                            [(i.0 / threshold_map_res_div) as usize])
                            / (threshold_map_res_div * threshold_map_res_div))
                            as u32;
                        let p3 = ((threshold_map[(i.1 / threshold_map_res_div) as usize]
                            [(i.0 / threshold_map_res_div) as usize + 1])
                            / (threshold_map_res_div * threshold_map_res_div))
                            as u32;
                        let p4 = ((threshold_map[(i.1 / threshold_map_res_div) as usize + 1]
                            [(i.0 / threshold_map_res_div) as usize + 1])
                            / (threshold_map_res_div * threshold_map_res_div))
                            as u32;
                        let x = i.0 % threshold_map_res_div;
                        let y = i.1 % threshold_map_res_div;
                        let p5 = p3 * x + p1 * (threshold_map_res_div - x);
                        let p6 = p4 * x + p2 * (threshold_map_res_div - x);
                        ((p6 * y + p5 * (threshold_map_res_div - y))
                            / (threshold_map_res_div * threshold_map_res_div))
                            as u8
                    }
                }]),
        );
    }

    //let downscaled = resize(input, thresold_map_dim.0 as u32, thresold_map_dim.1 as u32, FilterType::Triangle);
    //let output = resize(&downscaled, dim.0, dim.1, FilterType::Triangle);

    Some(output)
}

