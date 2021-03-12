use image::{ImageBuffer, ConvertBuffer, Rgb, Rgba, Luma, GenericImageView, GrayImage};
use crate::{create_threshold, add_border, segmentation, decode};
use std::time::Instant;
use std::collections::HashMap;
use imageproc::drawing::{draw_hollow_circle_mut, draw_text_mut, draw_antialiased_line_segment, draw_line_segment_mut, draw_antialiased_line_segment_mut, draw_hollow_rect, draw_hollow_rect_mut};
use imageproc::filter::{filter3x3, gaussian_blur_f32};
use rusttype::{FontCollection, Scale};
use crate::localize::{localize, draw_coords};
use cv_pinhole::CameraIntrinsics;
use crate::segmentation::FeatureVector;
use imageproc::region_labelling::{connected_components, Connectivity};
use crate::decode::LFTagComponents;
use opencv::core::dump_bool;
use imageproc::morphology::dilate;
use imageproc::distance_transform::Norm;
use imageproc::utils::gray_bench_image;
use imageproc::contrast::adaptive_threshold;
use imageproc::pixelops::interpolate;
use imageproc::gradients::{prewitt_gradients, sobel_gradients};
use imageproc::rect::Rect;

pub fn sharpen3x3(image: &GrayImage) -> GrayImage {
    let s = 0.1;
    let identity_minus_laplacian = [0.0, -s, 0.0, -s, 1.0 + 4.0 * s, -s, 0.0, -s, 0.0];
    filter3x3(image, &identity_minus_laplacian)
}

lazy_static! {
    static ref FONT: rusttype::Font<'static> = {
        let font = Vec::from(include_bytes!("DejaVuSans.ttf") as &[u8]);
        let font = FontCollection::from_bytes(font)
            .unwrap()
            .into_font()
            .unwrap();
        font
    };
}

pub fn find_topotags_debug(input: ImageBuffer<Rgb<u8>, Vec<u8>>, camera: CameraIntrinsics) -> Option<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let input: ImageBuffer<Luma<u8>, Vec<u8>> = input.convert();

    let dim = input.dimensions();

    // println!("Resolution: {}x{}", dim.0, dim.1);

    let threshold_map = create_threshold(&input)?;
    let mut output = ImageBuffer::new(dim.0, dim.1);

    let hard_thresh = 50; // handles saturation better

    for i in input.enumerate_pixels() {
        let thresh = threshold_map.get_pixel(i.0, i.1)[0];
        output.put_pixel(
            i.0,
            i.1,
            Luma ([{
                    if (i.2[0] > thresh || i.2[0] > (255 - hard_thresh))
                        && (i.2[0] > hard_thresh)
                    {
                        255
                    } else {
                        0
                    }
                }]),
        )
    }
    add_border(&mut output);
    let now = Instant::now();
    let mut topo = segmentation::to_topo(&output, &input);
    let len = (topo.len() - 1) as u32;
    //assert_eq!(area(&mut topo, len), 480 * 640);

    //filter(&mut topo, bg);
    //filter_img(&mut output, &mut topo, bg);
    let mut tags = HashMap::new();
    decode::detect_topotag(
        &mut topo,
        len,
        &mut tags,
        &vec![decode::TopotagClass::Topotag4x4, decode::TopotagClass::Topotag3x3],
    );
    //exit(1);
    //println!("{} connected components", topo.len());
    //println!("{}Mpix/s", (640 * 480)as f32 / now.elapsed().as_micros() as f32);
    let mut output: ImageBuffer<Rgba<u8>, Vec<u8>> = output.convert();

    for (k, v) in tags {
        if let decode::TopotagComponents::Root { nodes, .. } = &v {
            if let Some(decoded) = decode::DecodedTopotag::decode_topotag(&v) {
                println!("tag decoded: {:?}", decoded.data);
                let tr = decoded.top_right;
                let bl = decoded.bottom_left;

                let height = 20.0;
                let scale = Scale {
                    x: height,
                    y: height,
                };

                draw_text_mut(
                    &mut output,
                    Rgba([255, 0, 0, 255]),
                    k[0].0,
                    k[0].1,
                    scale,
                    &FONT,
                    &format!("id = {}", decoded.data),
                );

                for (idx, i) in decoded.node_pos.iter().enumerate() {
                    draw_text_mut(
                        &mut output,
                        Rgba([255, 0, 0, 255]),
                        i.0 as u32,
                        i.1 as u32,
                        scale,
                        &FONT,
                        &format!("{}", idx),
                    );
                }

                for (idx, i) in nodes.iter().enumerate() {
                    match i {
                        decode::TopotagComponents::Normal { node, data } => {
                            if tr == idx {
                                draw_hollow_circle_mut(
                                    &mut output,
                                    node.get_com(),
                                    5,
                                    Rgba([255, 255, 0, 255]),
                                );
                            }
                            if bl == idx {
                                draw_hollow_circle_mut(
                                    &mut output,
                                    node.get_com(),
                                    5,
                                    Rgba([255, 0, 255, 255]),
                                );
                            }
                            if *data {
                                draw_hollow_circle_mut(
                                    &mut output,
                                    node.get_com(),
                                    10,
                                    Rgba([255, 0, 0, 255]),
                                );
                            } else {
                                draw_hollow_circle_mut(
                                    &mut output,
                                    node.get_com(),
                                    10,
                                    Rgba([0, 0, 255, 255]),
                                );
                            }
                        }
                        decode::TopotagComponents::Baseline { nodes, .. } => {
                            draw_hollow_circle_mut(
                                &mut output,
                                nodes[decoded.top_left].get_com(),
                                5,
                                Rgba([0, 255, 255, 255]),
                            );
                            draw_hollow_circle_mut(
                                &mut output,
                                nodes[0].get_com(),
                                10,
                                Rgba([0, 255, 0, 255]),
                            );
                            draw_hollow_circle_mut(
                                &mut output,
                                nodes[1].get_com(),
                                10,
                                Rgba([0, 255, 0, 255]),
                            );
                        }
                        _ => {}
                    }
                }
                let pose = localize(&camera, &decoded);
                draw_coords(&mut output, &camera, &pose, 3.0);
                println!("pose: {:?}", pose);
            }
        }
        //draw_hollow_rect_mut(&mut output, Rect::at(k[1].0 as i32, k[1].1 as i32).of_size(k[0].0 - k[1].0, k[0].1 - k[1].1), Rgba{data:[255, 0, 0, 255]})
    }
    println!("done in {}us", now.elapsed().as_micros());
    Some(output)
}

pub fn dilate_fv(bin_input: &ImageBuffer<Luma<u8>, Vec<u8>>, input: &ImageBuffer<Luma<u8>, Vec<u8>>, fv: &mut FeatureVector) {
    let buffer = 3;
    let x = fv.bounding_box[0].0 - buffer;
    let y = fv.bounding_box[0].1 - buffer;
    let width = fv.bounding_box[1].0 - x + buffer * 2;
    let height = fv.bounding_box[1].1 - y + buffer * 2;

    if fv.bounding_box[0].0 <= buffer || fv.bounding_box[0].1 <= buffer || fv.bounding_box[1].0 + buffer * 2 >= bin_input.width() || fv.bounding_box[1].1 + buffer * 2 >= bin_input.height() {
        return
    }

    let roi = bin_input.view(x, y, width, height).to_image();
    let roi_grey = input.view(x, y, width, height);
    let mut connected = connected_components(&roi, Connectivity::Four, Luma([255]));
    let connected_id = connected.get_pixel(fv.max_x - x, fv.bounding_box[1].1 - y)[0];

    let mut v = Vec::new();

    connected.iter_mut().map(|p| {
        if *p == connected_id {
            v.push(255u8)
        } else {
            v.push(0)
        }
    }).last();

    let bin_connected =  ImageBuffer::from_vec(connected.width(), connected.height(), v).unwrap();
    let dialated = dilate(&bin_connected, Norm::LInf, 1);

    dialated.enumerate_pixels().map(|(x_r, y_r, p)| {
        if p[0] != 0 {
            if connected.get_pixel(x_r, y_r)[0] == 0 {
                fv.add_pixel_no_area(x + x_r, y + y_r, roi_grey.get_pixel(x_r, y_r)[0] as u32, fv.color)
            }
        }
    }).last();
}

pub fn find_lftags_debug(input_color: &ImageBuffer<Rgb<u8>, Vec<u8>>, camera: CameraIntrinsics) -> Option<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let now = Instant::now();

    let input: ImageBuffer<Luma<u8>, Vec<u8>> = input_color.convert();
    let mut gradient = &mut sobel_gradients(&input);
    let input = sharpen3x3(&input);

    let dim = input.dimensions();

    let threshold_map = create_threshold(&input)?;
    let mut output = ImageBuffer::new(dim.0, dim.1);
    let hard_thresh = 10; // handles saturation better
    for i in input.enumerate_pixels() {
        let thresh = threshold_map.get_pixel(i.0, i.1)[0];
        output.put_pixel(
            i.0,
            i.1,
            Luma([{
                    if (i.2[0] as u32 > thresh as u32 || i.2[0] > (255 - hard_thresh))
                        && (i.2[0] > hard_thresh) {
                        255
                    } else {
                        0
                    }
                }]),
        )
    }

    add_border(&mut output);

    let mut topo = segmentation::to_topo(&output, &input);
    let len = (topo.len() - 1) as u32;

    //assert_eq!(area(&mut topo, len), 480 * 640);

    let mut tags = HashMap::new();
    decode::detect_lftag(
        &mut topo,
        len,
        &mut tags,
        &vec![decode::LFTagClass::LFTag3x3, decode::LFTagClass::LFTag4x4],
        // &vec![decode::LFTagClass::LFTag3x3],
    );

    // if tags.len() > 10 {
    //     return None
    // }

    for (_, mut i) in &mut tags {
        match &mut i {
            LFTagComponents::Root { ref mut nodes , ..} => {
                for mut i in nodes {
                    match &mut i {
                        LFTagComponents::Normal { ref mut node } => {
                            dilate_fv(&output, &input, node);
                        },
                        _ => {}
                    }
                }
            },
            _ => {

            }
        }
    }

    let mut output: ImageBuffer<Rgba<u8>, Vec<u8>> = input_color.convert();
    //let mut output: ImageBuffer<Rgba<u8>, Vec<u8>> = output.convert();

    let height = 20.0;
    let scale = Scale {
        x: height,
        y: height,
    };

    // for i in topo {
    //     if i.max_x < output.width() && i.bounding_box[1].1 < output.height() {
    //         output.put_pixel(i.max_x, i.bounding_box[1].1, Rgba {
    //             data: [0, 0, 255, 255],
    //         }, )
    //     }
    // }

    for (k, v) in tags {
        // println!("potential tag");
        // draw_hollow_rect_mut(&mut output, Rect::at(k[0].0 as i32, k[0].1 as i32).of_size(k[1].0 - k[0].0, k[1].1 - k[0].1), Rgba { data: [255, 0, 0, 255], });

        if let Some(decoded) = decode::DecodedLFTag::decode_lftag(&v, camera, &mut gradient) {
            for (idx, i) in decoded.node_pos.iter().enumerate() {
                let vert_start = (i.0, i.1 - 5.0);
                let vert_end = (i.0, i.1 + 5.0);

                let horiz_start = (i.0 - 5.0, i.1);
                let horiz_end = (i.0 + 5.0, i.1);

                draw_line_segment_mut(&mut output,
                                      vert_start,
                                      vert_end,
                                      Rgba([255, 0, 0, 255]));
                draw_line_segment_mut(&mut output,
                                      horiz_start,
                                      horiz_end,
                                      Rgba([255, 0, 0, 255]));
            }

            for (idx, i) in decoded.expected_node_pos.iter().enumerate() {
                //draw_text_mut(
                //    &mut output,
                //    Rgba {
                //        data: [255, 0, 0, 255],
                //    },
                //    i.0 as u32,
                //    i.1 as u32,
                //    scale,
                //    &FONT,
                //    &format!("{}", idx),
                //);
                let vert_start = ((i.0 - 5.0) as i32, (i.1 - 5.0) as i32);
                let vert_end = ((i.0 + 5.0) as i32, (i.1 + 5.0) as i32);

                let horiz_start = ((i.0 - 5.0) as i32, (i.1 + 5.0) as i32);
                let horiz_end = ((i.0 + 5.0) as i32, (i.1 - 5.0) as i32);

                draw_antialiased_line_segment_mut(&mut output,
                                      vert_start,
                                      vert_end,
                                      Rgba([0, 255, 0, 255]), interpolate);
                draw_antialiased_line_segment_mut(&mut output,
                                      horiz_start,
                                      horiz_end,
                                      Rgba([0, 255, 0, 255]), interpolate);
            }
            let _ang = decoded.final_pose.0.rotation.euler_angles();
            draw_text_mut(
                &mut output,
                Rgba([255, 0, 0, 255]),
                k[1].0 as u32,
                k[1].1 as u32,
                scale,
                &FONT,
                &format!("id = {}", decoded.data),
            );
            let scale = (decoded.class.get_width() as f64 + 1.0) * 6.0;
            draw_coords(&mut output, &camera, &decoded.final_pose, scale);
        }

    }
    draw_text_mut(
        &mut output,
        Rgba([255, 0, 0, 255]),
        0,
        0,
        scale,
        &FONT,
        &format!("{}ms", now.elapsed().as_micros()/1000),
    );
    Some(output)
}