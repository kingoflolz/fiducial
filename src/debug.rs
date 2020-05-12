use image::{ImageBuffer, ConvertBuffer, Rgb, Rgba, Luma};
use crate::{create_threshold, add_border, segmentation, decode};
use std::time::Instant;
use std::collections::HashMap;
use imageproc::drawing::{draw_hollow_circle_mut, draw_text_mut};
use rusttype::{FontCollection, Scale};
use crate::localize::{localize, draw_coords};
use cv_pinhole::CameraIntrinsics;

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

pub fn find_tags_debug(input: ImageBuffer<Rgb<u8>, Vec<u8>>, camera: CameraIntrinsics) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let input: ImageBuffer<Luma<u8>, Vec<u8>> = input.convert();

    let dim = input.dimensions();

    // println!("Resolution: {}x{}", dim.0, dim.1);

    let threshold_map = create_threshold(&input);
    let mut output = ImageBuffer::new(dim.0, dim.1);

    let hard_thresh = 50; // handles saturation better

    for i in input.enumerate_pixels() {
        let thresh = threshold_map.get_pixel(i.0, i.1).data[0];
        output.put_pixel(
            i.0,
            i.1,
            Luma {
                data: [{
                    if (i.2.data[0] > thresh || i.2.data[0] > (255 - hard_thresh))
                        && (i.2.data[0] > hard_thresh)
                    {
                        255
                    } else {
                        0
                    }
                }],
            },
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
    decode::detect_tag(
        &mut topo,
        len,
        &mut tags,
        &vec![decode::TagClass::Topotag4x4, decode::TagClass::Topotag3x3],
    );
    //exit(1);
    //println!("{} connected components", topo.len());
    //println!("{}Mpix/s", (640 * 480)as f32 / now.elapsed().as_micros() as f32);
    let mut output: ImageBuffer<Rgba<u8>, Vec<u8>> = output.convert();

    for (k, v) in tags {
        if let decode::TagComponents::Root { nodes, .. } = &v {
            if let Some(decoded) = decode::DecodedTag::decode_tag(&v) {
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
                    Rgba {
                        data: [255, 0, 0, 255],
                    },
                    k[0].0,
                    k[0].1,
                    scale,
                    &FONT,
                    &format!("id = {}", decoded.data),
                );

                for (idx, i) in decoded.node_pos.iter().enumerate() {
                    draw_text_mut(
                        &mut output,
                        Rgba {
                            data: [255, 0, 0, 255],
                        },
                        i.0 as u32,
                        i.1 as u32,
                        scale,
                        &FONT,
                        &format!("{}", idx),
                    );
                }

                for (idx, i) in nodes.iter().enumerate() {
                    match i {
                        decode::TagComponents::Normal { node, data } => {
                            if tr == idx {
                                draw_hollow_circle_mut(
                                    &mut output,
                                    node.get_com(),
                                    5,
                                    Rgba {
                                        data: [255, 255, 0, 255],
                                    },
                                );
                            }
                            if bl == idx {
                                draw_hollow_circle_mut(
                                    &mut output,
                                    node.get_com(),
                                    5,
                                    Rgba {
                                        data: [255, 0, 255, 255],
                                    },
                                );
                            }
                            if *data {
                                draw_hollow_circle_mut(
                                    &mut output,
                                    node.get_com(),
                                    10,
                                    Rgba {
                                        data: [255, 0, 0, 255],
                                    },
                                );
                            } else {
                                draw_hollow_circle_mut(
                                    &mut output,
                                    node.get_com(),
                                    10,
                                    Rgba {
                                        data: [0, 0, 255, 255],
                                    },
                                );
                            }
                        }
                        decode::TagComponents::Baseline { nodes, .. } => {
                            draw_hollow_circle_mut(
                                &mut output,
                                nodes[decoded.top_left].get_com(),
                                5,
                                Rgba {
                                    data: [0, 255, 255, 255],
                                },
                            );
                            draw_hollow_circle_mut(
                                &mut output,
                                nodes[0].get_com(),
                                10,
                                Rgba {
                                    data: [0, 255, 0, 255],
                                },
                            );
                            draw_hollow_circle_mut(
                                &mut output,
                                nodes[1].get_com(),
                                10,
                                Rgba {
                                    data: [0, 255, 0, 255],
                                },
                            );
                        }
                        _ => {}
                    }
                }
                let pose = localize(&camera, &decoded);
                draw_coords(&mut output, &camera, &pose);
                println!("pose: {:?}", pose);
            }
        }
        //draw_hollow_rect_mut(&mut output, Rect::at(k[1].0 as i32, k[1].1 as i32).of_size(k[0].0 - k[1].0, k[0].1 - k[1].1), Rgba{data:[255, 0, 0, 255]})
    }
    println!("done in {}us", now.elapsed().as_micros());
    output
}