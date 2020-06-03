use crate::segmentation::{FeatureVector, area};
use euclid::default::{Vector2D};
use std::collections::HashMap;
use cv_pinhole::{CameraIntrinsics, NormalizedKeyPoint};
use arrsac::{Arrsac, Config};
use rand::prelude::SmallRng;
use rand::SeedableRng;
use lambda_twist::LambdaTwist;
use cv_core::{CameraModel, FeatureWorldMatch, KeyPoint, WorldPoint, sample_consensus::Consensus, WorldPose};
use nalgebra::{Point2, Point3, Isometry3, IsometryMatrix3, Translation, Vector3, Matrix3, Rotation3, distance};
use image::{ImageBuffer, Rgba, Luma};
use opencv::core::{Vector, CV_64F};
use opencv::prelude::*;
use itertools::iproduct;

use crate::localize::{opencv_localize, world_to_camera, world_to_camera_i32};
use imageproc::definitions::Image;
use imageproc::drawing::draw_antialiased_line_segment_mut;
use std::cell::RefCell;

#[derive(Debug, Clone, PartialEq)]
pub enum TopotagComponents {
    Root {
        nodes: Vec<TopotagComponents>,
        class: TopotagClass,
    },
    None,
    Baseline {
        nodes: [FeatureVector; 2],
    },
    Normal {
        data: bool,
        node: FeatureVector,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LFTagComponents {
    Root {
        nodes: Vec<LFTagComponents>,
        bg: FeatureVector,
        class: LFTagClass,
    },
    None,
    Normal {
        node: FeatureVector,
    },
    Baseline {
        node: FeatureVector,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedTopotag {
    pub data: usize,
    pub nodes: Vec<TopotagComponents>,
    pub top_right: usize,
    pub top_left: usize,
    pub bottom_left: usize,
    pub node_pos: Vec<(f32, f32)>,
    pub class: TopotagClass
}

impl DecodedTopotag {
    pub fn decode_topotag(root: &TopotagComponents) -> Option<DecodedTopotag> {
        if let TopotagComponents::Root { nodes, class } = root {
            if let TopotagComponents::Baseline {
                nodes: baseline_nodes,
            } = &nodes[0]
            {
                let mut node_pos = Vec::new();
                let mut top_right = &baseline_nodes[0];
                let mut top_right_id = 0;
                let mut min_angle = 999.9f32;
                let mut top_left = &baseline_nodes[0];
                let mut top_left_id = 0;

                for (idx, i) in nodes.iter().enumerate() {
                    if let TopotagComponents::Normal { node, .. } = i {
                        let baseline0: Vector2D<_> = baseline_nodes[0].get_com_f32().into();
                        let baseline1: Vector2D<_> = baseline_nodes[1].get_com_f32().into();
                        let node_vec: Vector2D<_> = node.get_com_f32().into();
                        let base_vec0 = baseline0 - node_vec;
                        let base_vec1 = baseline1 - node_vec;

                        let average_length = (base_vec0 + base_vec1) / 2.0f32;

                        let angle = (base_vec0.angle_to(base_vec1).radians.abs() + 0.02)
                            / average_length.length();
                        if angle < min_angle {
                            top_right_id = idx;
                            min_angle = angle;
                            top_right = node;

                            if base_vec1.length() > base_vec0.length() {
                                node_pos.clear();
                                node_pos.push(baseline_nodes[1].get_com_f32());
                                node_pos.push(baseline_nodes[0].get_com_f32());
                                top_left = &baseline_nodes[1];
                                top_left_id = 1;
                            } else {
                                node_pos.clear();
                                node_pos.push(baseline_nodes[0].get_com_f32());
                                node_pos.push(baseline_nodes[1].get_com_f32());
                                top_left = &baseline_nodes[0];
                                top_left_id = 0;
                            }
                        }
                    }
                }

                let mut max_angle = 0.0f32;

                let top_left_vec: Vector2D<_> = top_left.get_com_f32().into();
                let top_vec = top_left_vec - top_right.get_com_f32().into();

                for (_, i) in nodes.iter().enumerate() {
                    if let TopotagComponents::Normal { node, .. } = i {
                        let node_vec: Vector2D<_> = node.get_com_f32().into();
                        let angle = top_vec.angle_to(top_left_vec - node_vec).radians.abs();
                        if angle > max_angle {
                            max_angle = angle;
                        }
                    }
                }

                let mut distance = Vec::new();

                for (idx, i) in nodes.iter().enumerate() {
                    if let TopotagComponents::Normal { node, .. } = i {
                        let node_vec: Vector2D<_> = node.get_com_f32().into();
                        let angle = top_vec.angle_to(top_left_vec - node_vec).radians.abs();

                        let angle_diff = max_angle - angle;

                        if angle_diff < 0.1 {
                            distance.push(((top_left_vec - node_vec).length(), idx));
                        }
                    }
                }

                distance.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                if distance.len() != class.get_height() - 1 {
                    println!("wrong number of nodes in column");
                    return None;
                }

                let mut column = Vec::new();
                column.push(top_left);
                for (_, idx) in &distance {
                    if let TopotagComponents::Normal { node, .. } = &nodes[*idx] {
                        column.push(node);
                    }
                }

                if column.len() != class.get_height() {
                    println!("wrong height");
                    return None
                }

                let mut data = Vec::new();

                for row_start in column {
                    let mut row = Vec::new();
                    for (idx, i) in nodes.iter().enumerate() {
                        if let TopotagComponents::Normal { node, .. } = i {
                            let node_vec: Vector2D<_> = node.get_com_f32().into();
                            let row_start_vec: Vector2D<_> = row_start.get_com_f32().into();
                            let row_vec: Vector2D<_> = row_start_vec - node_vec;

                            if row_vec.length() < 0.1
                                || top_vec.angle_to(row_vec).radians.abs() < 0.1
                            {
                                row.push((row_vec.length(), idx));
                            }
                        }
                    }

                    row.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    //println!("row: {:?}", row);

                    if row.len() != class.get_width() && row.len() + 2 != class.get_width() {
                        println!("wrong row_width: {}", row.len());
                        return None;
                    }

                    for (_, i) in row {
                        if let TopotagComponents::Normal { node, data: point } = &nodes[i] {
                            node_pos.push(node.get_com_f32());
                            // println!("adding {}th node", i);
                            data.push(*point);
                        } else {
                            return None;
                        }
                    }
                }

                // println!("node_pos.len(): {}", node_pos.len());

                let mut data_int = 0;
                for i in data {
                    data_int <<= 1;
                    if i {
                        data_int |= 1;
                    }
                }

                if node_pos.len() != class.total_node_count() + 1 {
                    return None;
                }

                Some(DecodedTopotag {
                    data: data_int,
                    top_right: top_right_id,
                    nodes: nodes.clone(),
                    top_left: top_left_id,
                    bottom_left: distance.last().unwrap().1,
                    node_pos,
                    class: class.clone()
                })
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedLFTag {
    pub data: usize,
    // pub nodes: Vec<LFTagComponents>,
    pub node_pos: Vec<(f32, f32)>,
    pub expected_node_pos: Vec<(f32, f32)>,
    pub class: LFTagClass,
    pub initial_pose: WorldPose,
    pub initial_poses: Vec<WorldPose>,
    pub final_pose: WorldPose
}

fn get_corner_point(base0_fv: &FeatureVector, base1_fv: &FeatureVector, fvs: &[FeatureVector], class: &LFTagClass) -> Option<(usize, bool, bool)> {
    let base0: Vector2D<f32> = base0_fv.get_com_f32().into();
    let base1: Vector2D<f32> = base1_fv.get_com_f32().into();

    let baseline = base0 - base1;

    let mut possible_nodes: Vec<_> = fvs.iter().enumerate().filter(|(_, fv)| {
        *fv != base1_fv && *fv != base0_fv
    }).map( |(idx, fv)|{
        let point_vec: Vector2D<_> = fv.get_com_f32().into();
        let baseline_to_node = base0 - point_vec;
        let angle = baseline.angle_to(baseline_to_node).radians;
        let distance = baseline_to_node.length();

        (angle, distance, idx)
    }).collect();

    let mut angles: Vec<_> = possible_nodes.iter().map(|(a, _, _)| *a).collect();

    let angle_sum: f32 = possible_nodes.iter().map(|(a, _, _)| a).sum();

    angles.sort_by(|a, b| {
        if angle_sum > 0.0 {
            a.partial_cmp(&b).unwrap()
        } else {
            b.partial_cmp(&a).unwrap()
        }
    });

    let valid = angles.iter().skip(class.get_width() - 2).all(|a| {
        a.is_sign_positive() == angles.last().unwrap().is_sign_positive()
    });

    possible_nodes.sort_by(|a, b| {
        b.0.abs().partial_cmp(&a.0.abs()).unwrap()
    });

    // println!("angle sum {}", angle_sum);
    // dbg!(&angles);

    possible_nodes.truncate(class.get_height() - 1);

    // dbg!(&possible_nodes);

    possible_nodes.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap()
    });

    Some((possible_nodes[0].2, angle_sum > 0.0, valid))
}

fn find_lin_fit_error(input: &Vec<(f32, f32)>) -> f32 {
    let mean_x: f32 = input.iter().map(|(x, _)| x).sum::<f32>() / (input.len() as f32);
    let mean_y: f32 = input.iter().map(|(_, y)| y).sum::<f32>() / (input.len() as f32);

    let sxx: f32 = input.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f32>() / (input.len() as f32 - 1.0);
    let syy: f32 = input.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f32>() / (input.len() as f32 - 1.0);
    let sxy: f32 = input.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum::<f32>() / (input.len() as f32 - 1.0);

    let b1 = (syy - sxx + ((syy - sxx).powi(2) + 4.0 * sxy.powi(2)).powf(0.5)) / (2.0 * sxy);
    let b0 = mean_y - b1 * mean_x;

    input.iter().map(|(x, y)| {
        let c = b0;
        let a = b1;
        let b = -1.0;

        (a * x + b * y + c).abs() / (a.powi(2) + b.powi(2)).sqrt()
    }).sum()
}

impl DecodedLFTag {
    pub fn decode_lftag(root: &LFTagComponents, camera: CameraIntrinsics, gradient: &mut Image<Luma<u16>>) -> Option<DecodedLFTag> {
        if let LFTagComponents::Root { nodes, class, bg } = root {
            let mut fvs: Vec<_> = nodes.iter().map(|n| {
                if let LFTagComponents::Normal {node: fv} = n {
                    fv.clone()
                } else {
                    unreachable!()
                }
            }).collect();

            fvs.sort_by(|a, b| {
                (b.zom * b.area as f32).partial_cmp(&(a.zom * a.area as f32)).unwrap()
            });
            assert!(fvs.len() >= class.total_node_count());

            fvs.truncate(class.total_node_count());
            // println!("fvs: {:#?}", fvs);

            if fvs[0].zom / 5.0 > fvs.last().unwrap().zom {
                // println!("rejected due to area constraint");
                return None
            }


            let (a, zero_first, valid_a) = get_corner_point(&fvs[0], &fvs[1], &fvs, class)?;
            let (b, one_first, valid_b) = get_corner_point(&fvs[1], &fvs[0], &fvs, class)?;

            if !valid_a || !valid_b {
                // println!("rejected due to geometry constraint");
                return None
            }

            if zero_first == one_first {
                // println!("rejected due to angle constraint");
                return None
            }

            let centroids: Vec<_> = fvs.iter().map(|fv| {
                fv.get_com_f32()
            }).collect();

            if find_lin_fit_error(&centroids) < 20.0 {
                // println!("rejected due to collinearity constraint");
                return None
            }

            // dbg!(find_lin_fit_error(&centroids));

            // assert_ne!(zero_first, one_first);

            let tl: Vector2D<f32>;
            let tr: Vector2D<f32>;
            let bl: Vector2D<f32>;
            let br: Vector2D<f32>;

            if zero_first {
                tl = fvs[0].get_com_f32().into();
                tr = fvs[1].get_com_f32().into();

                bl = fvs[a].get_com_f32().into();
                br = fvs[b].get_com_f32().into();
            } else {
                tl = fvs[1].get_com_f32().into();
                tr = fvs[0].get_com_f32().into();

                bl = fvs[b].get_com_f32().into();
                br = fvs[a].get_com_f32().into();
            }

            let cam_points: Vec<(f32, f32)> = vec![tl.into(), tr.into(), bl.into(), br.into()];
            let all_cam_points: Vec<Point2::<f32>> = fvs[2..].iter().map(|fv| {
                let point = fv.get_com_f32();
                Point2::new(point.0, point.1)
            }).collect();

            let node_pos = class.get_keypoint_pos();
            let potential_node_pos = class.get_data_pos();

            let mut best_data = 0;
            let mut best_res = f32::MAX;
            let mut best_loc = None;
            let mut best_idx = 0;
            let mut best_data_pts = Vec::new();
            let mut best_data_pts_world = Vec::new();

            let mut i = 0;
            let mut poses = Vec::new();

            for (e_tl, e_tr, e_bl, e_br) in iproduct!(&node_pos[0], &node_pos[1], &node_pos[2], &node_pos[3]) {
                let mut data_pts = Vec::new();
                let mut data_pts_world = Vec::new();

                let expected_points: Vec<(f32, f32)> = vec![e_tl.clone(), e_tr.clone(), e_bl.clone(), e_br.clone()];
                // dbg!(&expected_points);

                let initial_pose = opencv_localize(&camera, &expected_points, &cam_points);
                poses.push(initial_pose.clone());

                let transformed: Vec<[Point2<f32>; 4]> = potential_node_pos.iter().map(|positions| {
                    let mut output = [Point2::<f32>::new(0.0, 0.0); 4];

                    for i in 0..4 {
                        let transformed_point = world_to_camera(&camera, &initial_pose, &WorldPoint(Point3::new(positions[i].0 as f64, positions[i].1 as f64, 0.0)));
                        output[i] = nalgebra::convert(Point2::new(transformed_point.0, transformed_point.1))
                    }
                    output
                }).collect();

                let mut data = 0;
                let mut total_bit_dist = 0.0;
                for (o_idx, potential_nodes) in transformed.iter().enumerate().rev() {
                    let mut min_bit_dist = f32::MAX;
                    let mut bit = 0;
                    let mut best_point = None;

                    for (idx, potential_node) in potential_nodes.iter().enumerate() {
                        let mut best_point_inner = None;
                        let mut min_dist = f32::MAX;
                        for data_bit in &all_cam_points {
                            let dist = nalgebra::distance(&data_bit, potential_node);

                            if dist < min_dist {
                                min_dist = dist;
                                best_point_inner = Some((data_bit.clone(), potential_node_pos[o_idx][idx].clone()))
                            }
                        }
                        if min_dist < min_bit_dist {
                            min_bit_dist = min_dist;
                            bit = idx;
                            best_point = best_point_inner.clone();
                        }
                    }

                    data <<= 2;
                    data |= bit;
                    total_bit_dist += min_bit_dist.powi(2);

                    data_pts_world.push(best_point?);
                    data_pts.push((potential_nodes[bit].coords[0], potential_nodes[bit].coords[1]));
                }

                let grad_sum = RefCell::new(0.0f32);
                let weight_total = RefCell::new(0.0f32);
                let sum_grad = |a: Luma<u16>, b: Luma<u16>, w| {
                    assert_eq!(a.data[0], 0);
                    weight_total.replace_with(|o| *o + w);
                    grad_sum.replace_with(|o| *o + b.data[0] as f32 * w);
                    b
                };

                let inner_edge = (class.get_width() as f64 + 1.0) * 6.0 - 2.0;
                // dbg!(inner_edge);

                let orig = world_to_camera_i32(&camera, &initial_pose, &WorldPoint(Point3::new(2.0, 2.0, 0.0)));
                let x = world_to_camera_i32(&camera, &initial_pose, &WorldPoint(Point3::new(2.0, inner_edge, 0.0)));
                let y = world_to_camera_i32(&camera, &initial_pose, &WorldPoint(Point3::new(inner_edge, 2.0, 0.0)));
                let xy = world_to_camera_i32(&camera, &initial_pose, &WorldPoint(Point3::new(inner_edge, inner_edge, 0.0)));
                draw_antialiased_line_segment_mut(gradient, orig, x, Luma {
                    data: [0],
                }, sum_grad);
                draw_antialiased_line_segment_mut(gradient, xy, x, Luma {
                    data: [0],
                }, sum_grad);
                draw_antialiased_line_segment_mut(gradient, orig, y, Luma {
                    data: [0],
                }, sum_grad);
                draw_antialiased_line_segment_mut(gradient, xy, y, Luma {
                    data: [0],
                }, sum_grad);

                let mean_grad = *grad_sum.borrow() / *weight_total.borrow();
                total_bit_dist /= mean_grad;

                if total_bit_dist < best_res {
                    best_res = total_bit_dist;
                    best_data = data;
                    best_loc = Some(initial_pose);
                    best_idx = i;
                    best_data_pts_world = data_pts_world;
                    best_data_pts = data_pts
                }
                i += 1;
                // println!("data: {}, res: {}", data, total_bit_dist);
                // dbg!(mean_grad*total_bit_dist);
            }

            let quality_metric = best_res / (bg.area as f32).sqrt();

            // println!("area: {}", (bg.area as f32).sqrt() / 3.0);
            match class {
                LFTagClass::LFTag3x3 => {
                    if quality_metric.abs() > 5e-4 {
                        // println!("rejected due to residual constraint 3x3");
                        return None
                    }
                },
                LFTagClass::LFTag4x4 => {
                    if quality_metric.abs() > 1e-3 {
                        // println!("rejected due to residual constraint 4x4");
                        return None
                    }
                },
                _ => {}
            }


            // dbg!(quality_metric);

            let mut final_camera_pts: Vec<_> = best_data_pts_world.iter().map(|x| (x.0[0], x.0[1])).collect();
            let mut final_world_pts: Vec<_> = best_data_pts_world.iter().map(|x| x.1).collect();

            final_camera_pts.push((tl.x, tl.y));
            final_camera_pts.push((tr.x, tr.y));

            final_world_pts.push(node_pos[0][0]);
            final_world_pts.push(node_pos[1][0]);

            let final_pose = opencv_localize(&camera, &final_world_pts, &final_camera_pts);

            // println!("best_idx: {} ,best_res: {}", best_idx, best_res);

            // let mut cam_points: Vec<(f32, f32)> = vec![base_tl.into(), base_tr.into(), fvs[cal_node_id].get_com_f32()];

            let actual_points: Vec<(f32, f32)> = fvs.iter().map(|fv| {
                fv.get_com_f32()
            }).collect();

            Some(DecodedLFTag{
                initial_pose: best_loc.unwrap(),
                final_pose,
                initial_poses: poses,
                data: best_data as usize,
                node_pos: final_camera_pts,
                expected_node_pos: best_data_pts,
                class: class.clone()
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LFTagClass {
    LFTag2x2,
    LFTag3x3,
    LFTag4x4,
    LFTag5x5,
}


impl LFTagClass {
    fn data_node_count(&self) -> usize {
        match self {
            LFTagClass::LFTag2x2 => 2,
            LFTagClass::LFTag3x3 => 7,
            LFTagClass::LFTag4x4 => 14,
            LFTagClass::LFTag5x5 => 23,
        }
    }

    fn total_node_count(&self) -> usize {
        self.data_node_count() + 2
    }

    fn get_height(&self) -> usize {
        match self {
            LFTagClass::LFTag2x2 => 2,
            LFTagClass::LFTag3x3 => 3,
            LFTagClass::LFTag4x4 => 4,
            LFTagClass::LFTag5x5 => 5,
        }
    }

    pub(crate) fn get_width(&self) -> usize {
        self.get_height()
    }

    fn get_pix_size(&self) -> f32 {
        ((self.get_height() + 1) * 6) as f32
    }

    fn get_keypoint_pos(&self) -> Vec<Vec<(f32, f32)>> {
        let pix = self.get_pix_size();
        // baseline
        let mut ret = vec![vec![(6.0, 6.0)], vec![(pix - 6.0, 6.0)]];

        // bottom left calibration dot
        ret.push(vec![
            (6.0 - 0.5, pix - 6.0 - 0.5),
            (6.0 + 0.5, pix - 6.0 - 0.5),
            (6.0 - 0.5, pix - 6.0 + 0.5),
            (6.0 + 0.5, pix - 6.0 + 0.5)]);

        // bottom right calibration dot
        ret.push(vec![
            (pix - 6.0 - 0.5, pix - 6.0 - 0.5),
            (pix - 6.0 + 0.5, pix - 6.0 - 0.5),
            (pix - 6.0 - 0.5, pix - 6.0 + 0.5),
            (pix - 6.0 + 0.5, pix - 6.0 + 0.5)]);

        ret
    }

    fn get_data_pos(&self) -> Vec<[(f32, f32);4]> {
        let mut ret = Vec::new();

        for j in 0..self.get_width() {
            for i in 0..self.get_height() {
                if (i, j) != (0, 0) && (i, j) != (self.get_width() - 1, 0) {
                    ret.push([
                        (i as f32 * 6.0 + 6.0 - 0.5, j as f32 * 6.0 + 6.0 - 0.5),
                        (i as f32 * 6.0 + 6.0 - 0.5, j as f32 * 6.0 + 6.0 + 0.5),
                        (i as f32 * 6.0 + 6.0 + 0.5, j as f32 * 6.0 + 6.0 - 0.5),
                        (i as f32 * 6.0 + 6.0 + 0.5, j as f32 * 6.0 + 6.0 + 0.5),
                    ])
                }
            }
        };
        ret
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum TopotagClass {
    Topotag3x3,
    Topotag4x4,
    Topotag5x5,
}

impl TopotagClass {
    fn normal_node_count(&self) -> usize {
        match self {
            TopotagClass::Topotag3x3 => 7,
            TopotagClass::Topotag4x4 => 14,
            TopotagClass::Topotag5x5 => 23,
        }
    }

    fn total_node_count(&self) -> usize {
        self.normal_node_count() + 1
    }

    fn get_height(&self) -> usize {
        match self {
            TopotagClass::Topotag3x3 => 3,
            TopotagClass::Topotag4x4 => 4,
            TopotagClass::Topotag5x5 => 5,
        }
    }

    fn get_width(&self) -> usize {
        match self {
            TopotagClass::Topotag3x3 => 3,
            TopotagClass::Topotag4x4 => 4,
            TopotagClass::Topotag5x5 => 5,
        }
    }

    pub fn get_expected_node_pos(&self) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for i in 0..self.get_height() {
            for j in 0..self.get_width() {
                if i == 0 && j == 1 {
                    out.push((1.5, 1.0))
                } else {
                    out.push((j as f32 + 1.0, i as f32 + 1.0))
                }
            }
        }
        out
    }
}

pub fn detect_topotag(
    topo: &Vec<FeatureVector>,
    start: u32,
    output: &mut HashMap<[(u32, u32); 2], TopotagComponents>,
    class: &Vec<TopotagClass>,
) -> TopotagComponents {
    let mut child = topo[start as usize].child;
    let current = &topo[start as usize];
    let mut children = Vec::new();
    let mut all_children = Vec::new();

    while let Some(c) = child {
        child = topo[c as usize].sibling;
        if topo[c as usize].area * 30 > current.area {
            children.push(c)
        }
        all_children.push(c)
    }

    if children.len() == 2 && !current.color {
        // possible baseline
        let mut has_grandchildren = false;
        for &c in &children {
            if topo[c as usize].child.is_some() {
                has_grandchildren = true;
            }
        }

        let nodes = [
            topo[children[0] as usize].clone(),
            topo[children[1] as usize].clone(),
        ];
        if !has_grandchildren {
            return TopotagComponents::Baseline { nodes };
        }
    }

    if children.len() < 2 && !current.color {
        // possible normal node
        if children.len() == 0 {
            return TopotagComponents::Normal {
                data: false,
                node: current.clone(),
            };
        }

        if children.len() == 1 {
            let mut has_grandchildren = false;
            for &c in &children {
                if topo[c as usize].child.is_some() {
                    has_grandchildren = true;
                }
            }
            if !has_grandchildren {
                return TopotagComponents::Normal {
                    data: true,
                    node: current.clone(),
                };
            }
        }
    }

    for tag_class in class {
        if children.len() == tag_class.total_node_count() && current.color {
            let mut normal_nodes = Vec::new();
            let mut baseline_nodes = Vec::new();
            for &c in &children {
                match detect_topotag(topo, c, output, &class) {
                    normal @ TopotagComponents::Normal { .. } => {
                        normal_nodes.push(normal.clone());
                    }
                    baseline @ TopotagComponents::Baseline { .. } => {
                        baseline_nodes.push(baseline.clone());
                    }
                    _ => (),
                }
            }

            if normal_nodes.len() == tag_class.normal_node_count() && baseline_nodes.len() == 1 {
                let mut nodes = Vec::new();
                nodes.push(baseline_nodes[0].clone());
                nodes.append(&mut normal_nodes);

                let root = TopotagComponents::Root {
                    nodes,
                    class: (*tag_class).clone(),
                };
                output.insert(current.bounding_box, root.clone());
                return root;
            }
        }
    }

    for &c in &all_children {
        detect_topotag(topo, c, output, &class);
    }

    return TopotagComponents::None;
}

pub fn detect_lftag(
    topo: &Vec<FeatureVector>,
    start: u32,
    output: &mut HashMap<[(u32, u32); 2], LFTagComponents>,
    class: &Vec<LFTagClass>,
) -> LFTagComponents {
    let mut child = topo[start as usize].child;
    let current = &topo[start as usize];
    let mut children = Vec::new();
    let mut node_children = Vec::new();
    let mut all_children = Vec::new();

    while let Some(c) = child {
        child = topo[c as usize].sibling;
        if topo[c as usize].area * 2 > current.area {
            children.push(c)
        }
        if topo[c as usize].area * 500 > current.area {
            node_children.push(c)
        }
        all_children.push(c)
    }

    for &c in &all_children {
        detect_lftag(topo, c, output, &class);
    }

    if children.len() == 0 && !current.color {
        return LFTagComponents::Normal {
            node: current.clone(),
        };
    }

    for tag_class in class {
        if node_children.len() >= tag_class.total_node_count() && current.color {
            let mut nodes = Vec::new();
            for &c in &node_children {
                match detect_lftag(topo, c, output, &class) {
                    n @ LFTagComponents::Normal{..} => {
                        nodes.push(n)
                    }
                    _ => {}
                }
            }

            if nodes.len() >= tag_class.total_node_count() && nodes.len() <= tag_class.total_node_count() + 3 {
                let root = LFTagComponents::Root {
                    nodes,
                    bg: current.clone(),
                    class: (*tag_class).clone(),
                };
                output.insert(current.bounding_box, root.clone());
                return root;
            }
        }
    }

    return LFTagComponents::None;
}