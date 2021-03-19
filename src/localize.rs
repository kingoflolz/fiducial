use crate::decode::DecodedTopotag;
use cv_core::{CameraModel, FeatureWorldMatch, KeyPoint, WorldPoint, sample_consensus::Consensus, WorldPose};
use cv_pinhole::{CameraIntrinsics, NormalizedKeyPoint};
use lambda_twist::LambdaTwist;
use arrsac::{Arrsac, Config};
use rand::{rngs::SmallRng, SeedableRng};
use image::{ImageBuffer, Rgba};
use imageproc::drawing::{draw_line_segment_mut, draw_antialiased_line_segment_mut};
use nalgebra::{Point2, Point3, Isometry3, IsometryMatrix3, Translation, Vector3, Matrix3, Rotation3};
use opencv::core::{Vector, CV_64F};
use opencv::prelude::*;
use imageproc::pixelops::interpolate;

pub fn opencv_localize(model: &CameraIntrinsics, world: &Vec<(f32, f32)>, camera: &Vec<(f32, f32)>) -> WorldPose {
    let image_points_vector = opencv::core::Vector::from_iter(camera
        .iter()
        .map(|(x, y)| opencv::core::Point2d::new(*x as f64, *y as f64)));

    let image_points: opencv::types::VectorOfPoint2d = image_points_vector.into();

    let world_points_vector = opencv::core::Vector::from_iter(world
        .iter()
        .map(|(x, y)| opencv::core::Point3d::new(*x as f64, *y as f64, 0.0)));

    let worldpoints: opencv::types::VectorOfPoint3d = world_points_vector.into();

    let mut camera_matrix = vec![vec![0.0; 3]; 3];

    for i in 0..3 {
        for j in 0..3 {
            camera_matrix[i][j] = *model.matrix().get((i, j)).unwrap();
        }
    }

    let mat = Mat::from_slice_2d(&camera_matrix).unwrap();

    let mut trans = Mat::from_slice(&vec![0.0, 0.0, 0.0]).unwrap();
    let mut rot = Mat::from_slice(&vec![0.0, 0.0, 0.0]).unwrap();
    let mut rot_mat = Mat::from_slice_2d(&camera_matrix).unwrap();

    opencv::calib3d::solve_pnp(&worldpoints, &image_points, &mat, &Mat::default().unwrap(), &mut rot, &mut trans, false, opencv::calib3d::SOLVEPNP_IPPE).unwrap();
    opencv::calib3d::rodrigues(&rot, &mut rot_mat, &mut Mat::default().unwrap()).unwrap();

    let trans_vec: Vector3<f64>;

    // I have no idea wtf this is for
    if trans.at_2d::<f64>(2, 0).is_ok() {
        trans_vec = Vector3::new(*trans.at_2d(0, 0).unwrap(),
                                 *trans.at_2d(1, 0).unwrap(),
                                 *trans.at_2d(2, 0).unwrap());
    } else {
        trans_vec = Vector3::new(*trans.at_2d(0, 0).unwrap(),
                                 *trans.at_2d(0, 1).unwrap(),
                                 *trans.at_2d(0, 2).unwrap());
    }


    let mut rot_mat_nalg: Matrix3<f64> = Matrix3::zeros();

    for i in 0..3 {
        for j in 0..3 {
            rot_mat_nalg[(i, j)] = *rot_mat.at_2d(i as i32, j as i32).unwrap();
        }
    }

    WorldPose(IsometryMatrix3::<f64> {
        rotation: Rotation3::from_matrix_unchecked(rot_mat_nalg),
        translation: Translation::from_vector(trans_vec)
    })
}

pub fn localize(model: &CameraIntrinsics, tag: &DecodedTopotag) -> WorldPose {
    let normalized_image_coordinates: Vec<NormalizedKeyPoint> = tag
        .node_pos
        .iter()
        .map(|(x, y)| model.calibrate(KeyPoint(Point2::new(*x as f64, *y as f64))))
        .collect();
    // println!("normalized_image_coordinates: {:?}", normalized_image_coordinates);
    // println!("normalized_image_coordinates.len(): {}", normalized_image_coordinates.len());

    let world_points: Vec<_> = tag.class
        .get_expected_node_pos()
        .iter()
        .map(|(x, y)| WorldPoint(Point3::new(*x as f64, *y as f64, 0.0)))
        .collect();
    // println!("world_points: {:?}", world_points);

    let samples: Vec<FeatureWorldMatch<NormalizedKeyPoint>> = world_points
        .iter()
        .zip(normalized_image_coordinates)
        .map(|(&world, image)| FeatureWorldMatch(image.into(), world.into()))
        .collect();
    println!("samples: {:?}", samples);

    let mut arrsac = Arrsac::new(Config::new(0.0001), SmallRng::from_seed([0; 16]));

    arrsac
        .model(&LambdaTwist::new(), samples.iter().cloned())
        .unwrap()
}

pub fn draw_coords(input: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, model: &CameraIntrinsics, pose: &WorldPose, scale: f64) {
    let origin = world_to_camera_i32(model, pose, &WorldPoint(Point3::new(2.0, 2.0, 0.0)));
    let x = world_to_camera_i32(model, pose, &WorldPoint(Point3::new(scale - 2.0, 2.0, 0.0)));
    let y = world_to_camera_i32(model, pose, &WorldPoint(Point3::new(2.0, scale - 2.0, 0.0)));
    let z = world_to_camera_i32(model, pose, &WorldPoint(Point3::new(2.0, 2.0, -scale)));
    let xy = world_to_camera_i32(model, pose, &WorldPoint(Point3::new(scale - 2.0, scale - 2.0, 0.0)));
    draw_antialiased_line_segment_mut(input, origin, x, Rgba([255, 0, 0, 255]), interpolate);
    draw_antialiased_line_segment_mut(input, origin, y, Rgba([0, 255, 0, 255]), interpolate);
    draw_antialiased_line_segment_mut(input, xy, x, Rgba([255, 0, 0, 255]), interpolate);
    draw_antialiased_line_segment_mut(input, xy, y, Rgba([0, 255, 0, 255]), interpolate);
    draw_antialiased_line_segment_mut(input, origin, z, Rgba([0, 0, 255, 255]), interpolate);
}

pub fn world_to_camera(model: &CameraIntrinsics, pose: &WorldPose, point: &WorldPoint) -> (f32, f32) {
    let camera_point = pose.transform(*point);
    let key_point = model.uncalibrate(camera_point.into());

    (key_point.0.coords[0] as f32, key_point.0.coords[1] as f32)
}

pub fn world_to_camera_i32(model: &CameraIntrinsics, pose: &WorldPose, point: &WorldPoint) -> (i32, i32) {
    let camera_point = pose.transform(*point);
    let key_point = model.uncalibrate(camera_point.into());

    (key_point.0.coords[0].round() as i32, key_point.0.coords[1].round() as i32)
}