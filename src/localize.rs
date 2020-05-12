use crate::decode::DecodedTag;
use cv_core::{CameraModel, FeatureWorldMatch, KeyPoint, WorldPoint, sample_consensus::Consensus, WorldPose};
use cv_pinhole::{CameraIntrinsics, NormalizedKeyPoint};
use nalgebra::{Point2, Point3};
use lambda_twist::LambdaTwist;
use arrsac::{Arrsac, Config};
use rand::{rngs::SmallRng, SeedableRng};
use image::{ImageBuffer, Rgba};
use imageproc::drawing::{draw_line_segment_mut};

pub fn localize(model: &CameraIntrinsics, tag: &DecodedTag) -> WorldPose {
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

pub fn draw_coords(input: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, model: &CameraIntrinsics, pose: &WorldPose) {
    let origin = world_to_camera(model, pose, &WorldPoint(Point3::new(0.0, 0.0, 0.0)));
    let x = world_to_camera(model, pose, &WorldPoint(Point3::new(3.0, 0.0, 0.0)));
    let y = world_to_camera(model, pose, &WorldPoint(Point3::new(0.0, 3.0, 0.0)));
    let z = world_to_camera(model, pose, &WorldPoint(Point3::new(0.0, 0.0, -3.0)));

    draw_line_segment_mut(input, origin, x, Rgba {
        data: [255, 0, 0, 255],
    });
    draw_line_segment_mut(input, origin, y, Rgba {
        data: [0, 255, 0, 255],
    });
    draw_line_segment_mut(input, origin, z, Rgba {
        data: [0, 0, 255, 255],
    });
}

pub fn world_to_camera(model: &CameraIntrinsics, pose: &WorldPose, point: &WorldPoint) -> (f32, f32) {
    let camera_point = pose.transform(*point);
    let key_point = model.uncalibrate(camera_point.into());

    (key_point.0.coords[0] as f32, key_point.0.coords[1] as f32)
}