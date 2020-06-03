#![feature(test)]
extern crate piston_window;
extern crate camera_capture;
extern crate rayon;

use image::{ConvertBuffer, open, DynamicImage, FilterType};
use nalgebra::{Vector2, Point2};
use cv_pinhole::CameraIntrinsics;
use rayon::prelude::*;

extern crate fiducial;
use fiducial::debug::{find_lftags_debug};
use std::path::Path;
use std::env;

extern crate test;
use test::Bencher;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();

    args.par_iter().enumerate().map(|(idx, filename)| {
        if idx % 100 == 0 {
            dbg!(idx);
        }
        let now = Instant::now();
        println!("opening {}", filename);

        if let Ok(im) = image::open(&Path::new(&filename)) {
            let im = im.resize_exact(640, 480, FilterType::Triangle);
            let o = find_lftags_debug(&im.to_rgb(), CameraIntrinsics {
                focals: Vector2::new(320.0, 320.0),
                principal_point: Point2::new(320.0, 240.0),
                skew: 0.0
            });
            println!("done in {} us {}", now.elapsed().as_micros(), &filename);
            // o.unwrap().save(&Path::new(&filename.replace(".jpg", "_dbg.png"))).unwrap()
        }
    }).count();
}

#[bench]
fn bench_find_lftags(b: &mut Bencher) {
    let im = image::open(&Path::new("examples/test12.png")).unwrap().to_rgb();
    b.iter(|| find_lftags_debug(&im, CameraIntrinsics {
        focals: Vector2::new(320.0, 320.0),
        principal_point: Point2::new(320.0, 240.0),
        skew: 0.0
    }));
}