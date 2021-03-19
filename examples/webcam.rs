extern crate eye;
extern crate minifb;

use std::{io, time};
use image::{ImageBuffer};
use nalgebra::{Vector2, Point2};
use cv_pinhole::CameraIntrinsics;
use eye::prelude::*;
extern crate fiducial;
use fiducial::debug::find_lftags_debug;
use minifb::{Window, WindowOptions};

fn main() -> io::Result<()> {
    let devices = Context::enumerate_devices();
    if devices.is_empty() {
        std::process::exit(1);
    }
    let dev = Device::with_uri(&devices[0]).unwrap();
    let desc = dev.preferred_stream(&|x, y| {
        if x.pixfmt == PixelFormat::Rgb(24) && y.pixfmt == PixelFormat::Rgb(24) {
            if x.interval > time::Duration::from_millis(33) {
                return y;
            }
            if x.width > y.width {
                x
            } else {
                y
            }
        } else if x.pixfmt == PixelFormat::Rgb(24) {
            x
        } else {
            y
        }
    })?;

    if desc.pixfmt != PixelFormat::Rgb(24) {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to set RGB format."
        ));
    } 

    println!("Stream: {:?}", desc);
    let mut stream = dev.start_stream(&desc)?;

    let mut window = Window::new(
        &devices[0],
        desc.width as usize,
        desc.height as usize,
        WindowOptions::default()
    ).expect("Could not open a window.");

    loop {
        let frame = stream.next().expect("Stream is dead.")?;
        
        let buf = frame.into_bytes().collect();
        let image : ImageBuffer<image::Rgb<u8>, Vec<u8>> = image::ImageBuffer::from_vec(
            desc.width, desc.height, buf).unwrap();

        let detected = find_lftags_debug(&image, CameraIntrinsics {
            focals: Vector2::new(712.44128286, 711.06570126),
            principal_point: Point2::new(316.80287675, 228.46397532),
            skew: 0.0
        }).unwrap();

        let pixels: Vec<u32> = detected.pixels().map(|rgb| {
            u32::from_be_bytes([0, rgb[0], rgb[1], rgb[2]])
        }).collect();
        
        window.update_with_buffer(
            &pixels,
            desc.width as usize, desc.height as usize).unwrap();
    }
}