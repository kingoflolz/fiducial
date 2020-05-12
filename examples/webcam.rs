extern crate piston_window;
extern crate camera_capture;

use image::ConvertBuffer;
use piston_window::{clear, PistonWindow, Texture, TextureSettings, WindowSettings};
use nalgebra::{Vector2, Point2};
use cv_pinhole::CameraIntrinsics;

extern crate topotag;
use topotag::debug::find_tags_debug;

fn main() {
    let mut window: PistonWindow = WindowSettings::new("topotagtest", [640, 480])
        .exit_on_esc(true)
        .build()
        .unwrap();
    let mut tex: Option<Texture<_>> = None;
    let (sender, receiver) = std::sync::mpsc::channel();
    let imgthread = std::thread::spawn(move || {
        let res1 = camera_capture::create(0);
        if let Err(e) = res1 {
            eprintln!("could not open camera: {}", e);
            std::process::exit(1);
        }
        let res2 = res1.unwrap().fps(30.0).unwrap().start();
        if let Err(e) = res2 {
            eprintln!("could retrieve data from camera: {}", e);
            std::process::exit(2);
        }
        let cam = res2.unwrap();
        // camera matrix from camera_cal
        // [[712.44128286   0.         316.80287675]
        //  [  0.         711.06570126 228.46397532]
        //  [  0.           0.           1.        ]]


        for frame in cam {
            if sender.send(find_tags_debug(frame.convert(), CameraIntrinsics {
                focals: Vector2::new(712.44128286, 711.06570126),
                principal_point: Point2::new(316.80287675, 228.46397532),
                skew: 0.0
            }).convert()).is_err() {
                break;
            }
        }
    });
    while let Some(e) = window.next() {
        if let Ok(frame) = receiver.try_recv() {
            if let Some(mut t) = tex {
                t.update(&mut window.encoder, &frame).unwrap();
                tex = Some(t);
            } else {
                tex =
                    Texture::from_image(&mut window.factory, &frame, &TextureSettings::new()).ok();
            }
        }
        window.draw_2d(&e, |c, g| {
            clear([1.0; 4], g);
            if let Some(ref t) = tex {
                piston_window::image(t, c.transform, g);
            }
        });
    }
    drop(receiver);
    imgthread.join().unwrap();
}