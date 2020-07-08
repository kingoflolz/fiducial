# Visual Fiducial Implementations (LFTag and TopoTag)

Implementation of two visual fiducial algorithms [LFTag](https://arxiv.org/abs/2006.00842) and [Topotag](https://arxiv.org/abs/1908.01450) algorithm in rust.

## Examples

An example is provided which finds, decodes and localizes LFTags in images from the webcam. (`examples/webcam.rs`)

Another example is provided which finds, decodes and localizes LFTags in a list of images provided through stdin. (`examples/lftag.rs`)

The accuracy of the localization is dependent on the specific camera parameters, which can be calculated with the `collect_img.py` and `camera_cal.py` in the `calibration` folder.

## Tag Generation

Tag generation is also implemented for the LFTag algorithm, with the `LFTag/generate.py` script.

## Status
This is a quick and dirty implementation, not very well tested, and the localization is somewhat dubious at times. This shouldn't be used for anything meaningful.

Or do, I'm not the boss of you 🤷
