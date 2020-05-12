# Topotag

Implementation of the [Topotag](https://arxiv.org/abs/1908.01450) algorithm in rust.

## Examples
An example is provided which finds, decodes and localizes topotags in images from the webcam.

The accuracy of the localization is dependent on the specific webcam parameters, which can be calculated with the `collect_img.py` and `camera_cal.py` in the `calibration` folder.

## Status
This is a quick and dirty implementation, not very well tested, and the localization is somewhat dubious at times. This shouldn't be used for anything meaningful.

Or do, I'm not the boss of you 🤷