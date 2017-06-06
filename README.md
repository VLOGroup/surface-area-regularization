# surface-area-regularization
This repository contains a python implementation of our method "Efficient Minimal-Surface Regularization of Perspective Depth Maps in Variational Stereo", CVPR 2015.

If you use this code, please cite the following publication:

```
@inproceedings{graber_cvpr2015,
author = {Gottfried Graber and Jonathan Balzer and Stefano Soatto and Thomas Pock},
title = {{Efficient Minimal-Surface Regularization of Perspective Depth Maps in Variational Stereo}},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
year = {2015}
}
```

The code has been written and tested under Linux using
python2.7.

Usage:
------
`$ python flowdepth.py`

Sample input data is provided, feel free to try it with your own data! The
framework expects a dictionary which contains the following:
- a key 'images', value: a list of 2 numpy-images, first one is the reference image
- a key 'G', value: list with the camera transformation matrices for the images
- a key 'K', value: intrinsic calibration matrix

See flowdepth.py line 89 how the dictionary is constructed.
