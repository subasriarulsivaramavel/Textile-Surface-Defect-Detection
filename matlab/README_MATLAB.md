# Textile Surface Defect Detection - MATLAB

## Requirements
- MATLAB R2020a or newer
- Image Processing Toolbox
- (Optional) Computer Vision Toolbox for `insertShape` annotated images

## Quick start
```matlab
cd('path/to/cv project/matlab')
generateSamples('../samples')   % optional
runDemo('../samples/sample_1.png')
runBatch('../samples', '../outputs')
```

## Main functions
- `detectDefects(src, sensitivity, minArea)` - returns original, mask, boxes, metrics.
- `runDemo(imagePath)` - displays a 6-panel result figure.
- `runBatch(folder, outFolder, sensitivity, minArea)` - processes a folder and writes CSV.
- `generateSamples(outFolder)` - creates synthetic fabric images.

## Notes
This MATLAB version mirrors the Python/OpenCV pipeline:
grayscale/blurring, background estimation via median filter, thresholding,
LAB color anomaly detection, morphology cleanup, and region extraction.
