function result = detectDefects(src, sensitivity, minArea)
%DETECTDEFECTS Classical OpenCV-style defect detection implemented in MATLAB.
%
%   result = detectDefects(src)
%   result = detectDefects(src, sensitivity, minArea)
%
%   src can be an image array or a path to an image file.
%   Requires the Image Processing Toolbox.

    if nargin < 2 || isempty(sensitivity)
        sensitivity = 1.2;
    end
    if nargin < 3 || isempty(minArea)
        minArea = 80;
    end

    if ischar(src) || isstring(src)
        I = imread(char(src));
    else
        I = src;
    end

    if size(I,3) == 1
        I = cat(3, I, I, I);
    elseif size(I,3) == 4
        I = I(:,:,1:3);
    end

    gray = rgb2gray(I);
    blur = imgaussfilt(gray, 1.0);

    % Estimate fabric background with a large median filter.
    bg = medfilt2(blur, [51 51]);

    % Intensity deviation from the background.
    diff = imabsdiff(blur, bg);
    thr = max(8, round(40 / max(0.3, sensitivity)));
    mask = diff > thr;

    % Color anomaly detection in LAB space.
    lab = rgb2lab(I);
    L = lab(:,:,1);
    bgL = medfilt2(imgaussfilt(L, 1.0), [51 51]);
    diffL = abs(L - bgL);
    thrL = max(6, round(30 / max(0.3, sensitivity)));
    maskL = diffL > thrL;
    mask = mask | maskL;

    % Morphological cleanup.
    mask = imclose(mask, strel('disk', 2));
    mask = imopen(mask, strel('disk', 1));

    % Extract defect regions.
    cc = bwconncomp(mask);
    stats = regionprops(cc, 'Area', 'BoundingBox');
    boxes = [];
    for k = 1:numel(stats)
        a = stats(k).Area;
        if a < minArea
            continue;
        end
        if a > 0.6 * numel(mask)
            continue;
        end
        boxes = [boxes; stats(k).BoundingBox]; %#ok<AGROW>
    end

    defectArea = sum(mask(:));
    totalPixels = numel(mask);
    defectPercent = 100 * defectArea / totalPixels;

    if isempty(boxes)
        severity = 'None';
    elseif defectPercent < 1.0 && size(boxes,1) <= 3
        severity = 'Low';
    elseif defectPercent < 5.0
        severity = 'Medium';
    else
        severity = 'High';
    end

    annotated = annotateImage(I, boxes);

    result = struct();
    result.original = I;
    result.gray = gray;
    result.background = bg;
    result.diff = diff;
    result.mask = mask;
    result.annotated = annotated;
    result.boxes = boxes;
    result.metrics = struct( ...
        'num_defects', size(boxes,1), ...
        'defect_area_px', defectArea, ...
        'total_pixels', totalPixels, ...
        'defect_percent', round(defectPercent, 3), ...
        'severity', severity, ...
        'image_size', sprintf('%dx%d', size(I,2), size(I,1)));
end

function A = annotateImage(I, boxes)
    A = I;
    if ~isempty(boxes) && exist('insertShape', 'file')
        A = insertShape(A, 'Rectangle', boxes, 'Color', 'yellow', 'LineWidth', 3);
    end
end