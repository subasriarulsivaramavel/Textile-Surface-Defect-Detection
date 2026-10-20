function runDemo(imagePath)
%RUNDEMO Run defect detection and display the result.
%
%   runDemo()
%   runDemo(imagePath)

    if nargin == 0
        [fileName, pathName] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', ...
            'Image Files'}, 'Select a fabric image');
        if isequal(fileName, 0)
            disp('No file selected.');
            return;
        end
        imagePath = fullfile(pathName, fileName);
    end

    I = imread(imagePath);
    result = detectDefects(I, 1.2, 80);

    figure('Name', 'Textile Surface Defect Detection', 'Color', 'w');
    subplot(2,3,1); imshow(result.original); title('Original');
    subplot(2,3,2); imshow(result.gray); title('Grayscale');
    subplot(2,3,3); imshow(result.diff); title('Deviation map');
    subplot(2,3,4); imshow(result.mask); title('Defect mask');
    subplot(2,3,5); imshow(result.annotated); title('Detected defects');

    ax = subplot(2,3,6);
    imshow(result.original); hold(ax, 'on');
    for i = 1:size(result.boxes,1)
        b = result.boxes(i,:);
        rectangle('Position', b, 'EdgeColor', 'y', 'LineWidth', 2);
        text(b(1), max(1, b(2)-5), sprintf('D%d', i), 'Color', 'y', 'FontWeight', 'bold');
    end
    hold(ax, 'off');
    title('Overlay');

    sgtitle(sprintf('Defects: %d | Area: %.3f%% | Severity: %s', ...
        result.metrics.num_defects, result.metrics.defect_percent, result.metrics.severity));

    disp(result.metrics);
end