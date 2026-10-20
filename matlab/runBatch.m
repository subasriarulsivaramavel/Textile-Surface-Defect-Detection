function runBatch(inputFolder, outputFolder, sensitivity, minArea)
%RUNBATCH Process all images in a folder and save annotated results.

    if nargin < 1 || isempty(inputFolder)
        inputFolder = 'samples';
    end
    if nargin < 2 || isempty(outputFolder)
        outputFolder = 'outputs';
    end
    if nargin < 3 || isempty(sensitivity)
        sensitivity = 1.2;
    end
    if nargin < 4 || isempty(minArea)
        minArea = 80;
    end

    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'};
    files = [];
    for e = 1:numel(exts)
        files = [files; dir(fullfile(inputFolder, ['*' exts{e}]))]; %#ok<AGROW>
    end

    rows = [];
    for k = 1:numel(files)
        imgPath = fullfile(files(k).folder, files(k).name);
        I = imread(imgPath);
        res = detectDefects(I, sensitivity, minArea);
        [~, name, ~] = fileparts(files(k).name);
        imwrite(res.annotated, fullfile(outputFolder, [name '_detected.png']));

        rows = [rows; {files(k).name, res.metrics.num_defects, ...
            res.metrics.defect_area_px, res.metrics.defect_percent, ...
            res.metrics.severity}]; %#ok<AGROW>
        fprintf('%s: %d defects, %.3f%% area, %s\n', files(k).name, ...
            res.metrics.num_defects, res.metrics.defect_percent, res.metrics.severity);
    end

    T = cell2table(rows, 'VariableNames', ...
        {'file', 'num_defects', 'defect_area_px', 'defect_percent', 'severity'});
    writetable(T, fullfile(outputFolder, 'report.csv'));
    disp('Batch report saved.');
end