function generateSamples(outFolder)
%GENERATESAMPLES Create synthetic fabric images for a quick MATLAB demo.

    if nargin < 1 || isempty(outFolder)
        outFolder = 'samples';
    end
    if ~exist(outFolder, 'dir')
        mkdir(outFolder);
    end

    rng(42);
    tones = [150, 170, 190, 160, 180];
    h = 480; w = 640;
    for t = 1:numel(tones)
        tone = tones(t);
        base = ones(h, w, 3, 'uint8') * tone;
        for y = 1:6:h
            base(y,:,:) = max(0, tone - 15);
        end
        for x = 1:6:w
            base(:,x,:) = min(255, tone + 10);
        end
        base = uint8(double(base) + (randi([-8 8], h, w, 3)));

        img = base;
        % Stains
        for k = 1:randi([1 3])
            cx = randi([60 w-60]); cy = randi([60 h-60]); r = randi([25 55]);
            color = [randi([80 130]), randi([60 110]), randi([90 140])];
            [X, Y] = meshgrid(1:w, 1:h);
            msk = (X-cx).^2 + (Y-cy).^2 <= r^2;
            for c = 1:3
                layer = img(:,:,c);
                layer(msk) = uint8(0.6*double(layer(msk)) + 0.4*double(color(c)));
                img(:,:,c) = layer;
            end
        end
        % Holes
        for k = 1:randi([1 2])
            cx = randi([60 w-60]); cy = randi([60 h-60]); r = randi([8 22]);
            [X, Y] = meshgrid(1:w, 1:h);
            msk = (X-cx).^2 + (Y-cy).^2 <= r^2;
            for c = 1:3
                layer = img(:,:,c);
                layer(msk) = 30;
                img(:,:,c) = layer;
            end
        end
        % Scratches omitted to avoid Computer Vision Toolbox dependency.


        imwrite(img, fullfile(outFolder, sprintf('sample_%d.png', t)));
    end

    clean = ones(h, w, 3, 'uint8') * 175;
    imwrite(clean, fullfile(outFolder, 'clean.png'));
    fprintf('Generated samples in %s\n', outFolder);
end