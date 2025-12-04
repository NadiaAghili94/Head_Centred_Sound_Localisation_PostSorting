 clc;
 close all;


folderPath ='F:\Spontage_results\Right\Sponge_Right__Group_1\recording_info.csv';
bad_block_path = ' ';

pick_cindx = [28]; % cluster

sampling_rate = 20000;
time_window = [0, 0.25];  % s
deg_map = [150 120 90 60 30 0 -30 -60 -90 -120 -150 -180];

% Smooth the mean and SEM curves (just to have a nice plot)
winSize = 5;      % odd number
sigma = 1;

% Adjustable parameters (for 2*1 subplot figure)
marginLeft = 0.18;
marginRight = 0.05;
marginTop = 0.05;
marginBottom = 0.10;
gapY = 0.10;   % bigger, adjustable gap between rows 

% Load spike & cluster data 
Cinfo = readtable('cluster_info.tsv','FileType','text','Delimiter','\t');
valid_clusters = Cinfo.cluster_id;
chan_of_clust = Cinfo.ch;

spike_times_all = double(readNPY('spike_times.npy'));
spike_clusters_all = double(readNPY('spike_clusters.npy'));
keep_idx = ismember(spike_clusters_all, valid_clusters);
spike_times_all = spike_times_all(keep_idx);
spike_clusters_all = spike_clusters_all(keep_idx);

channel_kept = zeros(size(spike_clusters_all));
for d = 1:length(spike_clusters_all)
    Cl = spike_clusters_all(d);
    CH = find(valid_clusters == Cl);
    channel_kept(d) = chan_of_clust(CH);
end

Sessionss = generateSessionCells(folderPath, bad_block_path);
McStStimOnsetTimes = onset_times(~cellfun('isempty',onset_times));
Behaviors = Behavior_(~cellfun('isempty',Behavior_));

%% main loop
for cl_id = pick_cindx

    trial_spikes = {};
    trial_head = [];
    trial_world = [];
    trial_session = [];
    trial_rate = [];
    trial_csr = [];

    % Collect trials across sessions 

    for s = 1:numel(Sessionss)
        if isempty(Behaviors{s}), continue, end

        beh = Behaviors{s};
        if width(beh) < 15, continue, end  % removing FRA 

        csr = beh.CenterSpoutRotation;
        sloc = beh.Speaker_Location;
        onset = McStStimOnsetTimes{s};
        [t0, t1] = bounds(Sessionss{s});

        if numel(sloc) ~= numel(onset), continue, end
        if any(isnan(sloc)), continue, end

        in_sess = (spike_clusters_all == cl_id) & ...
                  (spike_times_all >= t0) & (spike_times_all <= t1);
        spikes_s = (spike_times_all(in_sess) - t0) / sampling_rate;  % s

        for tr = 1:numel(onset)
            wa = deg_map(sloc(tr));                         % world angle
            ha = mod(wa - csr(tr) + 180, 360) - 180;        % head angle

            if abs(ha) == 180, ha = -180; end
            if abs(wa) == 180, wa = -180; end

            sp = spikes_s(spikes_s >= onset(tr) + time_window(1) & ...
                          spikes_s <= onset(tr) + time_window(2)) - onset(tr);

            trial_spikes{end+1} = sp;
            trial_head(end+1) = ha;
            trial_world(end+1) = wa;
            trial_session(end+1) = s;
            trial_rate(end+1) = numel(sp) / diff(time_window); % spikes/s
            trial_csr(end+1) = csr(tr);
        end
    end

    % Wrap CSR = 180 to -180
    trial_csr(trial_csr == 180) = -180;

    %% Rate map per CSR 
    unique_csrs = unique(trial_csr);
    n = numel(unique_csrs);
    nAngles = numel(deg_map);

    % Store absolute and normalised rate maps
    all_rate_maps_abs = cell(1, n);   % spikes/s
    all_rate_maps_norm = cell(1, n);   % 0-1

    % Per CSR's loop
    for i = 1:n
        csr_val = unique_csrs(i); 
        idx = trial_csr == unique_csrs(i);

        rate_map = nan(nAngles, nAngles);  % head * world
        count_map = zeros(nAngles, nAngles);

        trial_idx = find(idx);
        for t = trial_idx(:)'
            w = find(deg_map == trial_world(t), 1);
            h = find(deg_map == trial_head(t),  1);
            if ~isempty(w) && ~isempty(h)
                % Rows: head angle, Cols: world angle
                rate_map(h, w)  = nansum([rate_map(h, w), 0]) + trial_rate(t);
                count_map(h, w) = count_map(h, w) + 1;
            end
        end

        mask = count_map > 0;
        rate_map(mask) = rate_map(mask) ./ count_map(mask);  % mean spikes/s

        % Absolute map
        rate_map_abs = rate_map;

        % Normalised per CSR
        if all(isnan(rate_map_abs(:))) || max(rate_map_abs(:)) <= 0
            rate_map_norm = rate_map_abs;  % stays NaN
        else
            rate_map_norm = rate_map_abs / max(rate_map_abs(:));
        end

        all_rate_maps_abs{i} = rate_map_abs;
        all_rate_maps_norm{i} = rate_map_norm;
    end

    %%  Combined maps for heat map
    % Combine CSRs by taking max across 3rd dim
    combined_abs = max(cat(3, all_rate_maps_abs{:}),  [], 3);  % spikes/s
    combined_norm = max(cat(3, all_rate_maps_norm{:}), [], 3);  % 0-1

    % Fill NaNs 
    combined_abs = fill_nan_neighbours(combined_abs);
    combined_norm = fill_nan_neighbours(combined_norm);

    %% SRF curves with shaded SEM (from trial rate) 
    SRF_head_mu = nan(nAngles,1);
    SRF_head_sem = nan(nAngles,1);
    SRF_world_mu = nan(nAngles,1);
    SRF_world_sem = nan(nAngles,1);

    for a = 1:nAngles
        % Head-centred SRF (group by head angle)
        idxH = trial_head == deg_map(a);
        rH = trial_rate(idxH);
        nH = sum(~isnan(rH));
        if nH > 0
            SRF_head_mu(a) = mean(rH, 'omitnan');
            if nH > 1
                SRF_head_sem(a) = std(rH, 0, 'omitnan') / sqrt(nH);
            else
                SRF_head_sem(a) = NaN;
            end
        end

        % World-centred SRF (group by world angle)
        idxW = trial_world == deg_map(a);
        rW = trial_rate(idxW);
        nW = sum(~isnan(rW));
        if nW > 0
            SRF_world_mu(a) = mean(rW, 'omitnan');
            if nW > 1
                SRF_world_sem(a) = std(rW, 0, 'omitnan') / sqrt(nW);
            else
                SRF_world_sem(a) = NaN;
            end
        end
    end

    SRF_head_mu_smooth = circular_gauss_smooth(SRF_head_mu, winSize, sigma);
    SRF_head_sem_smooth = circular_gauss_smooth(SRF_head_sem, winSize, sigma);
    SRF_world_mu_smooth = circular_gauss_smooth(SRF_world_mu(:), winSize, sigma);
    SRF_world_sem_smooth = circular_gauss_smooth(SRF_world_sem(:), winSize, sigma);

    %%  2*1 plots (SRF;heatmap)
    col_head  = [0.10 0.20 0.50];
    col_world = [0.60 0.00 0.00];

    figure('Color','w','Position',[100 100 350 500]);

    axWidth = 1 - marginLeft - marginRight;
    axHeight = (1 - marginTop - marginBottom - gapY)/2;

    % Top: SRFs + SEM
    ax1 = axes('Position', [marginLeft, ...
                            marginBottom + axHeight + gapY, ...
                            axWidth, axHeight]);
    hold(ax1, 'on');

    x = deg_map;

    %  Head angle shaded SEM 
    head_upper = SRF_head_mu_smooth + SRF_head_sem_smooth;
    head_lower = SRF_head_mu_smooth - SRF_head_sem_smooth;

    head_patch_x = [x, fliplr(x)];
    head_patch_y = [head_upper', fliplr(head_lower')];

    hFillHead = fill(ax1, head_patch_x, head_patch_y, col_head, ...
        'FaceAlpha', 0.25, 'EdgeColor', 'none');

    % Head mean line
    hLineHead = plot(ax1, x, SRF_head_mu_smooth, '-', ...
        'Color', col_head, 'LineWidth', 1.8);

    % --- World angle shaded SEM ---
    world_upper = SRF_world_mu_smooth + SRF_world_sem_smooth;
    world_lower = SRF_world_mu_smooth - SRF_world_sem_smooth;

    world_patch_x = [x, fliplr(x)];
    world_patch_y = [world_upper', fliplr(world_lower')];

    hFillWorld = fill(ax1, world_patch_x, world_patch_y, col_world, ...
        'FaceAlpha', 0.25, 'EdgeColor', 'none');

    % World mean line
    hLineWorld = plot(ax1, x, SRF_world_mu_smooth, '-', ...
        'Color', col_world, 'LineWidth', 1.8);

    xlabel(ax1, 'Angle (deg)');
    ylabel(ax1, 'Firing rate (spikes/s)');
    title(ax1, sprintf('Spatial receptive fields – Cluster %d', cl_id));

    % Legend: only head & world frames
    legend(ax1, [hLineHead, hLineWorld], ...
        {'Head frame', 'World frame'}, ...
        'Location','best', 'Box','off');
xlim(ax1, [-180 180]);

    style_line_axes(ax1);

    %  Bottom: Normalized heatmap
    ax2 = axes('Position', [marginLeft, ...
                            marginBottom, ...
                            axWidth, axHeight]);

    imagesc(ax2, deg_map, deg_map, combined_norm);
    set(ax2, 'YDir','normal');
    axis(ax2, 'image', 'tight');

    colormap(ax2, cmap_neuro_blue(256));
    style_neuro_heatmap(ax2, 'World angle (deg)', 'Head angle (deg)');

    title(ax2, '');
    cb = colorbar(ax2);
    cb.Label.String   = 'Normalised firing rate';
    cb.Label.FontSize = 10;
    cb.FontSize       = 10;

end

%%  Functions

function B = fill_nan_neighbours(A)
    [H, W] = size(A);
    nanmask = isnan(A);
    [rows, cols] = find(nanmask);

    for k = 1:numel(rows)
        r = rows(k); 
        c = cols(k);
        neigh = [];

        % 4-neighbours
        if r > 1 && ~nanmask(r-1, c), neigh(end+1) = A(r-1, c); end
        if r < H && ~nanmask(r+1, c), neigh(end+1) = A(r+1, c); end
        if c > 1 && ~nanmask(r, c-1), neigh(end+1) = A(r, c-1); end
        if c < W && ~nanmask(r, c+1), neigh(end+1) = A(r, c+1); end

        % If none found, try 8-neighbours
        if isempty(neigh)
            for dr = -1:1
                for dc = -1:1
                    if dr == 0 && dc == 0, continue, end
                    rr = r + dr; cc = c + dc;
                    if rr >= 1 && rr <= H && cc >= 1 && cc <= W && ~nanmask(rr, cc)
                        neigh(end+1) = A(rr, cc);
                    end
                end
            end
        end
        if isempty(neigh)
            A(r, c) = mean(A(~nanmask), 'omitnan');
        else
            A(r, c) = mean(neigh, 'omitnan');
        end
    end

    B = A;
end

function y = circular_gauss_smooth(x, winSize, sigma)
    % x: column vector
    x = x(:);
    if nargin < 2 || isempty(winSize), winSize = 5; end
    if nargin < 3 || isempty(sigma),   sigma   = 1; end

    if mod(winSize,2) == 0
        error('winSize must be odd');
    end

    L   = numel(x);
    pad = floor(winSize/2);

    % Gaussian kernel
    k = -pad:pad;
    g = exp(-(k.^2) / (2*sigma^2));
    g = g / sum(g);

    % Circular padding
    x_ext = [x(end-pad+1:end); x; x(1:pad)];

    % Convolution
    y_ext = conv(x_ext, g, 'same');

    % Back to original length
    y = y_ext(pad+1 : pad+L);
end

function cmap = cmap_neuro_blue(n)
    if nargin < 1
        n = 256;
    end

    base = [ ...
        1.00 1.00 1.00;  
        0.88 0.93 1.00;  
        0.70 0.80 1.00;  
        0.40 0.60 0.98;
        0.10 0.20 0.50; 
    ];

    x = linspace(0, 1, size(base,1));
    xi = linspace(0, 1, n);
    cmap = interp1(x, base, xi, 'pchip');
end

function style_neuro_heatmap(ax, xlab, ylab)
    if nargin < 1 || isempty(ax); ax = gca; end
    if nargin >= 2 && ~isempty(xlab); ax.XLabel.String = xlab; end
    if nargin >= 3 && ~isempty(ylab); ax.YLabel.String = ylab; end

    ax.FontName = 'Arial';
    ax.FontSize = 10;
    ax.LineWidth = 0.75;
    ax.TickDir = 'out';
    ax.Box = 'off';
    ax.TickLength = [0.015 0.015];
end

function style_line_axes(ax)
    if nargin < 1 || isempty(ax); ax = gca; end
    ax.FontName = 'Arial';
    ax.FontSize = 10;
    ax.LineWidth = 0.75;
    ax.TickDir = 'out';
    ax.TickLength = [0.015 0.015];
    ax.Box = 'off';
end
