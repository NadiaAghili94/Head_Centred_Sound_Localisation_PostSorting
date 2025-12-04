clc;
close all;

folderPath = 'F:\Spontage_results\Right\Sponge_Right__Group_1\recording_info.csv';
bad_block_path = '';

sampling_rate = 20000;  % Hz
deg_map = [150 120 90 60 30 0 -30 -60 -90 -120 -150 -180];

post_win = [0, 0.15];       % seconds
late_post_win = [0.25, 0.4];

min_trials = 100;             % minimum pooled trials to fit models

gate_mode = 'full'; % Gate mode: 'paper'/'full'/'strict'/'permissive' ; paper = Stephen's

alpha_gate = 0.05;            

% preselected_ids = [6 7 10 13 14 19 20 21 26 28 29 30 36 37];
 preselected_ids = [28];  % leave [] for calculating all units


% Loading
Cinfo = readtable('cluster_info.tsv','FileType','text','Delimiter','\t');
valid_clusters = Cinfo.cluster_id;

spike_times_all = double(readNPY('spike_times.npy'));     % samples
spike_clusters_all = double(readNPY('spike_clusters.npy'));  % unit IDs

keep_idx = ismember(spike_clusters_all, valid_clusters);
spike_times_all = spike_times_all(keep_idx);
spike_clusters_all = spike_clusters_all(keep_idx);

Sessionss = generateSessionCells(folderPath, bad_block_path);
McStStimOnsetTimes = onset_times(~cellfun('isempty',onset_times));
Behaviors = Behavior_(~cellfun('isempty',Behavior_));

all_units = unique(spike_clusters_all(:))';
if isempty(preselected_ids)
    units_to_run = all_units;
else
    units_to_run = intersect(all_units, preselected_ids, 'stable');
end
nU = numel(units_to_run);

% Outputs
Label = strings(nU,1);       % "Head"/"World"/"NotTuned"/"FitError"/"No min trial"
AIC_H = nan(nU,1);
AIC_W = nan(nU,1);
Fit_head = nan(nU,1);           
Fit_world = nan(nU,1);          
ModelPref = nan(nU,1);          
nTrials = zeros(nU,1);
nTrials_l = zeros(nU,1);

% Main loop
for k = 1:nU
    cl_id = units_to_run(k);

    trial_head = [];
    trial_world = [];
    trial_count = [];
    trial_count_late = [];
    sess_id_all = [];

    %  Collect trials from all sessions & pool
    for s = 1:numel(Sessionss)
        if s > numel(McStStimOnsetTimes), continue; end
        if isempty(Sessionss{s}) || isempty(Behaviors{s}), continue; end
        beh = Behaviors{s};
        if width(beh) < 15, continue; end  % skip FRAs

        csr = beh.CenterSpoutRotation;   % deg
        sloc = beh.Speaker_Location;      % index into deg_map
        onset = McStStimOnsetTimes{s};

        [t0, t1] = bounds(Sessionss{s});
        in_sess = (spike_clusters_all == cl_id) & ...
                  (spike_times_all >= t0) & (spike_times_all <= t1);
        if ~any(in_sess), continue; end

        sp_s = (spike_times_all(in_sess) - t0) / sampling_rate;  % seconds

        th = []; tw = []; yc = []; ycl = [];
        for tr = 1:numel(onset)
            if isnan(sloc(tr)), continue; end

            wa = deg_map(sloc(tr));                    % world angle
            ha = mod(wa - csr(tr) + 180, 360) - 180;   % head angle [-180,180]
            if abs(ha) == 180, ha = -180; end
            if abs(wa) == 180, wa = -180; end

            s_rel = sp_s - onset(tr);
            cnt = sum(s_rel >= post_win(1) & s_rel < post_win(2));
            cnt_late = sum(s_rel >= late_post_win(1) & s_rel < late_post_win(2));

            th(end+1,1) = ha;
            tw(end+1,1) = wa;
            yc(end+1,1) = cnt;
            ycl(end+1,1) = cnt_late;
        end

        if ~isempty(yc)
            trial_head = [trial_head; th];
            trial_world = [trial_world; tw];
            trial_count = [trial_count; yc];
            trial_count_late = [trial_count_late; ycl];
            sess_id_all = [sess_id_all; s*ones(size(yc))];
        end
    end

    % counts
    nTrials(k) = numel(trial_count);
    nTrials_l(k) = numel(trial_count_late);

    if nTrials(k) < min_trials || nTrials_l(k) < min_trials
        Label(k) = "No min trial";
        continue;
    end

    Xh = [cosd(trial_head), sind(trial_head)];
    Xw = [cosd(trial_world), sind(trial_world)];
    y = trial_count;
    y_l = trial_count_late;

    try
        % Build tables with session as categorical (random intercept)
        tbl = table(y, Xh(:,1), Xh(:,2), Xw(:,1), Xw(:,2), categorical(sess_id_all), ...
                      'VariableNames', {'y','h1','h2','w1','w2','sess'});
        tbl_l = table(y_l, Xh(:,1), Xh(:,2), Xw(:,1), Xw(:,2), categorical(sess_id_all), ...
                      'VariableNames', {'y_1','h1','h2','w1','w2','sess'});

        % GLME fits with random intercept per session 

        % Post window
        mC = fitglme(tbl, 'y ~ 1 + (1|sess)', 'Distribution','Poisson','Link','log','FitMethod','Laplace');
        mFull = fitglme(tbl, 'y ~ h1+h2+w1+w2 + (1|sess)', 'Distribution','Poisson','Link','log','FitMethod','Laplace');

        % Late window
        mC_l = fitglme(tbl_l, 'y_1 ~ 1 + (1|sess)', 'Distribution','Poisson','Link','log','FitMethod','Laplace');
        mFull_l = fitglme(tbl_l, 'y_1 ~ h1+h2+w1+w2 + (1|sess)', 'Distribution','Poisson','Link','log','FitMethod','Laplace');

        % Full vs Constant (Simple manual calculation from the log-likelihoods, as there is no deviance function in fitmle)
        LR_F = max(2*(mFull.LogLikelihood - mC.LogLikelihood), 0);
        df_F = max(mFull.NumEstimatedCoefficients - mC.NumEstimatedCoefficients, 0);
        pF = (df_F==0 || LR_F==0) * 1 + (df_F>0 & LR_F>0) * (1 - chi2cdf(LR_F, df_F));

        LR_Fl = max(2*(mFull_l.LogLikelihood - mC_l.LogLikelihood), 0);
        df_Fl = max(mFull_l.NumEstimatedCoefficients - mC_l.NumEstimatedCoefficients, 0);
        pFl = (df_Fl==0 || LR_Fl==0) * 1 + (df_Fl>0 & LR_Fl>0) * (1 - chi2cdf(LR_Fl, df_Fl));

        tuned_full   = (pF < alpha_gate) && (LR_F > 0);
        tuned_full_l = (pFl < alpha_gate) && (LR_Fl > 0);

        if ~(tuned_full || tuned_full_l)
            Label(k) = "NotTuned";
            continue;
        end

        % If tuned, choose Head vs World by AIC 
        % Fit head-only and world-only (with random intercept) just to compare AICs
        if tuned_full
            mH = fitglme(tbl, 'y ~ h1+h2 + (1|sess)', 'Distribution','Poisson','Link','log','FitMethod','Laplace');
            mW = fitglme(tbl, 'y ~ w1+w2 + (1|sess)', 'Distribution','Poisson','Link','log','FitMethod','Laplace');

            AIC_H(k) = mH.ModelCriterion.AIC;
            AIC_W(k) = mW.ModelCriterion.AIC;

            if AIC_H(k) < AIC_W(k), Label(k) = "Head"; else, Label(k) = "World"; end

            % Model-fit rate using log-likelihoods (we do not have devience in fitglme) (relative to Full)
            LR_H = max(2*(mH.LogLikelihood - mC.LogLikelihood), 0);
            LR_W = max(2*(mW.LogLikelihood - mC.LogLikelihood), 0);
            denom = max(LR_F, eps);
            Fit_head(k)  = LR_H / denom;
            Fit_world(k) = LR_W / denom;
            ModelPref(k) = Fit_head(k) - Fit_world(k);

        else % tuned only in late window
            mH_l = fitglme(tbl_l, 'y_1 ~ h1+h2 + (1|sess)', 'Distribution','Poisson','Link','log','FitMethod','Laplace');
            mW_l = fitglme(tbl_l, 'y_1 ~ w1+w2 + (1|sess)', 'Distribution','Poisson','Link','log','FitMethod','Laplace');

            AIC_H(k) = mH_l.ModelCriterion.AIC;
            AIC_W(k) = mW_l.ModelCriterion.AIC;

            if AIC_H(k) < AIC_W(k), Label(k) = "Head"; else, Label(k) = "World"; end

            LR_Hl = max(2*(mH_l.LogLikelihood - mC_l.LogLikelihood), 0);
            LR_Wl = max(2*(mW_l.LogLikelihood - mC_l.LogLikelihood), 0);
            denom = max(LR_Fl, eps);
            Fit_head(k)  = LR_Hl / denom;
            Fit_world(k) = LR_Wl / denom;
            ModelPref(k) = Fit_head(k) - Fit_world(k);
        end

    catch 
        disp('Error occurred')
        Label(k) = "FitError";
    end
end

% Summary
head_ids = units_to_run(Label=="Head").';
world_ids = units_to_run(Label=="World").';
ntuned_ids = units_to_run(Label=="NotTuned").';

fprintf('\n Combined GLMM (random session intercept), alpha=%.3f \n', alpha_gate);
fprintf('Head=%d / World=%d / NotTuned=%d / TotalRun=%d\n', ...
    numel(head_ids), numel(world_ids), numel(ntuned_ids), nU);

% Plots
mask_plot = ~isnan(Fit_head) & ~isnan(Fit_world);
H = 100 * Fit_head(mask_plot);
W = 100 * Fit_world(mask_plot);
L = Label(mask_plot);

pref_thr = 0.2;
MP_masked = ModelPref(mask_plot);

is_tuned = (L=="Head") | (L=="World");
mixed = is_tuned & (abs(MP_masked) <= pref_thr);
strong_head = (L=="Head")  & (MP_masked >  pref_thr);
strong_world = (L=="World") & (MP_masked < -pref_thr);

figure('Color','w','Position',[260 260 420 420]); hold on; box on;
plot(H(mixed), W(mixed),'o', 'LineStyle','none', 'Color',[0.6 0.6 0.6]);
plot(H(strong_head), W(strong_head),'o','LineStyle','none', 'MarkerFaceColor',[0 0.45 1], 'MarkerEdgeColor','none');
plot(H(strong_world), W(strong_world), 'o', 'LineStyle','none', 'MarkerFaceColor',[1 0 0],   'MarkerEdgeColor','none');
plot([0 100],[0 100],'k--','LineWidth',1);
axis([0 100 0 100]); axis square; grid on
xlabel('Model Fit (%) - Head');
ylabel('Model Fit (%) - World');
title(sprintf('Model Preference (\\pm%.1f threshold)', pref_thr));

figure('Color','w','Position',[220 220 560 360]);
bar([numel(head_ids) numel(world_ids) numel(ntuned_ids)],'LineWidth',1.2); grid on
set(gca,'XTick',1:3,'XTickLabel',{'Head','World','Untuned'});
ylabel('Units (#)');


%% Bar plot (used when running a single unit; otherwise this section can be commented out)

col_head  = [0.10 0.20 0.50];   % navy blue
col_world = [0.60 0.00 0.00];   % dark red

figure('Color','w','Position',[100 100 350 350]);
ax = gca; hold(ax,'on');

bar(1, Fit_head(1),  'FaceColor', col_head,  'BarWidth', 0.6);
bar(2, Fit_world(1), 'FaceColor', col_world, 'BarWidth', 0.6);

% Zero line
plot([0.5 2.5], [0 0], 'k-', 'LineWidth', 0.75);

% Text for preference
ymax_ego = max([Fit_head, Fit_world]);
text(1.5, ymax_ego * 1.05, sprintf('Pref = %.3f', ModelPref(1)), ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 10, ...
    'FontName', 'Arial');

set(ax, 'XTick', [1 2], 'XTickLabel', {'Head','World'});
ylabel('GLM model fit');
title('Head-centred simulated unit');



