clc;
close all;

folderPath ='F:\Spontage_results\Right\Sponge_Right__Group_1\recording_info.csv';
bad_block_path = ' ';
sampling_rate = 20000;

baseline_win = [-0.05, 0];
early_win = [ 0, 0.05];
sound_dur = 0.25;

late_pre_win = [0.2, 0.25];
late_post_win = [0.25, 0.3];

alpha_glm = 0.05;
min_pairs = 20;
use_dz_gate = true;
d_min = 0.2;
min_sessions = 2;

n_sessions = sum(~cellfun('isempty', Behavior_));

% Loading
Cinfo = readtable('cluster_info.tsv','FileType','text','Delimiter','\t');
valid_clusters = Cinfo.cluster_id;
chan_of_clust = Cinfo.ch;

spike_times_all = double(readNPY('spike_times.npy'));
spike_clusters_all = double(readNPY('spike_clusters.npy'));
keep_idx = ismember(spike_clusters_all, valid_clusters);
spike_times_all = spike_times_all(keep_idx);
spike_clusters_all = spike_clusters_all(keep_idx);

Sessionss = generateSessionCells(folderPath, bad_block_path);
McStStimOnsetTimes = onset_times(~cellfun('isempty',onset_times));
Behaviors = Behavior_(~cellfun('isempty',Behavior_));

len_baseline = diff(baseline_win);
len_early = diff(early_win);
len_lpre = diff(late_pre_win);
len_lpost = diff(late_post_win);

all_units = unique(spike_clusters_all(:))';
nU = numel(all_units);

% Output
isDriven_glm = false(nU,1);
nSess_pass = zeros(nU,1);
sess_pass_flags = cell(nU,1);   

for ii = 1:nU
    uid = all_units(ii);
    sess_flags = false(numel(Sessionss),1);  % per-session driven?

    for s = 1:numel(Sessionss)

        if s > numel(McStStimOnsetTimes), continue; end
        if isempty(Sessionss{s}), continue; end
        if isempty(Behaviors{s}), continue, end

        beh = Behaviors{s};

        if width(beh) < 15, continue, end

        onset = McStStimOnsetTimes{s};

        if isempty(onset), continue; end

        [t0, t1] = bounds(Sessionss{s});
        in_sess = (spike_clusters_all==uid) & ...
            (spike_times_all>=t0) & (spike_times_all<=t1);

        if ~any(in_sess), continue; end

        sp_s = (spike_times_all(in_sess)-t0)/sampling_rate;

        % trial-wise counts
        pre_e=[]; post_e=[]; pre_l=[]; post_l=[];
        for tr = 1:numel(onset)
            r = sp_s - onset(tr);
            pre_e(end+1,1)  = sum(r >= baseline_win(1) & r < baseline_win(2));
            post_e(end+1,1) = sum(r >= early_win(1)    & r < early_win(2));
            pre_l(end+1,1)  = sum(r >= late_pre_win(1) & r < late_pre_win(2));
            post_l(end+1,1) = sum(r >= late_post_win(1)& r < late_post_win(2));
        end

        sig_flag = false;

        % Early GLM
        n_e = min(numel(pre_e),numel(post_e));

        if n_e >= min_pairs
            pre=pre_e(1:n_e);
            post=post_e(1:n_e);

            pre_fr=pre/len_baseline;
            post_fr=post/len_early;

            dvec=post_fr-pre_fr;
            sd_d=std(dvec,0);
            dz=mean(dvec)/(sd_d+(sd_d==0)*eps);

            if ~isfinite(dz), dz=0; end

            try
                y=[pre;post]; 
                epoch=[zeros(n_e,1);ones(n_e,1)];

                T=table(y,epoch,'VariableNames',{'y','epoch'});

                m0=fitglm(T,'y~1','Distribution','poisson','Link','log');
                m1=fitglm(T,'y~1+epoch','Distribution','poisson','Link','log');

                LR = m0.Deviance - m1.Deviance;

                if LR < 0
                    p = 1;              % model fits worse (definitely not significant)
                else
                    df = max(m1.NumEstimatedCoefficients - m0.NumEstimatedCoefficients, 0);
                    if df == 0
                        p = 1;
                    else
                        p = 1 - chi2cdf(LR, df);
                    end
                end

                if use_dz_gate
                    sig_flag = (isfinite(p) && p<=alpha_glm && abs(dz)>=d_min);
                else
                    sig_flag = (isfinite(p) && p<=alpha_glm);
                end
            catch
            end
        end

        % Late GLM 
        n_l = min(numel(pre_l),numel(post_l));
        if n_l >= min_pairs
            pre=pre_l(1:n_l);
            post=post_l(1:n_l);

            pre_fr=pre/len_lpre;
            post_fr=post/len_lpost;
            dvec=post_fr-pre_fr;
            sd_d=std(dvec,0);
            dz=mean(dvec)/(sd_d+(sd_d==0)*eps);
            if ~isfinite(dz), dz=0; end

            try
                y=[pre;post];
                epoch=[zeros(n_l,1);ones(n_l,1)];

                T=table(y,epoch,'VariableNames',{'y','epoch'});
                m0=fitglm(T,'y~1','Distribution','poisson','Link','log');
                m1=fitglm(T,'y~1+epoch','Distribution','poisson','Link','log');
                LR = m0.Deviance - m1.Deviance;

                if LR < 0
                    p = 1;              % model fits worse (definitely not significant)
                else
                    df = max(m1.NumEstimatedCoefficients - m0.NumEstimatedCoefficients, 0);
                    if df == 0
                        p = 1;
                    else
                        p = 1 - chi2cdf(LR, df);
                    end
                end


                if use_dz_gate
                    sig_flag = sig_flag | (isfinite(p) && p<=alpha_glm && abs(dz)>=d_min);
                else
                    sig_flag = sig_flag | (isfinite(p) && p<=alpha_glm);
                end
            catch
            end
        end

        sess_flags(s) = sig_flag;
    end

    % unit-level decision
    nSess_pass(ii) = sum(sess_flags);
    isDriven_glm(ii) = (nSess_pass(ii) >= min_sessions);
    sess_pass_flags{ii} = sess_flags;  
end

driven_list = all_units(isDriven_glm).';
fprintf('Session-by-session GLM (>= %d sessions significant): %d of %d units\n', ...
    min_sessions, sum(isDriven_glm), nU);
disp('Driven unit IDs:'); 
disp(driven_list');

%% Post Selection (remove sparse units by relative power)

kept_units = driven_list(:)';

if isempty(kept_units)
    final_driven_power = kept_units;
    fprintf('\n No units to evaluate.\n');
else
    % PSTH settings
    bin_s = 0.005;       % 5 ms bins
    sigma_ms = 2;        % smoothing
    sigma_bin = sigma_ms/(bin_s*1000);
    ker_half  = max(3, ceil(4*sigma_bin));
    xk = -ker_half:ker_half;
    gker = exp(-0.5*(xk/sigma_bin).^2);
    gker = gker/sum(gker);

    unit_power = nan(numel(kept_units),1);

    for u = 1:numel(kept_units)
        uid = kept_units(u);
        ii  = find(all_units==uid,1);
        if isempty(ii), unit_power(u)=0; continue; end

        pass_flags = sess_pass_flags{ii};         % GLM-passing sessions for this unit
        sess_peaks = [];                           % store per-session peak rates (Hz)

        for s = find(pass_flags(:))'
            if isempty(Sessionss{s}) || s > numel(McStStimOnsetTimes), continue; end
            onset = McStStimOnsetTimes{s}; if isempty(onset), continue; end
            [t0,t1] = bounds(Sessionss{s});
            in_sess = (spike_clusters_all==uid) & ...
                (spike_times_all>=t0) & (spike_times_all<=t1);
            if ~any(in_sess), continue; end

            sp_s = (spike_times_all(in_sess)-t0)/sampling_rate;
            ntr  = numel(onset);

  % pick the stronger window in this session (Early (0-50 ms) vs Late (250-300 ms))
            cntE = zeros(ntr,1); cntP = zeros(ntr,1);
            for tr = 1:ntr
                r = sp_s - onset(tr);
                cntE(tr) = sum(r>=early_win(1) & r<early_win(2));
                cntP(tr) = sum(r>=late_post_win(1) & r<late_post_win(2));
            end
            if mean(cntE)/diff(early_win) >= mean(cntP)/diff(late_post_win)
                W = early_win;
            else
                W = late_post_win;
            end

            % build PSTH for chosen window across all trials in this session
            edges = W(1):bin_s:W(2); 
            if edges(end)<W(2), edges(end+1)=W(2); end
            h = zeros(1,numel(edges)-1);
            for tr = 1:ntr
                r = sp_s - onset(tr);
                rr = r(r>=W(1) & r<W(2));
                h = h + histcounts(rr, edges);
            end

            hs = conv(h, gker, 'same'); % smoothed
            % peak firing rate in Hz (counts per bin / (bin * trials))
            peak_rate_hz = max(hs) / (bin_s * max(ntr,1));
            sess_peaks(end+1,1) = peak_rate_hz;
        end

        if isempty(sess_peaks)
            unit_power(u) = 0;
        else
            unit_power(u) = median(sess_peaks);
        end
    end

    medPow = median(unit_power(unit_power>0));
    cutPow = 0.2 * medPow;

    sparse_mask = unit_power < cutPow;
    sparse_ids  = kept_units(sparse_mask);
    final_driven_power = setdiff(kept_units, sparse_ids);

    fprintf('[Post selection] Units removed (sparse): %d\n', numel(sparse_ids)); disp(sparse_ids');
    fprintf('[Post selection] Final driven after filtering sparse units: %d\n', numel(final_driven_power));
    disp('Final Driven unit IDs:'); 
    disp(final_driven_power);
end
