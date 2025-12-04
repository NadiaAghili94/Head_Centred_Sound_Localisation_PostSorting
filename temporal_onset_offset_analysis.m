clc; 
close all;

folderPath ='F:\Spontage_results\Right\Sponge_Right__Group_1\recording_info.csv';
bad_block_path = ' ';
sampling_rate = 20000;

baseline_win = [-0.05, 0];   % for onset gate
early_win = [ 0, 0.05];   % onset (50 ms)
late_pre_win = [0.2, 0.25];  % for offset gate
late_post_win = [0.25, 0.3];  % offset (50 ms)

onset_win  = [0 0.15];
offset_win = [0.25 0.4]; 

preselected_driven = [6     7    10    13    14    19    20    21    26    28    29    30    36    37];  % already selected by driven_unit_selection
deg_map = [150 120 90 60 30 0 -30 -60 -90 -120 -150 -180];

alpha_glm = 0.05;
min_pairs = 20;
use_dz_gate = true;
d_min = 0.2;

Sessionss = generateSessionCells(folderPath, bad_block_path);
McStStimOnsetTimes = onset_times(~cellfun('isempty',onset_times));
Behaviors = Behavior_(~cellfun('isempty',Behavior_));

n_sessions = sum(~cellfun('isempty', Behavior_));
min_sessions = 2;   

Cinfo  = readtable('cluster_info.tsv','FileType','text','Delimiter','\t');
valid_clusters = Cinfo.cluster_id;
spike_times_all = double(readNPY('spike_times.npy'));
spike_clusters_all = double(readNPY('spike_clusters.npy'));
keep_idx = ismember(spike_clusters_all, valid_clusters);
spike_times_all = spike_times_all(keep_idx);
spike_clusters_all = spike_clusters_all(keep_idx);
Cinfo = readtable('cluster_info.tsv','FileType','text','Delimiter','\t');
valid_clusters = Cinfo.cluster_id;
chan_of_clust = Cinfo.ch;

len_baseline = diff(baseline_win);
len_early = diff(early_win);
len_lpre = diff(late_pre_win);
len_lpost = diff(late_post_win);

driven_units = preselected_driven(:)';
nU = numel(driven_units);

% Onset/Offset/Both
n_on   = zeros(nU,1);
n_off  = zeros(nU,1);
n_both = zeros(nU,1);
n_any  = zeros(nU,1);
Class  = strings(nU,1);  % "Both"/ "Onset-only"/ "Offset-only"/ "Mixed-across-sessions"/ "None"

% Main loop
for ui = 1:nU
    uid = driven_units(ui);
    on_s  = false(numel(Sessionss),1);
    off_s = false(numel(Sessionss),1);

    for s = 1:numel(Sessionss)
        if s > numel(McStStimOnsetTimes), continue; end

        if isempty(Sessionss{s}) || isempty(Behaviors{s}), continue; end

        beh = Behaviors{s}; 

        if width(beh) < 15, continue; end

        onset_s = McStStimOnsetTimes{s}; 

        if isempty(onset_s), continue; end

        [t0,t1] = bounds(Sessionss{s});

        in_sess = (spike_clusters_all==uid) & (spike_times_all>=t0) & (spike_times_all<=t1);

        if ~any(in_sess), continue; end

        sp_s = (spike_times_all(in_sess)-t0)/sampling_rate;

        % per-trial counts
        pre_on=[]; post_on=[]; pre_of=[]; post_of=[];
        for tr = 1:numel(onset_s)
            r = sp_s - onset_s(tr);
            pre_on(end+1,1) = sum(r >= baseline_win(1) & r < baseline_win(2));  % -50-0
            post_on(end+1,1) = sum(r >= early_win(1) & r < early_win(2));  % 0-50
            pre_of(end+1,1) = sum(r >= late_pre_win(1) & r < late_pre_win(2)); % 200-250
            post_of(end+1,1) = sum(r >= late_post_win(1) & r < late_post_win(2)); % 250-300
        end

        % onset test
        try
            n_e = min(numel(pre_on), numel(post_on));
            if n_e >= min_pairs
                pre  = pre_on(1:n_e); 
                post = post_on(1:n_e);

                pre_fr=pre/len_baseline;
                post_fr=post/len_early;

                dvec=post_fr-pre_fr;
                sd_d=std(dvec,0);
                dz=mean(dvec)/sd_d; % (sd_d+(sd_d==0)*eps);

                  if ~isfinite(dz), dz=0; end
                y=[pre;post]; 
                epoch=[zeros(n_e,1);ones(n_e,1)];

                T = table(y,epoch,'VariableNames',{'y','epoch'});
                m0=fitglm(T,'y~1','Distribution','poisson','Link','log');  % null
                m1=fitglm(T,'y~1+epoch','Distribution','poisson','Link','log');
                LR = m0.Deviance - m1.Deviance;

                if LR < 0
                    p = 1;    % model fits worse (not significant)
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
                on_s(s) = sig_flag;
            end
        catch
        end

        % offset test
        try
            n_l = min(numel(pre_of), numel(post_of));
            if n_l >= min_pairs

                pre_fr=pre_of/len_lpre;
                post_fr=post_of/len_lpost;
                dvec=post_fr-pre_fr;
                sd_d=std(dvec,0);
                dz=mean(dvec)/(sd_d+(sd_d==0)*eps);
                if ~isfinite(dz), dz=0; end

                y=[pre_of;post_of]; 
                epoch=[zeros(n_l,1);ones(n_l,1)];

                T=table(y,epoch,'VariableNames',{'y','epoch'});
                m0=fitglm(T,'y~1','Distribution','poisson','Link','log'); % null
                m1=fitglm(T,'y~1+epoch','Distribution','poisson','Link','log');

                LR = m0.Deviance - m1.Deviance;

                if LR < 0
                    p = 1;  % model fits worse  (not significant)
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
                off_s(s) = sig_flag;
             
            end
        catch
        end
    end

    both_s = on_s & off_s;
    any_s = on_s | off_s;

    n_on(ui) = sum(on_s);
    n_off(ui) = sum(off_s);
    n_both(ui) = sum(both_s);
    n_any(ui) = sum(any_s);

    if n_both(ui) >= min_sessions
        Class(ui) = "Both";
    elseif n_on(ui) >= min_sessions
        Class(ui) = "Onset-only";
    elseif n_off(ui) >= min_sessions
        Class(ui) = "Offset-only";
    elseif n_any(ui) >= min_sessions
        Class(ui) = "Mixed_across_session";
    else
        Class(ui) = "None";
    end
end

fprintf('\nOn/Off classification (session-wise): Both=%d | Onset-only=%d | Offset-only=%d |Mixed_across_session=%d | None=%d | Total=%d\n', ...
    sum(Class=="Both"), sum(Class=="Onset-only"), sum(Class=="Offset-only"), ...
    sum(Class=="Mixed_across_session"), sum(Class=="None"), nU);

Both_IDs = driven_units(Class=="Both").';
OnsetOnly_IDs = driven_units(Class=="Onset-only").';
OffsetOnly_IDs = driven_units(Class=="Offset-only").';
Mixed_across_session_IDs = driven_units(Class=="Mixed_across_session").';
NoneIDss = driven_units(Class=="None").';

%%  Population Bar (Onset-only / Offset-only / Both) 
figure('Color','w','Position',[160 160 420 300]);
bar([sum(Class=="Onset-only"), sum(Class=="Offset-only"), sum(Class=="Mixed_across_session"), sum(Class=="Both")], 'LineWidth',1.2);
set(gca,'XTickLabel',{'Onset-only','Offset-only','Mixed_across_session', 'Both'});
ylabel('Units'); grid on; title('Population (session-wise, 50 ms windows)');
labels = {'Onset-only','Offset-only','Mixed-across-sess','Both'};

%%  Tuning for "BOTH" units (Head / World / Mixed) 
      
min_trials_phase = 40;           % pooled across sessions
mix_thr = 0.2;                   

both_ids = Both_IDs(:)';
nb = numel(both_ids);

Label_On   = strings(nb,1); Label_Off = strings(nb,1);
FitHead_On = nan(nb,1);     FitWorld_On = nan(nb,1);     ModelPref_On = nan(nb,1);
FitHead_Off= nan(nb,1);     FitWorld_Off= nan(nb,1);     ModelPref_Off= nan(nb,1);

for k = 1:nb
    uid = both_ids(k);
    % pooled trials with session IDs
    th_on=[]; tw_on=[]; y_on=[]; sess_on=[];
    th_off=[];tw_off=[]; y_off=[]; sess_off=[];

    for s = 1:numel(Sessionss)
        if s>numel(McStStimOnsetTimes), continue; end
        if isempty(Sessionss{s}) || isempty(Behaviors{s}), continue; end
        beh = Behaviors{s}; if width(beh)<15, continue; end
        onset_s = McStStimOnsetTimes{s}; if isempty(onset_s), continue; end

        csr  = beh.CenterSpoutRotation;
        sloc = beh.Speaker_Location;

        [t0,t1] = bounds(Sessionss{s});
        in_sess = (spike_clusters_all==uid) & (spike_times_all>=t0) & (spike_times_all<=t1);
        if ~any(in_sess), continue; end
        sp_s = (spike_times_all(in_sess)-t0)/sampling_rate;

        for tr = 1:numel(onset_s)
            if isnan(sloc(tr)), continue; end
            wa = deg_map(sloc(tr));
            ha = mod(wa - csr(tr) + 180,360) - 180;
            if abs(ha)==180, ha=-180; end
            if abs(wa)==180, wa=-180; end

            r = sp_s - onset_s(tr);

            y_on(end+1,1) = sum(r>=onset_win(1)  & r<onset_win(2));
            th_on(end+1,1) = ha;
            tw_on(end+1,1) = wa; 
            sess_on(end+1,1) = s;

            y_off(end+1,1) = sum(r>=offset_win(1) & r<offset_win(2));
            th_off(end+1,1) = ha; 
            tw_off(end+1,1) = wa;
            sess_off(end+1,1) = s;
        end
    end

    [Label_On(k),  FitHead_On(k),  FitWorld_On(k),  ModelPref_On(k)]  = ...
        label_phase(y_on,  th_on,  tw_on,  sess_on,  mix_thr, min_trials_phase);
    [Label_Off(k), FitHead_Off(k), FitWorld_Off(k), ModelPref_Off(k)] = ...
        label_phase(y_off, th_off, tw_off, sess_off, mix_thr, min_trials_phase);
end

%%  Heatmap: Onset -> Offset labels (Both units)
cats = ["Head","World","Mixed","NotTuned"];

Label_On(~ismember(Label_On,  cats)) = "NotTuned";
Label_Off(~ismember(Label_Off, cats)) = "NotTuned";

co = categorical(Label_On,  cats);
cf = categorical(Label_Off, cats);
io = double(co); 
jo = double(cf);
nt = find(cats=="NotTuned");
io(isnan(io)) = nt;
jo(isnan(jo)) = nt;
C = accumarray([io,jo], 1, [numel(cats), numel(cats)]);

figure('Color','w','Position',[200 200 420 360]);
imagesc(C); axis image; colormap(parula); colorbar
set(gca,'XTick',1:numel(cats),'XTickLabel',cats,'YTick',1:numel(cats),'YTickLabel',cats);
xlabel('Offset label'); ylabel('Onset label');
title('Both units: Onset to Offset (Head/World/Mixed/Untuned)');
for r=1:numel(cats)
    for c=1:numel(cats)
        text(c,r,num2str(C(r,c)),'Color','w','FontWeight','bold','HorizontalAlignment','center');
    end
end


%%  Functions

function [lab,fh,fw,mp] = label_phase(y,th,tw,sess, mix_thr, min_trials_phase)

alpha_gate = 0.05;
    if numel(y) < min_trials_phase
        lab = "NotTuned"; 
        fh=NaN; 
        fw=NaN; 
        mp=NaN; 
        return;
    end
    Xh = [cosd(th), sind(th)];
    Xw = [cosd(tw), sind(tw)];
    tbl = table(y, Xh(:,1), Xh(:,2), Xw(:,1), Xw(:,2), categorical(sess), ...
                'VariableNames', {'y','h1','h2','w1','w2','sess'});
    try
        mC = fitglme(tbl,'y ~ 1 + (1|sess)','Distribution','Poisson','Link','log','FitMethod','Laplace');
        mF = fitglme(tbl,'y ~ h1+h2+w1+w2 + (1|sess)','Distribution','Poisson','Link','log','FitMethod','Laplace');
        mH = fitglme(tbl,'y ~ h1+h2 + (1|sess)','Distribution','Poisson','Link','log','FitMethod','Laplace');
        mW = fitglme(tbl,'y ~ w1+w2 + (1|sess)','Distribution','Poisson','Link','log','FitMethod','Laplace');
        LR_F = 2*(mF.LogLikelihood - mC.LogLikelihood);
        df_F = max(mF.NumEstimatedCoefficients - mC.NumEstimatedCoefficients, 0);

        if LR_F <= 0 || df_F==0
            pF = 1;
        else
            pF = 1 - chi2cdf(LR_F, df_F);
        end
        if ~(isfinite(pF) && pF < alpha_gate)
            lab = "NotTuned";
            fh=NaN;
            fw=NaN;
            mp=NaN;
            return;
        end

        % base label by AIC (Head vs World)
        base_label = "World";
        if mH.ModelCriterion.AIC < mW.ModelCriterion.AIC, base_label = "Head"; end

        % Model fits and preference
        Dc = -2*mC.LogLikelihood; 
        Df = -2*mF.LogLikelihood;
        Dh = -2*mH.LogLikelihood; 
        Dw = -2*mW.LogLikelihood;
        denom = max(Dc - Df, eps);
        fh = max(Dc - Dh, 0) / denom;
        fw = max(Dc - Dw, 0) / denom;
        mp = fh - fw;

        lab = base_label;
        if abs(mp) <= mix_thr
            lab = "Mixed";
        end
    catch
        lab = "NotTuned";
        fh=NaN; 
        fw=NaN;
        mp=NaN;
    end
end
