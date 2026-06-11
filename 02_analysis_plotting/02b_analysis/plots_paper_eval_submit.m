%% plots_paper_eval_singleBest_style.m
% Standalone evaluation-period paper-style analysis using one evaluated run
% per model x catchment x objective combination.
%
% Purpose
%   This script mirrors the retained-run paper plotting workflow, but replaces
%   the top-N calibration ensemble with a single evaluation-period run.
%
% Interpretation
%   The script assumes eval_index.tsv contains one evaluation result per
%   model x catchment x objective. Under that assumption, this is a
%   calibration-selected-run transfer analysis, not an evaluation top-N
%   equifinality analysis.
%
% Main choices
%   1) Uses calibration retained-run file only for metadata, objective order,
%      benchmark gates, and optional calibration-best comparison.
%   2) Loads evaluation OF and signature outputs from eval_index.tsv.
%   3) Recomputes observed signatures on the evaluation period.
%   4) Builds single-run evaluation arrays with the same dimensional logic as
%      the paper/top-N script where possible.
%   5) Produces evaluation-only and calibration-vs-evaluation figures/tables.
%   6) Keeps of_KGE_neg2_transf removed and uses the original paper objective
%      order: KGE, NSE, KGE0.2, log NSE, DE, KGE-NP, KGE Split, SHE.

clear; close all; clc;

%% USER SETTINGS
base_path       = '<LOCAL_DOWNLOADS>/all_fixed';
eval_path       = '<LOCAL_DOWNLOADS>/eval_fixed';
eval_index_file = '<LOCAL_DOWNLOADS>/final_results_combined_fixed/index/eval_index.tsv';
output_dir      = '<LOCAL_ROOT>/graphics_new_eval_singleBest';
path_nc         = '<LOCAL_ROOT>/ma_thesis/catchments_new';
hydrobm_threshold_file = '<LOCAL_ROOT>/benchmark_results/hydrobm_thresholds.mat';

make_png = true;

% Gate used for the main evaluation signature-analysis plots.
% Options:
%   'calibration' : include combos whose scalar calibration best passed benchmark
%   'evaluation'  : include combos whose evaluation OF passed the benchmark
%   'both'        : require both calibration and evaluation pass
%   'none'        : include any combo with finite evaluation data
benchmark_gate_mode = 'calibration';

addpath(genpath('<LOCAL_ROOT>/TOSSH-master/'))
addpath(genpath('<LOCAL_ROOT>/marrmot_211/'))
addpath(genpath('<LOCAL_ROOT>/ma_thesis/'))
if ~exist(output_dir,'dir'); mkdir(output_dir); end

%% CONSTANTS FROM plots_paper.m
catchments_aridity = {'camelsaus_143110A','camelsbr_60615000','camelsaus_607155','camels_12381400','camels_02017500', ...
    'camelsgb_39037','camels_03460000','hysets_01AF007','camelsgb_27035','lamah_200048'};
catchments_labels = {'AUS1','BR6','AUS6','C12','C02','GB3','C03','HYS','GB2','LAM'};

sorted_signatures = {'sig_TotalRR','sig_EventRR','sig_x_percentile_5per','sig_x_percentile_95per','sig_x_Q_duration_high','sig_x_Q_duration_low', ...
    'sig_x_Q_frequency_high','sig_x_Q_frequency_low','sig_HFD_mean','sig_FDC_slope','sig_VariabilityIndex','sig_BFI','sig_BaseflowRecessionK', ...
    'sig_FlashinessIndex','sig_RisingLimbDensity'};

sorted_sig_new = { ...
    'sig_TotalRR', 'sig_EventRR',  'sig_HFD_mean', ...
    'sig_x_percentile_95per', 'sig_x_Q_frequency_high','sig_x_Q_duration_high', ...
    'sig_x_percentile_5per', 'sig_x_Q_frequency_low','sig_x_Q_duration_low', ...
    'sig_BFI','sig_BaseflowRecessionK','sig_FDC_slope', ...
    'sig_FlashinessIndex', 'sig_VariabilityIndex', 'sig_RisingLimbDensity'};

label_signatures_new = { ...
    'Total RR (-)', 'Event RR (-)',  'MHFD (DOY)', ...
    'Q95 (mm/d)',  'HF Freq (-)','HF Dur (days)', ...
    'Q5 (mm/d)', 'LF Freq (-)','LF Dur (days)', ...
    'BFI (-)', 'BFRC (-)','FDC Slope (-)', ...
    'Flashiness Index (-)', 'Variability Index (-)', 'Rising Limb Density (-)'};

label_signatures_short = { ...
    'TRR','ERR','MHFD', ...
    'Q95','HFF','HFD', ...
    'Q5','LFF','LFD', ...
    'BFI','BFRC','FDC Slope', ...
    'FI','VI','RLD'};

file_sig_names = {'sig_FDC_slope','sig_RisingLimbDensity','sig_BaseflowRecessionK','sig_HFD_mean', ...
    'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex','sig_EventRR','sig_TotalRR', ...
    'sig_x_Q_duration_high','sig_x_Q_duration_low','sig_x_Q_frequency_high','sig_x_Q_frequency_low', ...
    'sig_x_percentile_5per','sig_x_percentile_95per'};

base_colors = NaN(8,3);
base_colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
base_colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
base_colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
base_colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
base_colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
base_colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
base_colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
base_colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];

%% LOAD CALIBRATION METADATA AND BEST CALIBRATION VALUES
fprintf('Loading calibration metadata and retained-run data...\n');
S = load(fullfile(base_path, 'seed_uncertainty_data.mat'));
required = {'of_all','sig_all','catchments','objectives','models'};
for r = 1:numel(required)
    if ~isfield(S, required{r})
        error('seed_uncertainty_data.mat is missing variable: %s', required{r});
    end
end

D.of_all     = double(S.of_all);
D.sig_all    = double(S.sig_all);
D.catchments = cellstr(string(S.catchments(:)))';
D.objectives = cellstr(string(S.objectives(:)))';
D.models     = cellstr(string(S.models(:)))';

% Remove neg2 transformation objective.
neg2_name = 'of_KGE_neg2_transf';
keep_obj = ~strcmp(D.objectives, neg2_name);
if numel(keep_obj) ~= size(D.of_all,3)
    error('Number of objective names (%d) does not match of_all objective dimension (%d).', ...
        numel(keep_obj), size(D.of_all,3));
end
D.objectives = D.objectives(keep_obj);
D.of_all     = D.of_all(:,:,keep_obj,:);
D.sig_all    = D.sig_all(:,:,keep_obj,:,:);

[nM,nC0,nO0,nCand] = size(D.of_all);
if ndims(D.sig_all) ~= 5
    error('Expected sig_all to be 5-D: model x catchment x objective x candidate x signature.');
end

OF_Plot = make_objective_labels_local(D.objectives);
if size(base_colors,1) >= nO0
    colors = base_colors(1:nO0,:);
else
    colors = lines(nO0);
end

% Reorder objectives to paper order.
desired_order_labels = {'KGE','NSE','KGE0.2','log NSE','DE','KGE-NP','KGE Split','SHE'};
new_idx = nan(1, numel(desired_order_labels));
for k = 1:numel(desired_order_labels)
    hit = find(strcmp(OF_Plot, desired_order_labels{k}), 1);
    if isempty(hit)
        warning('Desired objective label "%s" not found in OF_Plot.', desired_order_labels{k});
    else
        new_idx(k) = hit;
    end
end
new_idx = new_idx(~isnan(new_idx));
leftover = setdiff(1:numel(OF_Plot), new_idx, 'stable');
new_idx = [new_idx, leftover];

D.objectives = D.objectives(new_idx);
D.of_all     = D.of_all(:,:,new_idx,:);
D.sig_all    = D.sig_all(:,:,new_idx,:,:);
OF_Plot      = OF_Plot(new_idx);
if size(colors,1) >= max(new_idx)
    colors = colors(new_idx,:);
end
objective_functions = D.objectives;
nO = numel(OF_Plot);

% Align catchments to plots_paper.m order.
[cat_ok, cat_idx_in_D] = ismember(catchments_aridity, D.catchments);
if ~all(cat_ok)
    missing = catchments_aridity(~cat_ok);
    error('These plots_paper catchments are missing from new results: %s', strjoin(missing, ', '));
end
D.of_all     = D.of_all(:,cat_idx_in_D,:,:);
D.sig_all    = D.sig_all(:,cat_idx_in_D,:,:,:);
D.catchments = D.catchments(cat_idx_in_D);
nC = numel(D.catchments);
model_list = D.models;
model_name = model_list;

fprintf('Objectives used in evaluation script:\n');
for oi = 1:nO
    fprintf('  %2d: %-28s -> %s\n', oi, D.objectives{oi}, OF_Plot{oi});
end
fprintf('Loaded calibration retained-run data: %d models, %d catchments, %d objectives, %d retained candidates.\n', ...
    nM, nC, nO, nCand);

% Best calibration values for optional comparison and calibration benchmark gate.
OF_best_cal = squeeze(max(D.of_all, [], 4, 'omitnan')); % [model x catchment x objective]

% Best calibration signatures, paired with the calibration-best candidate.
nSigs = size(D.sig_all,5);
sig_cal_best = nan(nM,nC,nO,nSigs);
for mi = 1:nM
    for ci = 1:nC
        for oi = 1:nO
            vals = squeeze(D.of_all(mi,ci,oi,:));
            [~, best] = max(vals(:));
            if isfinite(vals(best))
                sig_cal_best(mi,ci,oi,:) = D.sig_all(mi,ci,oi,best,:);
            end
        end
    end
end

%% LOAD HYDROBM THRESHOLDS
bm_thr = load(hydrobm_threshold_file);
[obj_ok, obj_idx] = ismember(objective_functions, cellstr(string(bm_thr.objectives(:)))');
[cat_ok2, cat_idx] = ismember(D.catchments, cellstr(string(bm_thr.catchments(:)))');
if ~all(obj_ok)
    error('Objectives missing from hydrobm threshold file: %s', strjoin(objective_functions(~obj_ok), ', '));
end
if ~all(cat_ok2)
    error('Catchments missing from hydrobm threshold file: %s', strjoin(D.catchments(~cat_ok2), ', '));
end
threshold = bm_thr.threshold(obj_idx, cat_idx);   % [objective x catchment]

model_pass_cal = false(nM,nC,nO);
for oi = 1:nO
    for ci = 1:nC
        model_pass_cal(:,ci,oi) = isfinite(OF_best_cal(:,ci,oi)) & OF_best_cal(:,ci,oi) > threshold(oi,ci);
    end
end

%% SIGNATURE NAME MAPPING
if isfield(S, 'file_sig_names')
    stored_sig_names = cellstr(string(S.file_sig_names(:)))';
elseif isfield(S, 'signatures')
    stored_sig_names = cellstr(string(S.signatures(:)))';
elseif isfield(S, 'signature_names')
    stored_sig_names = cellstr(string(S.signature_names(:)))';
else
    stored_sig_names = {};
end

if ~isempty(stored_sig_names)
    if numel(stored_sig_names) ~= size(D.sig_all, 5)
        error('Signature-name vector has %d entries but sig_all has %d signature columns.', ...
            numel(stored_sig_names), size(D.sig_all,5));
    end
    [sig_ok, sig_col_in_D] = ismember(file_sig_names, stored_sig_names);
    if ~all(sig_ok)
        error('These signatures are missing from sig_all: %s', strjoin(file_sig_names(~sig_ok), ', '));
    end
    fprintf('Aligned calibration sig_all by signature name.\n');
else
    warning(['seed_uncertainty_data.mat does not contain a signature-name field. ', ...
             'Falling back to assumed order matching file_sig_names.']);
    sig_col_in_D = 1:numel(file_sig_names);
end

sorted_to_D_col = nan(1, numel(sorted_sig_new));
sorted_to_obs   = nan(1, numel(sorted_sig_new));
for si = 1:numel(sorted_sig_new)
    fsi = find(strcmp(file_sig_names, sorted_sig_new{si}), 1);
    if isempty(fsi)
        warning('Could not map signature %s into file_sig_names; leaving NaN.', sorted_sig_new{si});
        continue;
    end
    sorted_to_obs(si)   = fsi;
    sorted_to_D_col(si) = sig_col_in_D(fsi);
end

%% LOAD EVALUATION DATA
fprintf('Loading evaluation data from eval_index.tsv...\n');

localize_eval = @(p) strrep(char(string(p)), ...
    '/data/horse/ws/<HPC_USER>-marrmot_recal/output_seeded/eval_fixed', ...
    eval_path);

T_eval = readtable(eval_index_file, 'FileType','text', 'Delimiter','\t');
if ismember('status', T_eval.Properties.VariableNames)
    T_eval = T_eval(strcmp(T_eval.status, 'complete'), :);
end
fprintf('  %d complete evaluation rows after status filter.\n', height(T_eval));

of_eval  = nan(nM,nC,nO);
sig_eval = nan(nM,nC,nO,nSigs);

for ri = 1:height(T_eval)
    mi = find(strcmp(D.models,     T_eval.model{ri}),     1);
    ci = find(strcmp(D.catchments, T_eval.catchment{ri}), 1);
    oi = find(strcmp(D.objectives, T_eval.objective{ri}), 1);
    if isempty(mi) || isempty(ci) || isempty(oi)
        continue;
    end

    p_sum = localize_eval(T_eval.eval_summary_csv{ri});
    if isfile(p_sum)
        try
            Ts = readtable(p_sum, 'FileType','text','Delimiter',',', 'VariableNamingRule','preserve');
            if ismember('of_eval', Ts.Properties.VariableNames)
                of_eval(mi,ci,oi) = Ts.('of_eval')(1);
            else
                warning('of_eval column missing in %s', p_sum);
            end
        catch ME
            warning('Could not read eval summary %s: %s', p_sum, ME.message);
        end
    end

    p_sig = localize_eval(T_eval.eval_signatures_csv{ri});
    if isfile(p_sig)
        try
            Ts = readtable(p_sig, 'FileType','text','Delimiter',',');
            % Assumes eval_signatures.csv columns are s1..sN in the same order as D.sig_all.
            for si = 1:nSigs
                col = sprintf('s%d', si);
                if ismember(col, Ts.Properties.VariableNames)
                    sig_eval(mi,ci,oi,si) = Ts.(col)(1);
                end
            end
        catch ME
            warning('Could not read eval signatures %s: %s', p_sig, ME.message);
        end
    end
end

n_eval_of = sum(isfinite(of_eval(:)));
n_eval_sig = sum(isfinite(sig_eval(:)));
fprintf('Loaded finite evaluation OF values: %d / %d.\n', n_eval_of, numel(of_eval));
fprintf('Loaded finite evaluation signature values: %d / %d.\n', n_eval_sig, numel(sig_eval));

%% OBSERVED DATA AND PERIOD-SPECIFIC OBSERVED SIGNATURES
fprintf('Loading observed data and computing calibration/evaluation observed signatures...\n');
Obs = load_observed_data_local(D.catchments, path_nc);

obs_sig_cal  = nan(nC, numel(file_sig_names));
obs_sig_eval = nan(nC, numel(file_sig_names));
period_labels = cell(nC,1);

for ci = 1:nC
    catchment = D.catchments{ci};
    [cal_idx, eval_idx, period_label] = get_benchmark_period_indices_local(catchment, Obs.date_array_full);
    period_labels{ci} = period_label;
    obs_sig_cal(ci,:)  = compute_obs_signature_row_local(Obs.q(:,ci), Obs.precip(:,ci), cal_idx,  file_sig_names);
    obs_sig_eval(ci,:) = compute_obs_signature_row_local(Obs.q(:,ci), Obs.precip(:,ci), eval_idx, file_sig_names);
end

%% SINGLE-RUN EVALUATION ARRAYS
model_pass_eval = false(nM,nC,nO);
for oi = 1:nO
    for ci = 1:nC
        model_pass_eval(:,ci,oi) = isfinite(of_eval(:,ci,oi)) & of_eval(:,ci,oi) > threshold(oi,ci);
    end
end

switch lower(benchmark_gate_mode)
    case 'calibration'
        analysis_mask = model_pass_cal & isfinite(of_eval);
    case 'evaluation'
        analysis_mask = model_pass_eval & isfinite(of_eval);
    case 'both'
        analysis_mask = model_pass_cal & model_pass_eval & isfinite(of_eval);
    case 'none'
        analysis_mask = isfinite(of_eval);
    otherwise
        error('Unknown benchmark_gate_mode: %s', benchmark_gate_mode);
end

model_counter_cal_gate  = squeeze(sum(model_pass_cal, 1));   % [catchment x objective]
model_counter_eval_gate = squeeze(sum(model_pass_eval, 1));  % [catchment x objective]
model_counter_analysis  = squeeze(sum(analysis_mask, 1));    % [catchment x objective]
model_counter_cal_gate  = model_counter_cal_gate';           % [objective x catchment]
model_counter_eval_gate = model_counter_eval_gate';
model_counter_analysis  = model_counter_analysis';

nSigPlot = numel(sorted_sig_new);
OF_Eval_Array             = nan(nO,nC,nM);
Signatures_Eval_Array     = nan(nSigPlot,nO,nM,nC);
Error_Eval_Values         = nan(size(Signatures_Eval_Array));
Signature_eval_median     = nan(size(Signatures_Eval_Array));
OF_eval_single            = nan(nO,nC,nM);

OF_CalBest_Array          = nan(nO,nC,nM);
Signatures_CalBest_Array  = nan(nSigPlot,nO,nM,nC);
Error_CalBest_Values      = nan(size(Signatures_CalBest_Array));

for si = 1:nSigPlot
    d_col   = sorted_to_D_col(si);
    obs_col = sorted_to_obs(si);
    if isnan(d_col) || isnan(obs_col); continue; end

    for ci = 1:nC
        obs_eval_val = obs_sig_eval(ci, obs_col);
        obs_cal_val  = obs_sig_cal(ci, obs_col);

        for oi = 1:nO
            for mi = 1:nM
                if analysis_mask(mi,ci,oi)
                    val_of  = of_eval(mi,ci,oi);
                    val_sig = sig_eval(mi,ci,oi,d_col);
                    if isfinite(val_of) && isfinite(val_sig)
                        OF_Eval_Array(oi,ci,mi) = val_of;
                        OF_eval_single(oi,ci,mi) = val_of;
                        Signatures_Eval_Array(si,oi,mi,ci) = val_sig;
                        Error_Eval_Values(si,oi,mi,ci) = val_sig - obs_eval_val;
                        Signature_eval_median(si,oi,mi,ci) = val_sig;
                    end
                end

                % Calibration-best arrays for side-by-side degradation comparisons.
                if model_pass_cal(mi,ci,oi)
                    val_cal_of  = OF_best_cal(mi,ci,oi);
                    val_cal_sig = sig_cal_best(mi,ci,oi,d_col);
                    if isfinite(val_cal_of) && isfinite(val_cal_sig)
                        OF_CalBest_Array(oi,ci,mi) = val_cal_of;
                        Signatures_CalBest_Array(si,oi,mi,ci) = val_cal_sig;
                        Error_CalBest_Values(si,oi,mi,ci) = val_cal_sig - obs_cal_val;
                    end
                end
            end
        end
    end
end

%% NORMALIZED ERROR SUMMARIES
[Error_Eval_Norm_Values, Error_Eval_Norm_Values_range, Rank_Eval_values] = ...
    compute_single_run_error_summaries(Signature_eval_median, obs_sig_eval, sorted_to_obs, nO, nC);

% Calibration-best equivalent, useful for direct transfer degradation.
[Error_CalBest_Norm_Values, Error_CalBest_Norm_Values_range, Rank_CalBest_values] = ...
    compute_single_run_error_summaries(Signatures_CalBest_Array, obs_sig_cal, sorted_to_obs, nO, nC);

%% FIGURE 1: MODEL COUNTS FOR CALIBRATION/EVALUATION/ANALYSIS GATES
f = figure('units','normalized','outerposition',[0 0 0.9 0.45]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
bh = bar(model_counter_cal_gate'); set(bh, 'FaceColor', 'Flat');
for k = 1:min(nO,8); bh(k).CData = colors(k,:); end
yline(nM,'-','All models');
set(gca,'XTickLabel',catchments_labels); xtickangle(45);
title('Calibration-best benchmark pass'); ylabel('Number of models'); grid on;

nexttile;
bh = bar(model_counter_eval_gate'); set(bh, 'FaceColor', 'Flat');
for k = 1:min(nO,8); bh(k).CData = colors(k,:); end
yline(nM,'-','All models');
set(gca,'XTickLabel',catchments_labels); xtickangle(45);
title('Evaluation benchmark pass'); grid on;

nexttile;
bh = bar(model_counter_analysis'); set(bh, 'FaceColor', 'Flat');
for k = 1:min(nO,8); bh(k).CData = colors(k,:); end
yline(nM,'-','All models');
set(gca,'XTickLabel',catchments_labels); xtickangle(45);
title(sprintf('Analysis mask: %s gate', benchmark_gate_mode)); grid on;
legend(strrep(OF_Plot,'_','\_'),'Location','eastoutside');

fontsize(f,12,'points');
save_if_requested(f, output_dir, sprintf('eval_singleBest_model_counts_%sGate', benchmark_gate_mode), make_png);

%% FIGURE 2: EVALUATION OF VALUES BY CATCHMENT
f = figure('units','normalized','outerposition',[0 0 1 0.8]);
tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
for oi = 1:nO
    nexttile; hold on;
    X = [];
    G = [];
    for ci = 1:nC
        vals = squeeze(OF_Eval_Array(oi,ci,:));
        vals = finite_real_vector_local(vals);
        X = [X; vals]; %#ok<AGROW>
        G = [G; ci*ones(numel(vals),1)]; %#ok<AGROW>
    end
    if ~isempty(X)
        safe_violinplot_local(X, G, colors(oi,:));
    end
    for ci = 1:nC
        line([ci-0.35 ci+0.35], [threshold(oi,ci) threshold(oi,ci)], ...
            'Color','r','LineStyle','-','LineWidth',1.4);
    end
    ylim([-0.5 1]); xlim([0.5 nC+0.5]);
    title(OF_Plot{oi});
    set(gca,'XTick',1:nC,'XTickLabel',catchments_labels); xtickangle(45);
    grid minor;
end
fontsize(f,13,'points');
save_if_requested(f, output_dir, sprintf('eval_singleBest_OF_values_%sGate', benchmark_gate_mode), make_png);

%% FIGURE 3: CALIBRATION VS EVALUATION OF SCATTER
f = figure('units','normalized','outerposition',[0 0 1 0.8]);
tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
for oi = 1:nO
    nexttile; hold on;
    cal_vals  = squeeze(OF_best_cal(:,:,oi));
    eval_vals = squeeze(of_eval(:,:,oi));
    for ci = 1:nC
        valid = isfinite(cal_vals(:,ci)) & isfinite(eval_vals(:,ci));
        scatter(cal_vals(valid,ci), eval_vals(valid,ci), 22, 'filled', ...
            'MarkerFaceAlpha',0.55, 'DisplayName', catchments_labels{ci});
    end
    ax_lim = [-0.5 1];
    plot(ax_lim, ax_lim, 'k--', 'LineWidth',1);
    xlim(ax_lim); ylim(ax_lim);
    xlabel('Calibration best OF'); ylabel('Evaluation OF');
    title(OF_Plot{oi}); grid on;
    if oi == 1
        legend('Location','southeast','FontSize',7,'NumColumns',2);
    end
end
sgtitle('Calibration-best vs evaluation OF');
fontsize(f,11,'points');
save_if_requested(f, output_dir, 'eval_singleBest_OF_scatter_cal_vs_eval', make_png);

%% FIGURE 4: SIGNATURE COMPARISON WITH SINGLE-RUN EVALUATION VIOLINS
yrange_min = [0,0,80, 0,0,0, 0,0,0, 0,0,-25, 0,0,0];
yrange_max = [1,1,320, 10,0.5,80, 1.5,1,80, 1,0.7,0, 1,1.5,1];
box_color = [0.7 0.7 0.7]; observed_color = [0 0 0];

f = figure('Units','normalized','OuterPosition',[0 0 0.6 1]);
tiledlayout(f,5,3,'TileSpacing','compact','Padding','tight');
h_obj = gobjects(1,nO);
h_violin = [];
h_obs = [];

for si = 1:nSigPlot
    nexttile; hold on;
    all_data = [];
    group_labels = [];
    for oi = 1:nO
        plot_data = squeeze(Signatures_Eval_Array(si,oi,:,:)); % [model x catchment]
        for ci = 1:nC
            vals = finite_real_vector_local(plot_data(:,ci));
            all_data = [all_data; vals]; %#ok<AGROW>
            group_labels = [group_labels; ci*ones(numel(vals),1)]; %#ok<AGROW>
        end
    end
    if ~isempty(all_data)
        safe_violinplot_local(all_data, group_labels, box_color);
        if isempty(h_violin)
            h_violin = patch(NaN, NaN, box_color, 'EdgeColor', [0.4 0.4 0.4]);
        end
    end

    for oi = 1:nO
        med_by_c = squeeze(median(Signatures_Eval_Array(si,oi,:,:), 3, 'omitnan'));
        hs = scatter(1:nC, med_by_c, 46, colors(oi,:), 'filled');
        if si == 1; h_obj(oi) = hs; end
    end

    obs_col = sorted_to_obs(si);
    for ci = 1:nC
        observed_value = obs_sig_eval(ci,obs_col);
        if isfinite(observed_value)
            hl = line([ci-0.4, ci+0.4], [observed_value, observed_value], ...
                'Color', observed_color, 'LineWidth',2, 'LineStyle','-');
            if isempty(h_obs); h_obs = hl; end
        end
    end

    if si <= numel(yrange_min)
        ylim([yrange_min(si) yrange_max(si)]);
    end
    title(label_signatures_new{si}); grid minor;
    if si > 12
        set(gca,'XTick',1:nC,'XTickLabel',catchments_labels); xtickangle(45);
    else
        set(gca,'XTick',1:nC,'XTickLabel',[]);
    end
    xlim([0.5 nC+0.5]);
end
legend_handles = [h_obj, h_violin, h_obs];
legend_labels  = [OF_Plot, {'Single evaluated run'}, {'Observed eval'}];
valid = arrayfun(@(h) ~isempty(h) && isgraphics(h), legend_handles);
lgd = legend(legend_handles(valid), legend_labels(valid), 'NumColumns',5, 'Orientation','horizontal');
lgd.Layout.Tile = 'south';
fontsize(f,12,'points');
save_if_requested(f, output_dir, sprintf('eval_singleBest_signature_comparison_%sGate', benchmark_gate_mode), make_png);

%% FIGURE 5: NORMALIZED EVALUATION SIGNATURE ERROR PER OBJECTIVE
f = figure('units','normalized','outerposition',[0 0 0.75 0.5]); hold on;
for oi = 1:nO
    vals = squeeze(Error_Eval_Norm_Values_range(:,:,oi));
    vals = finite_real_vector_local(vals(:));
    if isempty(vals); continue; end
    safe_violinplot_local(vals, oi*ones(size(vals)), colors(oi,:));
end
yline(0,'k-');
set(gca,'XTick',1:nO,'XTickLabel',OF_Plot); xtickangle(45);
ylabel('Normalized median evaluation signature error');
grid minor;
title(sprintf('Evaluation-period normalized signature error (%s gate)', benchmark_gate_mode));
fontsize(f,14,'points');
save_if_requested(f, output_dir, sprintf('eval_singleBest_normalized_overall_violin_%sGate', benchmark_gate_mode), make_png);

%% FIGURE 6: HEATMAP OF MEAN ABS NORMALIZED EVALUATION ERROR
f = figure('units','normalized','outerposition',[0 0 0.65 0.75]);
heat = squeeze(mean(abs(Error_Eval_Norm_Values_range), 2, 'omitnan')); % [signature x objective]
imagesc(heat);
colormap(greenCenteredColormap(256)); colorbar;
set(gca,'XTick',1:nO,'XTickLabel',OF_Plot,'YTick',1:numel(label_signatures_new),'YTickLabel',label_signatures_new);
xtickangle(45); xlabel('Objective function'); ylabel('Signature');
title('Evaluation mean absolute normalized signature error');
fontsize(f,12,'points');
save_if_requested(f, output_dir, sprintf('eval_singleBest_normalized_error_heatmap_%sGate', benchmark_gate_mode), make_png);

%% FIGURE 7: CALIBRATION VS EVALUATION SIGNATURE ERROR CHANGE
% Positive = evaluation has larger absolute normalized error than calibration.
delta_abs_norm = abs(Error_Eval_Norm_Values_range) - abs(Error_CalBest_Norm_Values_range);

f = figure('units','normalized','outerposition',[0 0 1 0.6]); hold on;
X = [];
G = [];
for si = 1:nSigPlot
    vals = squeeze(delta_abs_norm(si,:,:));
    vals = finite_real_vector_local(vals(:));
    X = [X; vals]; %#ok<AGROW>
    G = [G; si*ones(numel(vals),1)]; %#ok<AGROW>
end
if ~isempty(X)
    safe_violinplot_local(X, G, [0.55 0.55 0.55]);
end
yline(0,'r--','LineWidth',1.5);
set(gca,'XTick',1:nSigPlot,'XTickLabel',label_signatures_short); xtickangle(45);
ylabel('Delta |normalized signature error|: eval - cal');
title('Signature-error degradation from calibration to evaluation');
grid on;
fontsize(f,12,'points');
save_if_requested(f, output_dir, sprintf('eval_singleBest_sig_error_delta_%sGate', benchmark_gate_mode), make_png);


%% =====================================================================
%  SUMMARY FIGURE H: PAIRED SIGNIFICANCE TEST BETWEEN OBJECTIVE FUNCTIONS
%  Calibration-best and evaluation single-run versions.
%
%  The calibration p-values define the signature filter used downstream in
%  Figure C/D/E.  Evaluation p-values are computed and plotted separately
%  for period comparison, but they do NOT redefine the filter.
%  =====================================================================

pval_alpha = 0.10;
nSig_sum = numel(sorted_sig_new);
[~, ~, nO_check] = size(Error_CalBest_Norm_Values);
nO_h = nO_check;

% Build list of unordered objective pairs.
pair_list = [];
for ii = 1:nO_h
    for jj = ii+1:nO_h
        pair_list = [pair_list; ii jj]; %#ok<AGROW>
    end
end
nPairs = size(pair_list, 1);

% Storage: calibration-best period.
save_p_value_cal  = nan(nSig_sum, nPairs);
save_t_value_cal  = nan(nSig_sum, nPairs);
save_mean_dif_cal = nan(nSig_sum, nPairs);
save_n_paired_cal = nan(nSig_sum, nPairs);

% Storage: evaluation period.
save_p_value_eval  = nan(nSig_sum, nPairs);
save_t_value_eval  = nan(nSig_sum, nPairs);
save_mean_dif_eval = nan(nSig_sum, nPairs);
save_n_paired_eval = nan(nSig_sum, nPairs);

for si = 1:nSig_sum
    for pi = 1:nPairs
        oi = pair_list(pi, 1);
        oj = pair_list(pi, 2);

        % ---- calibration-best paired test ----
        x = squeeze(Error_CalBest_Norm_Values(si, :, oi))';
        y = squeeze(Error_CalBest_Norm_Values(si, :, oj))';
        ok = isfinite(x) & isfinite(y);
        if sum(ok) >= 2
            try
                [~, p_tmp, ~, stats_tmp] = ttest(x(ok), y(ok));
                save_p_value_cal(si, pi)  = p_tmp;
                save_t_value_cal(si, pi)  = stats_tmp.tstat;
                save_mean_dif_cal(si, pi) = mean(x(ok) - y(ok), 'omitnan');
                save_n_paired_cal(si, pi) = sum(ok);
            catch ME
                warning('Calibration ttest failed sig %d pair (%d,%d): %s', si, oi, oj, ME.message);
            end
        end

        % ---- evaluation paired test ----
        x = squeeze(Error_Eval_Norm_Values(si, :, oi))';
        y = squeeze(Error_Eval_Norm_Values(si, :, oj))';
        ok = isfinite(x) & isfinite(y);
        if sum(ok) >= 2
            try
                [~, p_tmp, ~, stats_tmp] = ttest(x(ok), y(ok));
                save_p_value_eval(si, pi)  = p_tmp;
                save_t_value_eval(si, pi)  = stats_tmp.tstat;
                save_mean_dif_eval(si, pi) = mean(x(ok) - y(ok), 'omitnan');
                save_n_paired_eval(si, pi) = sum(ok);
            catch ME
                warning('Evaluation ttest failed sig %d pair (%d,%d): %s', si, oi, oj, ME.message);
            end
        end
    end
end

%% ---- Figure H1: calibration p-value violin plot ----
fH_cal = figure('units','normalized','outerposition',[0 0 0.85 0.55]);
hold on;
X = [];
G = [];
for si = 1:nSig_sum
    vals = save_p_value_cal(si, :)';
    vals = vals(isfinite(vals));
    if isempty(vals)
        X = [X; NaN]; G = [G; si]; %#ok<AGROW>
    else
        X = [X; vals]; %#ok<AGROW>
        G = [G; si*ones(numel(vals),1)]; %#ok<AGROW>
    end
end
try
    vp = violinplot(X, G, 'ShowMean', false, 'ShowData', true);
    for si = 1:numel(vp)
        if ~isempty(vp(si).ViolinPlot) && isvalid(vp(si).ViolinPlot)
            vp(si).ViolinPlot.FaceColor = [0.25 0.25 0.25];
            vp(si).ViolinPlot.FaceAlpha = 0.6;
        end
        if isprop(vp(si), 'ScatterPlot') && ~isempty(vp(si).ScatterPlot) && isvalid(vp(si).ScatterPlot)
            vp(si).ScatterPlot.SizeData = 12;
            vp(si).ScatterPlot.MarkerFaceAlpha = 0.5;
            vp(si).ScatterPlot.MarkerFaceColor = [0.25 0.25 0.25];
        end
    end
catch ME
    warning( ME.message,'Calibration p-value violinplot failed: %s');
    for si = 1:nSig_sum
        vals = save_p_value_cal(si, :)';
        vals = vals(isfinite(vals));
        if isempty(vals); continue; end
        jitter = (rand(size(vals))-0.5) * 0.25;
        scatter(si + jitter, vals, 14, [0.25 0.25 0.25], 'filled', 'MarkerFaceAlpha', 0.5);
    end
end
yline(0.05, 'r--', 'p = 0.05', 'LabelHorizontalAlignment','left', 'LineWidth', 2);
yline(0.10, 'k--', 'p = 0.10', 'LabelHorizontalAlignment','left', 'LineWidth', 2);
ylim([0 1]); xlim([0.5 nSig_sum + 0.5]);
set(gca, 'XTick', 1:nSig_sum, 'XTickLabel', label_signatures_new);
xtickangle(45);
ylabel('p-value (paired t-test between OF pairs)');
xlabel('Signature');
title(sprintf('Calibration-best objective-function differences per signature\n%d OF pairs per signature; %d catchments paired', nPairs, nC));
grid on;
fontsize(fH_cal, 13, 'points');
save_if_requested(fH_cal, output_dir, sprintf('cal_singleBest_summary_pvalue_violin_%sGate', benchmark_gate_mode), make_png);

%% ---- Figure H1b: evaluation p-value violin plot ----
fH_eval = figure('units','normalized','outerposition',[0 0 0.85 0.55]);
hold on;
X = [];
G = [];
for si = 1:nSig_sum
    vals = save_p_value_eval(si, :)';
    vals = vals(isfinite(vals));
    if isempty(vals)
        X = [X; NaN]; G = [G; si]; %#ok<AGROW>
    else
        X = [X; vals]; %#ok<AGROW>
        G = [G; si*ones(numel(vals),1)]; %#ok<AGROW>
    end
end
try
    vp = violinplot(X, G, 'ShowMean', false, 'ShowData', true);
    for si = 1:numel(vp)
        if ~isempty(vp(si).ViolinPlot) && isvalid(vp(si).ViolinPlot)
            vp(si).ViolinPlot.FaceColor = [0.25 0.25 0.25];
            vp(si).ViolinPlot.FaceAlpha = 0.6;
        end
        if isprop(vp(si), 'ScatterPlot') && ~isempty(vp(si).ScatterPlot) && isvalid(vp(si).ScatterPlot)
            vp(si).ScatterPlot.SizeData = 12;
            vp(si).ScatterPlot.MarkerFaceAlpha = 0.5;
            vp(si).ScatterPlot.MarkerFaceColor = [0.25 0.25 0.25];
        end
    end
catch ME
    warning(ME.message,'Evaluation p-value violinplot failed: %s');
    for si = 1:nSig_sum
        vals = save_p_value_eval(si, :)';
        vals = vals(isfinite(vals));
        if isempty(vals); continue; end
        jitter = (rand(size(vals))-0.5) * 0.25;
        scatter(si + jitter, vals, 14, [0.25 0.25 0.25], 'filled', 'MarkerFaceAlpha', 0.5);
    end
end
yline(0.05, 'r--', 'p = 0.05', 'LabelHorizontalAlignment','left', 'LineWidth', 2);
yline(0.10, 'k--', 'p = 0.10', 'LabelHorizontalAlignment','left', 'LineWidth', 2);
ylim([0 1]); xlim([0.5 nSig_sum + 0.5]);
set(gca, 'XTick', 1:nSig_sum, 'XTickLabel', label_signatures_new);
xtickangle(45);
ylabel('p-value (paired t-test between OF pairs)');
xlabel('Signature');
title(sprintf('Evaluation objective-function differences per signature\n%d OF pairs per signature; %d catchments paired', nPairs, nC));
grid on;
fontsize(fH_eval, 13, 'points');
save_if_requested(fH_eval, output_dir, sprintf('eval_singleBest_summary_pvalue_violin_%sGate', benchmark_gate_mode), make_png);

%% ---- Overview tables: median p, fraction < 0.10 and < 0.05 ----
median_p_cal      = median(save_p_value_cal, 2, 'omitnan');
frac_p_cal_lt_010 = mean(save_p_value_cal < 0.10, 2, 'omitnan');
frac_p_cal_lt_005 = mean(save_p_value_cal < 0.05, 2, 'omitnan');
median_p_eval      = median(save_p_value_eval, 2, 'omitnan');
frac_p_eval_lt_010 = mean(save_p_value_eval < 0.10, 2, 'omitnan');
frac_p_eval_lt_005 = mean(save_p_value_eval < 0.05, 2, 'omitnan');

T_p_overview_cal = table(median_p_cal, frac_p_cal_lt_010, frac_p_cal_lt_005, ...
    'VariableNames', {'median_p','frac_p_lt_0_10','frac_p_lt_0_05'}, ...
    'RowNames', label_signatures_new(:));
T_p_overview_eval = table(median_p_eval, frac_p_eval_lt_010, frac_p_eval_lt_005, ...
    'VariableNames', {'median_p','frac_p_lt_0_10','frac_p_lt_0_05'}, ...
    'RowNames', label_signatures_new(:));

T_p_overview_caleval = table(label_signatures_new(:), median_p_cal, median_p_eval, ...
    frac_p_cal_lt_010, frac_p_eval_lt_010, frac_p_cal_lt_005, frac_p_eval_lt_005, ...
    'VariableNames', {'signature','median_p_cal','median_p_eval', ...
    'frac_cal_p_lt_0_10','frac_eval_p_lt_0_10','frac_cal_p_lt_0_05','frac_eval_p_lt_0_05'});

writetable(T_p_overview_cal, fullfile(output_dir, sprintf('cal_singleBest_summary_pvalue_overview_%sGate.csv', benchmark_gate_mode)), 'WriteRowNames', true);
writetable(T_p_overview_eval, fullfile(output_dir, sprintf('eval_singleBest_summary_pvalue_overview_%sGate.csv', benchmark_gate_mode)), 'WriteRowNames', true);
writetable(T_p_overview_caleval, fullfile(output_dir, sprintf('caleval_singleBest_summary_pvalue_overview_%sGate.csv', benchmark_gate_mode)));

%% ---- Figure H2: heatmaps of which OF-pairs are significantly different ----
pair_frac_sig_cal = mean(save_p_value_cal < 0.05, 1, 'omitnan');
pair_frac_sig_eval = mean(save_p_value_eval < 0.05, 1, 'omitnan');

pair_matrix_cal = nan(nO_h, nO_h);
pair_matrix_eval = nan(nO_h, nO_h);
for pi = 1:nPairs
    ii = pair_list(pi,1);
    jj = pair_list(pi,2);
    pair_matrix_cal(ii, jj) = pair_frac_sig_cal(pi);
    pair_matrix_cal(jj, ii) = pair_frac_sig_cal(pi);
    pair_matrix_eval(ii, jj) = pair_frac_sig_eval(pi);
    pair_matrix_eval(jj, ii) = pair_frac_sig_eval(pi);
end

fH2 = figure('units','normalized','outerposition',[0 0 0.95 0.55]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile;
imagesc(pair_matrix_cal, [0 1]); axis square; colormap(parula); colorbar;
set(gca, 'XTick', 1:nO_h, 'XTickLabel', OF_Plot, 'YTick', 1:nO_h, 'YTickLabel', OF_Plot);
xtickangle(45); title('Calibration-best');
for ii = 1:nO_h
    for jj = 1:nO_h
        if isfinite(pair_matrix_cal(ii,jj))
            text(jj, ii, sprintf('%.2f', pair_matrix_cal(ii,jj)), 'HorizontalAlignment','center', 'FontSize',9, 'Color',[0 0 0]);
        end
    end
end
nexttile;
imagesc(pair_matrix_eval, [0 1]); axis square; colormap(parula); cb = colorbar;
cb.Label.String = 'Fraction of signatures with p < 0.05';
set(gca, 'XTick', 1:nO_h, 'XTickLabel', OF_Plot, 'YTick', 1:nO_h, 'YTickLabel', OF_Plot);
xtickangle(45); title('Evaluation');
for ii = 1:nO_h
    for jj = 1:nO_h
        if isfinite(pair_matrix_eval(ii,jj))
            text(jj, ii, sprintf('%.2f', pair_matrix_eval(ii,jj)), 'HorizontalAlignment','center', 'FontSize',9, 'Color',[0 0 0]);
        end
    end
end
fontsize(fH2, 12, 'points');
save_if_requested(fH2, output_dir, sprintf('caleval_singleBest_summary_pvalue_OFpair_heatmap_%sGate', benchmark_gate_mode), make_png);

%% ---- Significance gate for downstream filtering (used by Figure C/D/E) ----
% Critical: this uses calibration-best p-values only, so the same signature
% subset is applied to both calibration and evaluation aggregate comparisons.
sig_passes = false(1, nSig_sum);
for si = 1:nSig_sum
    sig_passes(si) = median(save_p_value_cal(si, :), 'omitnan') < pval_alpha;
end

fprintf('\nSignatures passing CALIBRATION significance gate (median p across %d OF pairs < %.3f):\n', nPairs, pval_alpha);
for si = 1:nSig_sum
    if sig_passes(si)
        status = 'KEEP';
    else
        status = 'drop';
    end
    fprintf('  %-30s %s  (cal median p = %.4f, eval median p = %.4f)\n', ...
        label_signatures_new{si}, status, median_p_cal(si), median_p_eval(si));
end

T_cal_filter = table(label_signatures_new(:), median_p_cal, median_p_eval, sig_passes(:), ...
    'VariableNames', {'signature','median_p_cal','median_p_eval','used_in_aggregate'});
writetable(T_cal_filter, fullfile(output_dir, sprintf('caleval_singleBest_calibration_filter_pvalues_%sGate.csv', benchmark_gate_mode)));

%% SUMMARY FIGURE C: STACKED HORIZONTAL BAR OF CUMULATIVE |NORMALIZED ERROR|
% Filtered to signatures where calibration-period objective-function
% differences are statistically significant.  This reproduces the original
% Figure C logic, then applies the same retained signatures to evaluation.

sig_idx_keep = find(sig_passes);
nSig_keep = numel(sig_idx_keep);

if nSig_keep == 0
    warning('No signatures passed the calibration significance gate; skipping Figure C/D/E aggregate plots.');
else
    median_abs_err_per_obj_sig_cal  = nan(nO, nSig_keep);
    median_abs_err_per_obj_sig_eval = nan(nO, nSig_keep);

    for oi = 1:nO
        for k = 1:nSig_keep
            si = sig_idx_keep(k);
            median_abs_err_per_obj_sig_cal(oi,k)  = median(abs(squeeze(Error_CalBest_Norm_Values_range(si,:,oi))), 'omitnan');
            median_abs_err_per_obj_sig_eval(oi,k) = median(abs(squeeze(Error_Eval_Norm_Values_range(si,:,oi))), 'omitnan');
        end
    end

    graydient = flipud(repmat(linspace(0.2, 0.9, nSig_keep)', 1, 3));

    % Evaluation-only direct analogue of the original Figure C.
    fC_eval = figure('units','normalized','outerposition',[0 0 0.5 0.7]);
    bh = barh(median_abs_err_per_obj_sig_eval, 'stacked', 'LineWidth', 1.0);
    for k = 1:nSig_keep
        bh(k).FaceColor = graydient(k,:);
        bh(k).EdgeColor = [0.25 0.25 0.25];
    end
    set(gca,'YDir','reverse','YTick',1:nO,'YTickLabel',OF_Plot);
    ylabel('Objective functions');
    xlabel('Cumulative median |normalized signature error|');
    legend(label_signatures_short(sig_idx_keep), 'Location','eastoutside');
    grid on;
    title('Evaluation cumulative normalized error per objective function');
    fontsize(fC_eval, 12, 'points');
    save_if_requested(fC_eval, output_dir, sprintf('eval_singleBest_summary_stacked_bar_normerr_%sGate', benchmark_gate_mode), make_png);

    % Calibration vs evaluation stacked comparison with the same signature segments.
    fC_cmp = figure('units','normalized','outerposition',[0 0 0.9 0.7]);
    tlC = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile; hold on;
    bh1 = barh(median_abs_err_per_obj_sig_cal, 'stacked', 'LineWidth', 1.0);
    for k = 1:nSig_keep
        bh1(k).FaceColor = graydient(k,:);
        bh1(k).EdgeColor = [0.25 0.25 0.25];
    end
    set(gca,'YDir','reverse','YTick',1:nO,'YTickLabel',OF_Plot);
    ylabel('Objective functions');
    xlabel('Cumulative median |normalized signature error|');
    title('Calibration best'); grid on;

    nexttile; hold on;
    bh2 = barh(median_abs_err_per_obj_sig_eval, 'stacked', 'LineWidth', 1.0);
    for k = 1:nSig_keep
        bh2(k).FaceColor = graydient(k,:);
        bh2(k).EdgeColor = [0.25 0.25 0.25];
    end
    set(gca,'YDir','reverse','YTick',1:nO,'YTickLabel',OF_Plot);
    xlabel('Cumulative median |normalized signature error|');
    title('Evaluation'); grid on;

    lgd = legend(bh2, label_signatures_short(sig_idx_keep), 'Orientation','horizontal', 'NumColumns', min(5,nSig_keep));
    lgd.Layout.Tile = 'south';
    title(tlC, sprintf('Aggregated normalized signature error using calibration significance filter (alpha = %.2f)', pval_alpha));
    fontsize(fC_cmp, 12, 'points');
    save_if_requested(fC_cmp, output_dir, sprintf('caleval_singleBest_summary_stacked_bar_normerr_%sGate', benchmark_gate_mode), make_png);

    % Export exact stacked-bar values.
    T_stack_cal = array2table(median_abs_err_per_obj_sig_cal, ...
        'VariableNames', matlab.lang.makeValidName(label_signatures_short(sig_idx_keep), 'ReplacementStyle','delete'), ...
        'RowNames', matlab.lang.makeValidName(OF_Plot));
    T_stack_eval = array2table(median_abs_err_per_obj_sig_eval, ...
        'VariableNames', matlab.lang.makeValidName(label_signatures_short(sig_idx_keep), 'ReplacementStyle','delete'), ...
        'RowNames', matlab.lang.makeValidName(OF_Plot));
    writetable(T_stack_cal, fullfile(output_dir, sprintf('caleval_singleBest_stacked_bar_values_calibration_%sGate.csv', benchmark_gate_mode)), 'WriteRowNames', true);
    writetable(T_stack_eval, fullfile(output_dir, sprintf('caleval_singleBest_stacked_bar_values_evaluation_%sGate.csv', benchmark_gate_mode)), 'WriteRowNames', true);

    %% SUMMARY FIGURE D: LINE PLOT OF CUMULATIVE NORMALIZED ERROR PER OBJECTIVE
    cum_err_cal = zeros(nSig_keep+1, nO);
    cum_err_eval = zeros(nSig_keep+1, nO);
    for k = 1:nSig_keep
        si = sig_idx_keep(k);
        for oi = 1:nO
            cum_err_cal(k+1,oi)  = cum_err_cal(k,oi)  + median(abs(squeeze(Error_CalBest_Norm_Values_range(si,:,oi))), 'omitnan');
            cum_err_eval(k+1,oi) = cum_err_eval(k,oi) + median(abs(squeeze(Error_Eval_Norm_Values_range(si,:,oi))), 'omitnan');
        end
    end

    fD = figure('units','normalized','outerposition',[0 0 0.95 0.65]);
    tlD = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile; hold on;
    for oi = 1:nO
        plot(0:nSig_keep, cum_err_cal(:,oi), '-', 'Color', colors(oi,:), 'LineWidth', 2.5);
    end
    set(gca,'XTick',0:nSig_keep,'XTickLabel',[{''}, label_signatures_short(sig_idx_keep)]);
    xtickangle(45); xlim([0 nSig_keep]);
    ylabel('Cumulative median |normalized signature error|');
    xlabel('Signatures retained by calibration filter');
    title('Calibration best'); grid on;

    nexttile; hold on;
    for oi = 1:nO
        plot(0:nSig_keep, cum_err_eval(:,oi), '-', 'Color', colors(oi,:), 'LineWidth', 2.5);
    end
    set(gca,'XTick',0:nSig_keep,'XTickLabel',[{''}, label_signatures_short(sig_idx_keep)]);
    xtickangle(45); xlim([0 nSig_keep]);
    xlabel('Signatures retained by calibration filter');
    title('Evaluation'); grid on;
    legend(OF_Plot, 'Location','northwest');

    title(tlD, 'Cumulative aggregate error comparison');
    fontsize(fD, 12, 'points');
    save_if_requested(fD, output_dir, sprintf('caleval_singleBest_summary_lineplot_cumulative_normerr_%sGate', benchmark_gate_mode), make_png);

    %% SUMMARY FIGURE E: BAR CHART OF OVERALL AGGREGATED ERROR PER OBJECTIVE
    overall_cal = sum(median_abs_err_per_obj_sig_cal, 2, 'omitnan');
    overall_eval = sum(median_abs_err_per_obj_sig_eval, 2, 'omitnan');

    fE = figure('units','normalized','outerposition',[0 0 0.5 0.55]);
    hb = bar([overall_cal overall_eval], 'grouped');
    hb(1).FaceColor = [0.4 0.4 0.4];
    hb(2).FaceColor = [0.75 0.75 0.75];
    set(gca,'XTick',1:nO,'XTickLabel',OF_Plot);
    xtickangle(45);
    ylabel('Cumulative median |normalized signature error|');
    legend({'Calibration best','Evaluation'}, 'Location','best');
    grid on;
    title(sprintf('Overall aggregated normalized error (%s gate)', benchmark_gate_mode));
    fontsize(fE, 12, 'points');
    save_if_requested(fE, output_dir, sprintf('caleval_singleBest_summary_overall_bar_normerr_%sGate', benchmark_gate_mode), make_png);

    T_aggregate = table(OF_Plot(:), overall_cal, overall_eval, overall_eval - overall_cal, overall_eval ./ overall_cal, ...
        'VariableNames', {'objective','aggregate_calibration','aggregate_evaluation','delta_eval_minus_cal','ratio_eval_to_cal'});
    writetable(T_aggregate, fullfile(output_dir, sprintf('caleval_singleBest_aggregate_normerr_by_objective_%sGate.csv', benchmark_gate_mode)));
end

%% FIGURE 8: PER-SIGNATURE OBJECTIVE PANELS OF EVALUATION ERROR
yrange_values = [0.3,0.3,50,5,0.6,80,1.5,1,80,1,0.7,25,1,1.5,1];
signature_list_plot_sorted = {'Total RR','Event RR','Mean Half Flow Date (DOY)', ...
    'Q95 (mm/d)','High Flow Frequency','High Flow Duration (d)', ...
    'Q5 (mm/d)','Low Flow Frequency','Low Flow Duration (d)', ...
    'Baseflow Index','BF Recession Coefficient (1/d)','FDC Slope', ...
    'Flashiness Index','Variability Index','Rising Limb Density (1/d)'};

for si = 1:nSigPlot
    f = figure('units','normalized','outerposition',[0 0 0.55 0.6]);
    tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

    for oi = 1:nO
        nexttile; hold on;
        plot_data = squeeze(Error_Eval_Values(si,oi,:,:)); % [model x catchment]
        X = [];
        G = [];
        n_per_catch = zeros(1,nC);
        for ci = 1:nC
            vals = finite_real_vector_local(plot_data(:,ci));
            n_per_catch(ci) = numel(vals);
            X = [X; vals]; %#ok<AGROW>
            G = [G; ci*ones(numel(vals),1)]; %#ok<AGROW>
        end
        if ~isempty(X)
            plot_fixed_position_violins(X, G, 1:nC, colors(oi,:));
        end
        yline(0,'k-','LineWidth',1.0);
        title(strrep(OF_Plot{oi},'_','\_'));
        xlim([0.5 nC+0.5]); xticks(1:nC);
        if oi > 4
            xticklabels(catchments_labels); xtickangle(45);
        else
            xticklabels([]);
        end
        if mod(oi-1,4) == 0
            ylabel('Evaluation signature error');
        end
        if si <= numel(yrange_values) && isfinite(yrange_values(si)) && yrange_values(si) > 0
            ylim([-yrange_values(si), yrange_values(si)]);
        end
        yl = ylim;
        y_text = yl(2) - 0.08 * range(yl);
        for ci = 1:nC
            if n_per_catch(ci) > 0
                text(ci, y_text, sprintf('%d', n_per_catch(ci)), ...
                    'Rotation',90, 'HorizontalAlignment','center', ...
                    'VerticalAlignment','middle', 'FontSize',6, 'Color',[0.8 0.8 0.8]);
            end
        end
        grid on; grid minor;
    end
    sgtitle(f, sprintf('Evaluation signature: %s', signature_list_plot_sorted{si}));
    fontsize(f,11,'points');
    save_if_requested(f, output_dir, sprintf('eval_singleBest_signature_model_%02d_%sGate', si, benchmark_gate_mode), make_png);
end

%% SUMMARY TABLES
fprintf('Writing summary tables...\n');

T_counts = array2table(model_counter_analysis, ...
    'VariableNames', matlab.lang.makeValidName(catchments_labels), ...
    'RowNames', matlab.lang.makeValidName(OF_Plot));
writetable(T_counts, fullfile(output_dir, sprintf('eval_singleBest_model_counts_%sGate.csv', benchmark_gate_mode)), ...
    'WriteRowNames', true);

medianSigObj_eval = nan(nSigPlot,nO);
stdSigObj_eval = nan(nSigPlot,nO);
for si = 1:nSigPlot
    for oi = 1:nO
        v = squeeze(Error_Eval_Norm_Values_range(si,:,oi));
        medianSigObj_eval(si,oi) = median(v, 'omitnan');
        stdSigObj_eval(si,oi) = std(v, 'omitnan');
    end
end

outMat = nan(2*nO,nSigPlot);
rowNames = cell(2*nO,1);
for oi = 1:nO
    outMat(2*oi-1,:) = medianSigObj_eval(:,oi).';
    outMat(2*oi,:) = stdSigObj_eval(:,oi).';
    rowNames{2*oi-1} = sprintf('%s eval median', objective_functions{oi});
    rowNames{2*oi}   = sprintf('%s eval sd', objective_functions{oi});
end
varNames = matlab.lang.makeValidName(label_signatures_new, 'ReplacementStyle','delete');
T_eval_stats = array2table(outMat, 'VariableNames', varNames, 'RowNames', rowNames);
writetable(T_eval_stats, fullfile(output_dir, sprintf('eval_singleBest_signature_error_stats_%sGate.csv', benchmark_gate_mode)), ...
    'WriteRowNames', true);

% Cell-level export for downstream inspection.
rows = {};
for mi = 1:nM
    for ci = 1:nC
        for oi = 1:nO
            rows(end+1,:) = { ...
                D.models{mi}, D.catchments{ci}, catchments_labels{ci}, ...
                objective_functions{oi}, OF_Plot{oi}, ...
                OF_best_cal(mi,ci,oi), of_eval(mi,ci,oi), ...
                threshold(oi,ci), model_pass_cal(mi,ci,oi), model_pass_eval(mi,ci,oi), ...
                analysis_mask(mi,ci,oi)}; %#ok<AGROW>
        end
    end
end
T_eval_cells = cell2table(rows, ...
    'VariableNames', {'model','catchment','catchment_label','objective','objective_label', ...
                      'of_cal_best','of_eval','threshold','pass_cal','pass_eval','used_in_analysis'});
writetable(T_eval_cells, fullfile(output_dir, sprintf('eval_singleBest_cell_summary_%sGate.csv', benchmark_gate_mode)));

%% SAVE WORKSPACE
save(fullfile(output_dir, sprintf('eval_singleBest_workspace_%sGate.mat', benchmark_gate_mode)), ...
    'D','S','of_eval','sig_eval','OF_best_cal','sig_cal_best','threshold', ...
    'model_pass_cal','model_pass_eval','analysis_mask','benchmark_gate_mode', ...
    'OF_Eval_Array','Signatures_Eval_Array','Error_Eval_Values','Signature_eval_median', ...
    'OF_CalBest_Array','Signatures_CalBest_Array','Error_CalBest_Values', ...
    'Error_Eval_Norm_Values','Error_Eval_Norm_Values_range','Rank_Eval_values', ...
    'Error_CalBest_Norm_Values','Error_CalBest_Norm_Values_range','Rank_CalBest_values', ...
    'obs_sig_cal','obs_sig_eval','file_sig_names','sorted_sig_new','sorted_to_D_col','sorted_to_obs', ...
    'objective_functions','OF_Plot','model_list','catchments_aridity','catchments_labels', ...
    'label_signatures_new','label_signatures_short','model_counter_analysis','T_eval_stats','T_eval_cells', ...
    '-v7.3');

fprintf('\nDone. Wrote single-run evaluation figures and tables to %s\n', output_dir);

%% =====================================================================
% LOCAL FUNCTIONS
% =====================================================================
function Obs = load_observed_data_local(catchments, path_nc)
    date_array_full = (datetime(1981,1,2):datetime(2020,12,31))';
    nT = numel(date_array_full);
    nC = numel(catchments);
    Obs.date_array_full = date_array_full;
    Obs.precip = nan(nT,nC);
    Obs.temp = nan(nT,nC);
    Obs.pet = nan(nT,nC);
    Obs.q = nan(nT,nC);
    for ci = 1:nC
        nc_file = fullfile(path_nc, [catchments{ci} '.nc']);
        if ~isfile(nc_file)
            error('NetCDF file missing: %s', nc_file);
        end
        q = ncread(nc_file,'streamflow');
        p = ncread(nc_file,'total_precipitation_sum');
        t = ncread(nc_file,'temperature_2m_mean');
        e = ncread(nc_file,'potential_evaporation_sum');
        q(q<0) = 0;
        T = min([nT,numel(q),numel(p),numel(t),numel(e)]);
        Obs.q(1:T,ci) = q(1:T);
        Obs.precip(1:T,ci) = p(1:T);
        Obs.temp(1:T,ci) = t(1:T);
        Obs.pet(1:T,ci) = e(1:T);
    end
    if mean(Obs.temp(:),'omitnan') > 200
        Obs.temp = Obs.temp - 273.15;
    end
end

function [cal_idx, eval_idx, period_label] = get_benchmark_period_indices_local(catchment, date_array)
    if strcmp(catchment, 'camelsaus_607155')
        cal_start  = datetime(1990,1,1);
        cal_end    = datetime(1999,12,31);
        eval_start = datetime(1982,1,1);
        eval_end   = datetime(1988,12,31);
    else
        cal_start  = datetime(2005,1,1);
        cal_end    = datetime(2014,12,31);
        eval_start = datetime(1994,1,1);
        eval_end   = datetime(2003,12,31);
    end
    cal_idx = find(date_array >= cal_start & date_array <= cal_end);
    eval_idx = find(date_array >= eval_start & date_array <= eval_end);
    period_label = sprintf('cal %s to %s; eval %s to %s', ...
        datestr(cal_start,'yyyy-mm-dd'), datestr(cal_end,'yyyy-mm-dd'), ...
        datestr(eval_start,'yyyy-mm-dd'), datestr(eval_end,'yyyy-mm-dd'));
end

function sig_row = compute_obs_signature_row_local(q, p, idx, file_sig_names)
    sig_row = nan(1, numel(file_sig_names));
    q = q(:);
    p = p(:);
    q(q < 0) = 0;
    q_idx  = idx(isfinite(q(idx)));
    qp_idx = idx(isfinite(q(idx)) & isfinite(p(idx)));

    for si = 1:numel(file_sig_names)
        sig = file_sig_names{si};
        try
            switch sig
                case 'sig_x_percentile_5per'
                    val = sig_x_percentile(q(q_idx), q_idx, 5);
                case 'sig_x_percentile_95per'
                    val = sig_x_percentile(q(q_idx), q_idx, 95);
                case 'sig_x_Q_duration_high'
                    val = sig_x_Q_duration(q(q_idx), q_idx, 'high');
                case 'sig_x_Q_duration_low'
                    val = sig_x_Q_duration(q(q_idx), q_idx, 'low');
                case 'sig_x_Q_frequency_high'
                    val = sig_x_Q_frequency(q(q_idx), q_idx, 'high');
                case 'sig_x_Q_frequency_low'
                    val = sig_x_Q_frequency(q(q_idx), q_idx, 'low');
                case 'sig_EventRR'
                    val = sig_EventRR(q(qp_idx), qp_idx, p(qp_idx));
                case 'sig_TotalRR'
                    val = sig_TotalRR(q(qp_idx), qp_idx, p(qp_idx));
                otherwise
                    fh = str2func(sig);
                    val = fh(q(q_idx), q_idx);
            end
            if numel(val) > 1
                val = val(1);
            end
            sig_row(si) = val;
        catch ME
            warning('Observed signature %s failed: %s', sig, ME.message);
            sig_row(si) = NaN;
        end
    end
end

function [Error_Norm_Values, Error_Norm_Values_range, Rank_values] = ...
    compute_single_run_error_summaries(Signature_values, obs_sig, sorted_to_obs, nO, nC)
    nSig = size(Signature_values,1);
    Error_Norm_Values = nan(nSig,nC,nO);
    Error_Norm_Values_range = nan(nSig,nC,nO);
    Rank_values = nan(nSig+1,nC+2,nO);

    for si = 1:nSig
        obs_col = sorted_to_obs(si);
        if isnan(obs_col); continue; end
        for ci = 1:nC
            obsval = obs_sig(ci, obs_col);
            for oi = 1:nO
                vals = squeeze(Signature_values(si,oi,:,ci));
                err = vals - obsval;
                Error_Norm_Values(si,ci,oi) = median(err, 'omitnan');
            end

            denom = max(abs(squeeze(Error_Norm_Values(si,ci,:))), [], 'omitnan');
            if ~isfinite(denom) || denom == 0
                denom = NaN;
            end
            for oi = 1:nO
                Error_Norm_Values_range(si,ci,oi) = Error_Norm_Values(si,ci,oi) ./ denom;
            end

            vals_for_rank = squeeze(abs(Error_Norm_Values_range(si,ci,:)));
            valid = isfinite(vals_for_rank);
            ranks = nan(nO,1);
            if any(valid)
                [~,ord] = sort(vals_for_rank(valid), 'ascend');
                tmp = nan(sum(valid),1);
                tmp(ord) = 1:sum(valid);
                ranks(valid) = tmp;
            end
            Rank_values(si,ci,:) = ranks;
        end
    end
    Rank_values(nSig+1,1:nC,:) = mean(Rank_values(1:nSig,1:nC,:), 1, 'omitnan');
    Rank_values(1:nSig,nC+2,:) = mean(Rank_values(1:nSig,1:nC,:), 2, 'omitnan');
end

function labels = make_objective_labels_local(objectives)
    objectives = cellstr(string(objectives(:)))';
    labels = cell(size(objectives));
    for ii = 1:numel(objectives)
        nm = objectives{ii};
        switch nm
            case 'of_KGE'
                labels{ii} = 'KGE';
            case 'of_NSE'
                labels{ii} = 'NSE';
            case {'of_KGE_log','of_log_KGE'}
                labels{ii} = 'log KGE';
            case {'of_NSE_log','of_log_NSE'}
                labels{ii} = 'log NSE';
            case {'of_KGE_sqrt','of_sqrt_KGE'}
                labels{ii} = 'sqrt KGE';
            case {'of_NSE_sqrt','of_sqrt_NSE'}
                labels{ii} = 'sqrt NSE';
            case {'of_DE','of_de'}
                labels{ii} = 'DE';
            case {'of_KGE_NP','of_KGE_np','of_KGE_nonparametric'}
                labels{ii} = 'KGE-NP';
            case {'of_KGE_split','of_KGE_Split'}
                labels{ii} = 'KGE Split';
            case {'of_SHE','of_SHE_transf'}
                labels{ii} = 'SHE';
            case 'of_KGE_neg2_transf'
                labels{ii} = 'KGE neg2';
            case 'of_KGE_02_transf'
                labels{ii} = 'KGE0.2';
            case 'of_KGE_non_parametric'
                labels{ii} = 'KGE-NP';
            case 'of_diagnostic_efficiency'
                labels{ii} = 'DE';
            otherwise
                tmp = regexprep(nm, '^of_', '');
                tmp = strrep(tmp, '_', ' ');
                labels{ii} = tmp;
        end
    end
end

function save_if_requested(fig_handle, output_dir, name, make_png)
    if ~make_png; return; end
    try
        exportgraphics(fig_handle, fullfile(output_dir, [name '.png']), 'Resolution', 300);
    catch
        try
            saveas(fig_handle, fullfile(output_dir, [name '.png']));
        catch ME
            warning('Could not save figure %s: %s', name, ME.message);
        end
    end
end

function vals = finite_real_vector_local(vals)
    if isempty(vals)
        vals = [];
        return;
    end
    if iscell(vals)
        vals = cellfun(@double_or_nan_local, vals(:));
    end
    if ~isnumeric(vals) && ~islogical(vals)
        vals = str2double(string(vals));
    end
    vals = double(vals(:));
    vals = real(vals);
    vals = vals(isfinite(vals));
end

function x = double_or_nan_local(v)
    try
        if isempty(v)
            x = NaN;
        elseif isnumeric(v) || islogical(v)
            vv = double(v(:));
            vv = real(vv(isfinite(vv)));
            if isempty(vv), x = NaN; else, x = vv(1); end
        else
            x = str2double(string(v));
        end
    catch
        x = NaN;
    end
end

function safe_violinplot_local(x, g, color)
    x = finite_real_vector_local(x);
    if isempty(x)
        return;
    end
    if isscalar(g)
        g = repmat(g, size(x));
    else
        g = double(g(:));
        n = min(numel(x), numel(g));
        x = x(1:n);
        g = g(1:n);
    end
    ok = isfinite(x) & isfinite(g) & isreal(x) & isreal(g);
    x = x(ok);
    g = g(ok);
    if isempty(x)
        return;
    end
    try
        violinplot(x, g, 'ViolinColor', color, 'ShowMean', false, 'ShowData', false);
    catch ME
        warning('safe_violinplot_local:violinFailed', ...
            'violinplot failed (%s). Falling back to jittered scatter/median.', ME.message);
        ug = unique(g(:))';
        for kk = 1:numel(ug)
            idx = g == ug(kk);
            xx = x(idx);
            if isempty(xx), continue; end
            jitter = (rand(size(xx))-0.5) * 0.12;
            scatter(ug(kk)+jitter, xx, 8, color, 'filled', 'MarkerFaceAlpha',0.18, 'MarkerEdgeAlpha',0.18);
            medx = median(xx, 'omitnan');
            plot([ug(kk)-0.18 ug(kk)+0.18], [medx medx], '-', 'Color', color, 'LineWidth',1.8);
        end
    end
end

function plot_fixed_position_violins(x, g, positions, color)
    x = double(x(:));
    g = double(g(:));
    ok = isfinite(x) & isfinite(g);
    x = x(ok);
    g = g(ok);
    max_width = 0.32;
    for kk = 1:numel(positions)
        pos = positions(kk);
        vals = x(g == pos);
        vals = vals(isfinite(vals));
        if numel(vals) < 2
            continue;
        end
        try
            [f, yi] = ksdensity(vals);
        catch
            continue;
        end
        if max(f) <= 0 || ~isfinite(max(f))
            continue;
        end
        f = f ./ max(f) .* max_width;
        patch([pos - f, fliplr(pos + f)], [yi, fliplr(yi)], color, ...
            'FaceAlpha',0.45, 'EdgeColor',[0.35 0.35 0.35], 'LineWidth',0.5);
        med = median(vals,'omitnan');
        plot([pos - 0.18, pos + 0.18], [med, med], 'Color',[0.25 0.25 0.25], 'LineWidth',1.0);
    end
end

function customColormap = greenCenteredColormap(n)
    if nargin < 1; n = 256; end
    half = floor(n / 2);
    blueToGreen = [linspace(0, 0, half)', linspace(0, 1, half)', linspace(1, 0, half)'];
    greenToRed = [linspace(0, 1, n - half)', linspace(1, 0, n - half)', linspace(0, 0, n - half)'];
    customColormap = [blueToGreen; greenToRed];
end
