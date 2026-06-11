%% plots_paper_topN_from_plots_paper_style.m
% Paper-figure reproduction using the NEW retained-run results, but keeping
% the original plots_paper.m figure logic/style as closely as possible.

clear; close all; clc;

%% USER SETTINGS
base_path  = '<LOCAL_DOWNLOADS>/all_fixed';
output_dir = '<LOCAL_ROOT>/graphics_new';
path_nc    = '<LOCAL_ROOT>/ma_thesis/catchments_new';
hydrobm_threshold_file = '<LOCAL_ROOT>/benchmark_results/hydrobm_thresholds.mat';

top_n_runs = 500;
make_png   = true;
use_relative_error_for_norm = false;

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
    'Q95 (mm/d)',  'HFF (-)','HFD (days)', ...
    'Q5 (mm/d)', 'LFF (-)','LFD (days)', ...
    'BFI (-)', 'BFRC (-)','FDC Slope (-)', ...
    'FI (-)', 'VI (-)', 'RLD (-)'};

signature_list_plot = {'FDC Slope','Rising Limb Density (1/d)','Baseflow Recession Coefficient (1/d)','Mean Half Flow Date', ...
    'Baseflow Index','Variability Index','Flashiness Index','Event RR','Total RR','High Flow Duration (d)','Low Flow Duration (d)', ...
    'High Flow Frequency','Low Flow Frequency','Q5 (mm/d)','Q95 (mm/d)'};

OF_Plot = {};  % filled dynamically from D.objectives after dropping removed objectives

base_colors = NaN(8,3);
base_colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
base_colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
base_colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
base_colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
base_colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
base_colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
base_colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
base_colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];
colors = base_colors;  % trimmed/extended after objectives are loaded

%% LOAD NEW RESULTS
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

% Pull top_n from the data file if present, otherwise fall back to user setting.
if isfield(S, 'top_n')
    top_n_runs = S.top_n;
    fprintf('Loaded top_n = %d from seed_uncertainty_data.mat\n', top_n_runs);
end

% Remove neg2 transformation objective and keep the remaining order.
neg2_name = 'of_KGE_neg2_transf';
keep_obj = ~strcmp(D.objectives, neg2_name);
if numel(keep_obj) ~= size(D.of_all,3)
    error('Number of objective names (%d) does not match of_all objective dimension (%d).', ...
        numel(keep_obj), size(D.of_all,3));
end
D.objectives = D.objectives(keep_obj);
D.of_all     = D.of_all(:,:,keep_obj,:);
D.sig_all    = D.sig_all(:,:,keep_obj,:,:);

% Make sure dimensions are [model catchment objective candidate signature].
[nM,nC,nO,nCand] = size(D.of_all);
if ndims(D.sig_all) ~= 5
    error('Expected sig_all to be 5-D: model x catchment x objective x candidate x signature.');
end
nSigs = size(D.sig_all,5);
if nO ~= 8
    warning('Expected 8 objectives after dropping %s, found %d.', neg2_name, nO);
end
if nSigs ~= 15
    warning('Expected 15 signatures, found %d.', nSigs);
end

fprintf('Candidate-dimension size after load: nCand = %d (expected %d for top-N pooling).\n', ...
        nCand, top_n_runs);

% Build objective display labels and starting color set from the post-drop list.
OF_Plot = make_objective_labels_local(D.objectives);
if size(base_colors,1) >= nO
    colors = base_colors(1:nO,:);
else
    colors = lines(nO);
end

%% ---- Reorder objectives to match original paper order ----
% Target order: KGE, NSE, KGE0.2, log NSE, DE, KGE-NP, KGE Split, SHE.
% Anything in the data that's not in this list gets appended at the end
% in its current order, so nothing is silently dropped.
desired_order_labels = {'KGE','NSE','KGE0.2','log NSE','DE','KGE-NP','KGE Split','SHE'};

new_idx = nan(1, numel(desired_order_labels));
for k = 1:numel(desired_order_labels)
    hit = find(strcmp(OF_Plot, desired_order_labels{k}), 1);
    if isempty(hit)
        warning('Desired objective label "%s" not found in OF_Plot. Check make_objective_labels_local.', ...
                desired_order_labels{k});
    else
        new_idx(k) = hit;
    end
end
new_idx = new_idx(~isnan(new_idx));
leftover = setdiff(1:numel(OF_Plot), new_idx, 'stable');
new_idx = [new_idx, leftover];

D.objectives        = D.objectives(new_idx);
D.of_all            = D.of_all(:,:,new_idx,:);
D.sig_all           = D.sig_all(:,:,new_idx,:,:);
OF_Plot             = OF_Plot(new_idx);
objective_functions = D.objectives;   % keep alias in sync for downstream code
if size(colors,1) >= max(new_idx)
    colors = colors(new_idx, :);
end
nO = numel(OF_Plot);

fprintf('Objectives used in plots:\n');
for oi_dbg = 1:nO
    fprintf('  %2d: %-28s -> %s\n', oi_dbg, D.objectives{oi_dbg}, OF_Plot{oi_dbg});
end

% Align to plots_paper.m catchment order where possible.
[cat_ok, cat_idx_in_D] = ismember(catchments_aridity, D.catchments);
if ~all(cat_ok)
    missing = catchments_aridity(~cat_ok);
    error('These plots_paper catchments are missing from new results: %s', strjoin(missing, ', '));
end
D.of_all     = D.of_all(:,cat_idx_in_D,:,:);
D.sig_all    = D.sig_all(:,cat_idx_in_D,:,:,:);
D.catchments = D.catchments(cat_idx_in_D);
nC = numel(D.catchments);

% Aliases for downstream code that uses old plots_paper.m variable names.
model_list        = D.models;
model_list_sorted = model_list;
model_name        = model_list;

fprintf('Loaded retained-run data: %d models, %d catchments, %d objectives, %d retained candidates, %d signatures.\n', ...
    nM, nC, nO, nCand, nSigs);

%% LOAD HYDROBM THRESHOLDS AND ALIGN
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

%% OBSERVED DATA AND SIGNATURES
% Recompute observed signatures to avoid depending on old signatures.mat.
Obs = load_observed_data_local(D.catchments, path_nc);
[obs_sig, obs_signatures_cali_bench] = compute_observed_signatures_local(D.catchments, Obs, sorted_signatures);

%% RECONSTRUCT OLD-STYLE SCALAR BENCHMARK VARIABLES + TOP-N ENSEMBLES
% Scalar best OF per model x catchment x objective. This replaces the old
% OF_value_cali.(catchment).(objective).(model).
OF_best = squeeze(max(D.of_all, [], 4, 'omitnan')); % [model x catchment x objective]


% D.of_all is already pre-sorted top-N descending; NaN fills any unused slot.
% So "top-N" reduces to "any finite candidate."
top_mask = isfinite(D.of_all);

% Benchmark gate is scalar/best-OF, matching the old design.
model_pass = false(nM,nC,nO);
for oi = 1:nO
    for ci = 1:nC
        model_pass(:,ci,oi) = isfinite(OF_best(:,ci,oi)) & OF_best(:,ci,oi) > threshold(oi,ci);
    end
end

% Final plotting mask: top-N candidates only where the model passed the scalar benchmark.
plot_mask = top_mask & repmat(permute(model_pass,[1 2 3]), [1 1 1 nCand]);

% Old-style structs for sections that still need scalar benchmark results.
OF_value_cali = struct();
OF_value_vali = struct();
OF_value_cali_benchmark = struct();
OF_value_vali_benchmark = struct();
model_counter = zeros(nO,nC);

for ci = 1:nC
    catchment = D.catchments{ci};
    for oi = 1:nO
        obj_fun = objective_functions{oi};
        for mi = 1:nM
            model = model_list{mi};
            val = OF_best(mi,ci,oi);
            OF_value_cali.(catchment).(obj_fun).(model) = val;
            OF_value_vali.(catchment).(obj_fun).(model) = NaN; % validation OF not in retained-run file unless added separately
            if model_pass(mi,ci,oi)
                OF_value_cali_benchmark.(catchment).(obj_fun).(model) = val;
                OF_value_vali_benchmark.(catchment).(obj_fun).(model) = NaN;
                model_counter(oi,ci) = model_counter(oi,ci) + 1;
            end
        end
    end
end

% Ensemble arrays used by the reproduced plots.
% Dimensions follow plots_paper.m where possible, but model dimension is
% expanded to model x top-candidate values. NaNs preserve unequal candidate counts.
%% RECONSTRUCT TOP-N ENSEMBLE ARRAYS (corrected)
maxK = top_n_runs;
OF_TopN_Array          = nan(nO, nC, nM, maxK);
Signatures_Array       = nan(numel(sorted_sig_new), nO, nM*maxK, nC);
Signatures_Array_vali  = nan(size(Signatures_Array));
Error_Values           = nan(size(Signatures_Array));
Signature_model_median = nan(numel(sorted_sig_new), nO, nM, nC);
OF_model_median        = nan(nO, nC, nM);

% Canonical signature ordering used by obs_sig and by all downstream plots.
file_sig_names = {'sig_FDC_slope','sig_RisingLimbDensity','sig_BaseflowRecessionK','sig_HFD_mean', ...
    'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex','sig_EventRR','sig_TotalRR', ...
    'sig_x_Q_duration_high','sig_x_Q_duration_low','sig_x_Q_frequency_high','sig_x_Q_frequency_low', ...
    'sig_x_percentile_5per','sig_x_percentile_95per'};

% --- BUGFIX 1: verify (and if possible, build) a name->column map for D.sig_all.
% If the .mat file stores signature names, align by name. Otherwise warn loudly
% rather than silently assuming the order matches file_sig_names.
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
        error(['Signature-name vector in seed_uncertainty_data.mat has %d entries ', ...
               'but sig_all has %d signature columns.'], ...
               numel(stored_sig_names), size(D.sig_all,5));
    end
    [sig_ok, sig_col_in_D] = ismember(file_sig_names, stored_sig_names);
    if ~all(sig_ok)
        error('These signatures are missing from sig_all: %s', ...
              strjoin(file_sig_names(~sig_ok), ', '));
    end
    fprintf('Aligned sig_all by signature name (verified against stored names).\n');
else
    warning(['seed_uncertainty_data.mat does not contain a signature-name field. ', ...
             'Falling back to ASSUMED order matching file_sig_names. ', ...
             'If signature errors look off, this is the first place to check.']);
    sig_col_in_D = 1:numel(file_sig_names);
end

% sorted_sig_new is a reordering of file_sig_names for plotting. Compute the
% column index in D.sig_all that corresponds to each sorted_sig_new entry once.
sorted_to_D_col = nan(1, numel(sorted_sig_new));
sorted_to_obs   = nan(1, numel(sorted_sig_new));
for si = 1:numel(sorted_sig_new)
    fsi = find(strcmp(file_sig_names, sorted_sig_new{si}), 1);
    if isempty(fsi)
        warning('Could not map signature %s into file_sig_names; leaving NaN.', sorted_sig_new{si});
        continue;
    end
    sorted_to_obs(si)   = fsi;              % column in obs_sig
    sorted_to_D_col(si) = sig_col_in_D(fsi);% column in D.sig_all
end

OF_TopN_Array_all = nan(nO, nC, nM, maxK);

for si = 1:numel(sorted_sig_new)
    if isnan(sorted_to_D_col(si)); continue; end
    d_col   = sorted_to_D_col(si);
    obs_col = sorted_to_obs(si);

    for ci = 1:nC
        obsval = obs_sig(ci, obs_col);
        if ~isfinite(obsval)
            warning('Observed signature %s is NaN for catchment %s; errors will be NaN.', ...
                    sorted_sig_new{si}, D.catchments{ci});
        end

        for oi = 1:nO
            for mi = 1:nM
                vals_sig = squeeze(D.sig_all(mi,ci,oi,:,d_col));
                vals_of  = squeeze(D.of_all(mi,ci,oi,:));
                keep_topN  = squeeze(top_mask(mi,ci,oi,:));    % ungated: top-N only
                keep_gated = squeeze(plot_mask(mi,ci,oi,:));   % gated: top-N + benchmark pass
        
                % --- ungated branch: fills OF_TopN_Array_all ---
                joint_u = keep_topN(:) & isfinite(vals_of(:));
                vals_of_u = vals_of(joint_u);
                if ~isempty(vals_of_u)
                    vals_of_u = sort(vals_of_u, 'descend');
                    nKeep_u = min(numel(vals_of_u), maxK);
                    OF_TopN_Array_all(oi,ci,mi,1:nKeep_u) = vals_of_u(1:nKeep_u);
                end
        
                % --- gated branch: fills OF_TopN_Array and the signature arrays as before ---
                joint_g = keep_gated(:) & isfinite(vals_sig(:)) & isfinite(vals_of(:));
                vals_sig_g = vals_sig(joint_g);
                vals_of_g  = vals_of(joint_g);
                if isempty(vals_of_g); continue; end
        
                [vals_of_g, ord] = sort(vals_of_g, 'descend');
                vals_sig_g = vals_sig_g(ord);
        
                nKeep_g = min(numel(vals_of_g), maxK);
                vals_of_g  = vals_of_g(1:nKeep_g);
                vals_sig_g = vals_sig_g(1:nKeep_g);
        
                OF_TopN_Array(oi,ci,mi,1:nKeep_g) = vals_of_g;
                OF_model_median(oi,ci,mi)         = median(vals_of_g, 'omitnan');
        
                idx0 = (mi-1)*maxK + (1:nKeep_g);
                Signatures_Array(si,oi,idx0,ci)      = vals_sig_g;
                Signatures_Array_vali(si,oi,idx0,ci) = NaN;
                Error_Values(si,oi,idx0,ci)          = vals_sig_g - obsval;
                Signature_model_median(si,oi,mi,ci)  = median(vals_sig_g, 'omitnan');
            end
        end
    end
end

nse_family_labels = {'NSE','log NSE'};
for nm = 1:numel(nse_family_labels)
    oi_fix = find(strcmp(OF_Plot, nse_family_labels{nm}));
    if isempty(oi_fix); continue; end
    slab = OF_TopN_Array_all(oi_fix,:,:,:);
    slab = real(slab);                     % drop complex parts
    n_bad = sum(slab(:) > 1);
    slab(slab > 1) = NaN;                  % NaN out impossible values
    if n_bad > 0
        fprintf('Sanitized %d impossible %s values (> 1) to NaN.\n', ...
                n_bad, nse_family_labels{nm});
    end
    OF_TopN_Array_all(oi_fix,:,:,:) = slab;
end

% Old-style top-N signature structs, mostly for debugging/backward checks.
sim_signatures_cali_bench = struct();
sim_signatures_eval_bench = struct();
for si = 1:numel(file_sig_names)
    signature = file_sig_names{si};
    for ci = 1:nC
        catchment = D.catchments{ci};
        for oi = 1:nO
            obj_fun = objective_functions{oi};
            for mi = 1:nM
                model = model_list{mi};
                vals = squeeze(D.sig_all(mi,ci,oi,:,si));
                keep = squeeze(plot_mask(mi,ci,oi,:));
                sim_signatures_cali_bench.(signature).(catchment).(model).(obj_fun) = vals(keep);
                sim_signatures_eval_bench.(signature).(catchment).(model).(obj_fun) = NaN(size(vals(keep)));
            end
        end
    end
end


%% STORE TOP-N RAW VALUES AND VARIABILITY SUMMARIES
% These objects preserve the behavioural/top-N spread even when later plots
% use only median values.  They are saved with the workspace at the end.
%
% TopNRaw.OF{mi,ci,oi} contains the retained top-N OF values for one model,
% catchment and objective.  TopNStats.OF(mi,ci,oi) stores summary statistics
% for the same values.
%
% TopNRaw.Signature{mi,ci,oi,si} contains the matching signature values.
% TopNStats.Signature(mi,ci,oi,si) stores the corresponding summaries.
[TopNRaw, TopNStats] = collect_topn_variability_stats(D, plot_mask, file_sig_names);

% Convenience arrays for later uncertainty-band plotting.
% Dimensions: model x catchment x objective x signature.
Signature_q05 = nan(nM,nC,nO,numel(file_sig_names));
Signature_q25 = nan(size(Signature_q05));
Signature_q50 = nan(size(Signature_q05));
Signature_q75 = nan(size(Signature_q05));
Signature_q95 = nan(size(Signature_q05));
Signature_IQR = nan(size(Signature_q05));
for mi = 1:nM
    for ci = 1:nC
        for oi = 1:nO
            for si = 1:numel(file_sig_names)
                st = TopNStats.Signature(mi,ci,oi,si);
                Signature_q05(mi,ci,oi,si) = st.q05;
                Signature_q25(mi,ci,oi,si) = st.q25;
                Signature_q50(mi,ci,oi,si) = st.median;
                Signature_q75(mi,ci,oi,si) = st.q75;
                Signature_q95(mi,ci,oi,si) = st.q95;
                Signature_IQR(mi,ci,oi,si) = st.iqr;
            end
        end
    end
end

OF_q05 = nan(nM,nC,nO);
OF_q25 = nan(nM,nC,nO);
OF_q50 = nan(nM,nC,nO);
OF_q75 = nan(nM,nC,nO);
OF_q95 = nan(nM,nC,nO);
OF_IQR = nan(nM,nC,nO);
for mi = 1:nM
    for ci = 1:nC
        for oi = 1:nO
            st = TopNStats.OF(mi,ci,oi);
            OF_q05(mi,ci,oi) = st.q05;
            OF_q25(mi,ci,oi) = st.q25;
            OF_q50(mi,ci,oi) = st.median;
            OF_q75(mi,ci,oi) = st.q75;
            OF_q95(mi,ci,oi) = st.q95;
            OF_IQR(mi,ci,oi) = st.iqr;
        end
    end
end

%% FIGURE 1: BENCHMARK MODEL COUNTS, as in plots_paper.m
f = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
bh = bar(model_counter(:,:)');
set(bh, 'FaceColor', 'Flat')
for k = 1:min(nO,8)
    bh(k).CData = colors(k,:);
end
yline(nM,'-','All Models Pass Benchmark')
fontsize(f,16,'points')
legend(strrep(OF_Plot,'_','\_'),'Location','eastoutside')
xticklabels(strrep(catchments_labels,'_','\_'))
xlabel('Catchments')
ylabel('Number of Models Outperforming Benchmark')
xtickangle(45)
title(sprintf('Scalar benchmark pass count; top-%d used only after pass gate', top_n_runs));
save_if_requested(f, output_dir, 'benchmark_counts_scalarBest_topN', make_png);

%% FIGURE 2: HEATMAP MODEL COUNTS, as in plots_paper.m
f = figure('units','normalized','outerposition',[0 0 0.65 0.55]);
imagesc(model_counter);
colormap(parula); colorbar;
set(gca,'XTick',1:nC,'XTickLabel',catchments_labels,'YTick',1:nO,'YTickLabel',OF_Plot);
xtickangle(45); xlabel('Catchments'); ylabel('Objective function');
title('Models outperforming hydrobm benchmark');
fontsize(f,14,'points');
save_if_requested(f, output_dir, 'benchmark_counts_heatmap_scalarBest_topN', make_png);


%% FIGURE 3: BOXPLOT/VIOLIN OF OBJECTIVE FUNCTION VALUES BY CATCHMENT
f = figure('units','normalized','outerposition',[0 0 1 0.8]);
t = tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

for oi = 1:nO
    nexttile; hold on;
    X = [];
    G = [];
    for ci = 1:nC
        vals = squeeze(OF_TopN_Array_all(oi,ci,:,:));
        vals = vals(:);
        vals = finite_real_vector_local(vals);
        X = [X; vals]; %#ok<AGROW>
        G = [G; ci*ones(numel(vals),1)]; %#ok<AGROW>
    end
    if ~isempty(X)
        safe_violinplot_local(X, G, colors(oi,:));
    end
    for ci = 1:nC
        yline_val = threshold(oi,ci);
        line([ci-0.35 ci+0.35], [yline_val yline_val], ...
             'Color', 'r', 'LineStyle', '-', 'LineWidth', 1.4);
    end

    % Annotate number of benchmark-passing models below each violin
    yl = [-0.5, 1];
    ylim(yl);
    xlim([0.5 10.5])
    y_text = yl(1) + 0.04 * (yl(2) - yl(1));   % just above the bottom of the panel
    for ci = 1:nC
        n_pass = sum(model_pass(:,ci,oi));
        text(ci, y_text, sprintf('%d', n_pass), ...
             'HorizontalAlignment','center', ...
             'VerticalAlignment','bottom', ...
             'FontSize', 9, ...
             'Color', [0.25 0.25 0.25]);
    end
    
    title(OF_Plot{oi});
    set(gca,'XTick',1:nC,'XTickLabel',catchments_labels);
    xtickangle(45);
    grid minor;
end

xlabel(t, 'Catchment', 'FontSize', 15);
ylabel(t, 'Objective function value', 'FontSize', 15);

fontsize(f,13,'points');
save_if_requested(f, output_dir, 'benchmark_violinplot_topN_from_original_style', make_png);

%% FIGURE 4: OBSERVED SIGNATURE VALUES, 4x4-style from plots_paper.m
f = figure('units','normalized','outerposition',[0 0 0.8 0.9]);
tiledlayout(4,4,'TileSpacing','compact','Padding','compact');
for si = 1:numel(sorted_signatures)
    nexttile; hold on;
    fsi = find(strcmp(file_sig_names, sorted_signatures{si}),1);
    vals = obs_sig(:,fsi);
    bar(vals, 'FaceColor',[0.7 0.7 0.7]);
    set(gca,'XTick',1:nC,'XTickLabel',catchments_labels); xtickangle(45);
    title(strrep(sorted_signatures{si},'_','\_'));
    grid minor;
end
fontsize(f,11,'points');
save_if_requested(f, output_dir, 'observed_signature_values', make_png);

%% FIGURE 5: SIGNATURE COMPARISON WITH UNDERLYING VIOLINS, as in plots_paper.m
yrange_min = [0,0,80, 0,0,0, 0,0,0, 0,0,-25, 0,0,0];
yrange_max = [1,1,320, 10,0.5,80, 1.5,1,80, 1,0.7,0, 1.5,1.5,1];
box_color = [0.7 0.7 0.7]; observed_color = [0 0 0];

f = figure('Units','normalized','OuterPosition',[0 0 0.6 1]);
t = tiledlayout(f,5,3,'TileSpacing','compact','Padding','tight');

% Handles for the legend (populated on first tile only)
h_obj    = gobjects(1, nO);
h_violin = [];
h_obs    = [];

for si = 1:numel(sorted_sig_new)
    nexttile; hold on;
    all_data = [];
    group_labels = [];
    for oi = 1:nO
        plot_data = squeeze(Signatures_Array(si,oi,:,:));
        for ci = 1:nC
            vals = plot_data(:,ci);
            vals = finite_real_vector_local(vals);
            all_data = [all_data; vals]; %#ok<AGROW>
            group_labels = [group_labels; ci*ones(numel(vals),1)]; %#ok<AGROW>
        end
    end
    if ~isempty(all_data)
        safe_violinplot_local(all_data, group_labels, box_color);
        if isempty(h_violin)
            % Dummy patch for the legend (off-axis NaN data)
            h_violin = patch(NaN, NaN, box_color, 'EdgeColor', [0.4 0.4 0.4]);
        end
    end

    % Colored objective medians
    for oi = 1:nO
        med_by_c = squeeze(median(Signatures_Array(si,oi,:,:), 3, 'omitnan'));
        hs = scatter(1:nC, med_by_c, 46, colors(oi,:), 'filled');
        if si == 1
            h_obj(oi) = hs;
        end
    end

    % Observed values
    fsi = find(strcmp(file_sig_names, sorted_sig_new{si}), 1);
    for ci = 1:nC
        observed_value = obs_sig(ci,fsi);
        if isfinite(observed_value)
            hl = line([ci-0.4, ci+0.4], [observed_value, observed_value], ...
                       'Color', observed_color, 'LineWidth', 2, 'LineStyle', '-');
            if isempty(h_obs)
                h_obs = hl;
            end
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
    xlim([0.5 10.5])
    ylabel("Signature Value")
end

% Build and attach legend to the layout
legend_handles = [h_obj, h_violin, h_obs];
legend_labels  = [OF_Plot, {'Top-500 ensemble'}, {'Observed'}];

% Drop any empty handles defensively
valid = arrayfun(@(h) ~isempty(h) && isgraphics(h), legend_handles);
legend_handles = legend_handles(valid);
legend_labels  = legend_labels(valid);

lgd = legend(legend_handles, legend_labels, ...
             'NumColumns', 5, 'Orientation', 'horizontal');
lgd.Layout.Tile = 'south';

fontsize(f,12,'points');
save_if_requested(f, output_dir, 'signature_comparison_underlying_violin_topN', make_png);
%% FIGURE 6: SIGNATURE MEDIAN LINE/PANEL PLOT, from plots_paper.m style
f = figure('Units','normalized','OuterPosition',[0 0 0.6 1]);
tiledlayout(f,5,3,'TileSpacing','compact','Padding','tight');
for si = 1:numel(sorted_sig_new)
    nexttile; hold on;
    for oi = 1:nO
        med_vals = squeeze(median(Signatures_Array(si,oi,:,:), 3, 'omitnan'));
        plot(1:nC, med_vals, '-o', 'Color', colors(oi,:), 'MarkerFaceColor', colors(oi,:), 'LineWidth', 1.2);
    end
    fsi = find(strcmp(file_sig_names, sorted_sig_new{si}), 1);
    plot(1:nC, obs_sig(:,fsi), 'k-', 'LineWidth', 2);
    title(label_signatures_new{si}); grid minor;
    if si > 12
        set(gca,'XTick',1:nC,'XTickLabel',catchments_labels); xtickangle(45);
    else
        set(gca,'XTick',1:nC,'XTickLabel',[]);
    end
end
legend([OF_Plot {'Observed'}], 'Location','eastoutside');
fontsize(f,12,'points');
save_if_requested(f, output_dir, 'signature_median_lines_topN', make_png);


%% FIGURE 7: NORMALIZED OVERALL ERROR VIOLIN, as in plots_paper.m
[Error_Norm_Values, Error_Norm_Values_range, Rank_values, median_catchment_error_range, mean_catchment_error_range, Signature_norm] = ...
    compute_original_style_error_summaries(Error_Values, obs_sig, sorted_sig_new, file_sig_names, nO, nC, ...
                                            Signature_model_median, sorted_to_obs);
f = figure('units','normalized','outerposition',[0 0 0.75 0.5]); hold on;
for oi = 1:nO
    vals = squeeze(Error_Norm_Values_range(:,:,oi));
    vals = vals(:);
    vals = finite_real_vector_local(vals);
    if isempty(vals); continue; end
    safe_violinplot_local(vals, oi*ones(size(vals)), colors(oi,:));
end
yline(0,'k-');
set(gca,'XTick',1:nO,'XTickLabel',OF_Plot); xtickangle(45);
ylabel('Normalized median signature error');
grid minor;
title('Normalized signature error across signatures/catchments');
fontsize(f,14,'points');
save_if_requested(f, output_dir, 'normalized_overall_violin_topN', make_png);


%% Make overview table

% TABLE: NORMALIZED SIGNATURE ERROR MEDIAN/SD
% Uses Error_Norm_Values_range:
%   [signature x catchment x objective]
% where each value was computed as:
%   1) median across retained top-N runs per model
%   2) median across models
%   3) range-normalized within signature x catchment across objectives

table_sig_order = { ...
    'sig_TotalRR', ...
    'sig_EventRR', ...
    'sig_HFD_mean', ...
    'sig_FDC_slope', ...
    'sig_x_percentile_5per', ...
    'sig_x_Q_duration_low', ...
    'sig_x_Q_frequency_low', ...
    'sig_BFI', ...
    'sig_x_percentile_95per', ...
    'sig_x_Q_duration_high', ...
    'sig_x_Q_frequency_high', ...
    'sig_BaseflowRecessionK', ...
    'sig_RisingLimbDensity', ...
    'sig_FlashinessIndex', ...
    'sig_VariabilityIndex'};

table_sig_labels = { ...
    'Total RR (-)', ...
    'Event RR (-)', ...
    'MHFD (DOY)', ...
    'FDC Slope (-)', ...
    'Q5 (mm/d)', ...
    'LF Duration (days)', ...
    'LF Frequency (-)', ...
    'Baseflow Index (-)', ...
    'Q95 (mm/d)', ...
    'HF Duration (days)', ...
    'HF Frequency (-)', ...
    'Baseflow Recession Coefficient (-)', ...
    'Rising Limb Density (-)', ...
    'Flashiness Index (-)', ...
    'Variability Index (-)'};

[ok_sig, table_sig_idx] = ismember(table_sig_order, sorted_sig_new);
if ~all(ok_sig)
    error('Could not map table signatures: %s', ...
        strjoin(table_sig_order(~ok_sig), ', '));
end

medianSigObj = nan(numel(table_sig_idx), nO);
stdSigObj    = nan(numel(table_sig_idx), nO);

for oi = 1:nO
    for jj = 1:numel(table_sig_idx)
        si = table_sig_idx(jj);

        % values across catchments after run->model collapse
        v = squeeze(Error_Norm_Values_range(si,:,oi));

        medianSigObj(jj,oi) = median(v, 'omitnan');
        stdSigObj(jj,oi)    = std(v, 0, 'omitnan');
    end
end

% Also write CSV for checking
outMat = nan(2*nO, numel(table_sig_idx));
rowNames = strings(2*nO,1);

for oi = 1:nO
    outMat(2*oi-1,:) = medianSigObj(:,oi).';
    outMat(2*oi,:)   = stdSigObj(:,oi).';

    rowNames(2*oi-1) = sprintf('%s median', OF_Plot{oi});
    rowNames(2*oi)   = sprintf('%s sd', OF_Plot{oi});
end

T_normerr = array2table(outMat, ...
    'VariableNames', matlab.lang.makeValidName(table_sig_labels), ...
    'RowNames', cellstr(rowNames));

writetable(T_normerr, ...
    fullfile(output_dir, 'table_normalized_signature_errors.csv'), ...
    'WriteRowNames', true);

% Write LaTeX table
tex_file = fullfile(output_dir, 'table_normalized_signature_errors.tex');
fid = fopen(tex_file, 'w');

fprintf(fid, '\\begin{landscape}\n');
fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '    \\centering\n');
fprintf(fid, '    \\caption{Medians and standard deviations of normalized signature errors for each objective function across 15 hydrological signatures.}\n');
fprintf(fid, '    \\renewcommand{\\arraystretch}{1.3}\n');
fprintf(fid, '    \\resizebox{\\linewidth}{!}{%%\n');
fprintf(fid, '    \\begin{tabular}{|l|%s|}\n', repmat('r',1,numel(table_sig_labels)));
fprintf(fid, '        \\hline\n');

fprintf(fid, '        Objective');
for jj = 1:numel(table_sig_labels)
    fprintf(fid, ' & %s', latex_escape_local(table_sig_labels{jj}));
end
fprintf(fid, ' \\\\\n');
fprintf(fid, '        \\hline\n');

for oi = 1:nO
    obj_label = latex_escape_local(OF_Plot{oi});

    fprintf(fid, '        %s median', obj_label);
    for jj = 1:numel(table_sig_idx)
        fprintf(fid, ' & %.3f', medianSigObj(jj,oi));
    end
    fprintf(fid, ' \\\\\n');

    fprintf(fid, '        %s sd', obj_label);
    for jj = 1:numel(table_sig_idx)
        fprintf(fid, ' & %.3f', stdSigObj(jj,oi));
    end
    fprintf(fid, ' \\\\ \\hline\n');
end

fprintf(fid, '    \\end{tabular}%%\n');
fprintf(fid, '    }\n');
fprintf(fid, '    \\renewcommand{\\arraystretch}{1.0}\n');
fprintf(fid, '    \\label{tab:obj_sig_stats_median}\n');
fprintf(fid, '\\end{table}\n');
fprintf(fid, '\\end{landscape}\n');

fclose(fid);

fprintf('Wrote normalized-error LaTeX table to %s\n', tex_file);


%% FIGURE 8: RANGE-LIMITED ERROR VALUES, original-style 5x3 panels
f = figure('Units','normalized','OuterPosition',[0 0 0.6 1]);
tiledlayout(f,5,3,'TileSpacing','compact','Padding','tight');
for si = 1:numel(sorted_sig_new)
    nexttile; hold on;
    for oi = 1:nO
        vals = squeeze(Error_Norm_Values_range(si,:,oi));
        plot(1:nC, vals, '-o', 'Color', colors(oi,:), 'MarkerFaceColor', colors(oi,:), 'LineWidth', 1.1);
    end
    yline(0,'k-'); title(label_signatures_new{si}); grid minor;
    if si > 12
        set(gca,'XTick',1:nC,'XTickLabel',catchments_labels); xtickangle(45);
    else
        set(gca,'XTick',1:nC,'XTickLabel',[]);
    end
end
fontsize(f,12,'points');
save_if_requested(f, output_dir, 'range_limited_error_values_topN', make_png);

%% FIGURE 9: HEATMAP OF MEAN ABS NORMALIZED ERROR BY SIGNATURE AND OBJECTIVE
f = figure('units','normalized','outerposition',[0 0 0.65 0.75]);
heat = squeeze(mean(abs(Error_Norm_Values_range), 2, 'omitnan')); % [signature x objective]
imagesc(heat);
colormap(greenCenteredColormap(256)); colorbar;
set(gca,'XTick',1:nO,'XTickLabel',OF_Plot,'YTick',1:numel(label_signatures_new),'YTickLabel',label_signatures_new);
xtickangle(45); xlabel('Objective function'); ylabel('Signature');
title('Mean absolute normalized signature error');
fontsize(f,12,'points');
save_if_requested(f, output_dir, 'normalized_error_heatmap_signature_objective_topN', make_png);

%% FIGURE 10: RANK SUMMARY, original-style lower-is-better ranks
f = figure('units','normalized','outerposition',[0 0 0.65 0.75]);
rank_heat = squeeze(mean(Rank_values(1:numel(sorted_sig_new),1:nC,:), 2, 'omitnan'));
imagesc(rank_heat);
colormap(flipud(parula)); colorbar;
set(gca,'XTick',1:nO,'XTickLabel',OF_Plot,'YTick',1:numel(label_signatures_new),'YTickLabel',label_signatures_new);
xtickangle(45); xlabel('Objective function'); ylabel('Signature');
title('Mean rank by signature and objective; lower is better');
fontsize(f,12,'points');
save_if_requested(f, output_dir, 'rank_summary_signature_objective_topN', make_png);

%% FIGURE 11: CORRELATION BETWEEN OBJECTIVE FUNCTIONS BASED ON NORMALIZED ERRORS
corr_input = reshape(permute(Error_Norm_Values_range,[2 1 3]), [], nO); % catchment/signature rows x objective cols
R = corr(corr_input, 'Rows','pairwise');
f = figure('units','normalized','outerposition',[0 0 0.55 0.55]);
imagesc(R,[-1 1]); axis square; colorbar; colormap(redblue_local(256));
set(gca,'XTick',1:nO,'XTickLabel',OF_Plot,'YTick',1:nO,'YTickLabel',OF_Plot);
xtickangle(45); title('Correlation of objective-function signature-error patterns');
fontsize(f,12,'points');
save_if_requested(f, output_dir, 'objective_error_correlation_topN', make_png);

%% FIGURE 12: IMPACT-LIKE SUMMARY FROM ORIGINAL SCRIPT
% Original impact sections mix multiple derived effects. Here the same idea is
% kept: summarize how much objective choice changes median normalized errors.
Impact_ObjectiveSpread = squeeze(max(Error_Norm_Values_range,[],3,'omitnan') - min(Error_Norm_Values_range,[],3,'omitnan'));
f = figure('units','normalized','outerposition',[0 0 0.65 0.75]);
imagesc(Impact_ObjectiveSpread);
colormap(parula); colorbar;
set(gca,'XTick',1:nC,'XTickLabel',catchments_labels,'YTick',1:numel(label_signatures_new),'YTickLabel',label_signatures_new);
xtickangle(45); xlabel('Catchment'); ylabel('Signature');
title('Range across objective functions in normalized median signature error');
fontsize(f,12,'points');
save_if_requested(f, output_dir, 'impact_objective_spread_topN', make_png);


%% FIGURE 13: 4x2 OBJECTIVE PANELS FOR EACH SIGNATURE, from plots_paper.m
% One figure per signature.  Each panel is one objective function; violins
% show top-N candidate signature errors by catchment for models passing the
% scalar benchmark gate.  This reproduces the old "signature_model_*.jpg"
% family, but uses the retained top-N candidates instead of one optimum per
% model.
yrange_values = [0.3,0.3,50,5,0.6,80,1.5,1,80,1,0.7,25,1,1.5,1];
signature_list_plot_sorted = {'Total RR','Event RR','Mean Half Flow Date (DOY)', ...
    'Q95 (mm/d)','High Flow Frequency','High Flow Duration (d)', ...
    'Q5 (mm/d)','Low Flow Frequency','Low Flow Duration (d)', ...
    'Baseflow Index','BF Recession Coefficient (1/d)','FDC Slope', ...
    'Flashiness Index','Variability Index','Rising Limb Density (1/d)'};

for si = 1:numel(sorted_sig_new)
    f = figure('units','normalized','outerposition',[0 0 0.55 0.6]);
    tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

    for oi = 1:nO
        nexttile; hold on;
        plot_data = squeeze(Error_Values(si,oi,:,:));  % [expanded model-candidate x catchment]
        X = [];
        G = [];
        n_per_catch = zeros(1,nC);
        
        for ci = 1:nC
            vals = finite_real_vector_local(plot_data(:,ci));
            n_per_catch(ci) = numel(vals);
        
            if ~isempty(vals)
                X = [X; vals]; %#ok<AGROW>
                G = [G; ci*ones(numel(vals),1)]; %#ok<AGROW>
            end
        end
        
        if ~isempty(X)
            plot_fixed_position_violins(X, G, 1:nC, colors(oi,:));
        end
        
        yline(0,'k-','LineWidth',1.0);
        title(strrep(OF_Plot{oi},'_','\_'));
                %for ci = 1:nC
        %    vals = finite_real_vector_local(plot_data(:,ci));
        %    X = [X; vals]; %#ok<AGROW>
        %    G = [G; ci*ones(numel(vals),1)]; %#ok<AGROW>
        %end


        xlim([0.5 nC+0.5]);
        xticks(1:nC);
        if oi > 4              % bottom row in 2x4 (panels 5-8)
            xticklabels(catchments_labels);
            xtickangle(45);
        else
            xticklabels([]);
        end
        if mod(oi-1, 4) == 0   % first column in 2x4 (panels 1, 5)
            ylabel('Signature error');
        end
        if si <= numel(yrange_values) && isfinite(yrange_values(si)) && yrange_values(si) > 0
            ylim([-yrange_values(si), yrange_values(si)]);
        end

        yl = ylim;
        y_text = yl(2) - 0.08 * range(yl);
        
        for ci = 1:nC
            if n_per_catch(ci) > 0
                if n_per_catch(ci) >= 1000
                    label_txt = sprintf('%.1fk', n_per_catch(ci)/1000);
                else
                    label_txt = sprintf('%d', n_per_catch(ci));
                end
        
                text(ci, y_text, label_txt, ...
                    'Rotation',90, ...
                    'HorizontalAlignment','center', ...
                    'VerticalAlignment','middle', ...
                    'FontSize',6, ...
                    'Color',[0.8 0.8 0.8]);
            end
        end

        grid on; grid minor;
    end
    sgtitle(f, sprintf('Signature: %s', signature_list_plot_sorted{si}));
    fontsize(f,11,'points');
    save_if_requested(f, output_dir, sprintf('signature_model_%02d_topN', si), make_png);
end

close all

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
            continue
        end

        try
            [f, yi] = ksdensity(vals);
        catch
            continue
        end

        if max(f) <= 0 || ~isfinite(max(f))
            continue
        end

        f = f ./ max(f) .* max_width;

        patch([pos - f, fliplr(pos + f)], ...
              [yi, fliplr(yi)], ...
              color, ...
              'FaceAlpha',0.45, ...
              'EdgeColor',[0.35 0.35 0.35], ...
              'LineWidth',0.5);

        med = median(vals,'omitnan');
        plot([pos - 0.18, pos + 0.18], [med, med], ...
             'Color',[0.25 0.25 0.25], ...
             'LineWidth',1.0);
    end
end

%% SAVE WORKSPACE FOR DEBUGGING
save(fullfile(base_path, 'plots_paper_topN_original_style_workspace.mat'), ...
    'D','top_mask','model_pass','plot_mask','OF_best','threshold','OF_value_cali','OF_value_cali_benchmark', ...
    'OF_TopN_Array','Signatures_Array','Error_Values','Error_Norm_Values','Error_Norm_Values_range', ...
    'TopNRaw','TopNStats','OF_q05','OF_q25','OF_q50','OF_q75','OF_q95','OF_IQR', ...
    'Signature_q05','Signature_q25','Signature_q50','Signature_q75','Signature_q95','Signature_IQR', ...
    'Rank_values','model_counter','objective_functions','model_list','catchments_aridity','catchments_labels', ...
    'sorted_sig_new','label_signatures_new','top_n_runs','-v7.3');

fprintf('\nDone. Wrote paper-style top-N figures to %s\n', output_dir);


%% =====================================================================
%  SUMMARY ERROR-METRIC PLOTS (appended)
%  Reproduces the "summary across all runs / models / catchments" figures
%  from the original plots_paper.m, adapted to the top-N retained-run
%  data structures built earlier in plots_paper_topN_from_plots_paper_style.m.
%
%  Inputs already in workspace:
%    - Error_Norm_Values        [nSig x nC x nO]  median (signed) normalized error per sig/catchment/objective
%    - Error_Norm_Values_range  [nSig x nC x nO]  per-catchment-normalized version of the above
%    - Signatures_Array         [nSig x nO x (nM*maxK) x nC] top-N signature values
%    - Error_Values             [nSig x nO x (nM*maxK) x nC] top-N raw signature errors
%    - obs_sig                  [nC x nSigFile]
%    - sorted_sig_new, label_signatures_new, file_sig_names
%    - OF_Plot, objective_functions, catchments_labels
%    - colors (nO x 3), output_dir, make_png
%  =====================================================================

nSig_sum = numel(sorted_sig_new);

% Short labels (used for the stacked/violin panels where space is tight)
label_signatures_short = { ...
    'TRR','ERR','MHFD', ...
    'Q95','HFF','HFD', ...
    'Q5','LFF','LFD', ...
    'BFI','BFRC','FDC Slope', ...
    'FI','VI','RLD'};
if numel(label_signatures_short) ~= nSig_sum
    label_signatures_short = label_signatures_new;  % fallback
end


%% SUMMARY FIGURE A: 4x2 violin grid of normalized signature error per objective
label_signatures_short = { ...
    'TRR','ERR','MHFD', ...
    'Q95','HFF','HFD', ...
    'Q5','LFF','LFD', ...
    'BFI','BFRC','FDC Slope', ...
    'FI','VI','RLD'};
if numel(label_signatures_short) ~= nSig_sum
    label_signatures_short = label_signatures_new;
end

nRowsA = ceil(nO/2);
fA = figure('units','normalized','outerposition',[0 0 0.6 1]);
tiledlayout(nRowsA, 2, 'TileSpacing','compact','Padding','compact');
for oi = 1:nO
    nexttile; hold on;
    X = [];
    G = [];
    counts_per_sig = zeros(1, nSig_sum);   % NEW
    
    for si = 1:nSig_sum
        vals = squeeze(Error_Norm_Values_range(si,:,oi));
        vals = finite_real_vector_local(vals);
        X = [X; vals]; %#ok<AGROW>
        G = [G; si*ones(numel(vals),1)]; %#ok<AGROW>
        counts_per_sig(si) = numel(vals);  % NEW
    end
    if ~isempty(X)
        safe_violinplot_local(X, G, colors(oi,:));
    end
    yline(0,'r-','Zero Error');
    ylim([-1 1]);
    xlim([0.5 15.5])
    set(gca,'XTick',1:nSig_sum,'XTickLabel',label_signatures_short);
    xtickangle(45);
    title(OF_Plot{oi});
    if mod(oi,2)==1
        ylabel('Normalized signature error');
    end
    grid on;

    % --- NEW: sample-size annotation above each violin ---
    for si = 1:nSig_sum
        if counts_per_sig(si) > 0
            text(si, 0.95, sprintf('%d', counts_per_sig(si)), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','top', ...
                'FontSize', 7, ...
                'Color', [0.25 0.25 0.25]);
        end
    end
end
sgtitle('Normalized signature error per objective (across catchments)');
fontsize(fA,10,'points');
save_if_requested(fA, output_dir, 'summary_violin_normerr_per_objective_topN', make_png);


%% SUMMARY FIGURE A1: 4x2 violin grid showing the FULL top-500 ensemble
% Same layout as Figure A (one panel per objective, one violin per signature),
% but each violin pools all top-100 run errors across catchments and passing
% models — no median-collapsing.
%
% Per-catchment range-normalization is applied so that signatures of
% different magnitudes (e.g. Q5 vs MHFD) sit on a comparable axis.

if ~exist('norm_denom', 'var')
    norm_denom = nan(nSig_sum, nC);
    for si = 1:nSig_sum
        for ci = 1:nC
            d = max(abs(squeeze(Error_Norm_Values(si, ci, :))), [], 'omitnan');
            if isfinite(d) && d > 0
                norm_denom(si, ci) = d;
            end
        end
    end
end

nRowsA1 = ceil(nO/2);
fA1 = figure('units','normalized','outerposition',[0 0 0.7 1]);
tiledlayout(nRowsA1, 2, 'TileSpacing','compact','Padding','compact');

top_n_runs = 500;
top_n_plot = 500;
for oi = 1:nO
    nexttile; hold on;
    X = [];
    G = [];

    for si = 1:nSig_sum
        d_col   = sorted_to_D_col(si);
        obs_col = sorted_to_obs(si);
        if isnan(d_col) || isnan(obs_col); continue; end

        vals_all = [];
        for ci = 1:nC
            obsval = obs_sig(ci, obs_col);
            denom  = norm_denom(si, ci);
            if ~isfinite(obsval) || ~isfinite(denom) || denom == 0
                continue;
            end

            for mi = 1:nM
                keep  = squeeze(plot_mask(mi, ci, oi, :));
                sig_v = squeeze(D.sig_all(mi, ci, oi, :, d_col));
                of_v  = squeeze(D.of_all(mi, ci, oi, :));

                keep = logical(keep(:));
                
                % Restrict to top_n_plot candidates only
                nUse = min(top_n_plot, numel(keep));
                keep_top = false(size(keep));
                keep_top(1:nUse) = keep(1:nUse);
                
                joint = keep_top & isfinite(of_v(:)) & isfinite(sig_v(:));
                if ~any(joint); continue; end
                
                err_norm = (sig_v(joint) - obsval) / denom;
                err_norm = err_norm(isfinite(err_norm) & abs(err_norm) <= 5);
                vals_all = [vals_all; err_norm(:)]; %#ok<AGROW>
            end
        end

        if ~isempty(vals_all)
            X = [X; vals_all];                          %#ok<AGROW>
            G = [G; si*ones(numel(vals_all), 1)];       %#ok<AGROW>
        end
    end

    if ~isempty(X)
        safe_violinplot_local(X, G, colors(oi,:));
    end

    yline(0,'r-','Zero Error');
    ylim([-1 1]);
    xlim([0.5 nSig_sum + 0.5]);
    set(gca,'XTick',1:nSig_sum,'XTickLabel',label_signatures_short);
    xtickangle(45);
    title(OF_Plot{oi});
    if mod(oi,2)==1
        ylabel('Range-normalized signature error');
    end
    grid on;
end

sgtitle(sprintf(['Full top-%d ensemble: per-run normalized signature error per objective\n', ...
                 '(pooled across passing models and catchments)'], top_n_plot));
fontsize(fA1, 10, 'points');
save_if_requested(fA1, output_dir, 'summary_violin_normerr_per_objective_topN_full_ensemble', make_png);

%%
plot_a1_with_subset(25,  D, plot_mask, obs_sig, sorted_to_D_col, sorted_to_obs, ...
                        norm_denom, colors, OF_Plot, label_signatures_short, ...
                        nO, nC, nM, nSig_sum, output_dir, make_png);

plot_a1_with_subset(500, D, plot_mask, obs_sig, sorted_to_D_col, sorted_to_obs, ...
                        norm_denom, colors, OF_Plot, label_signatures_short, ...
                        nO, nC, nM, nSig_sum, output_dir, make_png);
%%

plot_a1_with_subset(100, D, plot_mask, obs_sig, sorted_to_D_col, sorted_to_obs, ...
                        norm_denom, colors, OF_Plot, label_signatures_short, ...
                        nO, nC, nM, nSig_sum, output_dir, make_png);

%%
plot_a1_with_subset_balanced(100, D, plot_mask, OF_best, obs_sig, ...
    sorted_to_D_col, sorted_to_obs, norm_denom, colors, ...
    OF_Plot, label_signatures_short, nO, nC, nM, nSig_sum, output_dir, make_png);

%% SUMMARY FIGURE B: Per-signature violin grid showing all objectives
fB = figure('units','normalized','outerposition',[0 0 0.7 1]);
tiledlayout(4,4,'TileSpacing','compact','Padding','compact');

for si = 1:nSig_sum
    nexttile; hold on;

    % Collect data for all objectives in one vector
    X = [];
    G = [];
    for oi = 1:nO
        vals = squeeze(Error_Norm_Values_range(si,:,oi));
        vals = finite_real_vector_local(vals);
        if isempty(vals)
            % Inject a single NaN so the group still exists at position oi
            X = [X; NaN];
            G = [G; oi];
        else
            X = [X; vals];
            G = [G; oi*ones(numel(vals),1)];
        end
    end

    % Single violinplot call — positions 1..nO map to objectives 1..nO
    try
        vp = violinplot(X, G, 'ShowMean', false, 'ShowData', false);
        % Recolor each violin
        for oi = 1:numel(vp)
            if oi <= nO && ~isempty(vp(oi).ViolinPlot) && isvalid(vp(oi).ViolinPlot)
                vp(oi).ViolinPlot.FaceColor = colors(oi,:);
                vp(oi).ViolinPlot.FaceAlpha = 0.6;
            end
        end
    catch ME
        warning('violinplot failed for signature %d: %s', si, ME.message);
        % Scatter fallback
        for oi = 1:nO
            vals = finite_real_vector_local(squeeze(Error_Norm_Values_range(si,:,oi)));
            if isempty(vals); continue; end
            jitter = (rand(size(vals))-0.5)*0.15;
            scatter(oi+jitter, vals, 12, colors(oi,:), 'filled', 'MarkerFaceAlpha',0.5);
        end
    end

    % Overlay median line across objectives
    meds = nan(1,nO);
    for oi = 1:nO
        v = squeeze(Error_Norm_Values_range(si,:,oi));
        meds(oi) = median(v,'omitnan');
    end
    plot(1:nO, meds, 'k-', 'LineWidth', 1.2);

    yline(0,'r-');
    ylim([-1 1]);
    xlim([0.5 nO+0.5]);
    set(gca,'XTick',1:nO,'XTickLabel',OF_Plot);
    xtickangle(45);
    title(label_signatures_new{si});
    if mod(si-1, 4) == 0
        ylabel('Normalized error');
    end
    grid on;
end

sgtitle('Normalized signature error per signature (across catchments, all objectives)');
fontsize(fB,11,'points');
save_if_requested(fB, output_dir, 'summary_violin_normerr_per_signature_topN', make_png);


%% =====================================================================
%  SUMMARY FIGURE H: Paired significance test between objective functions
%  Reproduces the t-test block from plots_paper.m.
%
%  For each signature, for every unordered pair of objective functions
%  (OF_i, OF_j) with i<j, run a paired t-test on the per-catchment
%  median error vectors:
%
%      x = Error_Norm_Values(si, :, oi)'   % nC x 1 (median errors)
%      y = Error_Norm_Values(si, :, oj)'   % nC x 1
%      [h, p] = ttest(x, y)                % paired, same catchments
%
%  -> save_p_value [nSig x nPairs]   (with nPairs = nO*(nO-1)/2)
%
%  Then plot one violin per signature showing the distribution of p-values
%  across all OF-pairs, plus dashed reference lines at p = 0.05 and 0.10.
%
%  Also produce a summary table per signature with median p, fraction of
%  pairs with p < 0.10 and p < 0.05.
%
%  Workspace inputs:
%    - Error_Norm_Values [nSig x nC x nO]   (already in workspace)
%    - sorted_sig_new, label_signatures_new, objective_functions, OF_Plot
%    - output_dir, make_png
%  =====================================================================

nSig_sum = numel(sorted_sig_new);
[~, ~, nO_check] = size(Error_Norm_Values);
nO_h = nO_check;

% Build list of unordered objective pairs
pair_list = [];
for ii = 1:nO_h
    for jj = ii+1:nO_h
        pair_list = [pair_list; ii jj]; %#ok<AGROW>
    end
end
nPairs = size(pair_list, 1);

% Storage
save_p_value  = nan(nSig_sum, nPairs);
save_t_value  = nan(nSig_sum, nPairs);
save_mean_dif = nan(nSig_sum, nPairs);  % mean(x-y) for direction info

% Optional: per-pair sample size (catchments with both x and y finite)
save_n_paired = nan(nSig_sum, nPairs);

for si = 1:nSig_sum
    for pi = 1:nPairs
        oi = pair_list(pi, 1);
        oj = pair_list(pi, 2);

        x = squeeze(Error_Norm_Values(si, :, oi))';
        y = squeeze(Error_Norm_Values(si, :, oj))';

        % Paired t-test needs both finite at same catchment index
        ok = isfinite(x) & isfinite(y);
        if sum(ok) < 2
            continue;  % can't run a paired test with <2 valid pairs
        end

        try
            [~, p, ~, stats] = ttest(x(ok), y(ok));
            save_p_value(si, pi)  = p;
            save_t_value(si, pi)  = stats.tstat;
            save_mean_dif(si, pi) = mean(x(ok) - y(ok));
            save_n_paired(si, pi) = sum(ok);
        catch ME
            warning('ttest failed sig %d pair (%d,%d): %s', si, oi, oj, ME.message);
        end
    end
end

%% ---- Violin plot of p-values per signature ----
fH = figure('units','normalized','outerposition',[0 0 0.85 0.55]);
hold on;

X = [];
G = [];
for si = 1:nSig_sum
    vals = save_p_value(si, :)';
    vals = vals(isfinite(vals));
    if isempty(vals)
        % keep group present for axis alignment
        X = [X; NaN]; G = [G; si]; %#ok<AGROW>
    else
        X = [X; vals];                            %#ok<AGROW>
        G = [G; si*ones(numel(vals),1)];          %#ok<AGROW>
    end
end

try
    vp = violinplot(X, G, 'ShowMean', false, 'ShowData', true);
    % Uniform dark color to match the original
    for si = 1:numel(vp)
        if ~isempty(vp(si).ViolinPlot) && isvalid(vp(si).ViolinPlot)
            vp(si).ViolinPlot.FaceColor = [0.25 0.25 0.25];
            vp(si).ViolinPlot.FaceAlpha = 0.6;
        end
        if isprop(vp(si), 'ScatterPlot') && ~isempty(vp(si).ScatterPlot) ...
                && isvalid(vp(si).ScatterPlot)
            vp(si).ScatterPlot.SizeData = 12;
            vp(si).ScatterPlot.MarkerFaceAlpha = 0.5;
            vp(si).ScatterPlot.MarkerFaceColor = [0.25 0.25 0.25];
        end
    end
catch ME
    warning(ME.message,'p-value violinplot failed: %s');
    for si = 1:nSig_sum
        vals = save_p_value(si, :)';
        vals = vals(isfinite(vals));
        if isempty(vals); continue; end
        jitter = (rand(size(vals))-0.5) * 0.25;
        scatter(si + jitter, vals, 14, [0.25 0.25 0.25], 'filled', ...
                'MarkerFaceAlpha', 0.5);
    end
end

yline(0.05, 'r--', 'p = 0.05', 'LabelHorizontalAlignment','left', 'LineWidth', 2);
yline(0.10, 'k--', 'p = 0.10', 'LabelHorizontalAlignment','left', 'LineWidth', 2);

ylim([0 1]);
xlim([0.5 nSig_sum + 0.5]);
set(gca, 'XTick', 1:nSig_sum, 'XTickLabel', label_signatures_new);
xtickangle(45);
ylabel('p-value (paired t-test between OF pairs)');
xlabel('Signature');
title(sprintf(['Significance of objective-function differences per signature\n', ...
               '%d OF pairs per signature; %d catchments paired'], nPairs, nC));
grid on;
fontsize(fH, 13, 'points');
save_if_requested(fH, output_dir, 'summary_pvalue_violin_topN', make_png);

%% ---- Overview table: median p, fraction < 0.10 and < 0.05 ----
median_p       = median(save_p_value, 2, 'omitnan');
frac_p_lt_010  = mean(save_p_value < 0.10, 2, 'omitnan');
frac_p_lt_005  = mean(save_p_value < 0.05, 2, 'omitnan');

T_p_overview_topN = table(median_p, frac_p_lt_010, frac_p_lt_005, ...
    'VariableNames', {'median_p','frac_p_lt_0_10','frac_p_lt_0_05'}, ...
    'RowNames', label_signatures_new(:));

disp('Per-signature p-value overview (top-N):');
disp(T_p_overview_topN);

try
    writetable(T_p_overview_topN, ...
        fullfile(output_dir, 'summary_pvalue_overview_topN.csv'), ...
        'WriteRowNames', true);
catch ME
    warning(ME.message,'Could not write p-value overview CSV: %s');
end

%% TABLE: MEDIAN P-VALUES BY SIGNATURE

p_table_order = { ...
    'sig_TotalRR', ...
    'sig_EventRR', ...
    'sig_x_percentile_95per', ...
    'sig_x_percentile_5per', ...
    'sig_BFI', ...
    'sig_BaseflowRecessionK', ...
    'sig_FlashinessIndex', ...
    'sig_x_Q_frequency_low', ...
    'sig_x_Q_frequency_high', ...
    'sig_FDC_slope', ...
    'sig_VariabilityIndex', ...
    'sig_x_Q_duration_low', ...
    'sig_x_Q_duration_high', ...
    'sig_HFD_mean', ...
    'sig_RisingLimbDensity'};

p_table_labels = { ...
    'Total RR (-)', ...
    'Event RR (-)', ...
    'Q95 (mm/d)', ...
    'Q5 (mm/d)', ...
    'Baseflow Index (-)', ...
    'Baseflow Recession Coefficient (-)', ...
    'Flashiness Index (-)', ...
    'LF Frequency (-)', ...
    'HF Frequency (-)', ...
    'FDC Slope (-)', ...
    'Variability Index (-)', ...
    'LF Duration (days)', ...
    'HF Duration (days)', ...
    'MHFD (DOY)', ...
    'Rising Limb Density (-)'};

[ok_sig, p_sig_idx] = ismember(p_table_order, sorted_sig_new);
if ~all(ok_sig)
    error('Could not map p-value table signatures: %s', ...
        strjoin(p_table_order(~ok_sig), ', '));
end

median_p_table      = median_p(p_sig_idx);
frac_p_lt_010_table = 100 .* frac_p_lt_010(p_sig_idx);
frac_p_lt_005_table = 100 .* frac_p_lt_005(p_sig_idx);

T_p_latex = table( ...
    string(p_table_labels(:)), ...
    median_p_table(:), ...
    frac_p_lt_010_table(:), ...
    frac_p_lt_005_table(:), ...
    'VariableNames', {'Signature','median_p','pct_p_lt_0_10','pct_p_lt_0_05'});

writetable(T_p_latex, fullfile(output_dir, 'table_median_pvalues.csv'));

tex_file = fullfile(output_dir, 'table_median_pvalues.tex');
fid = fopen(tex_file, 'w');

fprintf(fid, '\\begin{table}[H]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Median $p$-values and fraction of tests with $p<0.10$ and $p<0.05$ across all objective function pairs for each hydrological signature.}\n');
fprintf(fid, '\\renewcommand{\\arraystretch}{1.2}\n');
fprintf(fid, '\\begin{tabular}{|l|c|c|c|}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, 'Signature & Median $p$ & $\\#$(p$<0.10$) [\\%%] & $\\#$(p$<0.05$) [\\%%] \\\\\n');
fprintf(fid, '\\hline\n');

for ii = 1:numel(p_sig_idx)
    fprintf(fid, '%s & %.3f & %.1f & %.1f \\\\\n', ...
        latex_escape_local(p_table_labels{ii}), ...
        median_p_table(ii), ...
        frac_p_lt_010_table(ii), ...
        frac_p_lt_005_table(ii));

    if ii == 8
        fprintf(fid, '\\hline\n');
    end
end

fprintf(fid, '\\hline\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\label{tab:median_pvals}\n');
fprintf(fid, '\\end{table}\n');

fclose(fid);

fprintf('Wrote p-value LaTeX table to %s\n', tex_file);

%% ---- Optional: heatmap of which OF-pairs are significantly different ----
% Average across signatures: for each pair, fraction of signatures with p < 0.05.
pair_frac_sig = mean(save_p_value < 0.05, 1, 'omitnan');  % 1 x nPairs

pair_matrix = nan(nO_h, nO_h);
for pi = 1:nPairs
    ii = pair_list(pi,1);
    jj = pair_list(pi,2);
    pair_matrix(ii, jj) = pair_frac_sig(pi);
    pair_matrix(jj, ii) = pair_frac_sig(pi);
end

fH2 = figure('units','normalized','outerposition',[0 0 0.5 0.55]);
imagesc(pair_matrix, [0 1]);
axis square;
colormap(parula);
cb = colorbar;
cb.Label.String = 'Fraction of signatures with p < 0.05';
set(gca, 'XTick', 1:nO_h, 'XTickLabel', OF_Plot, ...
         'YTick', 1:nO_h, 'YTickLabel', OF_Plot);
xtickangle(45);
title('OF-pair significance frequency across signatures');
for ii = 1:nO_h
    for jj = 1:nO_h
        if isfinite(pair_matrix(ii,jj))
            text(jj, ii, sprintf('%.2f', pair_matrix(ii,jj)), ...
                'HorizontalAlignment','center', 'FontSize', 9, ...
                'Color', [0 0 0]);
        end
    end
end
fontsize(fH2, 12, 'points');
save_if_requested(fH2, output_dir, 'summary_pvalue_OFpair_heatmap_topN', make_png);

%% ---- Significance gate for downstream filtering (used by Figure C) ----
pval_alpha = 0.1;   % gate threshold on MEDIAN p across all OF pairs

sig_passes = false(1, nSig_sum);
for si = 1:nSig_sum
    sig_passes(si) = median(save_p_value(si, :), 'omitnan') < pval_alpha;
end

fprintf('\nSignatures passing significance gate (median p across %d OF pairs < %.3f):\n', ...
        nPairs, pval_alpha);
for si = 1:nSig_sum
    if sig_passes(si)
        status = 'KEEP';
    else
        status = 'drop';
    end
    fprintf('  %-30s %s  (median p = %.4f, min p = %.4f)\n', ...
        label_signatures_new{si}, status, ...
        median(save_p_value(si,:), 'omitnan'), ...
        min(save_p_value(si,:), [], 'omitnan'));
end
%% SUMMARY FIGURE C: Stacked horizontal bar of cumulative |normalized error|
% Filtered to signatures where objective-function differences are statistically
% significant (any OF pair with p < pval_alpha in the paired t-test above).

sig_idx_keep   = find(sig_passes);
nSig_keep      = numel(sig_idx_keep);

if nSig_keep == 0
    warning('No signatures passed the significance gate; skipping Figure C.');
else
    median_abs_err_per_obj_sig = nan(nO, nSig_keep);
    for oi = 1:nO
        for k = 1:nSig_keep
            si = sig_idx_keep(k);
            median_abs_err_per_obj_sig(oi, k) = median(abs(squeeze(Error_Norm_Values_range(si,:,oi))), 'omitnan');
        end
    end

    graydient = flipud(repmat(linspace(0.2, 0.9, nSig_keep)', 1, 3));

    fC = figure('units','normalized','outerposition',[0 0 0.5 0.7]);
    bh = barh(median_abs_err_per_obj_sig, 'stacked', 'LineWidth', 1.0);
    for k = 1:nSig_keep
        bh(k).FaceColor = graydient(k,:);
        bh(k).EdgeColor = [0.25 0.25 0.25];
    end
    set(gca,'YDir','reverse','YTick',1:nO,'YTickLabel',OF_Plot);
    ylabel('Objective functions');
    xlabel('Cumulative median absolute normalized signature error');
    legend(label_signatures_short(sig_idx_keep), 'Location','eastoutside');
    grid on;
    title('Cumulative normalized error per objective function');
    fontsize(fC, 12, 'points');
    save_if_requested(fC, output_dir, 'summary_stacked_bar_normerr_topN', make_png);
end
%% SUMMARY FIGURE D: Line plot of cumulative normalized error per objective
% X-axis walks through signatures in order; y-axis is the running sum of the
% median |normalized error| across catchments. Reproduces "lineplot_new_error".

cum_err = zeros(nSig_sum+1, nO);
for si = 1:nSig_sum
    row = nan(1,nO);
    for oi = 1:nO
        row(oi) = median(abs(squeeze(Error_Norm_Values_range(si,:,oi))), 'omitnan');
    end
    cum_err(si+1,:) = cum_err(si,:) + row;
end

fD = figure('units','normalized','outerposition',[0 0 0.5 0.7]);
hold on;
for oi = 1:nO
    plot(0:nSig_sum, cum_err(:,oi), '-', 'Color', colors(oi,:), 'LineWidth', 3);
end
set(gca,'XTick',0:nSig_sum,'XTickLabel',[{''}, label_signatures_short(:)']);
xtickangle(45);
xlim([0 nSig_sum]);
ylabel('Cumulative median |normalized signature error|');
xlabel('Signatures (cumulative)');
legend(OF_Plot, 'Location','northwest');
grid on;
title(sprintf('Cumulative error walk per objective (top-%d retained runs)', top_n_runs));
fontsize(fD, 14, 'points');
save_if_requested(fD, output_dir, 'summary_lineplot_cumulative_normerr_topN', make_png);

%% SUMMARY FIGURE E: Bar chart of overall mean |normalized error| per objective
% A one-number-per-objective summary that collapses signatures and catchments.
overall_mean_abs_err = median(median_abs_err_per_obj_sig, 2, 'omitnan');  % [nO x 1]
overall_med_abs_err = nan(nO,1);
for oi = 1:nO
    vals = abs(squeeze(Error_Norm_Values_range(:,:,oi)));
    overall_med_abs_err(oi) = median(vals(:), 'omitnan');
end

fE = figure('units','normalized','outerposition',[0 0 0.45 0.55]);
hb = bar([overall_mean_abs_err overall_med_abs_err], 'grouped');
hb(1).FaceColor = [0.4 0.4 0.4];
hb(2).FaceColor = [0.75 0.75 0.75];
set(gca,'XTick',1:nO,'XTickLabel',OF_Plot);
xtickangle(45);
ylabel('|Normalized signature error|');
legend({'Mean over sigs x catchments','Median over sigs x catchments'}, 'Location','best');
grid on;
title(sprintf('Overall normalized signature-error summary per objective (top-%d)', top_n_runs));
fontsize(fE, 12, 'points');
save_if_requested(fE, output_dir, 'summary_overall_bar_normerr_topN', make_png);

%% OVERALL BAR PLOT WITH VARIABILITY FROM SIGNATURE TABLE

% Collapse signature-level table values per objective
% Use absolute median error per signature, then summarize across signatures
bar_mean_abs_median = mean(abs(medianSigObj), 1, 'omitnan');      % [1 x nO]
bar_median_abs_median = median(abs(medianSigObj), 1, 'omitnan'); % [1 x nO]

% Variability across signatures
err_sig_sd = std(abs(medianSigObj), 0, 1, 'omitnan');            % [1 x nO]

% Alternative variability: mean of within-signature SDs from table
err_within_sd = mean(stdSigObj, 1, 'omitnan');                   % [1 x nO]

fE = figure('units','normalized','outerposition',[0 0 0.45 0.55]);

Y = [bar_mean_abs_median(:), bar_median_abs_median(:)];
E = [err_sig_sd(:), err_within_sd(:)];

hb = bar(Y, 'grouped');
hb(1).FaceColor = [0.4 0.4 0.4];
hb(2).FaceColor = [0.75 0.75 0.75];
hold on;

% Add error bars manually for grouped bars
ngroups = size(Y,1);
nbars = size(Y,2);
x = nan(nbars, ngroups);

for ib = 1:nbars
    x(ib,:) = hb(ib).XEndPoints;
end

errorbar(x(1,:), Y(:,1), E(:,1), ...
    'k', 'linestyle','none', 'linewidth',0.8, 'capsize',6);

errorbar(x(2,:), Y(:,2), E(:,2), ...
    'k', 'linestyle','none', 'linewidth',0.8, 'capsize',6);

set(gca,'XTick',1:nO,'XTickLabel',OF_Plot);
xtickangle(45);
ylabel('|Normalized signature error|');

legend({'Mean abs. median across signatures', ...
        'Median abs. median across signatures', ...
        'SD across signature medians', ...
        'Mean within-signature SD'}, ...
        'Location','best');

grid on;
title(sprintf('Overall normalized signature-error summary per objective (top-%d)', top_n_runs));
fontsize(fE, 12, 'points');

save_if_requested(fE, output_dir, ...
    'summary_overall_bar_normerr_topN_with_variability', make_png);

%% SUMMARY TABLE: median & sd of normalized error per signature x objective
% Same shape as the T_stats table from the original script: rows alternate
% [median; sd] for each objective, columns are signatures.
medianSigObj = nan(nSig_sum, nO);
stdSigObj    = nan(nSig_sum, nO);
for si = 1:nSig_sum
    for oi = 1:nO
        v = squeeze(Error_Norm_Values_range(si,:,oi));
        medianSigObj(si,oi) = median(v, 'omitnan');
        stdSigObj(si,oi)    = std(v, 'omitnan');
    end
end

outMat = nan(2*nO, nSig_sum);
rowNames = cell(2*nO,1);
for oi = 1:nO
    outMat(2*oi-1,:) = medianSigObj(:,oi).';
    outMat(2*oi,  :) = stdSigObj(:,oi).';
    rowNames{2*oi-1} = sprintf('%s median', objective_functions{oi});
    rowNames{2*oi}   = sprintf('%s sd',     objective_functions{oi});
end
varNames = matlab.lang.makeValidName(label_signatures_new, 'ReplacementStyle','delete');
T_stats_topN = array2table(outMat, 'VariableNames', varNames, 'RowNames', rowNames);
disp('Median / SD of normalized signature error (rows alternate median, sd per objective):');
disp(T_stats_topN);
try
    writetable(T_stats_topN, fullfile(output_dir, 'summary_violin_errors_stats_topN.csv'), ...
        'WriteRowNames', true);
catch ME
    warning(ME.message,'Could not write summary stats CSV: %s');
end





%% =====================================================================
%  RANDOM FOREST IMPORTANCE: signature value AND signature error
%  =====================================================================
%  For each signature, fit two random forests:
%    1) Response = median simulated signature value (per cell)
%    2) Response = median signature error (per cell)
%  Predictors in both cases: Catchment, Model, OF (all categorical).
%
%  Out-of-bag permuted predictor importance scores are extracted,
%  normalized to sum to one within each signature, and reported as
%  the relative importance of each factor.
%  =====================================================================

rf_n_trees   = 500;
rf_seed      = 52;
rng(rf_seed);

nSig_rf = numel(sorted_sig_new);

% Storage: rows = signatures, columns = {Catchment, Model, OF}
RF.importance_value = nan(nSig_rf, 3);
RF.importance_error = nan(nSig_rf, 3);
RF.factor_names     = {'Catchment','Model','OF'};

fprintf('\n===== Random forest feature importance =====\n');

for si = 1:nSig_rf
    d_col   = sorted_to_D_col(si);
    obs_col = sorted_to_obs(si);
    if isnan(d_col) || isnan(obs_col); continue; end

    fprintf('  [%2d/%2d] %s\n', si, nSig_rf, label_signatures_new{si});

    % --- Collect per-cell medians (one row per Catchment × Model × OF) ---
    sig_val_list = [];
    sig_err_list = [];
    catch_list = []; model_list = []; of_list = [];

    for ci = 1:nC
        obs_val = obs_sig(ci, obs_col);
        if ~isfinite(obs_val); continue; end

        for mi = 1:nM
            for oi = 1:nO
                keep  = squeeze(plot_mask(mi, ci, oi, :));
                sig_v = squeeze(D.sig_all(mi, ci, oi, :, d_col));
                joint = logical(keep) & isfinite(sig_v);
                vals = sig_v(joint);
                if numel(vals) < 5; continue; end

                median_value = median(vals, 'omitnan');
                median_error = median_value - obs_val;

                sig_val_list = [sig_val_list; median_value]; %#ok<AGROW>
                sig_err_list = [sig_err_list; median_error]; %#ok<AGROW>
                catch_list   = [catch_list; ci];             %#ok<AGROW>
                model_list   = [model_list; mi];             %#ok<AGROW>
                of_list      = [of_list; oi];                %#ok<AGROW>
            end
        end
    end

    if numel(sig_val_list) < 50
        fprintf('    too few cells (%d), skipping\n', numel(sig_val_list));
        continue;
    end

    % Build predictor table with categorical factors
    X = table(categorical(catch_list), categorical(model_list), categorical(of_list), ...
              'VariableNames', {'Catchment','Model','OF'});

    % --- RF 1: predict signature value ---
    try
        rf_val = TreeBagger(rf_n_trees, X, sig_val_list, ...
            'Method', 'regression', ...
            'OOBPredictorImportance', 'on', ...
            'CategoricalPredictors', 'all');
        imp_val = rf_val.OOBPermutedPredictorDeltaError;
        imp_val(imp_val < 0) = 0;  % clip negatives (interpret as "no importance")
        if sum(imp_val) > 0
            RF.importance_value(si, :) = imp_val / sum(imp_val);
        end
    catch ME
        fprintf('    RF on value failed: %s\n', ME.message);
    end

    % --- RF 2: predict signature error ---
    try
        rf_err = TreeBagger(rf_n_trees, X, sig_err_list, ...
            'Method', 'regression', ...
            'OOBPredictorImportance', 'on', ...
            'CategoricalPredictors', 'all');
        imp_err = rf_err.OOBPermutedPredictorDeltaError;
        imp_err(imp_err < 0) = 0;
        if sum(imp_err) > 0
            RF.importance_error(si, :) = imp_err / sum(imp_err);
        end
    catch ME
        fprintf('    RF on error failed: %s\n', ME.message);
    end

    fprintf('    Value:  Catch=%.2f  Model=%.2f  OF=%.2f\n', RF.importance_value(si,:));
    fprintf('    Error:  Catch=%.2f  Model=%.2f  OF=%.2f\n', RF.importance_error(si,:));
end

%% ---- Plot: stacked bars for RF analyses, significant signatures only ----

rf_sig_idx = find(sig_passes);   % signatures passing median-p gate
nSig_rf_plot = numel(rf_sig_idx);

if nSig_rf_plot == 0
    warning('No signatures passed the significance gate; skipping RF importance plot.');
else
    cmap_rf = [ 0.30 0.70 0.40;   % Catchment
                0.20 0.50 0.80;   % Model
                0.85 0.20 0.20];  % OF

    data_val = RF.importance_value(rf_sig_idx, :);
    data_err = RF.importance_error(rf_sig_idx, :);

    data_val(~isfinite(data_val)) = 0;
    data_err(~isfinite(data_err)) = 0;

    fRF = figure('units','normalized','outerposition',[0 0 1 0.55]);
    tl = tiledlayout(1, 2, 'TileSpacing','compact','Padding','compact');

    % --- Signature value ---
    nexttile;
    b1 = bar(data_val, 'stacked');
    for s = 1:numel(b1)
        b1(s).FaceColor = cmap_rf(s,:);
        b1(s).EdgeColor = 'none';
    end
    xticks(1:nSig_rf_plot);
    xticklabels(label_signatures_short(rf_sig_idx));
    xtickangle(45);
    ylabel('Relative importance');
    ylim([0 1]);
    title('RF importance for signature value');
    grid on;
    box on;
    
    
    % --- Signature error ---
    nexttile;
    b2 = bar(data_err, 'stacked');
    for s = 1:numel(b2)
        b2(s).FaceColor = cmap_rf(s,:);
        b2(s).EdgeColor = 'none';
    end
    xticks(1:nSig_rf_plot);
    xticklabels(label_signatures_short(rf_sig_idx));
    xtickangle(45);
    ylabel('Relative importance');
    ylim([0 1]);
    title('RF importance for signature error');
    grid on;
    box on;


    lgd = legend(b2, {'Catchment','Model','OF'}, ...
        'Orientation','horizontal','Box','off');
    lgd.Layout.Tile = 'south';

    title(tl, 'Random forest feature importance for significant signatures only');
    fontsize(14, 'points');
    save_if_requested(fRF, output_dir, ...
        'rf_importance_value_vs_error_significant_only', make_png);
end

%% Random Forest without Benchmarking of Models

%% =====================================================================
%  RANDOM FOREST IMPORTANCE: FULL RETAINED SET
%  signature value AND signature error
%  =====================================================================

rf_n_trees   = 500;
rf_seed      = 52;
rng(rf_seed);

rf_mask = top_mask;          % FULL SET: no benchmark gate
rf_tag  = 'all_retained';

nSig_rf = numel(sorted_sig_new);

RF_all = struct();
RF_all.importance_value = nan(nSig_rf, 3);
RF_all.importance_error = nan(nSig_rf, 3);
RF_all.factor_names     = {'Catchment','Model','OF'};
RF_all.mask_description = 'Uses all finite retained top-N candidates; no benchmark pass gate.';

fprintf('\n===== Random forest feature importance: %s =====\n', rf_tag);

for si = 1:nSig_rf
    d_col   = sorted_to_D_col(si);
    obs_col = sorted_to_obs(si);
    if isnan(d_col) || isnan(obs_col); continue; end

    fprintf('  [%2d/%2d] %s\n', si, nSig_rf, label_signatures_new{si});

    sig_val_list = [];
    sig_err_list = [];
    catch_list   = [];
    model_list_i = [];
    of_list      = [];

    for ci = 1:nC
        obs_val = obs_sig(ci, obs_col);
        if ~isfinite(obs_val); continue; end

        for mi = 1:nM
            for oi = 1:nO
                keep  = squeeze(rf_mask(mi, ci, oi, :));
                sig_v = squeeze(D.sig_all(mi, ci, oi, :, d_col));

                joint = logical(keep(:)) & isfinite(sig_v(:));
                vals  = sig_v(joint);

                if numel(vals) < 5
                    continue;
                end

                median_value = median(vals, 'omitnan');
                median_error = median_value - obs_val;

                sig_val_list = [sig_val_list; median_value]; %#ok<AGROW>
                sig_err_list = [sig_err_list; median_error]; %#ok<AGROW>
                catch_list   = [catch_list; ci];             %#ok<AGROW>
                model_list_i = [model_list_i; mi];           %#ok<AGROW>
                of_list      = [of_list; oi];                %#ok<AGROW>
            end
        end
    end

    if numel(sig_val_list) < 50
        fprintf('    too few cells (%d), skipping\n', numel(sig_val_list));
        continue;
    end

    X = table(categorical(catch_list), ...
              categorical(model_list_i), ...
              categorical(of_list), ...
              'VariableNames', {'Catchment','Model','OF'});

    % --- RF 1: predict signature value ---
    try
        rf_val = TreeBagger(rf_n_trees, X, sig_val_list, ...
            'Method', 'regression', ...
            'OOBPredictorImportance', 'on', ...
            'CategoricalPredictors', 'all');

        imp_val = rf_val.OOBPermutedPredictorDeltaError;
        imp_val(imp_val < 0) = 0;

        if sum(imp_val) > 0
            RF_all.importance_value(si, :) = imp_val / sum(imp_val);
        end
    catch ME
        fprintf('    RF on value failed: %s\n', ME.message);
    end

    % --- RF 2: predict signature error ---
    try
        rf_err = TreeBagger(rf_n_trees, X, sig_err_list, ...
            'Method', 'regression', ...
            'OOBPredictorImportance', 'on', ...
            'CategoricalPredictors', 'all');

        imp_err = rf_err.OOBPermutedPredictorDeltaError;
        imp_err(imp_err < 0) = 0;

        if sum(imp_err) > 0
            RF_all.importance_error(si, :) = imp_err / sum(imp_err);
        end
    catch ME
        fprintf('    RF on error failed: %s\n', ME.message);
    end

    fprintf('    Value:  Catch=%.2f  Model=%.2f  OF=%.2f\n', ...
        RF_all.importance_value(si,:));
    fprintf('    Error:  Catch=%.2f  Model=%.2f  OF=%.2f\n', ...
        RF_all.importance_error(si,:));
end

%% ---- Plot: RF importance across ALL signatures (supports OF-insensitivity argument) ----
%  Uses the benchmark-gated RF struct, but plots every signature (no significance gate)
%  so that OF-insensitive signatures (small OF band) are visible alongside sensitive ones.

all_sig_idx   = 1:nSig_rf;
nSig_all_plot = nSig_rf;

cmap_rf = [ 0.30 0.70 0.40;   % Catchment
            0.20 0.50 0.80;   % Model
            0.85 0.20 0.20];  % OF

data_val = RF.importance_value(all_sig_idx, :);
data_err = RF.importance_error(all_sig_idx, :);

data_val(~isfinite(data_val)) = 0;
data_err(~isfinite(data_err)) = 0;

lab_ord = label_signatures_short(all_sig_idx);

fRF = figure('units','normalized','outerposition',[0 0 1 0.55]);
tl = tiledlayout(1, 2, 'TileSpacing','compact','Padding','compact');

nexttile;
b1 = bar(data_val, 'stacked');
for s = 1:numel(b1); b1(s).FaceColor = cmap_rf(s,:); b1(s).EdgeColor = 'none'; end
xticks(1:nSig_all_plot); xticklabels(lab_ord); xtickangle(45);
ylabel('Relative importance'); ylim([0 1]);
title('RF importance for signature value'); grid on; box on;

nexttile;
b2 = bar(data_err, 'stacked');
for s = 1:numel(b2); b2(s).FaceColor = cmap_rf(s,:); b2(s).EdgeColor = 'none'; end
xticks(1:nSig_all_plot); xticklabels(lab_ord); xtickangle(45);
ylabel('Relative importance'); ylim([0 1]);
title('RF importance for signature error'); grid on; box on;

lgd = legend(b2, {'Catchment','Model','OF'}, 'Orientation','horizontal','Box','off');
lgd.Layout.Tile = 'south';

title(tl, 'Random forest feature importance: all signatures');
fontsize(14, 'points');

save_if_requested(fRF, output_dir, ...
    'rf_importance_value_vs_error_all_signatures', make_png);

%% ---- Plot: stacked bars for RF analyses, significant signatures only ----

rf_sig_idx = find(sig_passes);
nSig_rf_plot = numel(rf_sig_idx);

if nSig_rf_plot == 0
    warning('No signatures passed the significance gate; skipping RF importance plot.');
else
    cmap_rf = [ 0.30 0.70 0.40;
                0.20 0.50 0.80;
                0.85 0.20 0.20];

    data_val = RF_all.importance_value(rf_sig_idx, :);
    data_err = RF_all.importance_error(rf_sig_idx, :);

    data_val(~isfinite(data_val)) = 0;
    data_err(~isfinite(data_err)) = 0;

    fRF = figure('units','normalized','outerposition',[0 0 1 0.55]);
    tl = tiledlayout(1, 2, 'TileSpacing','compact','Padding','compact');

    nexttile;
    b1 = bar(data_val, 'stacked');
    for s = 1:numel(b1)
        b1(s).FaceColor = cmap_rf(s,:);
        b1(s).EdgeColor = 'none';
    end
    xticks(1:nSig_rf_plot);
    xticklabels(label_signatures_short(rf_sig_idx));
    xtickangle(45);
    ylabel('Relative importance');
    ylim([0 1]);
    title('RF importance for signature value');
    grid on;
    box on;

    nexttile;
    b2 = bar(data_err, 'stacked');
    for s = 1:numel(b2)
        b2(s).FaceColor = cmap_rf(s,:);
        b2(s).EdgeColor = 'none';
    end
    xticks(1:nSig_rf_plot);
    xticklabels(label_signatures_short(rf_sig_idx));
    xtickangle(45);
    ylabel('Relative importance');
    ylim([0 1]);
    title('RF importance for signature error');
    grid on;
    box on;

    lgd = legend(b2, {'Catchment','Model','OF'}, ...
        'Orientation','horizontal','Box','off');
    lgd.Layout.Tile = 'south';
    fontsize(14, 'points');

    title(tl, 'Random forest feature importance: all retained top-N candidates');

    save_if_requested(fRF, output_dir, ...
        'rf_importance_value_vs_error_significant_only_all_retained', make_png);
end

%% =====================================================================
%  PARAMETER BOUNDARY HITS AMONG BENCHMARK-PASSING TOP-N RUNS
%  =====================================================================
fprintf('\n===== Parameter boundary-hit analysis =====\n');

tol_abs = 1e-8;
tol_rel = 1e-6;

param_field_candidates = {'par_all','param_all','theta_all','parameters_all','params_all'};
lower_field_candidates = {'par_lower','param_lower','theta_lower','lb','lower_bounds','param_min'};
upper_field_candidates = {'par_upper','param_upper','theta_upper','ub','upper_bounds','param_max'};

param_field = '';
for ii = 1:numel(param_field_candidates)
    if isfield(S, param_field_candidates{ii})
        param_field = param_field_candidates{ii};
        break;
    end
end

lower_field = '';
for ii = 1:numel(lower_field_candidates)
    if isfield(S, lower_field_candidates{ii})
        lower_field = lower_field_candidates{ii};
        break;
    end
end

upper_field = '';
for ii = 1:numel(upper_field_candidates)
    if isfield(S, upper_field_candidates{ii})
        upper_field = upper_field_candidates{ii};
        break;
    end
end

if isempty(param_field)
    warning('No parameter-value array found in S. Skipping boundary-hit analysis.');
elseif isempty(lower_field) || isempty(upper_field)
    warning('Parameter values found in S.%s, but bounds are missing. Skipping boundary-hit analysis.', param_field);
else
    fprintf('Using parameter values: S.%s\n', param_field);
    fprintf('Using parameter bounds: S.%s / S.%s\n', lower_field, upper_field);

    param_all = double(S.(param_field));

    if ndims(param_all) ~= 5
        warning('S.%s is not 5-D. Expected model x catchment x objective x candidate x parameter. Skipping.', param_field);
    else
        % Align parameter array to same objective/catchment ordering as D.of_all.
        param_all = param_all(:,:,keep_obj,:,:);
        param_all = param_all(:,:,new_idx,:,:);
        param_all = param_all(:,cat_idx_in_D,:,:,:);

        [nM_p,nC_p,nO_p,nCand_p,nP] = size(param_all);

        if nM_p ~= nM || nC_p ~= nC || nO_p ~= nO || nCand_p ~= nCand
            warning(['Parameter array dimensions after alignment do not match D.of_all. Skipping.\n', ...
                     'param_all: %d x %d x %d x %d x %d\n', ...
                     'D.of_all:  %d x %d x %d x %d'], ...
                     nM_p,nC_p,nO_p,nCand_p,nP, nM,nC,nO,nCand);
            return
        end

        % Parameter names
        if isfield(S, 'par_names')
            param_names = cellstr(string(S.par_names(:)))';
        elseif isfield(S, 'param_names')
            param_names = cellstr(string(S.param_names(:)))';
        elseif isfield(S, 'parameter_names')
            param_names = cellstr(string(S.parameter_names(:)))';
        else
            param_names = arrayfun(@(ii) sprintf('p%d', ii), 1:nP, 'UniformOutput', false);
        end

        if numel(param_names) ~= nP
            param_names = arrayfun(@(ii) sprintf('p%d', ii), 1:nP, 'UniformOutput', false);
        end

        % Actual parameter count per model.
        if isfield(S, 'nPars_by_model')
            nPars_by_model = double(S.nPars_by_model(:));
        else
            nPars_by_model = nan(nM,1);
            for mi = 1:nM
                tok = regexp(D.models{mi}, '_(\d+)p_', 'tokens', 'once');
                if isempty(tok)
                    warning('Could not parse parameter count from model name: %s. Using nP=%d.', D.models{mi}, nP);
                    nPars_by_model(mi) = nP;
                else
                    nPars_by_model(mi) = str2double(tok{1});
                end
            end
        end

        nPars_by_model = min(nPars_by_model, nP);

        param_lower = double(S.(lower_field));
        param_upper = double(S.(upper_field));

        if ~isequal(size(param_lower), [nM nP])
            param_lower = expand_bounds_inline(param_lower, nM, nP);
        end
        if ~isequal(size(param_upper), [nM nP])
            param_upper = expand_bounds_inline(param_upper, nM, nP);
        end

        % Validate real, non-padded bounds only.
        missing_bounds = false(nM,1);
        for mi = 1:nM
            pidx = 1:nPars_by_model(mi);
            missing_bounds(mi) = any(~isfinite(param_lower(mi,pidx))) || ...
                                 any(~isfinite(param_upper(mi,pidx)));
        end

        if any(missing_bounds)
            warning('Bounds missing for %d/%d real model parameter sets: %s', ...
                sum(missing_bounds), nM, strjoin(D.models(missing_bounds), ', '));
        end

        % Main storage
        n_pass_runs_boundary = zeros(nM,nC,nO);

        % Run-level: did any real parameter hit a boundary?
        frac_any_boundary = nan(nM,nC,nO);
        frac_low_boundary = nan(nM,nC,nO);
        frac_up_boundary  = nan(nM,nC,nO);

        % Parameter-level: did parameter pi hit a boundary?
        frac_param_any = nan(nM,nC,nO,nP);
        frac_param_low = nan(nM,nC,nO,nP);
        frac_param_up  = nan(nM,nC,nO,nP);

        % Parameter-normalized cell metric:
        % fraction of all real parameter-values at boundary within a cell.
        frac_param_values_boundary = nan(nM,nC,nO);

        % Main loop
        for mi = 1:nM
            nPars_this = nPars_by_model(mi);
            pidx = 1:nPars_this;

            lb = param_lower(mi,pidx);
            ub = param_upper(mi,pidx);

            valid_bounds = isfinite(lb) & isfinite(ub) & ub > lb;
            if ~any(valid_bounds)
                continue;
            end

            prange = ub - lb;
            tol = max(tol_abs, tol_rel .* abs(prange));

            for ci = 1:nC
                for oi = 1:nO
                    keep = logical(squeeze(plot_mask(mi,ci,oi,:)));
                    n_pass_runs_boundary(mi,ci,oi) = sum(keep);

                    if ~any(keep)
                        continue;
                    end

                    P = squeeze(param_all(mi,ci,oi,keep,pidx));

                    if isvector(P)
                        P = reshape(P, [], nPars_this);
                    end

                    P = double(P);

                    hit_low = false(size(P));
                    hit_up  = false(size(P));

                    for pp = 1:nPars_this
                        if valid_bounds(pp)
                            hit_low(:,pp) = abs(P(:,pp) - lb(pp)) <= tol(pp);
                            hit_up(:,pp)  = abs(P(:,pp) - ub(pp)) <= tol(pp);
                        end
                    end

                    hit_any = hit_low | hit_up;

                    valid_P = isfinite(P);
                    valid_hit = valid_P & repmat(valid_bounds, size(P,1), 1);

                    frac_any_boundary(mi,ci,oi) = mean(any(hit_any(:,valid_bounds), 2), 'omitnan');
                    frac_low_boundary(mi,ci,oi) = mean(any(hit_low(:,valid_bounds), 2), 'omitnan');
                    frac_up_boundary(mi,ci,oi)  = mean(any(hit_up(:,valid_bounds), 2), 'omitnan');

                    denom = sum(valid_hit(:));
                    if denom > 0
                        frac_param_values_boundary(mi,ci,oi) = sum(hit_any(valid_hit), 'omitnan') / denom;
                    end

                    for pp = 1:nPars_this
                        if valid_bounds(pp)
                            ok = isfinite(P(:,pp));
                            if any(ok)
                                frac_param_any(mi,ci,oi,pp) = mean(hit_any(ok,pp), 'omitnan');
                                frac_param_low(mi,ci,oi,pp) = mean(hit_low(ok,pp), 'omitnan');
                                frac_param_up(mi,ci,oi,pp)  = mean(hit_up(ok,pp),  'omitnan');
                            end
                        end
                    end
                end
            end
        end

        % Tables
        rows = {};
        for mi = 1:nM
            for ci = 1:nC
                for oi = 1:nO
                    rows(end+1,:) = { ...
                        D.models{mi}, D.catchments{ci}, catchments_labels{ci}, ...
                        objective_functions{oi}, OF_Plot{oi}, ...
                        nPars_by_model(mi), ...
                        n_pass_runs_boundary(mi,ci,oi), ...
                        frac_any_boundary(mi,ci,oi), ...
                        frac_param_values_boundary(mi,ci,oi), ...
                        frac_low_boundary(mi,ci,oi), ...
                        frac_up_boundary(mi,ci,oi)}; %#ok<AGROW>
                end
            end
        end

        T_boundary_summary = cell2table(rows, ...
            'VariableNames', {'model','catchment','catchment_label','objective','objective_label', ...
                              'n_parameters','n_runs_used', ...
                              'frac_runs_any_boundary', ...
                              'frac_parameter_values_boundary', ...
                              'frac_runs_any_lower_boundary', ...
                              'frac_runs_any_upper_boundary'});

        writetable(T_boundary_summary, ...
            fullfile(output_dir, 'boundary_hit_summary_model_catchment_objective.csv'));

        rows = {};
        for mi = 1:nM
            nPars_this = nPars_by_model(mi);
            for ci = 1:nC
                for oi = 1:nO
                    for pi = 1:nPars_this
                        rows(end+1,:) = { ...
                            D.models{mi}, D.catchments{ci}, catchments_labels{ci}, ...
                            objective_functions{oi}, OF_Plot{oi}, ...
                            param_names{pi}, pi, ...
                            n_pass_runs_boundary(mi,ci,oi), ...
                            frac_param_any(mi,ci,oi,pi), ...
                            frac_param_low(mi,ci,oi,pi), ...
                            frac_param_up(mi,ci,oi,pi)}; %#ok<AGROW>
                    end
                end
            end
        end

        T_boundary_by_param = cell2table(rows, ...
            'VariableNames', {'model','catchment','catchment_label','objective','objective_label', ...
                              'parameter','parameter_index','n_runs_used', ...
                              'frac_boundary','frac_lower_boundary','frac_upper_boundary'});

        writetable(T_boundary_by_param, ...
            fullfile(output_dir, 'boundary_hit_by_parameter_model_catchment_objective.csv'));

        % Weighted summaries
        overall_by_obj_any = nan(nO,1);
        overall_by_obj_paramnorm = nan(nO,1);
        overall_by_catch_any = nan(nC,1);
        overall_by_catch_paramnorm = nan(nC,1);
        overall_by_model_any = nan(nM,1);
        overall_by_model_paramnorm = nan(nM,1);

        for oi = 1:nO
            x = frac_any_boundary(:,:,oi);
            overall_by_obj_any(oi) = median(x(:), 'omitnan');

            x = frac_param_values_boundary(:,:,oi);
            overall_by_obj_paramnorm(oi) = median(x(:), 'omitnan');
        end

        for ci = 1:nC
            x = squeeze(frac_any_boundary(:,ci,:));
            overall_by_catch_any(ci) = median(x(:), 'omitnan');

            x = squeeze(frac_param_values_boundary(:,ci,:));
            overall_by_catch_paramnorm(ci) = median(x(:), 'omitnan');
        end

        for mi = 1:nM
            x = squeeze(frac_any_boundary(mi,:,:));
            overall_by_model_any(mi) = median(x(:), 'omitnan');

            x = squeeze(frac_param_values_boundary(mi,:,:));
            overall_by_model_paramnorm(mi) = median(x(:), 'omitnan');
        end

        % Parameter-specific weighted frequency by actual run count.
        param_hit_count = zeros(nP,1);
        param_run_count = zeros(nP,1);

        for mi = 1:nM
            nPars_this = nPars_by_model(mi);

            for pi = 1:nPars_this
                x = squeeze(frac_param_any(mi,:,:,pi));
                n = squeeze(n_pass_runs_boundary(mi,:,:));

                ok = isfinite(x) & n > 0;

                param_hit_count(pi) = param_hit_count(pi) + sum(x(ok) .* n(ok), 'omitnan');
                param_run_count(pi) = param_run_count(pi) + sum(n(ok), 'omitnan');
            end
        end

        overall_by_param_weighted = param_hit_count ./ param_run_count;

        T_boundary_param_overall = table(param_names(:), param_run_count, overall_by_param_weighted, ...
            'VariableNames', {'parameter','n_parameter_values','weighted_frac_boundary'});

        writetable(T_boundary_param_overall, ...
            fullfile(output_dir, 'boundary_hit_parameter_overall_weighted.csv'));

        T_boundary_objective_overall = table(OF_Plot(:), overall_by_obj_any, overall_by_obj_paramnorm, ...
            'VariableNames', {'objective','median_frac_runs_any_boundary','median_frac_parameter_values_boundary'});

        writetable(T_boundary_objective_overall, ...
            fullfile(output_dir, 'boundary_hit_objective_overall.csv'));

        % Plot 1: objective x catchment, run-level any boundary
        heat_any = nan(nO,nC);
        heat_paramnorm = nan(nO,nC);

        for oi = 1:nO
            for ci = 1:nC
                x = squeeze(frac_any_boundary(:,ci,oi));
                heat_any(oi,ci) = median(x(:), 'omitnan');

                x = squeeze(frac_param_values_boundary(:,ci,oi));
                heat_paramnorm(oi,ci) = median(x(:), 'omitnan');
            end
        end

        f = figure('Units','normalized','OuterPosition',[0 0 0.85 0.55]);
        tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

        nexttile;
        imagesc(heat_any, [0 1]);
        colormap(parula);
        colorbar;
        xticks(1:nC); xticklabels(catchments_labels); xtickangle(45);
        yticks(1:nO); yticklabels(OF_Plot);
        xlabel('Catchment'); ylabel('Objective');
        title('Run-level: at least one parameter at boundary');

        nexttile;
        imagesc(heat_paramnorm, [0 1]);
        colormap(parula);
        cb = colorbar;
        cb.Label.String = 'Median fraction';
        xticks(1:nC); xticklabels(catchments_labels); xtickangle(45);
        yticks(1:nO); yticklabels(OF_Plot);
        xlabel('Catchment'); ylabel('Objective');
        title('Parameter-normalized: parameter-values at boundary');

        fontsize(f, 11, 'points');
        save_if_requested(f, output_dir, ...
            'boundary_hit_objective_catchment_any_vs_paramnorm', make_png);

        % Plot 2: parameter-specific weighted boundary frequency
        valid_param = param_run_count > 0 & isfinite(overall_by_param_weighted);
        valid_idx = find(valid_param);

        [vals_sorted, ord0] = sort(overall_by_param_weighted(valid_param), 'descend');
        ord = valid_idx(ord0);

        f = figure('Units','normalized','OuterPosition',[0 0 0.65 0.75]);
        barh(vals_sorted, 'FaceColor', [0.45 0.45 0.45], 'EdgeColor', 'none');
        set(gca, 'YDir', 'reverse');
        yticks(1:numel(ord));
        yticklabels(strrep(param_names(ord), '_', '\_'));
        xlabel('Weighted fraction of parameter values at boundary');
        ylabel('Parameter index');
        title('Parameter-specific boundary-hit frequency');
        xlim([0 1]);
        grid on;
        fontsize(f, 12, 'points');

        save_if_requested(f, output_dir, ...
            'boundary_hit_parameter_weighted_ranked', make_png);

        % Plot 3: model-level comparison: any-boundary vs parameter-normalized
        [~, model_ord] = sort(overall_by_model_paramnorm, 'descend');

        f = figure('Units','normalized','OuterPosition',[0 0 0.75 0.85]);
        Y = [overall_by_model_any(model_ord), overall_by_model_paramnorm(model_ord)];
        barh(Y, 'grouped');
        set(gca, 'YDir', 'reverse');
        yticks(1:nM);
        yticklabels(strrep(D.models(model_ord), '_', '\_'));
        xlabel('Fraction');
        ylabel('Model');
        legend({'Any parameter at boundary per run', ...
                'Parameter-values at boundary'}, ...
                'Location','southeast');
        title('Model-level boundary-hit frequency');
        xlim([0 1]);
        grid on;
        fontsize(f, 11, 'points');

        save_if_requested(f, output_dir, ...
            'boundary_hit_model_any_vs_paramnorm_ranked', make_png);

        % Save workspace
        save(fullfile(output_dir, 'boundary_hit_analysis_workspace.mat'), ...
            'param_field','lower_field','upper_field','param_names','nPars_by_model', ...
            'param_lower','param_upper','tol_abs','tol_rel', ...
            'n_pass_runs_boundary', ...
            'frac_any_boundary','frac_low_boundary','frac_up_boundary', ...
            'frac_param_any','frac_param_low','frac_param_up', ...
            'frac_param_values_boundary', ...
            'T_boundary_summary','T_boundary_by_param', ...
            'T_boundary_param_overall','T_boundary_objective_overall', ...
            'overall_by_param_weighted','param_hit_count','param_run_count', ...
            'overall_by_obj_any','overall_by_obj_paramnorm', ...
            'overall_by_catch_any','overall_by_catch_paramnorm', ...
            'overall_by_model_any','overall_by_model_paramnorm', ...
            '-v7.3');

        fprintf('Boundary-hit analysis complete. Results written to %s\n', output_dir);
    end
end

%% Check GR4J specifically

%% =====================================================================
%  GR4J-SPECIFIC BOUNDARY-HIT ASSESSMENT (X1-X4)
%  ---------------------------------------------------------------------
%  Reads the per-parameter boundary table produced by the main script
%  and isolates GR4J to directly address the reviewer's claims about
%  X1 (min), X2 (interior), X3 (max in some cases), X4 (min).
%
%  Inputs (set ONE of the two source modes below):
%    MODE A: read the CSV  boundary_hit_by_parameter_model_catchment_objective.csv
%    MODE B: load the .mat  boundary_hit_analysis_workspace.mat
%
%  Output:
%    - T_gr4j           : tidy per-(catchment x OF x parameter) table, GR4J only
%    - T_gr4j_param     : per-parameter summary (X1-X4) across all cells
%    - overview figure  : gr4j_boundary_overview.png
%    - printed severity assessment
%  =====================================================================

clear gr4j*; close all;

% ---------------------------------------------------------------------
% USER SETTINGS
% ---------------------------------------------------------------------
output_dir = pwd;                 % <-- folder containing the boundary outputs
source_mode = 'csv';              % 'csv' or 'mat'
gr4j_token  = 'gr4j';             % substring used to match GR4J model name(s)
boundary_concern_threshold = 0.50;% >50% of runs at a boundary = flag as concern
% ---------------------------------------------------------------------

% --- Load the per-parameter table -----------------------------------
switch lower(source_mode)
    case 'csv'
        f = fullfile(output_dir, 'boundary_hit_by_parameter_model_catchment_objective.csv');
        assert(isfile(f), 'Cannot find %s', f);
        T = readtable(f, 'TextType','string');
    case 'mat'
        f = fullfile(output_dir, 'boundary_hit_analysis_workspace.mat');
        assert(isfile(f), 'Cannot find %s', f);
        L = load(f, 'T_boundary_by_param');
        T = L.T_boundary_by_param;
        % normalise types
        for v = {'model','catchment','catchment_label','objective','objective_label','parameter'}
            if ismember(v{1}, T.Properties.VariableNames)
                T.(v{1}) = string(T.(v{1}));
            end
        end
    otherwise
        error('source_mode must be ''csv'' or ''mat''.');
end

% --- Isolate GR4J ----------------------------------------------------
is_gr4j = contains(lower(T.model), lower(gr4j_token));
assert(any(is_gr4j), ['No model name contains "%s". ', ...
    'Check the model string and update gr4j_token.'], gr4j_token);
T_gr4j = T(is_gr4j, :);

fprintf('GR4J rows: %d  (models matched: %s)\n', height(T_gr4j), ...
    strjoin(unique(T_gr4j.model), ', '));

% --- Map MARRMoT parameter index -> X1..X4 ---------------------------
% MARRMoT GR4J parameter order: 1=x1 (production store, mm),
% 2=x2 (groundwater exchange, mm/d), 3=x3 (routing store, mm),
% 4=x4 (unit-hydrograph time base, d). If par_names already say x1..x4
% this still works; otherwise we relabel by parameter_index.
xlabels = ["X1 (prod. store)","X2 (exchange)","X3 (routing store)","X4 (UH time)"];
T_gr4j.Xname = strings(height(T_gr4j),1);
for i = 1:height(T_gr4j)
    idx = T_gr4j.parameter_index(i);
    if idx>=1 && idx<=4
        T_gr4j.Xname(i) = xlabels(idx);
    else
        T_gr4j.Xname(i) = sprintf("p%d", idx);
    end
end
T_gr4j = T_gr4j(T_gr4j.parameter_index>=1 & T_gr4j.parameter_index<=4, :);

% --- Per-parameter summary across all catchment x OF cells ----------
Xorder = xlabels(:);
nX = numel(Xorder);

med_any   = nan(nX,1);   % median fraction of runs with this param at *any* boundary
med_low   = nan(nX,1);
med_up    = nan(nX,1);
p_cells_concern = nan(nX,1);  % fraction of catchment-OF cells exceeding threshold
n_cells   = nan(nX,1);

% Boundary hits in GR4J are bimodal across cells: most catchment-OF cells
% are clean, a minority are pinned. The median therefore reads ~0 and hides
% the affected cells. We summarise with the MEAN, the 90th percentile, the
% MAX, and an explicit COUNT of affected cells instead.
mean_any  = nan(nX,1);
q90_any   = nan(nX,1);
max_any   = nan(nX,1);
n_cells_concern = nan(nX,1);   % # cells with >= threshold runs at a bound
dom_side  = strings(nX,1);     % which bound dominates the affected cells

for k = 1:nX
    m = T_gr4j.Xname == Xorder(k);
    fa = T_gr4j.frac_boundary(m);
    fl = T_gr4j.frac_lower_boundary(m);
    fu = T_gr4j.frac_upper_boundary(m);
    ok = isfinite(fa);

    med_any(k) = median(fa,'omitnan');
    med_low(k) = median(fl,'omitnan');
    med_up(k)  = median(fu,'omitnan');
    mean_any(k)= mean(fa(ok),'omitnan');
    q90_any(k) = quantile(fa(ok), 0.90);
    max_any(k) = max(fa(ok));
    n_cells(k) = sum(ok);
    p_cells_concern(k) = mean(fa(ok) >= boundary_concern_threshold);
    n_cells_concern(k) = sum(fa(ok) >= boundary_concern_threshold);

    % dominant side among the affected cells
    aff = ok & fa >= boundary_concern_threshold;
    if any(aff)
        if mean(fl(aff),'omitnan') >= mean(fu(aff),'omitnan')
            dom_side(k) = "lower";
        else
            dom_side(k) = "upper";
        end
    else
        dom_side(k) = "-";
    end
end

T_gr4j_param = table(Xorder, n_cells, med_any, mean_any, q90_any, max_any, ...
    n_cells_concern, p_cells_concern, dom_side, ...
    'VariableNames', {'parameter','n_cells','median_frac_any','mean_frac_any', ...
    'q90_frac_any','max_frac_any','n_cells_over_thr','frac_cells_over_thr','dominant_bound'});

writetable(T_gr4j_param, fullfile(output_dir,'gr4j_boundary_param_summary.csv'));
writetable(T_gr4j,       fullfile(output_dir,'gr4j_boundary_long.csv'));

disp(T_gr4j_param);

% --- list the specific offending cells (this is what the reviewer named) -
T_offenders = T_gr4j(T_gr4j.frac_boundary >= boundary_concern_threshold, ...
    {'catchment_label','objective_label','Xname', ...
     'frac_boundary','frac_lower_boundary','frac_upper_boundary','n_runs_used'});
T_offenders = sortrows(T_offenders, {'Xname','frac_boundary'}, {'ascend','descend'});
writetable(T_offenders, fullfile(output_dir,'gr4j_boundary_offending_cells.csv'));
fprintf('\nOffending cells (frac at bound >= %.2f): %d\n', ...
    boundary_concern_threshold, height(T_offenders));
disp(T_offenders);

% =====================================================================
%  OVERVIEW FIGURE
%  Left : grouped bars, median fraction at lower vs upper boundary per X
%  Right: heatmap of frac-at-any-boundary, OF (rows) x catchment (cols),
%         faceted is overkill; show the worst parameter (highest med_any)
% =====================================================================
ofs   = unique(T_gr4j.objective_label,'stable');
caps  = unique(T_gr4j.catchment_label,'stable');
nO = numel(ofs); nC = numel(caps);

% pick the parameter with the largest MEAN any-boundary fraction
% (median is ~0 for all because the distribution across cells is bimodal)
[~,worst_k] = max(mean_any);
worstX = Xorder(worst_k);

H = nan(nO,nC);
for oi = 1:nO
    for ci = 1:nC
        m = T_gr4j.Xname==worstX & ...
            T_gr4j.objective_label==ofs(oi) & ...
            T_gr4j.catchment_label==caps(ci);
        if any(m)
            H(oi,ci) = median(T_gr4j.frac_boundary(m),'omitnan');
        end
    end
end

f = figure('Units','normalized','OuterPosition',[0 0 0.95 0.55]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% --- panel 1: lower vs upper per parameter (MEAN across cells; median is ~0)
nexttile;
mean_low = nan(nX,1); mean_up = nan(nX,1);
for k = 1:nX
    m = T_gr4j.Xname == Xorder(k);
    mean_low(k) = mean(T_gr4j.frac_lower_boundary(m),'omitnan');
    mean_up(k)  = mean(T_gr4j.frac_upper_boundary(m),'omitnan');
end
Y = [mean_low, mean_up];
b = barh(Y,'grouped');
b(1).FaceColor = [0.20 0.45 0.70];   % lower
b(2).FaceColor = [0.80 0.40 0.20];   % upper
set(gca,'YDir','reverse');
yticks(1:nX); yticklabels(Xorder);
xlabel('Mean fraction of retained runs at boundary');
xlim([0 1]); grid on;
legend({'at lower bound','at upper bound'},'Location','southeast');
title('GR4J: boundary hits by parameter');

% --- panel 2: heatmap for the worst parameter
nexttile;
imagesc(H,[0 1]); colormap(parula);
cb = colorbar; cb.Label.String = 'Frac. runs at boundary';
xticks(1:nC); xticklabels(caps); xtickangle(45);
yticks(1:nO); yticklabels(ofs);
xlabel('Catchment'); ylabel('Objective function');
title(sprintf('Most-affected parameter: %s', worstX));

fontsize(f, 11, 'points');
exportgraphics(f, fullfile(output_dir,'gr4j_boundary_overview.png'), 'Resolution',200);

% =====================================================================
%  SEVERITY ASSESSMENT  (printed, ready to paste into response letter)
% =====================================================================
fprintf('\n================ GR4J BOUNDARY SEVERITY ================\n');
% Severity is judged on HOW MANY cells are pinned and HOW HARD (q90/max),
% not on the median (which is ~0 because most cells are clean).
verdict = strings(nX,1);
for k = 1:nX
    if p_cells_concern(k) >= 0.25 || q90_any(k) >= boundary_concern_threshold
        sev = "SUBSTANTIAL";
    elseif p_cells_concern(k) >= 0.10 || q90_any(k) >= 0.20
        sev = "moderate";
    else
        sev = "minor";
    end
    verdict(k) = sev;
    fprintf(['%-20s  mean=%.2f  q90=%.2f  max=%.2f  cells>thr=%d/%d (%.0f%%)  ' ...
             'dom=%s  -> %s\n'], ...
        Xorder(k), mean_any(k), q90_any(k), max_any(k), ...
        n_cells_concern(k), n_cells(k), 100*p_cells_concern(k), ...
        dom_side(k), sev);
end

frac_X_substantial = mean(verdict=="SUBSTANTIAL");
overall_mean = mean(T_gr4j.frac_boundary,'omitnan');
n_any_offender = sum(T_gr4j.frac_boundary >= boundary_concern_threshold);

fprintf('-------------------------------------------------------\n');
fprintf('Overall GR4J mean frac of runs at a boundary: %.2f\n', overall_mean);
fprintf('Total pinned cells (any X, >=%.0f%% of runs): %d of %d\n', ...
    100*boundary_concern_threshold, n_any_offender, height(T_gr4j));
fprintf('Parameters flagged SUBSTANTIAL/moderate: %d / %d\n', ...
    sum(verdict=="SUBSTANTIAL"), sum(verdict=="moderate"));
if frac_X_substantial >= 0.5
    fprintf(['VERDICT: Real, structural concern for GR4J. Multiple parameters are\n' ...
             '         pinned at bounds in a sizeable share of cases.\n']);
elseif any(verdict=="SUBSTANTIAL") || any(verdict=="moderate")
    fprintf(['VERDICT: Genuine but localised. The median says ~0 because most\n' ...
             '         catchment-OF cells are clean, but %s hit bounds in a\n' ...
             '         non-trivial minority of cells -- exactly the cases the\n' ...
             '         reviewer flagged. Report counts, not the median, and\n' ...
             '         inspect the offending cells against the PET/water-balance\n' ...
             '         investigation.\n'], ...
             strjoin(Xorder(verdict~="minor"),', '));
else
    fprintf('VERDICT: Limited concern once quantified per cell.\n');
end
fprintf('=======================================================\n');

%% =====================================================================
%  GR4J RAW PARAMETER VALUES, PER BASIN (X1-X4) AGAINST BOUNDS
%  ---------------------------------------------------------------------
%  Strip plot: each retained run is a dot. One column per catchment,
%  one row of panels per parameter (X1-X4). Dashed lines = lower/upper
%  bound, so parameters sitting on a bound are visible directly.
%
%  Requires (from the main script, still in the workspace):
%    param_all   : nM x nC x nO x nCand x nP, aligned to D.of_all
%    plot_mask   : nM x nC x nO x nCand logical (retained runs)
%    param_lower, param_upper : nM x nP   (after expand_bounds_inline)
%    nPars_by_model, D, OF_Plot, catchments_labels, output_dir
%  =====================================================================

assert(exist('param_all','var')==1, 'param_all not in workspace. Run the main script first.');
assert(exist('plot_mask','var')==1, 'plot_mask not in workspace.');

gr4j_token = 'gr4j';
mi = find(contains(lower(D.models), gr4j_token), 1);
assert(~isempty(mi), 'No GR4J model found in D.models.');
fprintf('GR4J model: %s (index %d)\n', D.models{mi}, mi);

nPars_this = min(4, nPars_by_model(mi));      % GR4J has 4 params (X1-X4)
xlabels = {'X1 (prod. store, mm)','X2 (exchange, mm/d)', ...
           'X3 (routing store, mm)','X4 (UH time, d)'};

[~,nC,nO,nCand,~] = size(param_all);

lb = param_lower(mi,1:nPars_this);
ub = param_upper(mi,1:nPars_this);

% one figure, nPars_this rows x 1, catchments along the x-axis of each row
f = figure('Units','normalized','OuterPosition',[0 0 0.95 0.95]);
tiledlayout(nPars_this,1,'TileSpacing','compact','Padding','compact');

for pp = 1:nPars_this
    nexttile; hold on;

    for ci = 1:nC
        % gather all retained runs for this catchment, across all OFs
        vals = [];
        for oi = 1:nO
            keep = logical(squeeze(plot_mask(mi,ci,oi,:)));
            if ~any(keep); continue; end
            v = squeeze(param_all(mi,ci,oi,keep,pp));
            vals = [vals; v(:)]; %#ok<AGROW>
        end
        vals = vals(isfinite(vals));
        if isempty(vals); continue; end

        % jittered strip
        x = ci + (rand(numel(vals),1)-0.5)*0.5;
        scatter(x, vals, 8, [0.30 0.40 0.55], 'filled', ...
            'MarkerFaceAlpha', 0.25);
        % median marker
        plot(ci, median(vals), 'd', 'MarkerSize',7, ...
            'MarkerFaceColor',[0.85 0.33 0.10], 'MarkerEdgeColor','k');
    end

    % bounds
    yline(lb(pp), '--', 'lower', 'Color',[0.2 0.2 0.2], ...
        'LabelHorizontalAlignment','left');
    yline(ub(pp), '--', 'upper', 'Color',[0.2 0.2 0.2], ...
        'LabelHorizontalAlignment','left');

    xlim([0.5 nC+0.5]);
    % pad y a little beyond bounds so on-bound dots are visible
    pad = 0.05*(ub(pp)-lb(pp));
    ylim([lb(pp)-pad, ub(pp)+pad]);
    xticks(1:nC); xticklabels(catchments_labels); xtickangle(45);
    ylabel(xlabels{pp});
    grid on; box on;
    if pp==1
        title('GR4J retained parameter values by catchment (all OFs pooled)');
    end
end

fontsize(f, 10, 'points');
exportgraphics(f, fullfile(output_dir,'gr4j_param_values_by_basin.png'), ...
    'Resolution', 200);
fprintf('Saved gr4j_param_values_by_basin.png\n');

%%
% Leave non-passing combinations as NaN
model_catch_boundary = nan(nM,nC);

for mi = 1:nM
    for ci = 1:nC

        x = squeeze(frac_param_values_boundary(mi,ci,:));
        x = x(:);
        x = x(isfinite(x));

        if ~isempty(x)
            model_catch_boundary(mi,ci) = mean(x,'omitnan');
        end
    end
end

[~,model_ord] = sort(model_num);

f = figure('Units','normalized','OuterPosition',[0 0 0.75 0.9]);
%f.Color = 'k';

ax = axes;
ax.Color = 'k';

A = model_catch_boundary(model_ord,:);

% Replace NaNs with a sentinel value
Aplot = A;
Aplot(isnan(Aplot)) = -0.01;

imagesc(Aplot,[-0.01 0.5]);

cmap = [0 0 0; parula(256)];
colormap(cmap);

cb = colorbar;
cb.Label.String = 'Mean fraction of parameter values at boundary';

xticks(1:nC);
xticklabels(catchments_labels);
xtickangle(45);

yticks(1:nM);
yticklabels(strrep(D.models(model_ord),'_','\_'));

xlabel('Catchment');
ylabel('Model');

title('Boundary pressure by model and catchment');
subtitle('Black = no benchmark-passing runs');

box on;
fontsize(f,10,'points');



%% =====================================================================
%  SIGNATURE VALUE CORRELATION: OBSERVED VS SIMULATED
%  Old plot order
%  =====================================================================

sig_corr_order_names = { ...
    'sig_TotalRR', ...
    'sig_EventRR', ...
    'sig_x_percentile_95per', ...
    'sig_x_percentile_5per', ...
    'sig_BFI', ...
    'sig_BaseflowRecessionK', ...
    'sig_FlashinessIndex', ...
    'sig_x_Q_frequency_low', ...
    'sig_x_Q_frequency_high', ...
    'sig_FDC_slope', ...
    'sig_VariabilityIndex', ...
    'sig_x_Q_duration_low', ...
    'sig_x_Q_duration_high', ...
    'sig_HFD_mean', ...
    'sig_RisingLimbDensity'};

label_signatures_corr_plot = { ...
    'Total RR (-)', ...
    'Event RR (-)', ...
    'Q95 (mm/d)', ...
    'Q5 (mm/d)', ...
    'Baseflow Index (-)', ...
    'Baseflow Recession Coefficient (-)', ...
    'Flashiness Index (-)', ...
    'LF Frequency (-)', ...
    'HF Frequency (-)', ...
    'FDC Slope (-)', ...
    'Variability Index (-)', ...
    'LF Duration (days)', ...
    'HF Duration (days)', ...
    'MHFD (DOY)', ...
    'Rising Limb Density (-)'};


nSigCorr = numel(sig_corr_order_names);

%% Map old-order signatures into obs_sig and Signature_model_median
corr_to_obs = nan(1,nSigCorr);
corr_to_sorted = nan(1,nSigCorr);

for jj = 1:nSigCorr
    corr_to_obs(jj) = find(strcmp(file_sig_names, sig_corr_order_names{jj}), 1);
    corr_to_sorted(jj) = find(strcmp(sorted_sig_new, sig_corr_order_names{jj}), 1);
end

if any(isnan(corr_to_obs))
    error('Some correlation signatures could not be mapped into file_sig_names.');
end

if any(isnan(corr_to_sorted))
    error('Some correlation signatures could not be mapped into sorted_sig_new.');
end

%% ---------------- OBSERVED SIGNATURE VALUES ----------------
% Rows = catchments
% Columns = signatures

X_obs = nan(nC, nSigCorr);

for jj = 1:nSigCorr
    X_obs(:,jj) = obs_sig(:, corr_to_obs(jj));
end

Corr_obs = corrcoef(X_obs, 'Rows', 'pairwise');

Corr_obs_plot = Corr_obs;
Corr_obs_plot(~tril(true(size(Corr_obs_plot)))) = NaN;

%% ---------------- SIMULATED SIGNATURE VALUES ----------------
% Rows = benchmark-passing model x catchment x objective cells
% Columns = signatures
%
% Signature_model_median:
%   [signature x objective x model x catchment]
%
% This uses median top-N simulated signature values, not errors.

X_sim = [];

for mi = 1:nM
    for ci = 1:nC
        for oi = 1:nO

            vals = nan(1,nSigCorr);

            for jj = 1:nSigCorr
                si_sorted = corr_to_sorted(jj);
                vals(jj) = Signature_model_median(si_sorted, oi, mi, ci);
            end

            if any(isfinite(vals))
                X_sim = [X_sim; vals]; %#ok<AGROW>
            end
        end
    end
end

Corr_sim = corrcoef(X_sim, 'Rows', 'pairwise');

Corr_sim_plot = Corr_sim;
Corr_sim_plot(~tril(true(size(Corr_sim_plot)))) = NaN;

%% ---------------- PLOT BOTH ----------------
try
    cmap = brewermap(19, '-RdBu');
catch
    cmap = redblue_local(19);
end

f = figure('units','normalized','outerposition',[0 0 1 0.75]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
h1 = heatmap( ...
    Corr_obs_plot, ...
    'Colormap', cmap, ...
    'ColorLimits', [-1 1], ...
    'MissingDataColor', 'w', ...
    'MissingDataLabel', " ", ...
    'GridVisible', 'off', ...
    'XData', label_signatures_corr_plot, ...
    'YData', label_signatures_corr_plot);

title(h1, 'Observed Signatures');
xlabel(h1, 'Signatures');
ylabel(h1, 'Signatures');
h1.CellLabelFormat = '%.2f';
h1.FontSize = 12;

nexttile;
h2 = heatmap( ...
    Corr_sim_plot, ...
    'Colormap', cmap, ...
    'ColorLimits', [-1 1], ...
    'MissingDataColor', 'w', ...
    'MissingDataLabel', " ", ...
    'GridVisible', 'off', ...
    'XData', label_signatures_corr_plot, ...
    'YData', label_signatures_corr_plot);

title(h2, 'Simulated Signatures');
xlabel(h2, 'Signatures');
ylabel(h2, 'Signatures');
h2.CellLabelFormat = '%.2f';
h2.FontSize = 12;

save_if_requested(f, output_dir, ...
    'correlation_observed_vs_simulated_signature_values', make_png);


%% compare_cal_eval.m
% Compares model performance between calibration and evaluation periods.
%
% For each model x catchment x objective:
%   - of_cal:  best-seed OF value from calibration (from seed_uncertainty_data.mat)
%   - of_eval: OF value on evaluation period      (from eval_index.tsv / eval_summary.csv)
%   - sig_cal: signatures from best calibration seed
%   - sig_eval: signatures from evaluation period  (from eval_signatures.csv)
%   - sig_obs:  observed signatures                (recomputed from NetCDF)
%
% Produces:
%   1. OF degradation violin plots (cal vs eval per objective, across models)
%   2. Signature error comparison (cal error vs eval error, per signature)
%   3. Heatmap of mean absolute signature error change (eval-cal) across models

clear; close all; clc;

%% USER SETTINGS  (same paths as your other scripts)
base_path  = '<LOCAL_DOWNLOADS>/all_fixed';
eval_path  = '<LOCAL_DOWNLOADS>/eval_fixed';
eval_index_file = '<LOCAL_DOWNLOADS>/final_results_combined_fixed/index/eval_index.tsv';
output_dir = '<LOCAL_ROOT>/graphics_new';
path_nc    = '<LOCAL_ROOT>/ma_thesis/catchments_new';

make_png = true;

addpath(genpath('<LOCAL_ROOT>/TOSSH-master/'))
addpath(genpath('<LOCAL_ROOT>/marrmot_211/'))
addpath(genpath('<LOCAL_ROOT>/ma_thesis/'))
if ~exist(output_dir,'dir'); mkdir(output_dir); end

%% CONSTANTS  (same as your other scripts)
catchments_aridity = {'camelsaus_143110A','camelsbr_60615000','camelsaus_607155', ...
    'camels_12381400','camels_02017500','camelsgb_39037','camels_03460000', ...
    'hysets_01AF007','camelsgb_27035','lamah_200048'};
catchments_labels = {'AUS1','BR6','AUS6','C12','C02','GB3','C03','HYS','GB2','LAM'};

sorted_signatures = {'sig_TotalRR','sig_EventRR','sig_x_percentile_5per', ...
    'sig_x_percentile_95per','sig_x_Q_duration_high','sig_x_Q_duration_low', ...
    'sig_x_Q_frequency_high','sig_x_Q_frequency_low','sig_HFD_mean', ...
    'sig_FDC_slope','sig_VariabilityIndex','sig_BFI','sig_BaseflowRecessionK', ...
    'sig_FlashinessIndex','sig_RisingLimbDensity'};

label_signatures_new = { ...
    'Total RR', 'Event RR', 'MHFD (DOY)', ...
    'Q95 (mm/d)', 'HF Frequency', 'HF Duration (d)', ...
    'Q5 (mm/d)', 'LF Frequency', 'LF Duration (d)', ...
    'Baseflow Index', 'Baseflow Recession Coefficient', 'FDC Slope', ...
    'Flashiness', 'Variability', 'Rising Limb'};

base_colors = [ ...
    0.1216 0.4667 0.7059; ...
    1.0000 0.4980 0.0549; ...
    0.1725 0.6275 0.1725; ...
    0.8392 0.1529 0.1569; ...
    0.5804 0.4039 0.7412; ...
    0.8902 0.4667 0.7608; ...
    0.7373 0.7412 0.1333; ...
    0.0902 0.7451 0.8118];

%% LOAD CALIBRATION DATA
fprintf('Loading calibration data...\n');
S = load(fullfile(base_path, 'seed_uncertainty_data.mat'));

D.of_all     = double(S.of_all);
D.sig_all    = double(S.sig_all);
D.catchments = cellstr(string(S.catchments(:)))';
D.objectives = cellstr(string(S.objectives(:)))';
D.models     = cellstr(string(S.models(:)))';

% Drop neg2
keep_obj     = ~strcmp(D.objectives, 'of_KGE_neg2_transf');
D.objectives = D.objectives(keep_obj);
D.of_all     = D.of_all(:,:,keep_obj,:);
D.sig_all    = D.sig_all(:,:,keep_obj,:,:);

% Reorder catchments
[~, cat_idx] = ismember(catchments_aridity, D.catchments);
D.of_all     = D.of_all(:,cat_idx,:,:);
D.sig_all    = D.sig_all(:,cat_idx,:,:,:);
D.catchments = D.catchments(cat_idx);

[nM, nC, nO, nCand] = size(D.of_all);
nSigs = size(D.sig_all, 5);
OF_Plot = make_objective_labels_local(D.objectives);
colors  = base_colors(1:nO, :);

fprintf('  %d models, %d catchments, %d objectives, %d seeds, %d signatures\n', ...
    nM, nC, nO, nCand, nSigs);

% Best-seed calibration OF and signatures (pick seed with max of_cal per combo)
of_cal_best  = nan(nM, nC, nO);
sig_cal_best = nan(nM, nC, nO, nSigs);
for mi = 1:nM
    for ci = 1:nC
        for oi = 1:nO
            vals = D.of_all(mi, ci, oi, :);
            [~, best] = max(vals(:));
            if isfinite(vals(best))
                of_cal_best(mi,ci,oi)     = vals(best);
                sig_cal_best(mi,ci,oi,:)  = D.sig_all(mi,ci,oi,best,:);
            end
        end
    end
end

%% LOAD EVALUATION DATA
fprintf('Loading evaluation data...\n');

localize_eval = @(p) strrep(char(string(p)), ...
    '/data/horse/ws/<HPC_USER>-marrmot_recal/output_seeded/eval_fixed', ...
    eval_path);

T_eval = readtable(eval_index_file, 'FileType','text', 'Delimiter','\t');
T_eval = T_eval(strcmp(T_eval.status, 'complete'), :);
fprintf('  %d complete eval combos\n', height(T_eval));

of_eval  = nan(nM, nC, nO);
sig_eval = nan(nM, nC, nO, nSigs);

for ri = 1:height(T_eval)
    mi = find(strcmp(D.models,     T_eval.model{ri}),     1);
    ci = find(strcmp(D.catchments, T_eval.catchment{ri}), 1);
    oi = find(strcmp(D.objectives, T_eval.objective{ri}), 1);
    if isempty(mi) || isempty(ci) || isempty(oi), continue; end

    % OF value
    p_sum = localize_eval(T_eval.eval_summary_csv{ri});
    if isfile(p_sum)
        try
            Ts = readtable(p_sum, 'FileType','text','Delimiter',',', ...
                'VariableNamingRule','preserve');
            of_eval(mi,ci,oi) = Ts.("of_eval")(1);
        catch; end
    end

    % Signatures
    p_sig = localize_eval(T_eval.eval_signatures_csv{ri});
    if isfile(p_sig)
        try
            Ts = readtable(p_sig, 'FileType','text','Delimiter',',');
            for si = 1:nSigs
                col = sprintf('s%d', si);
                if ismember(col, Ts.Properties.VariableNames)
                    sig_eval(mi,ci,oi,si) = Ts.(col)(1);
                end
            end
        catch; end
    end
end

%% OBSERVED SIGNATURES
fprintf('Computing observed signatures...\n');
Obs = load_observed_data_local(D.catchments, path_nc);
[obs_sig, ~] = compute_observed_signatures_local(D.catchments, Obs, sorted_signatures);
% obs_sig: (nC, nSigs)

%% SIGNATURE ERRORS
% Absolute error: sim - obs, normalized by |obs| (relative) or raw (absolute)
% Using relative error to make signatures comparable across units.
obs_mat = permute(repmat(obs_sig, [1,1,nM,nO]), [3,1,4,2]);  % (nM,nC,nO,nSigs)

sig_err_cal  = sig_cal_best - obs_mat;   % (nM,nC,nO,nSigs)
sig_err_eval = sig_eval     - obs_mat;

% Normalized by obs magnitude (replace 0 obs with NaN to avoid Inf)
obs_mag = abs(obs_mat);
obs_mag(obs_mag < 1e-10) = NaN;
sig_err_cal_norm  = sig_err_cal  ./ obs_mag;
sig_err_eval_norm = sig_err_eval ./ obs_mag;

fprintf('Data loaded. Starting plots...\n');

%% ========== PLOT 1: OF degradation (cal vs eval) per objective ==========
% For each objective: violin/boxplot of of_cal and of_eval across all
% model x catchment combinations.

h = figure('Units','normalized','OuterPosition',[0 0 1 0.5]);
tiledlayout(1, nO, 'TileSpacing','compact','Padding','compact');

for oi = 1:nO
    nexttile;

    cal_vals  = of_cal_best(:,:,oi);   % (nM,nC) → flatten
    eval_vals = of_eval(:,:,oi);

    cal_vals  = cal_vals(:);
    eval_vals = eval_vals(:);

    valid = isfinite(cal_vals) & isfinite(eval_vals);
    data  = [cal_vals(valid), eval_vals(valid)];

    boxplot(data, 'Labels', {'Cal','Eval'}, 'Whisker', 1.5);
    hold on;
    % Overlay mean markers
    plot(1, nanmean(cal_vals),  'r+', 'MarkerSize', 8, 'LineWidth', 2);
    plot(2, nanmean(eval_vals), 'r+', 'MarkerSize', 8, 'LineWidth', 2);
    hold off;

    title(OF_Plot{oi}, 'Interpreter','none');
    if oi == 1, ylabel('OF value'); end
    ylim([-1 1]);
    grid on;
end
sgtitle('Calibration vs Evaluation OF Performance');
fontsize(h, 11, 'points');
if make_png
    exportgraphics(h, fullfile(output_dir, 'caleval_OF_degradation.png'), 'Resolution', 200);
end

%% ========== PLOT 2: OF degradation scatter — cal vs eval ==========
% One panel per objective: scatter of of_cal vs of_eval (one point per
% model x catchment combo). Points below diagonal = degradation.

h = figure('Units','normalized','OuterPosition',[0 0 1 0.8]);
tiledlayout(2, 4, 'TileSpacing','compact','Padding','compact');

for oi = 1:nO
    nexttile;
    cal_vals  = of_cal_best(:,:,oi);
    eval_vals = of_eval(:,:,oi);

    hold on;
    for ci = 1:nC
        valid = isfinite(cal_vals(:,ci)) & isfinite(eval_vals(:,ci));
        scatter(cal_vals(valid,ci), eval_vals(valid,ci), 20, ...
            'filled', 'MarkerFaceAlpha', 0.5, ...
            'DisplayName', catchments_labels{ci});
    end
    % 1:1 line
    ax_lim = [-0.5 1];
    plot(ax_lim, ax_lim, 'k--', 'LineWidth', 1);
    hold off;

    xlim(ax_lim); ylim(ax_lim);
    xlabel('OF cal'); ylabel('OF eval');
    title(OF_Plot{oi}, 'Interpreter','none');
    grid on;

    if oi == 1
        legend('Location','southeast','FontSize',7,'NumColumns',2);
    end
end
sgtitle('Calibration vs Evaluation: per combo (below diagonal = degradation)');
fontsize(h, 11, 'points');
if make_png
    exportgraphics(h, fullfile(output_dir, 'caleval_OF_scatter.png'), 'Resolution', 200);
end

%% ========== PLOT 3: Signature error change (eval - cal) per signature ==========
% For each signature: boxplot of (|eval_err| - |cal_err|) across all combos.
% Positive = worse in eval, negative = better in eval.

h = figure('Units','normalized','OuterPosition',[0 0 1 0.6]);

delta_err = abs(sig_err_eval_norm) - abs(sig_err_cal_norm);  % (nM,nC,nO,nSigs)

data_box = nan(nM*nC*nO, nSigs);
for si = 1:nSigs
    d = delta_err(:,:,:,si);
    data_box(:,si) = d(:);
end
% Remove rows that are all NaN
data_box(all(isnan(data_box),2),:) = [];

boxplot(data_box, 'Labels', label_signatures_new, 'Whisker', 1.5, ...
    'Orientation','vertical');
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
hold off;

xtickangle(30);
ylabel('\Delta |norm. error|  (eval - cal)');
title('Signature Error Change: Evaluation vs Calibration (positive = worse in eval)');
grid on;
fontsize(h, 11, 'points');
if make_png
    exportgraphics(h, fullfile(output_dir, 'caleval_sig_error_delta.png'), 'Resolution', 200);
end

%% ========== PLOT 4: Heatmap — mean |error| change by catchment x signature ==========
% Rows = catchments, Cols = signatures. Value = mean over models x objectives.

h = figure('Units','normalized','OuterPosition',[0 0 0.9 0.5]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

for period = 1:2
    nexttile;
    if period == 1
        err_use = abs(sig_err_cal_norm);
        ttl = 'Calibration';
    else
        err_use = abs(sig_err_eval_norm);
        ttl = 'Evaluation';
    end

    % Mean over models and objectives → (nC, nSigs)
    hmap = squeeze(nanmean(nanmean(err_use, 1), 3));  % mean over nM (dim1) then nO (dim3 → dim2)
    % err_use is (nM,nC,nO,nSigs) → mean over dim1,3 → (nC,nSigs)
    hmap = squeeze(nanmean(nanmean(err_use, 3), 1));

    imagesc(hmap);
    set(gca,'YDir','normal');
    xticks(1:nSigs); xticklabels(label_signatures_new); xtickangle(35);
    yticks(1:nC);    yticklabels(catchments_labels);
    colorbar; clim([0 2]);
    colormap(hot);
    title(sprintf('Mean |norm. sig. error| — %s', ttl));
end
fontsize(h, 10, 'points');
if make_png
    exportgraphics(h, fullfile(output_dir, 'caleval_sig_error_heatmap.png'), 'Resolution', 200);
end

%% ========== PLOT 5: Per-objective OF degradation by catchment ==========
% Heatmap: rows=catchments, cols=objectives. Value = median(of_eval - of_cal).

h = figure('Units','normalized','OuterPosition',[0 0 0.6 0.4]);
delta_of = nanmedian(of_eval - of_cal_best, 1);   % median over models → (1,nC,nO)
delta_of = squeeze(delta_of);                       % (nC,nO)

imagesc(delta_of);
set(gca,'YDir','normal');
xticks(1:nO); xticklabels(OF_Plot); xtickangle(35);
yticks(1:nC); yticklabels(catchments_labels);
colorbar; colormap(redblue_local_2(256));
clim([-0.3 0.3]);
title('Median OF change (eval - cal) per catchment x objective');
fontsize(h, 11, 'points');
if make_png
    exportgraphics(h, fullfile(output_dir, 'caleval_OF_heatmap.png'), 'Resolution', 200);
end

fprintf('Done. Figures saved to %s\n', output_dir);



%% ========== PLOT: Observed signature representation change ==========
% Uses benchmark.m periods:
%   most basins: cal 2005-2014, eval 1994-2003
%   camelsaus_607155: cal 1990-1999, eval 1982-1988

file_sig_names = {'sig_FDC_slope','sig_RisingLimbDensity','sig_BaseflowRecessionK','sig_HFD_mean', ...
    'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex','sig_EventRR','sig_TotalRR', ...
    'sig_x_Q_duration_high','sig_x_Q_duration_low','sig_x_Q_frequency_high','sig_x_Q_frequency_low', ...
    'sig_x_percentile_5per','sig_x_percentile_95per'};

obs_sig_cal  = nan(nC, numel(file_sig_names));
obs_sig_eval = nan(nC, numel(file_sig_names));

for ci = 1:nC
    catchment = D.catchments{ci};

    [cal_idx, eval_idx, ~] = get_benchmark_period_indices_local(catchment, Obs.date_array_full);

    obs_sig_cal(ci,:)  = compute_obs_signature_row_local(Obs.q(:,ci), Obs.precip(:,ci), cal_idx,  file_sig_names);
    obs_sig_eval(ci,:) = compute_obs_signature_row_local(Obs.q(:,ci), Obs.precip(:,ci), eval_idx, file_sig_names);
end

obs_base = abs(obs_sig_cal);
obs_base(obs_base < 1e-10) = NaN;

delta_obs_sig_pct = 100 .* (obs_sig_cal - obs_sig_eval) ./ obs_base;

sig_plot_order = [9,8,1,4,14,11,13,5,15,10,12,3,2,7,6];

sig_titles = { ...
    'Total RR (-)', ...
    'Event RR (-)', ...
    'FDC Slope (-)', ...
    'MHFD (DOY)', ...
    'Q5 (mm/d)', ...
    'LF Duration (days)', ...
    'LF Frequency (-)', ...
    'Baseflow Index (-)', ...
    'Q95 (mm/d)', ...
    'HF Duration (days)', ...
    'HF Frequency (-)', ...
    'Baseflow Recession Coefficient (-)', ...
    'Rising Limb Density (-)', ...
    'Flashiness Index (-)', ...
    'Variability Index (-)'};

h = figure('Units','normalized','OuterPosition',[0 0 1 1]);
tiledlayout(4,4,'TileSpacing','compact','Padding','compact');

for jj = 1:numel(sig_plot_order)
    si = sig_plot_order(jj);
    nexttile;

    vals = delta_obs_sig_pct(:,si);

    bar(vals);
    hold on;
    yline(0, 'k-', 'LineWidth', 0.8);
    hold off;

    xticks(1:nC);
    xticklabels(catchments_labels);
    xtickangle(45);

    ylim([-100 100]);
    ylabel('\Delta observed signature (%)');
    xlabel('Catchments');
    title(sig_titles{jj}, 'Interpreter','none');
    grid on;
end

sgtitle('Observed Signature Change: Evaluation Period vs Calibration Period');

fontsize(h, 10, 'points');

if make_png
    exportgraphics(h, fullfile(output_dir, ...
        'delta_observed_signature_representation.png'), ...
        'Resolution', 250);
end% ========== PLOT: Observed signature change only ==========



%% ========== PLOT: Aggregated model differences in signatures ==========
% One point per model x signature:
%   median over catchments x objectives of normalized signature change
%
% Then one violin per signature shows the distribution across models.
%
% Positive = eval signature larger than calibration
% Negative = eval signature smaller than calibration

sig_base = abs(sig_cal_best);
sig_base(sig_base < 1e-10) = NaN;

delta_sig_pct = 100 .* (sig_cal_best - sig_eval) ./ sig_base;
% [model x catchment x objective x signature]

model_sig_delta = nan(nM, nSigs);

for mi = 1:nM
    for si = 1:nSigs
        vals = [];

        for ci = 1:nC
            for oi = 1:nO
                if ~model_pass(mi,ci,oi)
                    continue;
                end

                v = delta_sig_pct(mi,ci,oi,si);
                if isfinite(v)
                    vals = [vals; v]; %#ok<AGROW>
                end
            end
        end

        if ~isempty(vals)
            model_sig_delta(mi,si) = median(vals, 'omitnan');
        end
    end
end

% Optional visual clipping only
model_sig_delta_plot = model_sig_delta;
model_sig_delta_plot(model_sig_delta_plot > 100)  = 100;
model_sig_delta_plot(model_sig_delta_plot < -100) = -100;

sig_plot_order = [9,8,1,4,14,11,13,5,15,10,12,3,2,7,6];

sig_titles = { ...
    'Total RR', ...
    'Event RR', ...
    'FDC Slope', ...
    'MHFD', ...
    'Q5', ...
    'LF Dur', ...
    'LF Freq', ...
    'BFI', ...
    'Q95', ...
    'HF Dur', ...
    'HF Freq', ...
    'BFRC', ...
    'RLD', ...
    'Flashiness', ...
    'Variability'};

y = [];
g = [];

for jj = 1:numel(sig_plot_order)
    si = sig_plot_order(jj);

    vals = model_sig_delta_plot(:,si);
    vals = vals(isfinite(vals));

    y = [y; vals]; %#ok<AGROW>
    g = [g; jj .* ones(numel(vals),1)]; %#ok<AGROW>
end

h = figure('Units','normalized','OuterPosition',[0 0 0.95 0.55]);
hold on;

violinplot(y, categorical(g, 1:numel(sig_plot_order), sig_titles));

yline(0, 'k-', 'LineWidth', 1.2);

ylim([-100 100]);
ylabel('Median \Delta simulated signature per model (%)');
xlabel('Signature');
title('Model-level signature change from calibration to evaluation');
subtitle('Each point is one model, aggregated as median across benchmark-passing catchment-objective combinations');

xtickangle(45);
grid on;
box on;

fontsize(h, 12, 'points');

if make_png
    exportgraphics(h, fullfile(output_dir, ...
        'caleval_model_aggregated_signature_change_violins.png'), ...
        'Resolution', 300);
end


%% ========== PLOT: Signature changes by catchment ==========
% One subplot per signature.
% X-axis = catchments.
% Each violin pools benchmark-passing model x objective values.
%
% Positive = eval signature larger than calibration
% Negative = eval signature smaller than calibration

sig_base = abs(sig_cal_best);
sig_base(sig_base < 1e-10) = NaN;

delta_sig_pct = 100 .* (sig_cal_best - sig_eval) ./ sig_base;
% [model x catchment x objective x signature]

sig_plot_order = [9,8,1,4,14,11,13,5,15,10,12,3,2,7,6];

sig_titles = { ...
    'Total RR', ...
    'Event RR', ...
    'FDC Slope', ...
    'MHFD', ...
    'Q5', ...
    'LF Dur', ...
    'LF Freq', ...
    'BFI', ...
    'Q95', ...
    'HF Dur', ...
    'HF Freq', ...
    'BFRC', ...
    'Rising Limb', ...
    'Flashiness', ...
    'Variability'};

h = figure('Units','normalized','OuterPosition',[0 0 1 1]);
tiledlayout(4,4,'TileSpacing','compact','Padding','compact');

for jj = 1:numel(sig_plot_order)
    si = sig_plot_order(jj);
    nexttile; hold on;

    y = [];
    g = [];

    for ci = 1:nC
        vals_catch = [];

        for mi = 1:nM
            for oi = 1:nO
                if ~model_pass(mi,ci,oi)
                    continue;
                end

                v = delta_sig_pct(mi,ci,oi,si);

                if isfinite(v)
                    vals_catch = [vals_catch; v]; %#ok<AGROW>
                end
            end
        end

        vals_catch(vals_catch > 100)  = 100;
        vals_catch(vals_catch < -100) = -100;

        if ~isempty(vals_catch)
            y = [y; vals_catch]; %#ok<AGROW>
            g = [g; ci .* ones(numel(vals_catch),1)]; %#ok<AGROW>
        end
    end

    if ~isempty(y)        
        violinplot(y, categorical(g, 1:nC, catchments_labels), ...
            'ShowMean', false, ...
            'ShowData', false, 'ViolinColor' , [0.5 0.5 0.5]);
    else
        text(0.5,0.5,'No valid data','HorizontalAlignment','center');
    end

    yline(0, 'k-', 'LineWidth', 0.8);
    xlim([0.5 10.5])
    ylim([-100 100]);
    xtickangle(45);

    title(sig_titles{jj}, 'Interpreter','none');
    ylabel('\Delta signature (%)');
    xlabel('Catchment');

    grid on;
    box on;
end

sgtitle('Signature change by catchment: evaluation vs calibration');

fontsize(h, 9, 'points');

if make_png
    exportgraphics(h, fullfile(output_dir, ...
        'caleval_signature_change_by_catchment_violins.png'), ...
        'Resolution', 300);
end




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ONLY LOCAL FUNCTIONS BELOW, END OF MAIN SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LOCAL FUNCTIONS
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
    % Period definitions copied from benchmark.m.
    % Special Australian basin (catchmentIdx == 3 in benchmark.m):
    %   calibration: 1990-01-01 to 1999-12-31
    %   evaluation:  1982-01-01 to 1988-12-31
    % All other basins:
    %   calibration: 2005-01-01 to 2014-12-31
    %   evaluation:  1994-01-01 to 2003-12-31
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
    period_label = sprintf('%s to %s', datestr(cal_start,'yyyy-mm-dd'), datestr(cal_end,'yyyy-mm-dd'));
end

function [obs_sig, obs_struct] = compute_observed_signatures_local(catchments, Obs, sorted_signatures)
    file_sig_names = {'sig_FDC_slope','sig_RisingLimbDensity','sig_BaseflowRecessionK','sig_HFD_mean', ...
        'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex','sig_EventRR','sig_TotalRR', ...
        'sig_x_Q_duration_high','sig_x_Q_duration_low','sig_x_Q_frequency_high','sig_x_Q_frequency_low', ...
        'sig_x_percentile_5per','sig_x_percentile_95per'};
    nC = numel(catchments);
    obs_sig = nan(nC, numel(file_sig_names));
    obs_struct = struct();
    for ci = 1:nC
        q = Obs.q(:,ci);
        p = Obs.precip(:,ci);
        q = q(:);
        p = p(:);
        q(q<0) = 0;

        % Match benchmark.m exactly: most catchments use 2005-2014 for
        % calibration signatures/benchmarks, while camelsaus_607155 uses
        % 1990-1999. Evaluation indices are returned by the helper too,
        % but this paper plotting script currently uses calibration-period
        % retained results/signatures.
        [cal_period_idx, ~, period_label] = get_benchmark_period_indices_local(catchments{ci}, Obs.date_array_full);
        fprintf('Observed signatures for %s use calibration period %s.\n', catchments{ci}, period_label);

        q_idx  = cal_period_idx(isfinite(q(cal_period_idx)));
        qp_idx = cal_period_idx(isfinite(q(cal_period_idx)) & isfinite(p(cal_period_idx)));

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
                if numel(val) > 1; val = val(1); end
                obs_sig(ci,si) = val;
            catch ME
                warning('Observed signature %s failed for %s: %s', sig, catchments{ci}, ME.message);
                obs_sig(ci,si) = NaN;
            end
            obs_struct.(sig).(catchments{ci}) = obs_sig(ci,si);
        end
    end
    % Also ensure sorted_signatures fields exist if different ordering requested.
    for si = 1:numel(sorted_signatures)
        sig = sorted_signatures{si};
        if ~isfield(obs_struct, sig)
            for ci = 1:nC; obs_struct.(sig).(catchments{ci}) = NaN; end
        end
    end
end

function [Error_Norm_Values, Error_Norm_Values_range, Rank_values, median_catchment_error_range, mean_catchment_error_range, Signature_norm] = ...
    compute_original_style_error_summaries(Error_Values, obs_sig, sorted_sig_new, file_sig_names, nO, nC, ...
                                            Signature_model_median, sorted_to_obs)
    nSig = numel(sorted_sig_new);
    Error_Norm_Values = nan(nSig,nC,nO);
    Error_Norm_Values_range = nan(nSig,nC,nO);
    Rank_values = nan(nSig+1,nC+2,nO);
    median_catchment_error_range = nan(nSig+1,nO);
    mean_catchment_error_range = nan(nSig+1,nO);
    Signature_norm = nan(size(Error_Values));
    for si = 1:nSig
        obs_col = sorted_to_obs(si);
        for ci = 1:nC
            obsval = obs_sig(ci, obs_col);
            for oi = 1:nO
                % SEQUENTIAL: per-model median error, then median across models
                per_model_med_err = squeeze(Signature_model_median(si,oi,:,ci)) - obsval;  % [nM x 1]
                Error_Norm_Values(si,ci,oi) = median(per_model_med_err, 'omitnan');
            end
            denom = max(abs(squeeze(Error_Norm_Values(si,ci,:))), [], 'omitnan');
            if ~isfinite(denom) || denom == 0; denom = NaN; end
            for oi = 1:nO
                Error_Norm_Values_range(si,ci,oi) = Error_Norm_Values(si,ci,oi) ./ denom;
                Signature_norm(si,oi,:,ci) = Error_Values(si,oi,:,ci) ./ denom;   % UNCHANGED — run-level
            end
            % ... rank computation below stays exactly the same ...
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
        for oi = 1:nO
            median_catchment_error_range(si+1,oi) = median(abs(Error_Norm_Values_range(si,:,oi)), 'omitnan');
            mean_catchment_error_range(si+1,oi) = mean(abs(Error_Norm_Values_range(si,:,oi)), 'omitnan');
        end
    end
    Rank_values(nSig+1,1:nC,:) = mean(Rank_values(1:nSig,1:nC,:), 1, 'omitnan');
    Rank_values(1:nSig,nC+2,:) = mean(Rank_values(1:nSig,1:nC,:), 2, 'omitnan');
end


function [TopNRaw, TopNStats] = collect_topn_variability_stats(D, plot_mask, file_sig_names)
    [nM,nC,nO,nCand] = size(D.of_all);
    nSig = numel(file_sig_names);

    TopNRaw = struct();
    TopNRaw.OF = cell(nM,nC,nO);
    TopNRaw.Signature = cell(nM,nC,nO,nSig);
    TopNRaw.signature_names = file_sig_names;
    TopNRaw.models = D.models;
    TopNRaw.catchments = D.catchments;
    TopNRaw.objectives = D.objectives;

    empty_stats = summarize_vector_local([]);
    TopNStats = struct();
    TopNStats.OF = repmat(empty_stats, nM,nC,nO);
    TopNStats.Signature = repmat(empty_stats, nM,nC,nO,nSig);
    TopNStats.signature_names = file_sig_names;
    TopNStats.models = D.models;
    TopNStats.catchments = D.catchments;
    TopNStats.objectives = D.objectives;

    for mi = 1:nM
        for ci = 1:nC
            for oi = 1:nO
                keep = squeeze(plot_mask(mi,ci,oi,:));
                keep = logical(keep(:));
                if numel(keep) ~= nCand
                    error('plot_mask candidate dimension mismatch at model=%d catchment=%d objective=%d.', mi,ci,oi);
                end

                of_vals_raw = squeeze(D.of_all(mi,ci,oi,:));
                of_vals_raw = of_vals_raw(:);
                nAlignOF = min(numel(of_vals_raw), numel(keep));
                if nAlignOF == 0
                    of_vals = [];
                else
                    raw_subset = real(double(of_vals_raw(1:nAlignOF)));
                    keep_of = logical(keep(1:nAlignOF));
                    joint = keep_of & isfinite(raw_subset);
                    of_vals = raw_subset(joint);
                end
                TopNRaw.OF{mi,ci,oi} = of_vals(:);
                TopNStats.OF(mi,ci,oi) = summarize_vector_local(of_vals);

                for si = 1:nSig
                    sig_vals_raw = squeeze(D.sig_all(mi,ci,oi,:,si));
                    sig_vals_raw = sig_vals_raw(:);
                    nAlign = min([numel(sig_vals_raw), numel(keep), nCand]);
                    if nAlign == 0
                        sig_vals = [];
                    else
                        sig_raw = real(double(sig_vals_raw(1:nAlign)));
                        keep_sig = logical(keep(1:nAlign));
                        joint_sig = keep_sig & isfinite(sig_raw);
                        sig_vals = sig_raw(joint_sig);
                    end
                    sig_vals = sig_vals(:);
                    TopNRaw.Signature{mi,ci,oi,si} = sig_vals(:);
                    TopNStats.Signature(mi,ci,oi,si) = summarize_vector_local(sig_vals);
                end
            end
        end
    end
end

function S = summarize_vector_local(vals)
    vals = coerce_numeric_vector_local(vals);
    S = struct();
    S.n = numel(vals);
    if isempty(vals)
        S.mean = NaN;
        S.median = NaN;
        S.std = NaN;
        S.min = NaN;
        S.max = NaN;
        S.q05 = NaN;
        S.q25 = NaN;
        S.q75 = NaN;
        S.q95 = NaN;
        S.iqr = NaN;
        S.range = NaN;
        return;
    end
    S.mean = mean(vals, 'omitnan');
    S.median = median(vals, 'omitnan');
    S.std = std(vals, 'omitnan');
    S.min = min(vals, [], 'omitnan');
    S.max = max(vals, [], 'omitnan');
    qs = prctile(vals, [5 25 75 95]);
    S.q05 = qs(1);
    S.q25 = qs(2);
    S.q75 = qs(3);
    S.q95 = qs(4);
    S.iqr = S.q75 - S.q25;
    S.range = S.max - S.min;
end

function vals = coerce_numeric_vector_local(vals)
    % Convert cells/tables/timetables/string-like scalar containers to a finite real double vector.
    % This prevents prctile from crashing when a retained candidate slot is [], NaN, or a cell scalar.
    if isempty(vals)
        vals = [];
        return;
    end

    if istable(vals) || istimetable(vals)
        vals = table2array(vals);
    end

    if iscell(vals)
        tmp = [];
        for ii = 1:numel(vals)
            v = vals{ii};
            if istable(v) || istimetable(v)
                v = table2array(v);
            end
            if iscell(v)
                v = coerce_numeric_vector_local(v);
            elseif isstring(v) || ischar(v)
                v = str2double(v);
            end
            if isnumeric(v) || islogical(v)
                tmp = [tmp; double(v(:))]; %#ok<AGROW>
            end
        end
        vals = tmp;
    elseif isstring(vals) || ischar(vals)
        vals = str2double(vals);
    elseif islogical(vals)
        vals = double(vals);
    elseif isnumeric(vals)
        vals = double(vals);
    else
        vals = [];
    end

    vals = vals(:);
    vals = vals(isreal(vals) & isfinite(vals));
end


function labels = make_objective_labels_local(objectives)
% Convert objective variable names from the result file into display labels.
% This is intentionally conservative: unknown names are displayed as their
% actual variable names rather than reusing stale labels.
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

function customColormap = greenCenteredColormap(n)
    if nargin < 1; n = 256; end
    half = floor(n / 2);
    blueToGreen = [linspace(0, 0, half)', linspace(0, 1, half)', linspace(1, 0, half)'];
    greenToRed = [linspace(0, 1, n - half)', linspace(1, 0, n - half)', linspace(0, 0, n - half)'];
    customColormap = [blueToGreen; greenToRed];
    if mod(n, 2) == 1
        customColormap = [customColormap; [0 1 0]];
    end
end

function cmap = redblue_local(m)
    if nargin < 1; m = 256; end
    x = linspace(0,1,m)';
    cmap = [min(1,2*x), 1-abs(2*x-1), min(1,2*(1-x))];
end


function vals = finite_real_vector_local(vals)
% Return finite real double column vector suitable for plotting/stats.
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
% Defensive wrapper around violinplot. Drops invalid values and falls back
% to scatter + median if the third-party Violin class cannot handle a group.
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
    % Remove groups with no real spread only from violin call; draw them as points.
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
            scatter(ug(kk)+jitter, xx, 8, color, 'filled', 'MarkerFaceAlpha', 0.18, 'MarkerEdgeAlpha', 0.18);
            medx = median(xx, 'omitnan');
            plot([ug(kk)-0.18 ug(kk)+0.18], [medx medx], '-', 'Color', color, 'LineWidth', 1.8);
        end
    end
end

function pct_var = run_anova_two_stage(y, g_of, g_model, g_catch, verbose)
% ANOVA on cell-level summary with all main effects + 2-way + 3-way interaction.

if nargin < 5; verbose = false; end
pct_var = nan(1, 8);   % was 7

ok = isfinite(y);
if sum(ok) < 20; return; end
y = y(ok); g_of = g_of(ok); g_model = g_model(ok); g_catch = g_catch(ok);

% Full factorial: 3 mains + 3 two-way + 1 three-way
model_terms = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 1 1 1];

try
    [~, tbl] = anovan(y, {g_of, g_model, g_catch}, ...
        'model', model_terms, ...
        'varnames', {'OF','Model','Catch'}, ...
        'sstype', 2, ...
        'display', 'off');
catch ME
    if verbose, fprintf('    ANOVA failed: %s\n', ME.message); end
    return;
end

if verbose
    fprintf('    ANOVA source terms returned by anovan:\n');
    for kk = 2:size(tbl,1)-1
        fprintf('      "%s"  SS=%.3g\n', tbl{kk,1}, tbl{kk,2});
    end
end

src = tbl(2:end-1, 1);
ss  = cell2mat(tbl(2:end-1, 2));

% Match by which factor names are present in the source term name
ss_row = nan(1, 8);
for k = 1:numel(src)
    name = src{k};
    has_of = contains(name, 'OF');
    has_m  = contains(name, 'Model');
    has_c  = contains(name, 'Catch');
    n_factors = has_of + has_m + has_c;

    if strcmpi(name, 'Error')
        ss_row(8) = ss(k);
    elseif n_factors == 1
        if has_of, ss_row(1) = ss(k);
        elseif has_m, ss_row(2) = ss(k);
        elseif has_c, ss_row(3) = ss(k);
        end
    elseif n_factors == 2
        if has_of && has_m, ss_row(4) = ss(k);
        elseif has_of && has_c, ss_row(5) = ss(k);
        elseif has_m && has_c, ss_row(6) = ss(k);
        end
    elseif n_factors == 3
        ss_row(7) = ss(k);
    end
end

total_ss = sum(ss_row, 'omitnan');
if total_ss > 0
    pct_var = 100 * ss_row / total_ss;
end
end


function [raw_var, pct_var] = extract_lme_variance_local(lme)
% Extract variance components from a fitted LinearMixedModel.
% Returns 8 components: OF, Model, Catch, OFxM, OFxC, MxC, Cell, Residual.
%
% Uses lme.VariableInfo to robustly identify which variable each
% coefficient belongs to, rather than substring-matching level strings.

raw_var = zeros(1, 8);
pct_var = zeros(1, 8);

X = designMatrix(lme, 'Fixed');
beta = fixedEffects(lme);
coef_names = lme.CoefficientNames;

% Variable names in the table (excluding the response 'y' and the
% grouping variable 'CellID' which is the random effect)
var_names = lme.PredictorNames;
% Should be {'OF','Model','Catch','CellID'} but we only care about the
% three fixed-effect predictors.

% Identify each coefficient by which fixed-effect variables it involves
nC_coef = numel(coef_names);
involves = false(nC_coef, 3);   % columns: OF, Model, Catch

for k = 1:nC_coef
    nm = coef_names{k};
    if strcmp(nm, '(Intercept)'), continue; end

    % Split the coefficient name by ':' to get interaction parts
    parts = strsplit(nm, ':');

    for p = 1:numel(parts)
        part = parts{p};
        % Each part is of the form 'VariableName_LevelName' or similar.
        % We test which variable's NAME comes first in the part.
        if startsWith(part, 'OF_')
            involves(k, 1) = true;
        elseif startsWith(part, 'Model_')
            involves(k, 2) = true;
        elseif startsWith(part, 'Catch_')
            involves(k, 3) = true;
        end
    end
end

% Now classify each coefficient by how many factors it involves
which_term = zeros(nC_coef, 1);
for k = 1:nC_coef
    n_facs = sum(involves(k, :));
    if n_facs == 1
        if     involves(k,1), which_term(k) = 1;   % OF main
        elseif involves(k,2), which_term(k) = 2;   % Model main
        elseif involves(k,3), which_term(k) = 3;   % Catch main
        end
    elseif n_facs == 2
        if     involves(k,1) && involves(k,2), which_term(k) = 4;  % OF x Model
        elseif involves(k,1) && involves(k,3), which_term(k) = 5;  % OF x Catch
        elseif involves(k,2) && involves(k,3), which_term(k) = 6;  % Model x Catch
        end
    elseif n_facs == 3
        % If a three-way slipped in (shouldn't with our formulas), park it in the cell variance later
        which_term(k) = 0;
    end
end

% Variance of fitted contribution from each grouping
for fac = 1:6
    mask = (which_term == fac);
    if any(mask)
        contrib = X(:, mask) * beta(mask);
        raw_var(fac) = var(contrib, 1);
    end
end

% Random-effect variance (cell intercept)
try
    [psi, ~, ~] = covarianceParameters(lme);
    v_cell = psi{1};
    if ~isscalar(v_cell), v_cell = v_cell(1,1); end
    raw_var(7) = v_cell;
catch
    raw_var(7) = NaN;
end

% Residual variance (within-cell equifinality)
raw_var(8) = lme.MSE;

total = sum(raw_var, 'omitnan');
if total > 0
    pct_var = 100 * raw_var / total;
end
end


%% SUMMARY FIGURE A1 variants: top-25 and top-500 runs

function plot_a1_with_subset(subset_n, D, plot_mask, obs_sig, sorted_to_D_col, sorted_to_obs, ...
                              norm_denom, colors, OF_Plot, label_signatures_short, ...
                              nO, nC, nM, nSig_sum, output_dir, make_png)
    nRowsA1 = ceil(nO/2);
    fA1 = figure('units','normalized','outerposition',[0 0 0.7 1]);
    tiledlayout(nRowsA1, 2, 'TileSpacing','compact','Padding','compact');
    
    for oi = 1:nO
        nexttile; hold on;
        X = [];
        G = [];
        counts_per_sig = zeros(1, nSig_sum);   % NEW: track n per violin
        
        for si = 1:nSig_sum
            d_col   = sorted_to_D_col(si);
            obs_col = sorted_to_obs(si);
            if isnan(d_col) || isnan(obs_col); continue; end
            vals_all = [];
            for ci = 1:nC
                obsval = obs_sig(ci, obs_col);
                denom  = norm_denom(si, ci);
                if ~isfinite(obsval) || ~isfinite(denom) || denom == 0
                    continue;
                end
                for mi = 1:nM
                    keep  = squeeze(plot_mask(mi, ci, oi, :));
                    sig_v = squeeze(D.sig_all(mi, ci, oi, :, d_col));
                    of_v  = squeeze(D.of_all(mi, ci, oi, :));
                    keep = logical(keep(:));
                    joint = keep & isfinite(of_v(:)) & isfinite(sig_v(:));
                    if ~any(joint); continue; end
                    joint_top = joint;
                    joint_top(subset_n+1:end) = false;
                    if ~any(joint_top); continue; end
                    err_norm = (sig_v(joint_top) - obsval) / denom;
                    err_norm = err_norm(isfinite(err_norm) & abs(err_norm) <= 5);
                    vals_all = [vals_all; err_norm(:)]; %#ok<AGROW>
                end
            end
            if ~isempty(vals_all)
                X = [X; vals_all];                          %#ok<AGROW>
                G = [G; si*ones(numel(vals_all), 1)];       %#ok<AGROW>
                counts_per_sig(si) = numel(vals_all);       % NEW
            end
        end
        if ~isempty(X)
            safe_violinplot_local(X, G, colors(oi,:));
        end
        yline(0,'r-','Zero Error');
        ylim([-1.5 1.5]);
        xlim([0.5 nSig_sum + 0.5]);
        set(gca,'XTick',1:nSig_sum,'XTickLabel',label_signatures_short);
        xtickangle(45);
        title(OF_Plot{oi});
        if mod(oi,2)==1
            ylabel('Range-normalized signature error');
        end
        grid on;

        ax = gca;
        y_top = ax.YLim(2);
        y_off = 0.04 * range(ax.YLim);
        
        for si = 1:nSig_sum
            if counts_per_sig(si) > 0
        
                if counts_per_sig(si) >= 1000
                    label_txt = sprintf('%.1fk', counts_per_sig(si)/1000);
                else
                    label_txt = sprintf('%d', counts_per_sig(si));
                end
        
                text(si, y_top - y_off, label_txt, ...
                    'Rotation',90, ...
                    'HorizontalAlignment','right', ...
                    'VerticalAlignment','middle', ...
                    'FontSize',6, ...
                    'Color',[0.5 0.5 0.5], ...
                    'Clipping','on');
            end
        end

        %ax = gca;
        %ax.Clipping = 'off';
    end
    %sgtitle(sprintf(['Top-%d ensemble: per-run normalized signature error per objective\n', ...
    %                 '(pooled across passing models and catchments)'], subset_n));
    fontsize(fA1, 8, 'points');
    save_if_requested(fA1, output_dir, ...
        sprintf('summary_violin_normerr_per_objective_top%d_ensemble', subset_n), make_png);
end



function plot_a1_with_subset_balanced(subset_n, D, plot_mask, OF_best, obs_sig, ...
                                       sorted_to_D_col, sorted_to_obs, ...
                                       norm_denom, colors, OF_Plot, label_signatures_short, ...
                                       nO, nC, nM, nSig_sum, output_dir, make_png)
% Like plot_a1_with_subset but balances catchment contributions by
% capping the number of passing models per catchment to the minimum
% across catchments (for each OF). For each catchment, the top-k
% passing models are selected by best OF value.

    nRowsA1 = ceil(nO/2);
    fA1 = figure('units','normalized','outerposition',[0 0 0.7 1]);
    tiledlayout(nRowsA1, 2, 'TileSpacing','compact','Padding','compact');
    
    for oi = 1:nO
        nexttile; hold on;
        X = [];
        G = [];
        counts_per_sig = zeros(1, nSig_sum);

        % --- Determine cap: minimum number of passing models across catchments ---
        n_pass_per_catch = zeros(1, nC);
        for ci = 1:nC
            % Count models with at least one valid retained run
            n_here = 0;
            for mi = 1:nM
                if any(squeeze(plot_mask(mi, ci, oi, :)))
                    n_here = n_here + 1;
                end
            end
            n_pass_per_catch(ci) = n_here;
        end
        n_models_cap = max(1, min(n_pass_per_catch(n_pass_per_catch > 0)));

        % --- For each catchment, identify the top-n_models_cap models by best OF ---
        selected_models_per_catch = cell(1, nC);
        for ci = 1:nC
            best_of = OF_best(:, ci, oi);  % [nM x 1]
            % Mask out models that didn't pass
            pass_mask = false(nM, 1);
            for mi = 1:nM
                pass_mask(mi) = any(squeeze(plot_mask(mi, ci, oi, :)));
            end
            best_of_passing = best_of;
            best_of_passing(~pass_mask) = -Inf;
            [~, ord] = sort(best_of_passing, 'descend');
            selected_models_per_catch{ci} = ord(1:min(n_models_cap, sum(pass_mask)));
        end

        for si = 1:nSig_sum
            d_col   = sorted_to_D_col(si);
            obs_col = sorted_to_obs(si);
            if isnan(d_col) || isnan(obs_col); continue; end
            vals_all = [];
            for ci = 1:nC
                obsval = obs_sig(ci, obs_col);
                denom  = norm_denom(si, ci);
                if ~isfinite(obsval) || ~isfinite(denom) || denom == 0
                    continue;
                end
                selected_models = selected_models_per_catch{ci};
                for mi_idx = 1:numel(selected_models)
                    mi = selected_models(mi_idx);
                    keep  = squeeze(plot_mask(mi, ci, oi, :));
                    sig_v = squeeze(D.sig_all(mi, ci, oi, :, d_col));
                    of_v  = squeeze(D.of_all(mi, ci, oi, :));
                    keep = logical(keep(:));
                    joint = keep & isfinite(of_v(:)) & isfinite(sig_v(:));
                    if ~any(joint); continue; end
                    joint_top = joint;
                    joint_top(subset_n+1:end) = false;
                    if ~any(joint_top); continue; end
                    err_norm = (sig_v(joint_top) - obsval) / denom;
                    err_norm = err_norm(isfinite(err_norm) & abs(err_norm) <= 5);
                    vals_all = [vals_all; err_norm(:)]; %#ok<AGROW>
                end
            end
            if ~isempty(vals_all)
                X = [X; vals_all];                          %#ok<AGROW>
                G = [G; si*ones(numel(vals_all), 1)];       %#ok<AGROW>
                counts_per_sig(si) = numel(vals_all);
            end
        end
        if ~isempty(X)
            safe_violinplot_local(X, G, colors(oi,:));
        end
        yline(0,'r-','Zero Error');
        ylim([-1.5 1.5]);
        xlim([0.5 nSig_sum + 0.5]);
        set(gca,'XTick',1:nSig_sum,'XTickLabel',label_signatures_short);
        xtickangle(45);
        title(sprintf('%s (top-%d models per catch)', OF_Plot{oi}, n_models_cap));
        if mod(oi,2)==1
            ylabel('Range-normalized signature error');
        end
        grid on;

        ax = gca;
        y_top = ax.YLim(2);
        y_off = 0.03 * range(ax.YLim);
        
        for si = 1:nSig_sum
            if counts_per_sig(si) > 0
        
                if counts_per_sig(si) >= 1000
                    label_txt = sprintf('%.1fk', counts_per_sig(si)/1000);
                else
                    label_txt = sprintf('%d', counts_per_sig(si));
                end
        
                text(si, y_top + y_off, label_txt, ...
                    'Rotation',90, ...
                    'HorizontalAlignment','left', ...
                    'VerticalAlignment','bottom', ...
                    'FontSize',6, ...
                    'Color',[0.8 0.8 0.8], ...
                    'Clipping','off');
            end
        end

        %ax = gca;
        %ax.Clipping = 'off';
    end
    %sgtitle(sprintf(['Top-%d ensemble: balanced catchment contribution\n', ...
    %                 '(top models per catchment selected by best OF)'], subset_n));
    fontsize(fA1, 8, 'points');
    save_if_requested(fA1, output_dir, ...
        sprintf('summary_violin_normerr_per_objective_top%d_balanced', subset_n), make_png);
end


function plot_a1_with_subset_balanced_diagnostic(subset_n, D, plot_mask, OF_best, obs_sig, ...
                                                   sorted_to_D_col, sorted_to_obs, ...
                                                   norm_denom, colors, OF_Plot, label_signatures_short, ...
                                                   nO, nC, nM, nSig_sum, output_dir, make_png)

    fprintf('\n===== Diagnostic: where are the missing runs? =====\n');

    for oi = 1:nO
        % --- Compute cap as before ---
        n_pass_per_catch = zeros(1, nC);
        for ci = 1:nC
            n_here = 0;
            for mi = 1:nM
                if any(squeeze(plot_mask(mi, ci, oi, :)))
                    n_here = n_here + 1;
                end
            end
            n_pass_per_catch(ci) = n_here;
        end
        n_models_cap = max(1, min(n_pass_per_catch(n_pass_per_catch > 0)));

        selected_models_per_catch = cell(1, nC);
        for ci = 1:nC
            best_of = OF_best(:, ci, oi);
            pass_mask = false(nM, 1);
            for mi = 1:nM
                pass_mask(mi) = any(squeeze(plot_mask(mi, ci, oi, :)));
            end
            best_of_passing = best_of;
            best_of_passing(~pass_mask) = -Inf;
            [~, ord] = sort(best_of_passing, 'descend');
            selected_models_per_catch{ci} = ord(1:min(n_models_cap, sum(pass_mask)));
        end

        expected_n_per_catch = n_models_cap * subset_n;
        expected_n_per_violin = n_models_cap * subset_n * nC;

        fprintf('\n--- %s (cap=%d, expected per violin=%d) ---\n', ...
            OF_Plot{oi}, n_models_cap, expected_n_per_violin);

        % --- Diagnose a few signatures ---
        sigs_to_diagnose = [1, 6, 11];   % adjust if you want different ones
        for si_idx = sigs_to_diagnose
            si = si_idx;
            d_col   = sorted_to_D_col(si);
            obs_col = sorted_to_obs(si);
            if isnan(d_col) || isnan(obs_col); continue; end

            % Counters
            n_dropped_no_obs = 0;
            n_dropped_no_denom = 0;
            n_dropped_no_pass_mask = 0;
            n_dropped_nan_run = 0;
            n_dropped_outlier_clip = 0;
            n_kept = 0;
            n_catch_skipped = 0;

            for ci = 1:nC
                obsval = obs_sig(ci, obs_col);
                denom = norm_denom(si, ci);

                if ~isfinite(obsval)
                    n_dropped_no_obs = n_dropped_no_obs + expected_n_per_catch;
                    n_catch_skipped = n_catch_skipped + 1;
                    continue;
                end
                if ~isfinite(denom) || denom == 0
                    n_dropped_no_denom = n_dropped_no_denom + expected_n_per_catch;
                    n_catch_skipped = n_catch_skipped + 1;
                    continue;
                end

                selected_models = selected_models_per_catch{ci};
                for mi_idx = 1:numel(selected_models)
                    mi = selected_models(mi_idx);
                    keep  = squeeze(plot_mask(mi, ci, oi, :));
                    sig_v = squeeze(D.sig_all(mi, ci, oi, :, d_col));
                    of_v  = squeeze(D.of_all(mi, ci, oi, :));
                    keep = logical(keep(:));

                    % How many keep slots exist in this model's top-N?
                    keep_topN = keep;
                    keep_topN(subset_n+1:end) = false;
                    n_expected_here = sum(keep_topN);

                    if n_expected_here == 0
                        n_dropped_no_pass_mask = n_dropped_no_pass_mask + subset_n;
                        continue;
                    end

                    % Now apply finite checks
                    joint = keep_topN & isfinite(of_v(:)) & isfinite(sig_v(:));
                    n_dropped_nan = n_expected_here - sum(joint);
                    n_dropped_nan_run = n_dropped_nan_run + n_dropped_nan;

                    if ~any(joint); continue; end

                    err_norm = (sig_v(joint) - obsval) / denom;
                    n_before_clip = numel(err_norm);
                    err_norm = err_norm(isfinite(err_norm) & abs(err_norm) <= 5);
                    n_dropped_outlier_clip = n_dropped_outlier_clip + (n_before_clip - numel(err_norm));

                    n_kept = n_kept + numel(err_norm);

                    % Also count slots in this model's keep that fell outside top subset_n
                    n_outside_subset = sum(keep) - sum(keep_topN);
                    % These weren't dropped, they just weren't asked for; ignore.
                                % After computing n_dropped_nan, accumulate per (mi, ci)
                    if n_dropped_nan > 0 && si == 1
                        fprintf('      NaN at (oi=%d, ci=%s, mi=%s): %d runs\n', ...
                            oi, D.catchments{ci}, D.models{mi}, n_dropped_nan);
                    end
                end
                
            end

            total_loss = expected_n_per_violin - n_kept;
            fprintf('  sig %-15s expected=%d  kept=%d  lost=%d (%.1f%%)\n', ...
                label_signatures_short{si}, expected_n_per_violin, n_kept, total_loss, ...
                100 * total_loss / expected_n_per_violin);
            fprintf('    catchments skipped (no obs/denom):  %d   (loss: %d)\n', ...
                n_catch_skipped, n_dropped_no_obs + n_dropped_no_denom);
            fprintf('    runs not in plot_mask:              loss: %d\n', n_dropped_no_pass_mask);
            fprintf('    runs with NaN OF or signature:      loss: %d\n', n_dropped_nan_run);
            fprintf('    runs dropped by |err|>5 outlier:    loss: %d\n', n_dropped_outlier_clip);
        end
    end
end




%% ========== LOCAL HELPERS ==========
function cmap = redblue_local_2(n)
    r = [linspace(0.7,1,n/2), linspace(1,1,n/2)]';
    g = [linspace(0.7,0.3,n/2), linspace(0.3,0.7,n/2)]';
    b = [linspace(1,1,n/2),   linspace(1,0.7,n/2)]';
    cmap = [r,g,b];
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

function B = expand_bounds_inline(Braw, nM, nP)
    if iscell(Braw)
        B = nan(nM, nP);
        for mi = 1:nM
            b = double(Braw{mi});
            b = b(:)';
            B(mi,:) = b;
        end
        return;
    end

    Braw = double(squeeze(Braw));

    if isvector(Braw)
        b = Braw(:)';
        B = repmat(b, nM, 1);
    elseif isequal(size(Braw), [nM nP])
        B = Braw;
    elseif isequal(size(Braw), [nP nM])
        B = Braw';
    else
        error('Could not interpret bounds array.');
    end
end

function s = latex_escape_local(s)
    s = char(string(s));
    s = strrep(s, '\', '\textbackslash{}');
    s = strrep(s, '_', '\_');
    s = strrep(s, '%', '\%');
    s = strrep(s, '&', '\&');
    s = strrep(s, '#', '\#');
end