%% hydrobm_thresholds.m
% Evaluate all hydrobm benchmarks against every objective function and
% retain the best-performing benchmark score as the threshold per
% catchment/objective. Drop-in replacement for build_interannual_benchmark
% + compute_thresholds in bm_plots_new_runs.m.
%
% Output:
%   threshold       [nO x nC]   best benchmark score per objective/catchment
%   best_benchmark  {nO x nC}   name of the winning benchmark
%   catchments      {nC x 1}    gauge IDs (same order as threshold columns)
%   objectives      {nO x 1}    objective function names (same order as rows)

clear; close all; clc;

%% ========================= SETTINGS =========================
cfg.bm_dir   = '<LOCAL_ROOT>/benchmark_results';
cfg.out_file = fullfile(cfg.bm_dir, 'hydrobm_thresholds.mat');

addpath(genpath('<LOCAL_ROOT>/marrmot_211/'))

objectives = { ...
    'of_KGE', ...
    'of_NSE', ...
    'of_log_NSE', ...
    'of_KGE_non_parametric', ...
    'of_KGE_split', ...
    'of_SHE', ...
    'of_diagnostic_efficiency', ...
    'of_KGE_02_transf'};

%% ========================= DISCOVER BASINS =========================
bm_files   = dir(fullfile(cfg.bm_dir, '*_benchmarks.mat'));
catchments = cell(numel(bm_files), 1);
for i = 1:numel(bm_files)
    catchments{i} = strrep(bm_files(i).name, '_benchmarks.mat', '');
end

nC = numel(catchments);
nO = numel(objectives);

threshold      = nan(nO, nC);
best_benchmark = cell(nO, nC);
all_scores     = nan(nO, nC, 0);   % will grow to [nO x nC x nBM] on first load

fprintf('Found %d catchments, %d objectives.\n', nC, nO);

%% ========================= MAIN LOOP =========================
for ci = 1:nC
    gauge_id = catchments{ci};
    fprintf('[%d/%d] %s\n', ci, nC, gauge_id);

    bm = load(fullfile(cfg.bm_dir, [gauge_id '_benchmarks.mat']));

    q_obs      = double(bm.q_obs(:));
    precip_obs = double(bm.precip_obs(:));
    bm_names   = cellstr(bm.benchmark_names);
    bm_flows   = double(bm.benchmark_flows);   % [T x nBM]
    nBM        = numel(bm_names);

    % Initialise score cube on first basin
    if isempty(all_scores) || size(all_scores, 3) == 0
        all_scores = nan(nO, nC, nBM);
    end

    for oi = 1:nO
        obj_fun = objectives{oi};
        scores  = nan(1, nBM);

        for bi = 1:nBM
            sim = bm_flows(:, bi);
            sim(isnan(sim)) = 0;   % replace NaN with 0 (some benchmarks)

            try
                scores(bi) = compute_objective_value(obj_fun, q_obs, sim, precip_obs);
            catch ME
                warning('  Skipped %s | %s | %s: %s', gauge_id, obj_fun, bm_names{bi}, ME.message);
            end
        end

        all_scores(oi, ci, :) = scores;

        [best_val, best_idx] = max(scores, [], 'omitnan');
        if ~isnan(best_val)
            threshold(oi, ci)      = best_val;
            best_benchmark{oi, ci} = bm_names{best_idx};
        end
    end
end

%% ========================= LOWER BOUNDS =========================
% KGE: floor at 0  (benchmark must beat mean flow)
idx_kge = find(strcmp(objectives, 'of_KGE'), 1);
if ~isempty(idx_kge)
    threshold(idx_kge, threshold(idx_kge,:) < 0) = 0;
end

% NSE: floor at -0.4142  (1 - sqrt(2), i.e. benchmark must beat mean flow)
idx_nse = find(strcmp(objectives, 'of_NSE'), 1);
if ~isempty(idx_nse)
    threshold(idx_nse, threshold(idx_nse,:) < -0.4142) = -0.4142;
end

%% ========================= SAVE =========================
save(cfg.out_file, 'threshold', 'best_benchmark', 'all_scores', ...
     'catchments', 'objectives', 'cfg');
fprintf('\nSaved to %s\n', cfg.out_file);

%% ========================= SUMMARY =========================
fprintf('\n--- Best benchmark per objective (across all catchments) ---\n');
for oi = 1:nO
    counts = struct();
    for ci = 1:nC
        bm = best_benchmark{oi, ci};
        if ~isempty(bm)
            key = matlab.lang.makeValidName(bm);
            if isfield(counts, key)
                counts.(key) = counts.(key) + 1;
            else
                counts.(key) = 1;
            end
        end
    end
    names  = fieldnames(counts);
    vals   = cellfun(@(f) counts.(f), names);
    [~, ord] = sort(vals, 'descend');
    fprintf('  %-30s  top: %s (%dx)\n', objectives{oi}, strrep(names{ord(1)},'_',' '), vals(ord(1)));
end

%% ========================= LOCAL FUNCTIONS =========================
function val = compute_objective_value(obj_fun, obs, sim, precip)
if isstring(obj_fun), obj_fun = char(obj_fun); end
if contains(obj_fun, 'SHE', 'IgnoreCase', true)
    val = feval(obj_fun, obs, sim, precip);
else
    val = feval(obj_fun, obs, sim);
end
end
