%% Seed Uncertainty Analysis — TOP-N POOLED ACROSS SEEDS + PARAMETERS
clear; close all; clc;

%% USER SETTINGS
base_path  = '<LOCAL_DOWNLOADS>/all_fixed';
output_dir = '<LOCAL_ROOT>/graphics_new';
top_n      = 500;

marrmot_path = '<LOCAL_ROOT>/marrmot_211';

addpath(genpath(marrmot_path));

if ~exist(output_dir, 'dir'); mkdir(output_dir); end

localize = @(p) strrep(strrep(char(string(p)), ...
    '/data/horse/ws/<HPC_USER>-marrmot_recal/output_seeded/all_fixed/', ...
    '<LOCAL_DOWNLOADS>/all_fixed/'), ...
    '/data/horse/ws/<HPC_USER>-marrmot_recal/output_seeded/australian_fixed/', ...
    '<LOCAL_DOWNLOADS>/all_fixed/');

%% SIGNATURE MAPPING
sig_names = {'FDC Slope','Rising Limb Density','BF Recession K', ...
             'Mean Half Flow Date','BFI','Variability Index', ...
             'Flashiness Index','Event RR','Total RR', ...
             'High Flow Duration','Low Flow Duration', ...
             'High Flow Frequency','Low Flow Frequency', ...
             'Q5 (mm/d)','Q95 (mm/d)'};
nSigs = 15;

%% LOAD INDEX
fprintf('Loading index...\n');
T = readtable(fullfile(base_path,'index','combined_seed_index.tsv'), ...
    'FileType','text','Delimiter','\t');

T = T(strcmp(T.status,'complete'), :);
fprintf('  %d complete runs loaded\n', height(T));

catchments = unique(T.catchment, 'stable');
objectives = unique(T.objective, 'stable');
models     = unique(T.model, 'stable');

nC = numel(catchments);
nO = numel(objectives);
nM = numel(models);

fprintf('  %d catchments, %d objectives, %d models\n', nC, nO, nM);

%% DETECT MAX PARAMETER COUNT FROM MODEL NAMES
% MARRMoT model names encode parameter count, e.g. m_23_lascam_24p_3s.

nPars_by_model = nan(nM,1);

for mi = 1:nM
    tok = regexp(models{mi}, '_(\d+)p_', 'tokens', 'once');
    if isempty(tok)
        error('Could not parse parameter count from model name: %s', models{mi});
    end
    nPars_by_model(mi) = str2double(tok{1});
end

nPars_max = max(nPars_by_model);

par_names = arrayfun(@(i) sprintf('p%d', i), 1:nPars_max, 'UniformOutput', false);
nPars = nPars_max;

fprintf('Using padded parameter dimension nPars = %d\n', nPars);
fprintf('Parameter counts by model: min=%d, max=%d\n', min(nPars_by_model), max(nPars_by_model));
%% ALLOCATE ARRAYS
of_all  = nan(nM, nC, nO, top_n);
sig_all = nan(nM, nC, nO, top_n, nSigs);
par_all = nan(nM, nC, nO, top_n, nPars);

n_pooled_per_combo = zeros(nM, nC, nO);

%% PARAMETER BOUNDS FROM MARRMoT MODEL FILES
fprintf('Extracting parameter bounds from MARRMoT model files...\n');

par_lower = nan(nM, nPars);
par_upper = nan(nM, nPars);


for mi = 1:nM
    model_name = models{mi};
    nPars_this = nPars_by_model(mi);

    try
        [lb, ub, ~] = get_marrmot_bounds_local(model_name);

        if numel(lb) ~= nPars_this
            warning('Model %s expected %d parameters from name, but bounds returned %d.', ...
                model_name, nPars_this, numel(lb));
        end

        k = min([numel(lb), numel(ub), nPars_this]);

        par_lower(mi,1:k) = lb(1:k);
        par_upper(mi,1:k) = ub(1:k);

    catch ME
        warning('Could not extract bounds for model %s: %s', model_name, ME.message);
    end
end

if any(isnan(par_lower(:))) || any(isnan(par_upper(:)))
    warning(['Some parameter bounds are NaN. Boundary-hit analysis will be incomplete ', ...
             'unless these are fixed. Inspect par_lower/par_upper after extraction.']);
end

%% EXTRACT TOP-N POOLED ACROSS SEEDS
fprintf('Extracting top-%d runs pooled across seeds...\n', top_n);
tic;

n_files_read = 0;
n_files_failed = 0;

for mi = 1:nM
    if mod(mi, 5) == 0 || mi == 1
        fprintf('  Model %d/%d (%s) — %.1f min elapsed\n', ...
            mi, nM, models{mi}, toc/60);
    end

    for ci = 1:nC
        for oi = 1:nO

            rows = T(strcmp(T.model, models{mi}) & ...
                     strcmp(T.catchment, catchments{ci}) & ...
                     strcmp(T.objective, objectives{oi}), :);

            if isempty(rows)
                continue;
            end

            pooled_of  = [];
            pooled_sig = [];
            pooled_par = [];

            for ri = 1:height(rows)
                p_siglog = localize(rows.signature_log{ri});
                p_siglog_csv = strrep(p_siglog, '.mat', '.csv');

                if contains(p_siglog, '.csv')
                    p_siglog_csv = p_siglog;
                end

                if ~isfile(p_siglog_csv)
                    n_files_failed = n_files_failed + 1;
                    continue;
                end

                try
                    Slog = readtable(p_siglog_csv);

                    if ~ismember('fitness', Slog.Properties.VariableNames)
                        n_files_failed = n_files_failed + 1;
                        continue;
                    end

                    of_vec = -Slog.fitness;

                    sig_mat = nan(height(Slog), nSigs);
                    for si_idx = 1:nSigs
                        col_name = sprintf('s%d', si_idx);
                        if ismember(col_name, Slog.Properties.VariableNames)
                            sig_mat(:, si_idx) = Slog.(col_name);
                        end
                    end

                    par_mat = nan(height(Slog), nPars);
                    for pi_idx = 1:nPars
                        if ismember(par_names{pi_idx}, Slog.Properties.VariableNames)
                            par_mat(:, pi_idx) = Slog.(par_names{pi_idx});
                        end
                    end

                    pooled_of  = [pooled_of;  of_vec];  %#ok<AGROW>
                    pooled_sig = [pooled_sig; sig_mat]; %#ok<AGROW>
                    pooled_par = [pooled_par; par_mat]; %#ok<AGROW>

                    n_files_read = n_files_read + 1;

                catch ME
                    n_files_failed = n_files_failed + 1;
                    warning('Failed reading %s: %s', p_siglog_csv, ME.message);
                end
            end

            if isempty(pooled_of)
                continue;
            end

            valid = isfinite(pooled_of);
            pooled_of  = pooled_of(valid);
            pooled_sig = pooled_sig(valid, :);
            pooled_par = pooled_par(valid, :);

            if isempty(pooled_of)
                continue;
            end

            n_pooled_per_combo(mi, ci, oi) = numel(pooled_of);

            [pooled_of_sorted, ord] = sort(pooled_of, 'descend');
            pooled_sig_sorted = pooled_sig(ord, :);
            pooled_par_sorted = pooled_par(ord, :);

            nKeep = min(top_n, numel(pooled_of_sorted));

            of_all(mi, ci, oi, 1:nKeep)       = pooled_of_sorted(1:nKeep);
            sig_all(mi, ci, oi, 1:nKeep, :)   = pooled_sig_sorted(1:nKeep, :);
            par_all(mi, ci, oi, 1:nKeep, :)   = pooled_par_sorted(1:nKeep, :);
        end
    end
end

elapsed = toc;

fprintf('Done extracting (%.1f minutes)\n', elapsed/60);
fprintf('  Files read OK:    %d\n', n_files_read);
fprintf('  Files failed:     %d\n', n_files_failed);
fprintf('  Mean candidates per combo: %.1f\n', mean(n_pooled_per_combo(n_pooled_per_combo>0)));
fprintf('  Median candidates per combo: %.1f\n', median(n_pooled_per_combo(n_pooled_per_combo>0)));

%% STATS
of_mean = real(mean(of_all, 4, 'omitnan'));
of_std  = real(std(of_all, 0, 4, 'omitnan'));
of_q05  = quantile(of_all, 0.05, 4);
of_q50  = quantile(of_all, 0.50, 4);
of_q95  = quantile(of_all, 0.95, 4);

sig_mean = real(mean(sig_all, 4, 'omitnan'));
sig_std  = real(std(sig_all, 0, 4, 'omitnan'));
sig_q05  = quantile(sig_all, 0.05, 4);
sig_q50  = quantile(sig_all, 0.50, 4);
sig_q95  = quantile(sig_all, 0.95, 4);

par_mean = real(mean(par_all, 4, 'omitnan'));
par_std  = real(std(par_all, 0, 4, 'omitnan'));
par_q05  = quantile(par_all, 0.05, 4);
par_q50  = quantile(par_all, 0.50, 4);
par_q95  = quantile(par_all, 0.95, 4);

%% SAVE
save(fullfile(base_path, 'seed_uncertainty_data.mat'), ...
    'of_all','sig_all','par_all', ...
    'of_mean','of_std','of_q05','of_q50','of_q95', ...
    'sig_mean','sig_std','sig_q05','sig_q50','sig_q95', ...
    'par_mean','par_std','par_q05','par_q50','par_q95', ...
    'catchments','objectives','models', ...
    'sig_names','par_names','par_lower','par_upper', ...
    'nC','nO','nM','nSigs','nPars','nPars_by_model','top_n', ...
    'n_pooled_per_combo', '-v7.3');

fprintf('Data saved to seed_uncertainty_data.mat\n');

%% SUMMARY
fprintf('\n========== TOP-%d POOLED ENSEMBLE SUMMARY ==========\n', top_n);
fprintf('  of_all:  [%d x %d x %d x %d]\n', nM, nC, nO, top_n);
fprintf('  sig_all: [%d x %d x %d x %d x %d]\n', nM, nC, nO, top_n, nSigs);
fprintf('  par_all: [%d x %d x %d x %d x %d]\n', nM, nC, nO, top_n, nPars);

fill_frac = sum(~isnan(of_all(:))) / numel(of_all);
fprintf('\nOF fill rate: %.1f%% (%d / %d slots populated)\n', ...
    100*fill_frac, sum(~isnan(of_all(:))), numel(of_all));

fprintf('\nDone.\n');

%% =====================================================================
%% LOCAL FUNCTIONS

function [lb, ub, par_names] = get_marrmot_bounds_local(model_name)
    lb = [];
    ub = [];
    par_names = {};

    % Strategy 1: instantiate model directly.
    obj = [];
    try
        obj = feval(model_name);
    catch
        % Some model names may include file/class mismatches.
    end

    if ~isempty(obj)
        candidates = {'parRanges','ParRanges','parameterRanges','ParameterRanges', ...
                      'bounds','Bounds','thetaRanges','ThetaRanges'};

        for i = 1:numel(candidates)
            f = candidates{i};

            if isprop(obj, f)
                R = obj.(f);
                [lb, ub] = parse_range_matrix_local(R);
                if ~isempty(lb)
                    par_names = try_get_param_names_local(obj);
                    return;
                end
            elseif isfield_safe_local(obj, f)
                R = obj.(f);
                [lb, ub] = parse_range_matrix_local(R);
                if ~isempty(lb)
                    par_names = try_get_param_names_local(obj);
                    return;
                end
            end
        end

        % Common MARRMoT-style methods
        method_candidates = {'getParameterRanges','get_parameter_ranges', ...
                             'getParameters','get_parameter_bounds'};

        for i = 1:numel(method_candidates)
            m = method_candidates{i};
            if ismethod(obj, m)
                try
                    R = obj.(m)();
                    [lb, ub] = parse_range_matrix_local(R);
                    if ~isempty(lb)
                        par_names = try_get_param_names_local(obj);
                        return;
                    end
                catch
                end
            end
        end
    end

    % Strategy 2: parse the model file text.
    model_path = which(model_name);
    if isempty(model_path)
        error('Could not locate model file for %s with which().', model_name);
    end

    txt = fileread(model_path);

    patterns = { ...
        'parRanges\s*=\s*\[([^\]]+)\]', ...
        'ParRanges\s*=\s*\[([^\]]+)\]', ...
        'parameterRanges\s*=\s*\[([^\]]+)\]', ...
        'thetaRanges\s*=\s*\[([^\]]+)\]'};

    for p = 1:numel(patterns)
        tok = regexp(txt, patterns{p}, 'tokens', 'once');
        if ~isempty(tok)
            R = str2num(tok{1}); %#ok<ST2NM>
            [lb, ub] = parse_range_matrix_local(R);
            if ~isempty(lb)
                par_names = parse_param_names_from_text_local(txt);
                return;
            end
        end
    end

    error('Could not extract bounds from model %s.', model_name);
end

function tf = isfield_safe_local(obj, name)
    tf = false;
    try
        tf = isstruct(obj) && isfield(obj, name);
    catch
    end
end

function [lb, ub] = parse_range_matrix_local(R)
    lb = [];
    ub = [];

    if isempty(R)
        return;
    end

    if istable(R)
        R = table2array(R);
    end

    R = double(R);

    if size(R,2) == 2
        lb = R(:,1)';
        ub = R(:,2)';
    elseif size(R,1) == 2
        lb = R(1,:);
        ub = R(2,:);
    end
end

function names = try_get_param_names_local(obj)
    names = {};

    candidates = {'parNames','ParNames','parameterNames','ParameterNames', ...
                  'thetaNames','ThetaNames','inputNames','InputNames'};

    for i = 1:numel(candidates)
        f = candidates{i};
        try
            if isprop(obj, f)
                names = cellstr(string(obj.(f)(:)))';
                return;
            end
        catch
        end
    end
end

function names = parse_param_names_from_text_local(txt)
    names = {};

    patterns = { ...
        'parNames\s*=\s*\{([^\}]+)\}', ...
        'ParNames\s*=\s*\{([^\}]+)\}', ...
        'parameterNames\s*=\s*\{([^\}]+)\}', ...
        'thetaNames\s*=\s*\{([^\}]+)\}'};

    for p = 1:numel(patterns)
        tok = regexp(txt, patterns{p}, 'tokens', 'once');
        if isempty(tok)
            continue;
        end

        raw = tok{1};
        q = regexp(raw, '''([^'']+)''', 'tokens');
        if ~isempty(q)
            names = cellfun(@(c)c{1}, q, 'UniformOutput', false);
            return;
        end
    end
end