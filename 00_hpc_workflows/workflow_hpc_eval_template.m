function workflow_hpc_eval(catchment, objective_function, path_nc, cali_root, eval_root)
% Evaluation-period forward run using the best calibrated parameter set.
%
% Finds the seed with highest of_cal across calibration seeds, loads
% par_opt, runs the model on the evaluation period, computes the same
% 15 TOSSH signatures, and writes eval_summary.csv + eval_signatures.csv.
%
% Only __MODEL_NAME__ is a compile-time placeholder (injected by compile script).
% All other arguments are passed at runtime so ONE binary per model covers
% all catchment × objective combinations.
%
% Args:
%   catchment          e.g. 'camels_02017500'
%   objective_function e.g. 'of_KGE'
%   path_nc            path to NetCDF directory
%   cali_root          root of calibration results (output_seeded/all_fixed or australian_fixed)
%   eval_root          root for eval output (output_seeded/eval_fixed)

model = '__MODEL_NAME__';

if nargin < 1
    error('workflow_hpc_eval:MissingArgs', ...
        'Usage: workflow_hpc_eval catchment objective path_nc cali_root eval_root');
end

pet_variable = 'potential_evaporation_sum_FAO_PENMAN_MONTEITH';

% ------------------------------------------------------------------
% Eval period dates (catchment-specific)
% ------------------------------------------------------------------
if strcmp(catchment, 'camelsaus_607155')
    % AUS6: data gap after 1999 — eval 1981-1988 (warmup 1980)
    start_date_eval_inc = datetime(1980, 1, 1);
    end_date_eval_inc   = datetime(1988, 12, 31);
    start_date_eval_exc = datetime(1981, 1, 1);
    end_date_eval_exc   = datetime(1988, 12, 31);
else
    % All other catchments: 1993-2003 (warmup 1993, eval 1994-2003)
    start_date_eval_inc = datetime(1993, 1, 1);
    end_date_eval_inc   = datetime(2003, 12, 31);
    start_date_eval_exc = datetime(1994, 1, 1);
    end_date_eval_exc   = datetime(2003, 12, 31);
end

% ------------------------------------------------------------------
% Find best calibration seed
% ------------------------------------------------------------------
cali_dir  = fullfile(cali_root, catchment, objective_function, model);
best_of   = -Inf;
best_seed_dir = '';

for seed = 1:5
    sd  = fullfile(cali_dir, sprintf('seed_%04d', seed));
    csv = fullfile(sd, 'summary.csv');
    if ~isfile(csv), continue; end
    try
        T      = readtable(csv, 'FileType', 'text', 'Delimiter', ',', ...
            'VariableNamingRule', 'preserve');
        of_val = T.("of_cal")(1);
        if isfinite(of_val) && of_val > best_of
            best_of       = of_val;
            best_seed_dir = sd;
        end
    catch
    end
end

if isempty(best_seed_dir)
    error('workflow_hpc_eval:NoCalibration', ...
        'No completed calibration found for %s / %s / %s', ...
        catchment, objective_function, model);
end

% Load par_opt
mat_file = fullfile(best_seed_dir, 'summary.mat');
if isfile(mat_file)
    S       = load(mat_file, 'summary');
    par_opt = double(S.summary.par_opt(:));
else
    T       = readtable(fullfile(best_seed_dir, 'summary.csv'), ...
        'FileType', 'text', 'Delimiter', ',', 'VariableNamingRule', 'preserve');
    parts   = strsplit(strtrim(strrep(char(T.("par_opt")(1)), '"', '')), ';');
    par_opt = cellfun(@str2double, parts(~cellfun(@isempty, parts)));
    par_opt = par_opt(:);
end

fprintf('[%s/%s/%s] Best seed: %s  of_cal=%.4f  nParams=%d\n', ...
    catchment, objective_function, model, best_seed_dir, best_of, numel(par_opt));

% ------------------------------------------------------------------
% Read NetCDF
% ------------------------------------------------------------------
fn_nc = fullfile(path_nc, strcat(catchment, '.nc'));
if ~isfile(fn_nc)
    error('workflow_hpc_eval:MissingNetcdf', 'NetCDF not found: %s', fn_nc);
end

precip     = double(ncread(fn_nc, 'total_precipitation_sum'));
temp       = double(ncread(fn_nc, 'temperature_2m_mean'));
pet        = double(ncread(fn_nc, pet_variable));
streamflow = double(ncread(fn_nc, 'streamflow'));

precip     = precip(:);  temp = temp(:);
pet        = pet(:);     streamflow = streamflow(:);

date_array = read_nc_dates(fn_nc, numel(precip));

% ------------------------------------------------------------------
% Derive eval indices
% ------------------------------------------------------------------
assert(date_array(1) <= start_date_eval_inc, ...
    'workflow_hpc_eval:DateRangeError', ...
    '%s starts %s — before eval warmup start %s', ...
    catchment, char(date_array(1)), char(start_date_eval_inc));
assert(date_array(end) >= end_date_eval_inc, ...
    'workflow_hpc_eval:DateRangeError', ...
    '%s ends %s — before eval end %s', ...
    catchment, char(date_array(end)), char(end_date_eval_inc));

t_eval_inc = find(date_array >= start_date_eval_inc & date_array <= end_date_eval_inc);
t_eval_exc = find(date_array >= start_date_eval_exc & date_array <= end_date_eval_exc);

q_obs_eval = streamflow(t_eval_exc);
n_valid    = sum(~isnan(q_obs_eval));
coverage   = n_valid / numel(q_obs_eval);
fprintf('[%s] Eval coverage: %d / %d days (%.1f%%)\n', ...
    catchment, n_valid, numel(q_obs_eval), 100*coverage);

% ------------------------------------------------------------------
% Run model forward on eval period (inc warmup)
% ------------------------------------------------------------------
input_solver_opts.resnorm_tolerance = 0.1;
input_solver_opts.resnorm_maxiter   = 6;

m_eval             = feval(model);
m_eval.solver_opts = input_solver_opts;

input_eval.precip  = precip(t_eval_inc);
input_eval.temp    = temp(t_eval_inc);
input_eval.pet     = pet(t_eval_inc);
input_eval.delta_t = 1;
input_eval.t       = date_array(t_eval_inc);

t_start = tic;
[output_ex, ~, ~, ~] = m_eval.get_output( ...
    input_eval, zeros(m_eval.numStores, 1), par_opt, input_solver_opts);
walltime_s = toc(t_start);

Q_sim_full = output_ex.Q(:);

% Post-warmup indices within the eval sub-array
t_exc_local = find(date_array(t_eval_inc) >= start_date_eval_exc & ...
                   date_array(t_eval_inc) <= end_date_eval_exc);
Q_eval      = Q_sim_full(t_exc_local);
t_eval_dates = date_array(t_eval_exc);
P_eval       = precip(t_eval_exc);

% ------------------------------------------------------------------
% Objective function value on eval period
% ------------------------------------------------------------------
of_eval = compute_of(Q_eval, q_obs_eval, P_eval, objective_function);
fprintf('[%s/%s/%s] of_eval=%.4f  walltime=%.1fs\n', ...
    catchment, objective_function, model, of_eval, walltime_s);

% ------------------------------------------------------------------
% Signatures on eval period
% ------------------------------------------------------------------
sig_eval = compute_signatures(Q_eval, P_eval, t_eval_dates);

% ------------------------------------------------------------------
% Write results
% ------------------------------------------------------------------
out_dir = fullfile(eval_root, catchment, objective_function, model);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% eval_summary.csv
fid = fopen(fullfile(out_dir, 'eval_summary.csv'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, ['catchment,model,objective_function,of_cal_best,of_eval,' ...
    'best_seed_dir,eval_start,eval_end,eval_coverage_frac,walltime_s,timestamp\n']);
fprintf(fid, '"%s","%s","%s",%.17g,%.17g,"%s","%s","%s",%.6f,%.1f,"%s"\n', ...
    catchment, model, objective_function, best_of, of_eval, best_seed_dir, ...
    char(start_date_eval_exc), char(end_date_eval_exc), coverage, walltime_s, ...
    char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd HH:mm:ss Z')));
clear cleanup

% eval_signatures.csv
sig_names = {'s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15'};
fid2 = fopen(fullfile(out_dir, 'eval_signatures.csv'), 'w');
cleanup2 = onCleanup(@() fclose(fid2));
fprintf(fid2, '%s\n', strjoin(sig_names, ','));
fprintf(fid2, '%s\n', strjoin(arrayfun(@(x) sprintf('%.17g',x), sig_eval, 'UniformOutput', false), ','));
clear cleanup2

fprintf('[%s/%s/%s] Done. Results in %s\n', catchment, objective_function, model, out_dir);
end

% ======================================================================
function date_array = read_nc_dates(fn_nc, n_expected)
    date_raw   = double(ncread(fn_nc, 'date'));
    t_units    = ncreadatt(fn_nc, 'date', 'units');
    tok        = regexp(t_units, '(\d{4}-\d{2}-\d{2})', 'tokens', 'once');
    epoch      = datetime(tok{1}, 'InputFormat', 'yyyy-MM-dd');
    date_array = epoch + days(date_raw(:));
    assert(numel(date_array) == n_expected, 'Date length %d != precip length %d', ...
        numel(date_array), n_expected);
end

% ======================================================================
function sig = compute_signatures(Q, P, t)
% Same 15-signature order as sig_calc_cali_fixed.m
    sig = nan(1, 15);
    k = 0;
    for fn = {'sig_FDC_slope','sig_RisingLimbDensity','sig_BaseflowRecessionK', ...
              'sig_HFD_mean','sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex'}
        k = k+1;
        try [sig(k),~,~] = feval(fn{1}, Q, t); catch; end
    end
    for fn = {'sig_EventRR','sig_TotalRR'}
        k = k+1; sig(k) = safe_call(fn{1}, Q, t, P);
    end
    k = k+1; sig(k) = safe_call('sig_x_Q_duration',  Q, t, 'high');
    k = k+1; sig(k) = safe_call('sig_x_Q_duration',  Q, t, 'low');
    k = k+1; sig(k) = safe_call('sig_x_Q_frequency', Q, t, 'high');
    k = k+1; sig(k) = safe_call('sig_x_Q_frequency', Q, t, 'low');
    k = k+1; sig(k) = safe_call('sig_x_percentile',  Q, t, 5);
    k = k+1; sig(k) = safe_call('sig_x_percentile',  Q, t, 95);
end

function v = safe_call(fname, varargin)
    try; [raw,~] = feval(fname, varargin{:}); catch
    try;  raw    = feval(fname, varargin{:}); catch; raw = NaN; end; end
    if isnumeric(raw) && isscalar(raw); v = double(raw); else; v = NaN; end
end

% ======================================================================
function of_val = compute_of(Q_sim, Q_obs, P, of_name)
    try
        % of_SHE requires precipitation as 3rd argument
        if strcmp(of_name, 'of_SHE')
            of_val = feval(of_name, Q_obs, Q_sim, P);
        else
            of_val = feval(of_name, Q_obs, Q_sim);
        end
    catch
        of_val = NaN;
    end
end
