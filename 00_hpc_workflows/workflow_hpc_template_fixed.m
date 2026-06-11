function workflow_hpc(seed_str, results_root)
% Compiled MARRMoT entrypoint for one model/objective/catchment setup.
% Placeholders are injected by the compile script before mcc is called.
%
% FIX (2026-05): date_array is now read from the NetCDF 'date' variable and
% its 'units' attribute instead of being anchored to a hard-coded 1981-01-02
% start date.  The old approach silently calibrated on the wrong calendar
% years whenever a file's epoch differed from 1981.

if nargin < 1
    error('workflow_hpc:MissingSeed', 'A seed argument is required.');
end

seed = str2double(seed_str);
if ~isfinite(seed)
    error('workflow_hpc:InvalidSeed', 'Seed must be numeric, got "%s".', seed_str);
end

if nargin < 2 || strlength(string(results_root)) == 0
    results_root = pwd;
end

model = '__MODEL_NAME__';
objective_function = '__OBJECTIVE_NAME__';
catchment = '__CATCHMENT__';
path_nc = '__PATH_NC__';
pet_variable = 'potential_evaporation_sum_FAO_PENMAN_MONTEITH';

warmup = 365;

start_date_cali = datetime(2005, 1, 1);
end_date_cali   = datetime(2014, 12, 31);

start_date_cali_inc = datetime(2004, 1, 1);   % includes warmup year
end_date_cali_inc   = datetime(2014, 12, 31);

results_dir = fullfile(results_root, sprintf('seed_%04d', seed));
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

fn_nc = fullfile(path_nc, strcat(catchment, '.nc'));
if ~isfile(fn_nc)
    error('workflow_hpc:MissingNetcdf', 'NetCDF not found: %s', fn_nc);
end

% ------------------------------------------------------------------
% Read climate and streamflow variables
% ------------------------------------------------------------------
precip     = ncread(fn_nc, 'total_precipitation_sum');
temp       = ncread(fn_nc, 'temperature_2m_mean');
pet        = ncread(fn_nc, pet_variable);
streamflow = ncread(fn_nc, 'streamflow');

precip     = precip(:);
temp       = temp(:);
pet        = pet(:);
streamflow = streamflow(:);

% ------------------------------------------------------------------
% Build date_array from the file's own time axis
% ------------------------------------------------------------------
date_array = read_nc_dates(fn_nc, numel(precip));

% Sanity-check: the file must actually cover the calibration window
assert(date_array(1)   <= start_date_cali_inc, ...
    'workflow_hpc:DateRangeError', ...
    '%s starts on %s — before warmup start %s', ...
    catchment, char(date_array(1)), char(start_date_cali_inc));
assert(date_array(end) >= end_date_cali, ...
    'workflow_hpc:DateRangeError', ...
    '%s ends on %s — before calibration end %s', ...
    catchment, char(date_array(end)), char(end_date_cali));

% ------------------------------------------------------------------
% Derive calibration indices from the real time axis
% ------------------------------------------------------------------
t_cali_exl_warmup = find(date_array >= start_date_cali     & date_array <= end_date_cali);
t_cali_inc_warmup = find(date_array >= start_date_cali_inc & date_array <= end_date_cali_inc);

% Log coverage (fraction of non-NaN Q days in calibration window)
q_cali   = streamflow(t_cali_exl_warmup);
n_valid  = sum(~isnan(q_cali));
n_total  = numel(q_cali);
coverage = n_valid / n_total;
fprintf('[%s] Cal coverage: %d / %d days (%.1f%%)\n', catchment, n_valid, n_total, 100*coverage);
if coverage < 0.5
    warning('workflow_hpc:LowCoverage', ...
        '%s has only %.0f%% valid Q in calibration window — check data.', ...
        catchment, 100*coverage);
end

% ------------------------------------------------------------------
% Assemble MARRMoT input structures
% ------------------------------------------------------------------
input_climatology.precip  = precip;
input_climatology.temp    = temp;
input_climatology.pet     = pet;
input_climatology.delta_t = 1;
input_climatology.t       = date_array(:);

input_solver_opts.resnorm_tolerance = 0.1;
input_solver_opts.resnorm_maxiter   = 6;

m = feval(model);
parRanges = m.parRanges;
par_ini   = mean(parRanges, 2);

optim_opts.insigma          = 0.5 * (parRanges(:,2) - parRanges(:,1));
optim_opts.LBounds          = parRanges(:,1);
optim_opts.UBounds          = parRanges(:,2);
optim_opts.PopSize          = max(20, 4 + floor(10 * log(m.numParams)));
optim_opts.TolX             = 1e-3 * min(optim_opts.insigma);
optim_opts.TolFun           = 1e-3;
optim_opts.TolHistFun       = 1e-3;
optim_opts.MaxFunEvals      = 25000;
optim_opts.SaveFilename     = 'cmaesvars.mat';
optim_opts.LogFilenamePrefix = 'cmaes_';
optim_opts.Seed             = seed;
optim_opts.EvalParallel     = false;

% Runtime signature logging from the updated CMA-ES implementation.
optim_opts.AuxReturnAux = 'yes';
optim_opts.AuxLogToDisk = 'yes';
optim_opts.AuxLogPath   = fullfile(results_dir, 'signature_log.mat');

m.input_climate = input_climatology;
m.solver_opts   = input_solver_opts;
m.S0            = zeros(m.numStores, 1);

old_dir     = pwd;
cd(results_dir);
cleanup_obj = onCleanup(@() cd(old_dir)); %#ok<NASGU>

if isfile(optim_opts.AuxLogPath)
    delete(optim_opts.AuxLogPath);
end

t_start = tic;
[par_opt, of_cal, stopflag, output] = m.calibrate( ...
    streamflow, ...
    t_cali_exl_warmup, ...
    'my_cmaes', ...
    par_ini, ...
    optim_opts, ...
    objective_function, ...
    1, ...
    1);
walltime_s = toc(t_start);

m_eval          = feval(model);
m_eval.solver_opts = input_solver_opts;
input_s0        = zeros(m_eval.numStores, 1);
input_theta     = par_opt;

input_eval.precip  = precip(t_cali_inc_warmup);
input_eval.temp    = temp(t_cali_inc_warmup);
input_eval.pet     = pet(t_cali_inc_warmup);
input_eval.delta_t = 1;
input_eval.t       = date_array(t_cali_inc_warmup);

[output_ex, output_in, output_ss, output_waterbalance] = m_eval.get_output( ...
    input_eval, ...
    input_s0, ...
    input_theta, ...
    input_solver_opts);

summary = struct();
summary.catchment          = catchment;
summary.model              = model;
summary.objective_function = objective_function;
summary.seed               = seed;
summary.pet_variable       = pet_variable;
summary.of_cal             = of_cal;
summary.par_opt            = par_opt(:);
summary.numParams          = numel(par_opt);
summary.stopflag           = stringify_stopflag(stopflag);
summary.walltime_s         = walltime_s;
summary.results_dir        = results_dir;
summary.signature_log      = optim_opts.AuxLogPath;
summary.timestamp          = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd HH:mm:ss Z'));
summary.optim_output       = output;
summary.nc_date_start      = char(date_array(1));   % record for traceability
summary.nc_date_end        = char(date_array(end));
summary.cal_coverage_frac  = coverage;

save(fullfile(results_dir, 'summary.mat'), 'summary', 'output_ex', 'output_in', 'output_ss', 'output_waterbalance');
write_summary_csv(summary, fullfile(results_dir, 'summary.csv'));
end

% ======================================================================
function date_array = read_nc_dates(fn_nc, n_expected)
% Read the 'date' variable from the NetCDF and convert to datetime array.
% Falls back to a legacy 1981-01-02 anchor with a loud warning if the
% variable is absent (should never happen for CARAVAN-style files).

    try
        date_raw = double(ncread(fn_nc, 'date'));
        t_units  = ncreadatt(fn_nc, 'date', 'units');

        % Parse epoch from e.g. "days since 1951-01-01" or "days since 1951-01-01 00:00:00"
        tok = regexp(t_units, '(\d{4}-\d{2}-\d{2})', 'tokens', 'once');
        if isempty(tok)
            error('Cannot parse epoch from units string: "%s"', t_units);
        end
        epoch      = datetime(tok{1}, 'InputFormat', 'yyyy-MM-dd');
        date_array = epoch + days(date_raw(:));

        if numel(date_array) ~= n_expected
            error('workflow_hpc:DateLengthMismatch', ...
                'NetCDF ''date'' has %d entries but ''precip'' has %d.', ...
                numel(date_array), n_expected);
        end

    catch ME
        warning('workflow_hpc:FallbackDate', ...
            ['Could not read ''date'' from NetCDF (%s). ' ...
             'Falling back to hard-coded 1981-01-02 anchor — ' ...
             'calibration indices may be WRONG. Fix the NetCDF.'], ME.message);
        date_array = datetime(1981, 1, 2) + days((0:n_expected-1)');
    end
end

% ======================================================================
function write_summary_csv(summary, csv_path)
header = {'catchment','model','objective_function','seed','pet_variable','of_cal','stopflag', ...
    'walltime_s','numParams','par_opt','results_dir','signature_log','timestamp', ...
    'nc_date_start','nc_date_end','cal_coverage_frac'};

row = {summary.catchment, summary.model, summary.objective_function, summary.seed, summary.pet_variable, ...
    summary.of_cal, summary.stopflag, summary.walltime_s, summary.numParams, ...
    parameter_vector_to_text(summary.par_opt), summary.results_dir, summary.signature_log, summary.timestamp, ...
    summary.nc_date_start, summary.nc_date_end, summary.cal_coverage_frac};

fid = fopen(csv_path, 'w');
if fid == -1
    error('workflow_hpc:CsvOpenFailed', 'Could not open %s for writing.', csv_path);
end
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '%s\n', strjoin(header, ','));
fprintf(fid, '%s\n', csv_escape_row(row));
end

function out = csv_escape_row(values)
parts = strings(1, numel(values));
for i = 1:numel(values)
    value = values{i};
    if isstring(value)
        text = char(value);
    elseif ischar(value)
        text = value;
    elseif isnumeric(value) || islogical(value)
        text = num2str(value, 17);
    else
        text = char(string(value));
    end
    text = strrep(text, '"', '""');
    parts(i) = string(strcat('"', text, '"'));
end
out = strjoin(parts, ',');
end

function out = stringify_stopflag(stopflag)
if isstring(stopflag)
    out = char(strjoin(stopflag(:)', ' '));
elseif iscell(stopflag)
    out = char(strjoin(string(stopflag), ' '));
else
    out = char(string(stopflag));
end
end

function out = parameter_vector_to_text(values)
out = sprintf('%.17g;', values(:));
if ~isempty(out)
    out(end) = [];
end
end
