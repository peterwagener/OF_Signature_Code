function sig = sig_calc_cali_fixed(Q_sim, input_climate, cal_idx, t_sim_start)
% Returns numeric 1x15 signature vector for the calibration period.
% NOTE: This must be fast + deterministic. Do NOT return structs/cells.
%
% FIX (2026-05): The previous version (sig_calc_cali) hard-coded the
% simulation start date as 1981-01-02, which produced wrong calendar dates
% for any file whose actual epoch differs from 1981.  This caused sig_HFD_mean
% (half-flow date, uses day-of-year) to be computed against the wrong season.
%
% USAGE:
%   sig = sig_calc_cali_fixed(Q_sim, input_climate, cal_idx, t_sim_start)
%
%   t_sim_start  -- datetime scalar, the calendar date of Q_sim(1).
%                   In the workflow scripts this is date_array(1).
%                   Pass as the 4th arg; if omitted a warning is issued and
%                   the old 1981-01-02 fallback is used.
%
% REQUIRED CHANGE IN MARRMoT_model.m (line ~533):
%   OLD:  sig = sig_calc_cali(Q_sim, obj.input_climate, cal_idx);
%   NEW:  sig = sig_calc_cali_fixed(Q_sim, obj.input_climate, cal_idx, obj.sim_t_start);
%
%   and set  obj.sim_t_start = date_array(1)  in the workflow before calling
%   m.calibrate.  Alternatively, if MARRMoT stores input_climate as a struct
%   with a .t field, derive it as obj.input_climate.t(1).

persistent tosshAdded
if isempty(tosshAdded)
    % addpath is only effective in non-compiled MATLAB sessions.
    % In compiled (mcc) deployments TOSSH must be included at compile time.
    % The path here is intentionally left as a reminder; adjust to your system.
    addpath(genpath('<LOCAL_ROOT>/TOSSH-master/'));
    tosshAdded = true;
end

% ------------------------------------------------------------------
% Resolve simulation start date
% ------------------------------------------------------------------
if nargin < 4 || isempty(t_sim_start)
    warning('sig_calc_cali_fixed:NoStartDate', ...
        ['t_sim_start not provided — falling back to 1981-01-02. ' ...
         'sig_HFD_mean and other time-dependent signatures will be WRONG ' ...
         'unless the file actually starts on 1981-01-02.']);
    t_sim_start = datetime(1981, 1, 2);
end

% ------------------------------------------------------------------
% Inputs
% ------------------------------------------------------------------
Q_sim  = double(Q_sim(:));
P_full = double(input_climate(:,1));

% Build time vector anchored at the actual simulation start date.
t_full = t_sim_start + days((0:numel(Q_sim)-1)');

% ------------------------------------------------------------------
% Subset to calibration period
% ------------------------------------------------------------------
if islogical(cal_idx); cal_idx = find(cal_idx); end
Q = Q_sim(cal_idx);
P = P_full(cal_idx);
t = t_full(cal_idx);

if numel(Q) ~= numel(t)
    error("sig_calc_cali_fixed: numel(Q)=%d, numel(t)=%d (must match).", numel(Q), numel(t));
end

% ------------------------------------------------------------------
% Signature lists (must stay in fixed order!)
% ------------------------------------------------------------------
signatures_Q        = {'sig_FDC_slope','sig_RisingLimbDensity', ...
                       'sig_BaseflowRecessionK','sig_HFD_mean', ...
                       'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex'};
signatures_Q_P      = {'sig_EventRR','sig_TotalRR'};
signatures_high_low = {'sig_x_Q_duration','sig_x_Q_frequency'};
signatures_percentage = {'sig_x_percentile'};

% Build numeric vector in fixed order: 7 + 2 + 4 + 2 = 15
sig = nan(1,15);
k   = 0;

% Q-only (7)
for i = 1:numel(signatures_Q)
    k = k + 1;
    [sig(k), ~, ~] = feval(signatures_Q{i}, Q, t);
end

% Q+P (2)
for i = 1:numel(signatures_Q_P)
    k = k + 1;
    sig(k) = call_tossh_val(signatures_Q_P{i}, Q, t, P);
end

% high/low (2*2 = 4)
for i = 1:numel(signatures_high_low)
    k = k + 1;
    sig(k) = call_tossh_val(signatures_high_low{i}, Q, t, 'high');
    k = k + 1;
    sig(k) = call_tossh_val(signatures_high_low{i}, Q, t, 'low');
end

% percentiles (2)
for i = 1:numel(signatures_percentage)
    k = k + 1;
    sig(k) = call_tossh_val(signatures_percentage{i}, Q, t, 5);
    k = k + 1;
    sig(k) = call_tossh_val(signatures_percentage{i}, Q, t, 95);
end

end

% ======================================================================
function v = call_tossh_val(fname, varargin)
% Robust call wrapper: returns scalar double or NaN.
try
    [raw, flag] = feval(fname, varargin{:}); %#ok<ASGLU>
catch
    try
        raw  = feval(fname, varargin{:});
        flag = 0; %#ok<NASGU>
    catch
        raw = NaN;
    end
end

if isnumeric(raw) && isscalar(raw)
    v = double(raw);
else
    v = NaN;
end
end
