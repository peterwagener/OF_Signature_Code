function sig = sig_calc_cali(Q_sim, input_climate, cal_idx)
% Returns numeric 1x15 signature vector for the calibration period.
% NOTE: This must be fast + deterministic. Do NOT return structs/cells.

persistent tosshAdded
if isempty(tosshAdded)
    addpath(genpath('<LOCAL_ROOT>/TOSSH-master/'));
    tosshAdded = true;
end

% inputs are numeric inside MARRMoT
Q_sim  = double(Q_sim(:));
P_full = double(input_climate(:,1));

% daily time vector (same convention as your scripts)
t0 = datetime(1981,1,2);
t_full = t0 + days((0:numel(Q_sim)-1)');

% subset
if islogical(cal_idx); cal_idx = find(cal_idx); end
Q = Q_sim(cal_idx);
P = P_full(cal_idx);
t = t_full(cal_idx);

if numel(Q) ~= numel(t)
    error("sig_calc_cali: numel(Q)=%d, numel(t)=%d (must match).", numel(Q), numel(t));
end


% signature lists (must stay fixed order!)
signatures_Q = {'sig_FDC_slope','sig_RisingLimbDensity',...
    'sig_BaseflowRecessionK','sig_HFD_mean',...
    'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex'};
signatures_Q_P = {'sig_EventRR','sig_TotalRR'};
signatures_high_low = {'sig_x_Q_duration','sig_x_Q_frequency'};
signatures_percentage = {'sig_x_percentile'};

% Build numeric vector in fixed order: 7 + 2 + 4 + 2 = 15
sig = nan(1,15);
k = 0;

% Q-only (7)
for i = 1:numel(signatures_Q)
    k = k + 1;
    [sig(k),err,e_str] = feval(signatures_Q{i}, Q, t);
    %disp(sig(k))
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

function v = call_tossh_val(fname, varargin)
% Robust call wrapper: returns scalar double or NaN.

try
    % Most TOSSH signatures: [val, error_flag, error_str, ...]
    [raw, flag] = feval(fname, varargin{:}); %#ok<ASGLU>
catch
    try
        raw = feval(fname, varargin{:});
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