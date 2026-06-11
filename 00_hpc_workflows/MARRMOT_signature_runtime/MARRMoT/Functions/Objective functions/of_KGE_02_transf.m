function [val,c,idx,w] = of_KGE_02_transf(obs, sim, idx, w)
% of_KGE_02_transf Calculates Kling-Gupta Efficiency on power-0.2
% transformed streamflow.
%
% This version uses a soft penalty for invalid negative flows during
% calibration, instead of throwing an error that aborts the optimization.
%
% In:
% obs       - time series of observations       [nx1]
% sim       - time series of simulations        [nx1]
% idx       - optional vector of indices to use for calculation
% w         - optional weights of components    [3x1]
%
% Out:
% val       - transformed KGE value             [1x1]
% c         - components on transformed flows   [r,alpha,beta]
% idx       - indices used for the calculation
% w         - weights                           [wr,wa,wb]

%% Check inputs and select timesteps
if nargin < 2
    error('Not enough input arguments')
end

if nargin < 3
    idx = [];
end

[sim, obs, idx] = check_and_select(sim, obs, idx);

%% Set weights
w_default = [1,1,1];

if nargin < 4 || isempty(w)
    w = w_default;
else
    if ~isvector(w) || numel(w) ~= 3
        error('Weights should be a 3x1 or 1x3 vector.')
    end
    w = w(:).'; % keep indexing simple
end

%% Soft penalty for invalid flows
% Negative simulated flows can occur for bad parameter proposals during
% optimization. Do not crash the calibration; return a very poor score.
if any(obs < 0) || any(sim < 0)
    val = -Inf;
    c = [NaN; NaN; NaN];
    return
end

%% Apply streamflow transformation
lambda = 0.2;
obs_t = obs .^ lambda;
sim_t = sim .^ lambda;

%% Calculate KGE components on transformed flows
c = nan(3,1);
c(1) = corr(obs_t, sim_t);         % r
c(2) = std(sim_t) / std(obs_t);    % alpha
c(3) = mean(sim_t) / mean(obs_t);  % beta

%% Guard against invalid transformed statistics
if any(~isfinite(c))
    val = -Inf;
    return
end

%% Calculate weighted KGE
val = 1 - sqrt( ...
    (w(1) * (c(1) - 1))^2 + ...
    (w(2) * (c(2) - 1))^2 + ...
    (w(3) * (c(3) - 1))^2 );

end