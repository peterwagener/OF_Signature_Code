function [val,c,idx,w] = of_KGE_neg2_transf(obs, sim, idx, w)

if nargin < 2
    error('Not enough input arguments')
end

if nargin < 3
    idx = [];
end

[sim, obs, idx] = check_and_select(sim, obs, idx);

% Remove any invalid observed values
valid = ~isnan(obs) & ~isnan(sim) & obs >= 0;
obs = obs(valid);
sim = sim(valid);
idx = idx(valid);

if isempty(obs)
    val = -Inf;
    c = [NaN; NaN; NaN];
    return
end

% Clamp simulated negative flows to zero
sim(sim < 0) = 0;

% Set weights
w_default = [1,1,1];
if nargin < 4 || isempty(w)
    w = w_default;
else
    if ~isvector(w) || numel(w) ~= 3
        error('Weights should be a 3x1 or 1x3 vector.')
    end
    w = w(:).';
end

% Regularization near zero
obs_pos = obs(obs > 0);
if isempty(obs_pos)
    val = -Inf;
    c = [NaN; NaN; NaN];
    return
end

eps_q = max(1e-6, 0.01 * mean(obs_pos));

obs_t = (obs + eps_q).^(-2);
sim_t = (sim + eps_q).^(-2);

c = nan(3,1);
c(1) = corr(obs_t, sim_t);
c(2) = std(sim_t) / std(obs_t);
c(3) = mean(sim_t) / mean(obs_t);

val = 1 - sqrt( ...
    (w(1) * (c(1) - 1))^2 + ...
    (w(2) * (c(2) - 1))^2 + ...
    (w(3) * (c(3) - 1))^2 );
end