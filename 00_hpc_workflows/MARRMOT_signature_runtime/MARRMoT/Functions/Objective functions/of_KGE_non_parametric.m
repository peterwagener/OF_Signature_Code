% Implementation of OF based on https://doi.org/10.1080/02626667.2018.1552002

function [val,c,idx,w] = of_KGE_non_parametric(sim, obs,idx,w)

%% Check inputs and select timesteps
if nargin < 2
    error('Not enugh input arguments')    
end

if nargin < 3; idx = []; end
[sim, obs, idx] = check_and_select(sim, obs, idx);

%% Set weights
w_default = [1,1,1];          % default weights

% update defaults weights if needed  
if nargin < 4 || isempty(w)
    w = w_default;
else
    if ~min(size(w)) == 1 || ~max(size(w)) == 3                            % check weights variable for size
        error('Weights should be a 3x1 or 1x3 vector.')                    % or throw error        
    end
end   

%% Calculate Non-Parametric KGE
% Remove NaN values
indices = ~isnan(sim) & ~isnan(obs);
sim = sim(indices);
obs = obs(indices);

% Calculate mean sim and obs
mean_sim = mean(sim, 'omitnan');
mean_obs = mean(obs, 'omitnan');

% Calculate normalized flow duration curves
fdc_sim = sort(sim / (mean_sim * length(sim)));
fdc_obs = sort(obs / (mean_obs * length(obs)));

% Calculate alpha component
c(1) = 1 - 0.5 * sum(abs(fdc_sim - fdc_obs));

% Calculate beta component
c(2) = mean_sim / mean_obs;

% Calculate r component
c(3) = corr(sim, obs, 'Type', 'Spearman');

% Return Non-Parametric Efficiency value
val = 1 - sqrt((c(1) - 1)^2 + (c(2) - 1)^2 + (c(3) - 1)^2);

end