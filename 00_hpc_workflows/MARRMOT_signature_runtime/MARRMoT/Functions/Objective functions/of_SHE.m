% Implementation of OF based on https://doi.org/10.1029/2023WR035321

function [val,c,idx,w] = of_SHE(obs, sim, precip, idx, w)

%% Check inputs and select timesteps
if nargin < 3
    error('Not enugh input arguments')    
end

if nargin < 4; idx = []; end
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

%% calculate SHE index
% Runoff Ratio Mean
c(1) = (mean(sim)/mean(precip))/(mean(obs)/mean(precip));
% Runoff Ratio Variance
c(2) = (std(sim)/std(precip))/(std(obs)/std(precip));
% Pearson Correlation
c(3) = corr(sim, obs, 'Type', 'Spearman');

val = 1 - sqrt((c(1) - 1)^2 + (c(2) - 1)^2 + (c(3) - 1)^2);

end