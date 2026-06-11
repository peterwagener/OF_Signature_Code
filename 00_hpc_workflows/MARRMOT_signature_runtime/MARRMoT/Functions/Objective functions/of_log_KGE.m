function [val,c,idx,w] = of_log_KGE(obs, sim, idx, w)

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

%% Find the constant e
e = mean(obs)/100;

%% Apply constant and transform flows
obs = log(obs+e);
sim = log(sim+e);

%% calculate components
c(1) = corr(obs,sim);                                             % r: linear correlation
c(2) = std(sim)/std(obs);                                         % alpha: ratio of standard deviations
c(3) = mean(sim)/mean(obs);                                       % beta: bias 

%% calculate value
val = 1-sqrt((w(1)*(c(1)-1))^2 + (w(2)*(c(2)-1))^2 + (w(3)*(c(3)-1))^2);    % weighted KGE

end