% Implementation of OF based on https://doi.org/10.1029/2017WR022466

function [val,c,idx,w] = of_KGE_split(obs, sim, idx, w)

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

%% create subyear arrays
number_years = floor(length(obs)/365);
store_KGE_yearly = NaN(number_years,1);

for i = 1:number_years
    index_low = 1+365*(i-1);
    index_high = 365+365*(i-1);
    obs_year = obs(index_low:index_high);
    sim_year = sim(index_low:index_high);

    % calculate and store KGE values
    c(1) = corr(obs_year,sim_year);                                             % r: linear correlation
    c(2) = std(sim_year)/std(obs_year);                                         % alpha: ratio of standard deviations
    c(3) = mean(sim_year)/mean(obs_year);                                       % beta: bias

    store_KGE_yearly(i,1) = 1-sqrt((w(1)*(c(1)-1))^2 + (w(2)*(c(2)-1))^2 + (w(3)*(c(3)-1))^2);
end


%% calculate mean KGE
val = mean(store_KGE_yearly, 'omitnan');

end