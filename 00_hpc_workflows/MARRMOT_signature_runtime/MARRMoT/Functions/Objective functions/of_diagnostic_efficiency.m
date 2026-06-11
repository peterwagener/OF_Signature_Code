% Implementation of OF based on https://doi.org/10.5194/hess-25-2187-2021

function [val,c,idx,w] = of_diagnostic_efficiency(obs, sim, idx, w)

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

%% calculate diagnostic efficiency (DE)
% Calculate mean relative bias
sim_sort = sort(sim);
obs_sort = sort(obs);

zero_indices = (obs_sort == 0);

if any(zero_indices)
    % Handle division by zero
    warning('Division by zero in Q_rel calculation. Setting Q_rel to NaN for corresponding elements.');
    Q_rel = zeros(size(sim_sort));  % Set Q_rel to zeros or any other appropriate value
    Q_rel(~zero_indices) = (sim_sort(~zero_indices) - obs_sort(~zero_indices)) ./ obs_sort(~zero_indices);
else
    % No division by zero, proceed with the calculation
    Q_rel = (sim_sort - obs_sort) ./ obs_sort;
end

c(1) =  mean(Q_rel,'omitnan');
% calculate area of residual bias
% use integral function
Q_res = Q_rel-c(1);

perc = linspace(0, 1, length(Q_res));

% Calculate the area of absolute bias using the trapezoidal rule
c(2) = trapz(perc, abs(Q_res));
%c(2) = integral(@(exceed_prob) abs(Q_res(exceed_prob)),0,1);
if c(2)<0.001
    c(2)=0;
end
% calculat pearson correlation
c(3) = corr(sim, obs, 'Type', 'Pearson');
if isnan(c(3))
    c(3)=1000;
end
val = 1-sqrt(c(1)^2+c(2)^2+(c(3)-1)^2);
%disp(c(1))
%disp(c(2))
%disp(c(3))
%disp(val)

end