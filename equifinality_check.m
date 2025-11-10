%% Create Benchmark models with Equifinality Analysis
clear
close all

% Add paths for required functions
addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/TOSSH-master/'))
addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/marrmot_211/'))
addpath(genpath('/Users/peterwagener/Desktop/ma_thesis_dump/ma_thesis/'))

% Load additional data (assumes objective_functions, etc. are defined here)
load('additional.mat')
load('signatures.mat')

% Define catchments and signature names
catchments_aridity = {'camelsaus_143110A','camelsbr_60615000','camelsaus_607155',...
    'camels_12381400','camels_02017500','camelsgb_39037','camels_03460000',...
    'hysets_01AF007','camelsgb_27035','camelsgb_8013','lamah_200048'};

signatures_Q = {'sig_FDC_slope','sig_RisingLimbDensity',...
    'sig_BaseflowRecessionK','sig_HFD_mean',...
    'sig_BFI','sig_VariabilityIndex','sig_FlashinessIndex'};
signatures_Q_P = {'sig_EventRR','sig_TotalRR'};
signatures_high_low = {'sig_x_Q_duration','sig_x_Q_frequency'};
signatures_percentage = {'sig_x_percentile'};

current_signatures = [signatures_Q, signatures_Q_P, signatures_high_low, signatures_percentage];

%% Read data from NetCDF files (if needed for further analysis)
path_nc = '/Users/peterwagener/Desktop/ma_thesis_dump/ma_thesis/catchments_new/';
Files = dir(fullfile(path_nc,'*.nc'));

precip = zeros(14609,length(catchments_aridity));
temp = zeros(14609,length(catchments_aridity));
pet = zeros(14609,length(catchments_aridity));
streamflow = zeros(14609,length(catchments_aridity));

for h = 1:length(catchments_aridity)
    ncFile = fullfile(path_nc, [catchments_aridity{h}, '.nc']);
    precip(:,h)   = ncread(ncFile, 'total_precipitation_sum');
    temp(:,h)     = ncread(ncFile, 'temperature_2m_mean');
    pet(:,h)      = ncread(ncFile, 'potential_evaporation_sum');
    streamflow(:,h) = ncread(ncFile, 'streamflow');
end

%% Define run folders and corresponding base directories
% Run "output3" is in the desktop dump, the others are in Downloads.
runFolders = {'output3','output4','output5','output6','output7','output8'};
baseDirs = cell(size(runFolders));
baseDirs{1} = '/Users/peterwagener/Desktop/ma_thesis_dump/output3';
for k = 2:length(runFolders)
    baseDirs{k} = fullfile('/Users/peterwagener/Downloads', runFolders{k});
end

%% Define model list (as in your original script)
model_list = {'m_07_gr4j_4p_2s'};
%m_29_hymod_5p_5s'};
%,'m_07_gr4j_4p_2s','m_01_collie1_1p_1s'};

%% --- Part 1: Load and Store Optimal Parameters for Multiple Runs ---
% The structure is organized as:
%   optimal_parameters_all.(catchment).(objective_function).(model).(runFolder)
% Make sure that the variable "objective_functions" exists (e.g. from additional.mat)

optimal_parameters_all = struct();

for runIdx = 1:length(runFolders)
    currentRun = runFolders{runIdx};
    baseDirectory = baseDirs{runIdx};
    
    for catchmentIdx = 1:length(catchments_aridity)
        catchment = catchments_aridity{catchmentIdx};
        
        for funcIdx = 1:length(objective_functions)
            obj_fun = objective_functions{funcIdx};
            
            for modelIdx = 1:length(model_list)
                model = model_list{modelIdx};
                directory = fullfile(baseDirectory, catchment, obj_fun, model);
                resultsFile = fullfile(directory, 'results.mat');
                
                if exist(resultsFile, 'file')
                    loadedData = load(resultsFile);
                    if isfield(loadedData, 'optimal_parameters')
                        % Store under the run name
                        optimal_parameters_all.(catchment).(obj_fun).(model).(currentRun) = loadedData.optimal_parameters;
                    else
                        warning('Field "optimal_parameters" not found in %s', resultsFile);
                    end
                else
                    warning('results.mat not found in %s', directory);
                end
            end
        end
    end
end

%% --- Part 2: Plot Parameter Variation for Each Model across Equifinal Runs ---
% Get list of catchments and objective functions from the structure
catchments = fieldnames(optimal_parameters_all);
obj_funcs  = fieldnames(optimal_parameters_all.(catchments{1}));
nCatch = length(catchments);
nObj   = length(obj_funcs);
nRuns  = length(runFolders);
nModels = length(model_list);

for m = 1:nModels
    currentModel = model_list{m};
    
    % --- Determine number of parameters automatically for the current model ---
    nParamsModel = [];
    found = false;
    for i = 1:nCatch
        for j = 1:nObj
            if isfield(optimal_parameters_all.(catchments{i}).(obj_funcs{j}), currentModel)
                runStruct = optimal_parameters_all.(catchments{i}).(obj_funcs{j}).(currentModel);
                runNames = fieldnames(runStruct);
                if ~isempty(runNames)
                    temp = runStruct.(runNames{1});
                    if iscell(temp)
                        temp = cell2mat(temp);
                    end
                    nParamsModel = numel(temp);
                    found = true;
                    break;
                end
            end
        end
        if found, break; end
    end
    if isempty(nParamsModel)
        warning('No parameter data found for model %s', currentModel);
        continue;
    end
    
    % --- Aggregate parameter values for the current model ---
    % We will have one line per combination of objective function and run.
    nLines = nObj * nRuns;
    aggregated_data = nan(nCatch, nLines, nParamsModel);
    lineLabels = cell(nLines, 1);
    
    lineIdx = 0;
    for j = 1:nObj
        for r = 1:nRuns
            lineIdx = lineIdx + 1;
            % Create a label for the line (objective_function and run)
            lineLabels{lineIdx} = sprintf('%s_%s', obj_funcs{j}, runFolders{r});
            for i = 1:nCatch
                % Check if data exists for this catchment, objective function, model, and run.
                if isfield(optimal_parameters_all.(catchments{i}).(obj_funcs{j}).(currentModel), runFolders{r})
                    temp = optimal_parameters_all.(catchments{i}).(obj_funcs{j}).(currentModel).(runFolders{r});
                    if iscell(temp)
                        temp = cell2mat(temp);
                    end
                    if numel(temp) ~= nParamsModel
                        warning('Parameter vector length mismatch for catchment %s, objective %s, model %s, run %s', ...
                            catchments{i}, obj_funcs{j}, currentModel, runFolders{r});
                    end
                    aggregated_data(i, lineIdx, :) = temp;
                end
            end
        end
    end
    
    % --- Create a tiled layout for subplots for each parameter ---
    nRows = ceil(sqrt(nParamsModel));
    nCols = ceil(nParamsModel / nRows);
    
    figure;
    t = tiledlayout(nRows, nCols, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    title(t, sprintf('Parameter Variation for Model: %s', currentModel), 'FontSize', 12);
    
    for p = 1:nParamsModel
        ax = nexttile;
        hold(ax, 'on');
        % Plot one line per combination (objective function and run)
        for l = 1:nLines
            ydata = squeeze(aggregated_data(:, l, p));
            plot(ax, 1:nCatch, ydata, '-o', 'LineWidth', 2);
        end
        set(ax, 'XTick', 1:nCatch, 'XTickLabel', catchments, 'XTickLabelRotation', 45);
        xlabel(ax, 'Catchment');
        ylabel(ax, sprintf('Parameter %d Value', p));
        title(ax, sprintf('Variation of Parameter %d', p));
        hold(ax, 'off');
    end
    
    % --- Add a common legend using one of the subplot axes (ax) ---
    legend(ax, lineLabels, 'Orientation', 'horizontal', 'Location', 'southoutside');
end


%% Get benchmark values for all models

% --- Calculate and Store Signatures for Each Run ---

runFolders = {'output3','output4','output5','output6','output7','output8'};
baseDirs = cell(size(runFolders));
baseDirs{1} = '/Users/peterwagener/Desktop/ma_thesis_dump/output3';
for k = 2:length(runFolders)
    baseDirs{k} = fullfile('/Users/peterwagener/Downloads', runFolders{k});
end
start_year = 1981;
end_year = 2020;
date_array = datetime(start_year, 1, 2):datetime(end_year, 12, 31);

catchments = catchments_aridity; % from your script
nCatch = length(catchments);
model_list = {'m_07_gr4j_4p_2s'};
obj_funcs  = objective_functions; % from your loaded data

% Initialize signature storage structures
sim_signatures_cali_runs = struct();
sim_signatures_eval_runs = struct();

for runIdx = 1:length(runFolders)
    currentRun = runFolders{runIdx};
    baseDirectory = baseDirs{runIdx};

    for catchmentIdx = 1:nCatch
        catchment = catchments{catchmentIdx};

        % Define calibration and evaluation periods
        if catchmentIdx == 3
            start_date = datetime(1990, 1, 1);
            end_date = datetime(1999, 12, 31);
            mask_cali_idx = (date_array >= start_date) & (date_array <= end_date);
            cal_idx = find(mask_cali_idx);

            start_date = datetime(1982, 1, 1);
            end_date = datetime(1988, 12, 31);
            mask_eval_idx = (date_array >= start_date) & (date_array <= end_date);
            eval_idx = find(mask_eval_idx);
        else
            start_date = datetime(2005, 1, 1);
            end_date = datetime(2014, 12, 31);
            mask_cali_idx = (date_array >= start_date) & (date_array <= end_date);
            cal_idx = find(mask_cali_idx);

            start_date = datetime(1994, 1, 1);
            end_date = datetime(2003, 12, 31);
            mask_eval_idx = (date_array >= start_date) & (date_array <= end_date);
            eval_idx = find(mask_eval_idx);
        end

        for funcIdx = 1:length(obj_funcs)
            obj_fun = obj_funcs{funcIdx};

            for modelIdx = 1:length(model_list)
                model = model_list{modelIdx};

                directory = fullfile(baseDirectory, catchment, obj_fun, model);
                resultsFile = fullfile(directory, 'results.mat');

                try
                    if exist(resultsFile, 'file')
                        loadedData = load(resultsFile);
                    else
                        error('Results file missing');
                    end

                    % Get simulated runoff
                    runoff_sim_eval = loadedData.runoff_sim_eval{1,1};
                    if catchmentIdx == 3
                        runoff_sim = loadedData.runoff_sim{1,1};
                    else
                        runoff_sim = loadedData.runoff_sim{1,1}(2:3653);
                    end

                    % Clean negative values
                    runoff_sim_eval(runoff_sim_eval < 0) = 0;
                    runoff_sim(runoff_sim < 0) = 0;

                    % ----------- CALCULATE SIGNATURES ------------
                    for p = 1:length(current_signatures)
                        signature = current_signatures{p};

                        % Calibration signatures
                        if any(strcmp(signatures_Q,signature))
                            sim_signatures_cali_runs.(signature).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim, cal_idx);

                        elseif any(strcmp(signatures_Q_P,signature))
                            sim_signatures_cali_runs.(signature).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim, cal_idx, precip(cal_idx,catchmentIdx));

                        elseif any(strcmp(signatures_high_low,signature))
                            % high
                            dummy = strcat(signature, '_high');
                            sim_signatures_cali_runs.(dummy).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim, cal_idx, 'high');
                            % low
                            dummy2 = strcat(signature, '_low');
                            sim_signatures_cali_runs.(dummy2).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim, cal_idx, 'low');

                        else % percentage signatures
                            dummy = strcat(signature, '_5per');
                            sim_signatures_cali_runs.(dummy).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim, cal_idx, 5);
                            dummy2 = strcat(signature, '_95per');
                            sim_signatures_cali_runs.(dummy2).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim, cal_idx, 95);
                        end

                        % Evaluation signatures
                        if any(strcmp(signatures_Q,signature))
                            sim_signatures_eval_runs.(signature).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim_eval, eval_idx);

                        elseif any(strcmp(signatures_Q_P,signature))
                            sim_signatures_eval_runs.(signature).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim_eval, eval_idx, precip(eval_idx,catchmentIdx));

                        elseif any(strcmp(signatures_high_low,signature))
                            dummy = strcat(signature, '_high');
                            sim_signatures_eval_runs.(dummy).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim_eval, eval_idx, 'high');
                            dummy2 = strcat(signature, '_low');
                            sim_signatures_eval_runs.(dummy2).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim_eval, eval_idx, 'low');

                        else % percentage signatures
                            dummy = strcat(signature, '_5per');
                            sim_signatures_eval_runs.(dummy).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim_eval, eval_idx, 5);
                            dummy2 = strcat(signature, '_95per');
                            sim_signatures_eval_runs.(dummy2).(catchment).(model).(obj_fun).(currentRun) = ...
                                feval(signature, runoff_sim_eval, eval_idx, 95);
                        end
                    end % end signature loop

                catch ME
                    fprintf('Signature calculation failed for %s, %s, %s, %s: %s\n', ...
                        catchment, obj_fun, model, currentRun, ME.message);
                end
            end
        end
    end
end

% Save results for later plotting
save signatures_runs.mat sim_signatures_cali_runs sim_signatures_eval_runs

%% Plotting against observed values

% Example: Plot simulated signature (all runs) vs. observed, for a signature/catchment/model/OF
load('signatures_runs.mat')
load('signatures.mat') % for observed signatures (structure names may need adjusting)

signature = 'sig_TotalRR';
catchment = 'camels_02017500';
model = 'm_07_gr4j_4p_2s';
obj_fun = 'of_KGE';

runs = {'output3','output4','output5','output6','output7','output8'};

sim_vals = zeros(1, length(runs));
for k = 1:length(runs)
    if isfield(sim_signatures_cali_runs.(signature).(catchment).(model).(obj_fun), runs{k})
        sim_vals(k) = sim_signatures_cali_runs.(signature).(catchment).(model).(obj_fun).(runs{k});
    else
        sim_vals(k) = NaN;
    end
end

% Get observed value (adjust struct as needed for your obs_signatures)
obs_val = obs_signatures_cali_bench.(signature).(catchment);

figure;
bar(1:length(runs), sim_vals, 'FaceColor', [0.2 0.4 0.8]);
hold on
yline(obs_val, 'r-', 'LineWidth', 2, 'DisplayName','Observed');
xticks(1:length(runs)); xticklabels(runs);
ylabel(signature); title([signature ' - ' catchment]);
legend({'Simulated (each run)','Observed'});
hold off

%% Automatised for all catchments and objective functions

% Adjust these lists if you add/remove signatures:
% Build list of all signatures (including high, low, 5per, 95per)
all_signatures = [ ...
    signatures_Q, ...
    signatures_Q_P, ...
    strcat(signatures_high_low, '_high'), ...
    strcat(signatures_high_low, '_low'), ...
    strcat(signatures_percentage, '_5per'), ...
    strcat(signatures_percentage, '_95per') ...
    ];
signature_labels = strrep(all_signatures, '_', '\_');signature_labels = strrep(all_signatures, '_', '\_'); % for nicer subplot titles

catchments = fieldnames(sim_signatures_cali_runs.(all_signatures{1}));
runs = {'output3','output4','output5','output6','output7','output8'};
model = 'm_07_gr4j_4p_2s'; % or loop over models if you want
obj_fun = 'of_log_KGE';        % or loop over OFs if you want

for c = 1:length(catchments)
    catchment = catchments{c};
    
    figure('Position', [100, 100, 1600, 900]);
    tiledlayout(3, 5, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    for s = 1:length(all_signatures)
        signature = all_signatures{s};
        
        sim_vals = nan(1, length(runs));
        for k = 1:length(runs)
            try
                % Some signatures have '_high', '_low', '_5per', etc. in the structure names!
                if isfield(sim_signatures_cali_runs, signature) && ...
                   isfield(sim_signatures_cali_runs.(signature), catchment) && ...
                   isfield(sim_signatures_cali_runs.(signature).(catchment), model) && ...
                   isfield(sim_signatures_cali_runs.(signature).(catchment).(model), obj_fun) && ...
                   isfield(sim_signatures_cali_runs.(signature).(catchment).(model).(obj_fun), runs{k})
                    sim_vals(k) = sim_signatures_cali_runs.(signature).(catchment).(model).(obj_fun).(runs{k});
                end
            catch
                sim_vals(k) = NaN;
            end
        end

        % Observed value: try-catch in case it's missing (for new signature types)
        try
            obs_val = obs_signatures_cali_bench.(signature).(catchment);
        catch
            obs_val = NaN;
        end

        nexttile;
        bar(1:length(runs), sim_vals, 'FaceColor', [0.2 0.4 0.8]);
        hold on;
        yline(obs_val, 'r-', 'LineWidth', 2, 'DisplayName','Observed');
        hold off;
        xticks(1:length(runs)); xticklabels(runs);
        ylabel('');
        title(signature_labels{s}, 'Interpreter', 'tex', 'FontSize', 10);
        if s == 1
            legend({'Simulated (each run)','Observed'},'Location','north');
        end
        set(gca, 'FontSize', 9);
    end

    sgtitle(sprintf('Signatures (Calibration) for %s and log KGE', strrep(catchment, '_', '\_')), 'FontSize', 16);
    
    % Optional: save automatically
    saveas(gcf, sprintf('signatures_bars_%s.png', catchment));
    % or use exportgraphics(gcf, sprintf(...), 'Resolution',300);

    % Or: pause/display for manual browsing, then close
    % pause; close(gcf);
end

%% SETTINGS & DEFINITIONS

% Define objective functions and their plotting order/labels.
% (Make sure these names match the functions you use for performance calculation.)
%objective_functions = {'KGE','NSE','log KGE','log NSE','DE','KGE-NP','KGE Split','SHE'};
OF_Plot = {'KGE','NSE','log KGE','log NSE','DE','KGE-NP','KGE Split','SHE'};

% Define colors (one row per objective function in the order of OF_Plot)
colors = NaN(8,3);
colors(1,:) = [0.1216, 0.4667, 0.7059];
colors(2,:) = [1.0000, 0.4980, 0.0549];
colors(3,:) = [0.1725, 0.6275, 0.1725];
colors(4,:) = [0.8392, 0.1529, 0.1569];
colors(5,:) = [0.5804, 0.4039, 0.7412];
colors(6,:) = [0.8902, 0.4667, 0.7608];
colors(7,:) = [0.7373, 0.7412, 0.1333];
colors(8,:) = [0.0902, 0.7451, 0.8118];

% Define catchments and models (adjust as needed)
catchments_aridity = {'camelsaus_143110A','camelsbr_60615000','camelsaus_607155',...
    'camels_12381400','camels_02017500','camelsgb_39037','camels_03460000',...
    'hysets_01AF007','camelsgb_27035','camelsgb_8013','lamah_200048'};

model_list = {'m_07_gr4j_4p_2s'};
    %'m_29_hymod_5p_5s','m_07_gr4j_4p_2s','m_01_collie1_1p_1s'};

% Define run folders and their corresponding base directories.
runFolders = {'output3','output4','output5','output6','output7','output8'};
baseDirs = cell(size(runFolders));
baseDirs{1} = '/Users/peterwagener/Desktop/ma_thesis_dump/output3';
for k = 2:length(runFolders)
    baseDirs{k} = fullfile('/Users/peterwagener/Downloads', runFolders{k});
end

% Generate datetime array from 1981-01-02 to 2020-12-31 (used for indexing)
start_year = 1981;
end_year = 2020;
date_array = datetime(start_year, 1, 2):datetime(end_year, 12, 31);

% Pre-allocate cell arrays to store simulated runoff (calibration and evaluation)
numCatchments = numel(catchments_aridity);
numFunctions = numel(objective_functions);
numModels = numel(model_list);

Q_sim_cali = cell(numCatchments, numFunctions, numModels);
Q_sim_vali = cell(numCatchments, numFunctions, numModels);

% Pre-allocate structures to store performance values (OF)
OF_value_cali = struct();
OF_value_vali = struct();

%% CALCULATE PERFORMANCE (OF) FOR EACH MODEL, CATCHMENT, OBJECTIVE FUNCTION, AND RUN

for catchmentIdx = 1:numCatchments
    catchment = catchments_aridity{catchmentIdx};
    
    % Define calibration and evaluation periods (note different date ranges for catchment 3)
    if catchmentIdx == 3   % adjust if new data become available
        % For catchment 3:
        start_date = datetime(1990, 1, 1);
        end_date = datetime(1999, 12, 31);
        mask_cali_idx = (date_array >= start_date) & (date_array <= end_date);
        cal_idx = find(mask_cali_idx);

        start_date = datetime(1982, 1, 1);
        end_date = datetime(1988, 12, 31);
        mask_eval_idx = (date_array >= start_date) & (date_array <= end_date);
        eval_idx = find(mask_eval_idx);
    else
        % For all other catchments:
        start_date = datetime(2005, 1, 1);
        end_date = datetime(2014, 12, 31);
        mask_cali_idx = (date_array >= start_date) & (date_array <= end_date);
        cal_idx = find(mask_cali_idx);

        start_date = datetime(1994, 1, 1);
        end_date = datetime(2003, 12, 31);
        mask_eval_idx = (date_array >= start_date) & (date_array <= end_date);
        eval_idx = find(mask_eval_idx);
    end
    
    for funcIdx = 1:numFunctions
        obj_fun = objective_functions{funcIdx};
        
        for modelIdx = 1:numModels
            model = model_list{modelIdx};
            
            % Loop over run folders
            for runIdx = 1:length(runFolders)
                currentRun = runFolders{runIdx};
                baseDirectoryRun = baseDirs{runIdx};
                directory = fullfile(baseDirectoryRun, catchment, obj_fun, model);
                %print(directory)
                resultsFile = fullfile(directory, 'results.mat');
                
                % Try loading results.mat
                try
                    if exist(resultsFile, 'file')
                        loadedData = load(resultsFile);
                    else
                        error('File does not exist');
                    end
                    
                    % Get simulated runoff and streamflow (adjust indices for catchment 3)
                    runoff_sim_eval = loadedData.runoff_sim_eval{1,1};
                    if catchmentIdx == 3
                        runoff_sim = loadedData.runoff_sim{1,1};
                    else
                        runoff_sim = loadedData.runoff_sim{1,1}(2:3653);
                    end
                    streamflow_single = loadedData.streamflow;
                    
                    % Remove negative values
                    runoff_sim_eval(runoff_sim_eval < 0) = 0;
                    runoff_sim(runoff_sim < 0) = 0;
                    streamflow_single(streamflow_single < 0) = 0;
                    
                    % Save runoff (if needed for later use)
                    Q_sim_cali{catchmentIdx, funcIdx, modelIdx} = runoff_sim;
                    Q_sim_vali{catchmentIdx, funcIdx, modelIdx} = runoff_sim_eval;
                    %print('test')
                    % Calculate the performance value (OF) using feval.
                    % For "of_SHE" also pass precipitation.
                    if strcmp(obj_fun, "of_SHE")
                        perf_cali = feval(obj_fun, streamflow_single(cal_idx), runoff_sim, precip(cal_idx, catchmentIdx));
                        perf_vali = feval(obj_fun, streamflow_single(eval_idx), runoff_sim_eval, precip(eval_idx, catchmentIdx));
                    else
                        perf_cali = feval(obj_fun, streamflow_single(cal_idx), runoff_sim);
                        perf_vali = feval(obj_fun, streamflow_single(eval_idx), runoff_sim_eval);
                    end
                    
                    % Store performance values into a nested structure.
                    OF_value_cali.(catchment).(obj_fun).(model).(currentRun) = perf_cali;
                    OF_value_vali.(catchment).(obj_fun).(model).(currentRun) = perf_vali;
                    
                catch ME
                    fprintf('Error for catchment %s, OF %s, model %s, run %s:\n%s\n', ...
                        catchment, obj_fun, model, currentRun, ME.message);
                end
                
            end  % end runFolders loop
        end  % end model loop
    end  % end objective functions loop
end  % end catchment loop



%% Compare performance to benchmark

%% --- ASSUMPTIONS & PRELIMINARIES ---
% Assume the following variables are defined:
%   - catchments_aridity: cell array of catchment names (e.g., 11 catchments)
%   - objective_functions: cell array of objective function names (8 items)
%   - OF_Plot: cell array with the desired order/labels for plotting (same as objective_functions)
%   - model_list: cell array of model names
%   - OF_value_cali: a nested structure with calibration performance values
%       e.g., OF_value_cali.(catchment).(obj_fun).(model).(runFolder)
%   - benchmarkMatrix: an 8x11 matrix with benchmarks. Row i corresponds to OF_Plot{i},
%       column j corresponds to catchments_aridity{j}.
%
% Also, define a colors matrix (one row per objective function):
colors = NaN(8,3);
colors(1,:) = [0.1216, 0.4667, 0.7059];
colors(2,:) = [1.0000, 0.4980, 0.0549];
colors(3,:) = [0.1725, 0.6275, 0.1725];
colors(4,:) = [0.8392, 0.1529, 0.1569];
colors(5,:) = [0.5804, 0.4039, 0.7412];
colors(6,:) = [0.8902, 0.4667, 0.7608];
colors(7,:) = [0.7373, 0.7412, 0.1333];
colors(8,:) = [0.0902, 0.7451, 0.8118];

%% --- COMPARE PERFORMANCE TO BENCHMARK ---
% Create a structure to store the comparison (higher or lower)
OF_comparison = struct();

numCatchments = numel(catchments_aridity);
numFunctions = numel(objective_functions);
runFolders = {'output3','output4','output5','output6','output7','output8'};  % assumed defined earlier

% Loop over catchments, objective functions, and runs for a selected model
% (Here, we illustrate for a chosen model; you can extend to all models as needed.)
selectedModel = model_list{1};

for catchmentIdx = 1:numCatchments
    catchment = catchments_aridity{catchmentIdx};
    for funcIdx = 1:numFunctions
        obj_fun = objective_functions{funcIdx};
        % Get the benchmark value for this objective function and catchment
        benchmark_val = threshold(funcIdx, catchmentIdx);
        for runIdx = 1:length(runFolders)
            currentRun = runFolders{runIdx};
            % Check if performance exists for this combination
            if isfield(OF_value_cali.(catchment).(obj_fun).(selectedModel), currentRun)
                perf = OF_value_cali.(catchment).(obj_fun).(selectedModel).(currentRun);
                if perf > benchmark_val
                    OF_comparison.(catchment).(obj_fun).(selectedModel).(currentRun) = 'higher';
                elseif perf < benchmark_val
                    OF_comparison.(catchment).(obj_fun).(selectedModel).(currentRun) = 'lower';
                else
                    OF_comparison.(catchment).(obj_fun).(selectedModel).(currentRun) = 'equal';
                end
            end
        end
    end
end

%% --- SETTINGS & PRELIMINARIES ---
% Assume the following are defined in your workspace:
%   - optimal_parameters_all: structure with optimal parameter values.
%   - catchments_aridity: cell array of catchment names (11 catchments)
%   - objective_functions and OF_Plot: cell arrays of objective function names (8 items)
%   - model_list: cell array of model names.
%   - runFolders: cell array of run names (e.g., {'output3','output4',...})
%   - benchmarkMatrix: 8x11 matrix; row i corresponds to OF_Plot{i} and column j to catchments_aridity{j}.
%
% Also define colors for each objective function (row order same as OF_Plot):
colors = NaN(8,3);
colors(1,:) = [0.1216, 0.4667, 0.7059];
colors(2,:) = [1.0000, 0.4980, 0.0549];
colors(3,:) = [0.1725, 0.6275, 0.1725];
colors(4,:) = [0.8392, 0.1529, 0.1569];
colors(5,:) = [0.5804, 0.4039, 0.7412];
colors(6,:) = [0.8902, 0.4667, 0.7608];
colors(7,:) = [0.7373, 0.7412, 0.1333];
colors(8,:) = [0.0902, 0.7451, 0.8118];

% For this example, we choose one model to plot.
selectedModel = model_list{1};

%% --- SETTINGS & PRELIMINARIES ---
% Assume the following are defined in your workspace:
%   - optimal_parameters_all: structure with optimal parameter values.
%   - catchments_aridity: cell array of catchment names (11 catchments)
%   - objective_functions and OF_Plot: cell arrays of objective function names (8 items)
%   - model_list: cell array of model names.
%   - runFolders: cell array of run names (e.g., {'output3','output4',...})
%   - benchmarkMatrix: 8x11 matrix; row i corresponds to OF_Plot{i} and column j to catchments_aridity{j}.
%
% Also define colors for each objective function (row order same as OF_Plot):
colors = NaN(8,3);
colors(1,:) = [0.1216, 0.4667, 0.7059];
colors(2,:) = [1.0000, 0.4980, 0.0549];
colors(3,:) = [0.1725, 0.6275, 0.1725];
colors(4,:) = [0.8392, 0.1529, 0.1569];
colors(5,:) = [0.5804, 0.4039, 0.7412];
colors(6,:) = [0.8902, 0.4667, 0.7608];
colors(7,:) = [0.7373, 0.7412, 0.1333];
colors(8,:) = [0.0902, 0.7451, 0.8118];

% For this example, we choose one model to plot.
selectedModel = model_list{1};

%% --- PLOT PARAMETER VARIATION PER CATCHMENT WITH BENCHMARK HIGHLIGHTING ---
% For each catchment, create a figure with one subplot per parameter.
% In each subplot, the x-axis is the objective functions (ordered by OF_Plot).
% For each objective function, for each run, the parameter value is plotted.
% A patch behind the marker is drawn: light green if the parameter value is higher than the 
% benchmark (for that objective function and catchment) and light red if lower.

catchmentNames = fieldnames(optimal_parameters_all);
numCatchments = numel(catchmentNames);
numOF = numel(OF_Plot);  % should be 8

for c = 1:numCatchments
    currentCatchment = catchmentNames{c};
    
    % Determine number of parameters for this catchment and selectedModel.
    % We search through the objective functions until we find one with data.
    nParams = [];
    for j = 1:numOF
        if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}), selectedModel)
            runStruct = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel);
            runNames = fieldnames(runStruct);
            if ~isempty(runNames)
                temp = runStruct.(runNames{1});
                if iscell(temp)
                    temp = cell2mat(temp);
                end
                nParams = numel(temp);
                break;
            end
        end
    end
    if isempty(nParams)
        warning('No parameter data found for catchment %s', currentCatchment);
        continue;
    end
    
    % Create figure with a tiled layout: one subplot per parameter.
    nRows = ceil(sqrt(nParams));
    nCols = ceil(nParams / nRows);
    fig = figure;
    t = tiledlayout(nRows, nCols, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    title(t, sprintf('Parameter Variation for Catchment: %s (Model: %s)', currentCatchment, selectedModel), 'FontSize', 12);
    
    % Loop over parameters (each subplot corresponds to one parameter index)
    for p = 1:nParams
        ax = nexttile;
        hold(ax, 'on');
        
        % For each objective function (x-axis position)
        for j = 1:numOF
            % The corresponding color for this objective function.
            currentColor = colors(j,:);
            % Get the benchmark value for this objective function and catchment.
            benchmark_val = threshold(j, c);
            
            % Loop over each run to plot the p-th parameter value.
            for r = 1:length(runFolders)
                currentRun = runFolders{r};
                % Check if data exists for this catchment, objective function, model, and run.
                if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel), currentRun)
                    temp = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel).(currentRun);
                    if iscell(temp)
                        temp = cell2mat(temp);
                    end
                    if numel(temp) < p
                        warning('Parameter vector too short for catchment %s, objective %s, run %s', ...
                            currentCatchment, OF_Plot{j}, currentRun);
                        continue;
                    end
                    param_value = temp(p);
                    
                    % Define a small horizontal offset for this run.
                    offset = (r - (length(runFolders)+1)/2)*0.1;
                    x_center = j + offset;
                    
                    % Draw a patch behind the marker to indicate benchmark comparison.
                    % If parameter is higher than benchmark -> light green, else light red.
                    if param_value >= benchmark_val
                        patch_color = [0.8, 1, 0.8];  % light green
                    else
                        patch_color = [1, 0.8, 0.8];  % light red
                    end
                    patch_width = 0.12;
                    x_patch = [x_center - patch_width/2, x_center + patch_width/2, ...
                               x_center + patch_width/2, x_center - patch_width/2];
                    % Create patch from benchmark to parameter value.
                    if param_value >= benchmark_val
                        y_patch = [benchmark_val, benchmark_val, param_value, param_value];
                    else
                        y_patch = [param_value, param_value, benchmark_val, benchmark_val];
                    end
                    patch(ax, x_patch, y_patch, patch_color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
                    
                    % Plot the marker for the parameter value on top.
                    plot(ax, x_center, param_value, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', currentColor);
                end
            end
        end
        
        % Format subplot: set x-axis to the objective functions.
        set(ax, 'XTick', 1:numOF, 'XTickLabel', OF_Plot, 'XTickLabelRotation', 45);
        xlabel(ax, 'Objective Function');
        ylabel(ax, sprintf('Parameter %d Value', p));
        title(ax, sprintf('Parameter %d', p));
        hold(ax, 'off');
    end
    
    % (Optional) Create a custom legend for the objective function colors.
    %h = gobjects(numOF,1);
    for j = 1:numOF
    %    h(j) = plot(nan, nan, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', colors(j,:));
    end
    %legend(ax, h, OF_Plot, 'Orientation', 'horizontal', 'Location', 'southoutside');
end

%% Define colors and objective function order
colors = NaN(8,3);
colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];

OF_Plot = {'KGE','NSE','log KGE','log NSE','DE','KGE-NP','KGE Split','SHE'};

%% Selected model for plotting
currentModel = model_list{1};

%% Adjust this fraction as needed (here 5% of the range)
patchFraction = 20;  

% Loop through each catchment
catchmentNames = fieldnames(optimal_parameters_all);
for c = 1:numel(catchmentNames)
    currentCatchment = catchmentNames{c};
    
    % Determine the number of parameters for this catchment and selectedModel.
    nParams = [];
    for j = 1:numel(OF_Plot)
        if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}), selectedModel)
            runStruct = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel);
            runNames = fieldnames(runStruct);
            if ~isempty(runNames)
                temp = runStruct.(runNames{1});
                if iscell(temp)
                    temp = cell2mat(temp);
                end
                nParams = numel(temp);
                break;
            end
        end
    end
    if isempty(nParams)
        warning('No parameter data found for catchment %s', currentCatchment);
        continue;
    end
    
    % Create a figure with a tiled layout (one subplot per parameter)
    nRows = ceil(sqrt(nParams));
    nCols = ceil(nParams / nRows);
    fig = figure;
    t = tiledlayout(nRows, nCols, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    title(t, sprintf('Parameter Variation for Catchment: %s (Model: %s)', currentCatchment, currentModel), 'FontSize', 12);
    
    % Loop over each parameter (each subplot corresponds to one parameter index)
    for p = 1:nParams
        ax = nexttile;
        hold(ax, 'on');
        
        % --- Determine dynamic patch height for parameter p ---
        allValues = [];
        for j = 1:numel(OF_Plot)
            for r = 1:length(runFolders)
                currentRun = runFolders{r};
                if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel), currentRun)
                    paramData = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel).(currentRun);
                    if iscell(paramData)
                        paramData = cell2mat(paramData);
                    end
                    if numel(paramData) >= p
                        allValues = [allValues; paramData(p)];
                    end
                end
            end
        end
        if isempty(allValues)
            dynPatchHeight = 5;  % fallback value
        else
            rangeVal = max(allValues) - min(allValues);
            if rangeVal == 0
                dynPatchHeight = 0.05;  % minimal height if no variation
            else
                dynPatchHeight = patchFraction * rangeVal;
            end
        end
        
        % Loop over each objective function (x-axis positions 1:numel(OF_Plot))
        for j = 1:numel(OF_Plot)
            currentColor = colors(j,:);
            % Retrieve benchmark value for this objective and catchment.
            benchmark_val = threshold(j, c);  % Ensure 'threshold' holds your benchmark matrix.
            
            % Loop over each run (with horizontal offset to avoid overlapping markers)
            for r = 1:length(runFolders)
                currentRun = runFolders{r};
                if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel), currentRun)
                    paramData = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel).(currentRun);
                    if iscell(paramData)
                        paramData = cell2mat(paramData);
                    end
                    if numel(paramData) < p
                        warning('Parameter vector too short for catchment %s, objective %s, run %s', ...
                            currentCatchment, OF_Plot{j}, currentRun);
                        continue;
                    end
                    param_value = paramData(p);
                    
                    % Retrieve the performance value for the same catchment/objective/model/run.
                    if isfield(OF_value_cali.(currentCatchment).(objective_functions{j}).(selectedModel), currentRun)
                        perf = OF_value_cali.(currentCatchment).(objective_functions{j}).(selectedModel).(currentRun);
                    else
                        continue;  % Skip if performance is not available.
                    end
                    
                    % Determine patch color based on performance vs. benchmark.
                    if perf >= benchmark_val
                        patch_color = [0.8, 1, 0.8];  % light green if performance is above or equal
                    else
                        patch_color = [1, 0.8, 0.8];  % light red if performance is below
                    end
                    
                    % Define a small horizontal offset for this run.
                    offset = (r - (length(runFolders)+1)/2) * 0.1;
                    x_center = j + offset;
                    
                    % Draw a patch behind the marker.
                    patch_width = 0.12;
                    x_patch = [x_center - patch_width/2, x_center + patch_width/2, ...
                               x_center + patch_width/2, x_center - patch_width/2];
                    y_patch = [param_value - dynPatchHeight/2, param_value - dynPatchHeight/2, ...
                               param_value + dynPatchHeight/2, param_value + dynPatchHeight/2];
                    patch(ax, x_patch, y_patch, patch_color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
                    
                    % Plot the parameter marker on top.
                    plot(ax, x_center, param_value, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', currentColor);
                end
            end
        end
        
        % For GR4J, limit y-axis ranges for the first 4 parameters using specified bounds.
        if strcmp(currentModel, 'm_07_gr4j_4p_2s')
            parRanges = [1, 2000;    % Parameter 1 [mm]
                         -20, 20;    % Parameter 2 [mm/d]
                          1, 300;    % Parameter 3 [mm]
                         0.5, 15];   % Parameter 4 [d]
            if p <= size(parRanges,1)
                ylim(ax, parRanges(p, :));
            end
        end
        
        % Format the subplot: set x-axis tick positions and labels.
        set(ax, 'XTick', 1:numel(OF_Plot), 'XTickLabel', OF_Plot, 'XTickLabelRotation', 45);
        xlabel(ax, 'Objective Function');
        ylabel(ax, sprintf('Parameter %d Value', p));
        title(ax, sprintf('Parameter %d', p));
        hold(ax, 'off');
    end
    
    % Create a common legend using an invisible axes so as not to disturb the tiled layout.
    %lgdAx = axes('Position', [0 0 1 1], 'Visible', 'off');
    %h = gobjects(numel(OF_Plot),1);
    for j = 1:numel(OF_Plot)
    %    h(j) = plot(lgdAx, nan, nan, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', colors(j,:));
    end
    %legend(lgdAx, h, OF_Plot, 'Orientation', 'horizontal', 'Location', 'southoutside');
    print(fig, sprintf('Catchment_%s.jpeg', currentCatchment), '-djpeg', '-r300');

end

%% ADDED 
% =======================================================================
%  PARAMETER–VS–OBJECTIVE PANELS  —  VALUE-BASED GREY  (BEST = BLACK)
% =======================================================================

% -------------- user knobs ----------------------------------------------
selectedModel = model_list{1};     % e.g. 'm_07_gr4j_4p_2s'
patchFraction = 20;                % stripe height as % of parameter range
nCmap         = 256;               % greyscale resolution

% GREY RAMP: white  →  black  (row nCmap = black)
perfCmap = flipud(gray(nCmap));

% 1)  ➤  SHORT CATCHMENT NAMES  ------------------------------------------
% Put your abbreviations **in the SAME ORDER** as catchments_aridity.
catchments_labels = {'AUS1','BR6','AUS6','C12','C02',...
                     'GB3','C03','HYS','GB2','GB8','LAM'};   % 10 labels

% Create a mapping:  longID  ➜  shortID
labelMap = containers.Map(catchments_aridity, catchments_labels);

% 2)  ➤  HUMAN-FRIENDLY MODEL LABEL  -------------------------------------
modelLabel = 'GR4J';

catchmentNames = fieldnames(optimal_parameters_all);
numCatchments  = numel(catchmentNames);
numOF          = numel(OF_Plot);

for c = 1:numCatchments
    ct = catchmentNames{c};
    label_c = catchments_labels{c};

    % --------- how many parameters in this catchment --------------------
    nParams = [];
    for j = 1:numOF
        fld = optimal_parameters_all.(ct).(objective_functions{j});
        if isfield(fld, selectedModel)
            fn = fieldnames(fld.(selectedModel));
            if ~isempty(fn)
                pv = fld.(selectedModel).(fn{1});
                if iscell(pv), pv = cell2mat(pv); end
                nParams = numel(pv);  break
            end
        end
    end
    if isempty(nParams), warning('No parameters for %s – skipped',ct); continue; end

    % -------------------- figure & layout -------------------------------
    nRows = ceil(sqrt(nParams));  nCols = ceil(nParams/nRows);
    fig = figure('Color','w', ...                       % white background
             'Units','normalized', ...              % use screen fractions
             'OuterPosition',[0.02 0.05 0.96 0.90]);% [left bottom width height]
    t     = tiledlayout(nRows,nCols,'TileSpacing','compact','Padding','compact');
    title(t,sprintf('Parameter variation – %s  (%s)',label_c,'GR4J'),...
             'FontWeight','bold');

    % -------------------- parameter panels ------------------------------
    for p = 1:nParams
        ax = nexttile;  hold(ax,'on');

        % stripe height
        allVals = [];
        for j = 1:numOF
            for r = 1:numel(runFolders)
                rn = runFolders{r};
                fld = optimal_parameters_all.(ct).(objective_functions{j}).(selectedModel);
                if isfield(fld,rn)
                    v = fld.(rn); if iscell(v), v = cell2mat(v); end
                    if numel(v)>=p, allVals(end+1)=v(p); end
                end
            end
        end
        dynH = max(0.05, patchFraction*range(allVals));

        % ------------ stripes (objective functions) ---------------------
        for j = 1:numOF
            % collect performances for this stripe
            perfRuns = nan(1,numel(runFolders));
            for r = 1:numel(runFolders)
                rn = runFolders{r};
                if isfield(OF_value_cali.(ct).(objective_functions{j}).(selectedModel), rn)
                    perfRuns(r) = OF_value_cali.(ct).(objective_functions{j}).(selectedModel).(rn);
                end
            end
            pMin = min(perfRuns,[],'omitnan');
            pMax = max(perfRuns,[],'omitnan');
            rng  = pMax - pMin;

            % plot each run ------------------------------------------------
            for r = 1:numel(runFolders)
                rn = runFolders{r};
                op = optimal_parameters_all.(ct).(objective_functions{j}).(selectedModel);
                if ~isfield(op,rn), continue, end
                v = op.(rn); if iscell(v), v = cell2mat(v); end
                if numel(v)<p, continue, end
                paramVal = v(p);

                if ~isfield(OF_value_cali.(ct).(objective_functions{j}).(selectedModel), rn)
                    continue
                end
                perf = OF_value_cali.(ct).(objective_functions{j}).(selectedModel).(rn);

                % value-based grey (relative to min/max in this stripe)
                if isnan(rng) || rng==0
                    patchCol = [0.7 0.7 0.7];              % all equal → mid-grey
                else
                    pStar = (perf - pMin) / rng;           % 0 … 1
                    idx   = max(1, min(nCmap, round(pStar*(nCmap-1))+1));
                    patchCol = perfCmap(idx,:);           % darkest = best
                end

                xC = j + (r-(numel(runFolders)+1)/2)*0.1;

                % background stripe
                patch(ax,[xC-0.06 xC+0.06 xC+0.06 xC-0.06], ...
                         [paramVal-dynH/2 paramVal-dynH/2 paramVal+dynH/2 paramVal+dynH/2], ...
                         patchCol,'EdgeColor','none','FaceAlpha',0.9);

                % marker
                plot(ax,xC,paramVal,'o', ...
                     'MarkerFaceColor','w', 'MarkerEdgeColor', colors(j,:), ...
                     'MarkerSize',8,'LineWidth',1.3);
            end
        end

        % optional GR4J y-limits
        if strcmp(selectedModel,'m_07_gr4j_4p_2s')
            lim = [1 2000; -20 20; 1 300; 0.5 15];
            if p<=size(lim,1), ylim(ax,lim(p,:)); end
        end

        set(ax,'XTick',1:numOF,'XTickLabel',OF_Plot,'XTickLabelRotation',45);
        xlabel(ax,'Objective Function');  ylabel(ax,sprintf('Parameter %d',p));
        box(ax,'on'); grid(ax,'on');
    end

    % ---------------- colour-bar ----------------------------------------
    colormap(fig,perfCmap);
    cb = colorbar('southoutside');
    cb.Label.String = 'Relative performance  (best = black)';
    cb.TickDirection = 'out';
end


%% ADDED WITH ABSOLUTE SCALE

% =======================================================================
%  PARAMETER–VS–OBJECTIVE PANELS  —  ABSOLUTE PERFORMANCE SCALE [0–1]
% =======================================================================

% -------------------- user knobs ---------------------------------------
selectedModel = model_list{1};     % e.g. 'm_07_gr4j_4p_2s'
patchFraction = 20;                % stripe height as % of parameter range
nCmap         = 256;               % greyscale resolution

% GREY RAMP: white  →  black  (row nCmap = black)
perfCmap = flipud(gray(nCmap));

% 1)  ➤  SHORT CATCHMENT NAMES  -----------------------------------------
% Put your abbreviations **in the SAME ORDER** as catchments_aridity.
catchments_labels = {'AUS1','BR6','AUS6','C12','C02',...
                     'GB3','C03','HYS','GB2','GB8','LAM'};   % 11 labels

% Create a mapping:  longID  ➜  shortID
labelMap = containers.Map(catchments_aridity, catchments_labels);

% 2)  ➤  HUMAN-FRIENDLY MODEL LABEL  ------------------------------------
modelLabel = 'GR4J';

catchmentNames = fieldnames(optimal_parameters_all);
numCatchments  = numel(catchmentNames);
numOF          = numel(OF_Plot);

for c = 1:numCatchments
    ct      = catchmentNames{c};
    label_c = catchments_labels{c};

    % -------- determine number of parameters in this catchment ----------
    nParams = [];
    for j = 1:numOF
        fld = optimal_parameters_all.(ct).(objective_functions{j});
        if isfield(fld, selectedModel)
            fn = fieldnames(fld.(selectedModel));
            if ~isempty(fn)
                pv = fld.(selectedModel).(fn{1});
                if iscell(pv), pv = cell2mat(pv); end
                nParams = numel(pv);
                break
            end
        end
    end
    if isempty(nParams)
        warning('No parameters for %s – skipped', ct);
        continue
    end

    % ---------------------- figure & layout -----------------------------
    nRows = ceil(sqrt(nParams));
    nCols = ceil(nParams/nRows);
    fig   = figure('Color',          'w', ...
                   'Units',          'normalized', ...
                   'OuterPosition',  [0.02 0.05 0.96 0.90]);
    t = tiledlayout(nRows, nCols, 'TileSpacing', 'compact', ...
                                     'Padding',     'compact');
    title(t, sprintf('Parameter variation – %s  (%s)', label_c, modelLabel), ...
             'FontWeight', 'bold');

    % -------------------- parameter panels ------------------------------
    for p = 1:nParams
        ax = nexttile;  hold(ax, 'on');

        % stripe height for this parameter
        allVals = [];
        for j = 1:numOF
            for r = 1:numel(runFolders)
                rn  = runFolders{r};
                fld = optimal_parameters_all.(ct).(objective_functions{j}).(selectedModel);
                if isfield(fld, rn)
                    v = fld.(rn); if iscell(v), v = cell2mat(v); end
                    if numel(v) >= p, allVals(end+1) = v(p); end
                end
            end
        end
        dynH = max(0.05, patchFraction * range(allVals));

        % ----------- stripes (objective functions) ----------------------
        for j = 1:numOF
            % absolute performance bounds (0 = worst, 1 = best)
            for r = 1:numel(runFolders)
                rn = runFolders{r};

                % skip if no parameters
                op = optimal_parameters_all.(ct).(objective_functions{j}).(selectedModel);
                if ~isfield(op, rn), continue, end
                v = op.(rn); if iscell(v), v = cell2mat(v); end
                if numel(v) < p, continue, end
                paramVal = v(p);

                % skip if no performance value
                if ~isfield(OF_value_cali.(ct).(objective_functions{j}).(selectedModel), rn)
                    continue
                end
                perf = OF_value_cali.(ct).(objective_functions{j}).(selectedModel).(rn);

                % value-based grey (absolute scale 0–1)
                if isnan(perf)
                    patchCol = [0.7 0.7 0.7];          % NaNs → mid-grey
                else
                    pStar = min(max(perf, 0), 1);      % clip to [0,1]
                    idx   = max(1, min(nCmap, round(pStar*(nCmap-1))+1));
                    patchCol = perfCmap(idx, :);
                end

                % horizontal offset to separate runs visually
                xC = j + (r - (numel(runFolders)+1)/2) * 0.1;

                % background stripe
                patch(ax, [xC-0.06 xC+0.06 xC+0.06 xC-0.06], ...
                           [paramVal-dynH/2 paramVal-dynH/2 ...
                            paramVal+dynH/2 paramVal+dynH/2], ...
                           patchCol, 'EdgeColor', 'none', 'FaceAlpha', 0.9);

                % marker
                plot(ax, xC, paramVal, 'o', ...
                     'MarkerFaceColor', 'w', ...
                     'MarkerEdgeColor', colors(j, :), ...
                     'MarkerSize',      8, ...
                     'LineWidth',       1.3);
            end
        end

        % optional GR4J y-limits
        if strcmp(selectedModel, 'm_07_gr4j_4p_2s')
            lim = [1 2000; -20 20; 1 300; 0.5 15];
            if p <= size(lim,1), ylim(ax, lim(p, :)); end
        end

        set(ax, 'XTick', 1:numOF, ...
                'XTickLabel', OF_Plot, ...
                'XTickLabelRotation', 45);
        xlabel(ax, 'Objective Function');
        ylabel(ax, sprintf('Parameter %d', p));
        box(ax, 'on');
        grid(ax, 'on');
    end

    % ---------------------- colour-bar ----------------------------------
    colormap(fig, perfCmap);
    cb = colorbar('southoutside');
    cb.Ticks = 0:0.2:1;
    cb.Label.String = 'Performance (0 = worst  …  1 = best)';
    cb.TickDirection = 'out';
end



%% =====================================================================

%% --- PLOT PARAMETER VARIATION PER CATCHMENT WITH PERFORMANCE-HIGHLIGHTED BACKGROUND ---
% For each catchment, create a figure with one subplot per parameter.
% In each subplot, the x-axis is the objective functions (ordered by OF_Plot).
% For each objective function and each run, the parameter value is plotted.
% Behind each marker, a patch is drawn:
%   - Light green if the run's performance (from OF_value_cali) is >= the benchmark (from threshold)
%   - Light red if the performance is below the benchmark

% Adjust patchHeight to a value that is visible relative to your parameter scale.
patchHeight = 5;  % Change this value as needed

catchmentNames = fieldnames(optimal_parameters_all);
numCatchments = numel(catchmentNames);
numOF = numel(OF_Plot);  % should be 8

for c = 1:numCatchments
    currentCatchment = catchmentNames{c};
    
    % Determine number of parameters for this catchment and selectedModel.
    % We search through the objective functions until we find one with data.
    nParams = [];
    for j = 1:numOF
        if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}), selectedModel)
            runStruct = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel);
            runNames = fieldnames(runStruct);
            if ~isempty(runNames)
                temp = runStruct.(runNames{1});
                if iscell(temp)
                    temp = cell2mat(temp);
                end
                nParams = numel(temp);
                break;
            end
        end
    end
    if isempty(nParams)
        warning('No parameter data found for catchment %s', currentCatchment);
        continue;
    end
    
    % Create figure with a tiled layout: one subplot per parameter.
    nRows = ceil(sqrt(nParams));
    nCols = ceil(nParams / nRows);
    fig = figure;
    t = tiledlayout(nRows, nCols, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    title(t, sprintf('Parameter Variation for Catchment: %s (Model: %s)', currentCatchment, selectedModel), 'FontSize', 12);
    
    % Loop over each parameter (each subplot corresponds to one parameter index)
    for p = 1:nParams
        ax = nexttile;
        hold(ax, 'on');
        
        % For each objective function (x-axis positions given by 1:numOF)
        for j = 1:numOF
            % Marker color for this objective function.
            currentColor = colors(j,:);
            
            % Loop over each run (with a small horizontal offset to avoid overlap)
            for r = 1:length(runFolders)
                currentRun = runFolders{r};
                % Check if optimal parameter data exists for this catchment, objective, model, and run.
                if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel), currentRun)
                    paramData = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(selectedModel).(currentRun);
                    if iscell(paramData)
                        paramData = cell2mat(paramData);
                    end
                    if numel(paramData) < p
                        warning('Parameter vector too short for catchment %s, objective %s, run %s', ...
                            currentCatchment, OF_Plot{j}, currentRun);
                        continue;
                    end
                    param_value = paramData(p);
                    
                    % Retrieve performance for the same catchment, objective, model, and run.
                    if isfield(OF_value_cali.(currentCatchment).(objective_functions{j}).(selectedModel), currentRun)
                        perf = OF_value_cali.(currentCatchment).(objective_functions{j}).(selectedModel).(currentRun);
                    else
                        continue;
                    end
                    
                    % Get the benchmark value from your benchmark variable (here named 'threshold').
                    % (Row = objective function index, column = catchment index)
                    benchmark_val = threshold(j, c);
                    
                    % Determine patch color based on performance vs. benchmark.
                    if perf >= benchmark_val
                        patch_color = [0.8, 1, 0.8];  % light green if performance is above or equal
                    else
                        patch_color = [1, 0.8, 0.8];  % light red if below
                    end
                    
                    % Define a small horizontal offset for this run.
                    offset = (r - (length(runFolders)+1)/2) * 0.1;
                    x_center = j + offset;
                    
                    % Draw a patch behind the marker.
                    % The patch is drawn centered at the parameter value with a fixed height (patchHeight).
                    y_patch = [param_value - patchHeight/2, param_value - patchHeight/2, ...
                               param_value + patchHeight/2, param_value + patchHeight/2];
                    patch_width = 0.12;
                    x_patch = [x_center - patch_width/2, x_center + patch_width/2, ...
                               x_center + patch_width/2, x_center - patch_width/2];
                    patch(ax, x_patch, y_patch, patch_color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
                    
                    % Plot the parameter marker on top.
                    plot(ax, x_center, param_value, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', currentColor);
                end
            end
        end
        
        % Format the subplot: x-axis with objective function labels.
        set(ax, 'XTick', 1:numOF, 'XTickLabel', OF_Plot, 'XTickLabelRotation', 45);
        xlabel(ax, 'Objective Function');
        ylabel(ax, sprintf('Parameter %d Value', p));
        title(ax, sprintf('Parameter %d', p));
        hold(ax, 'off');
    end
    
    % Create a common legend using an invisible axes so as not to disturb the tiled layout.
    %lgdAx = axes('Position', [0 0 1 1], 'Visible', 'off');
    %h = gobjects(numOF,1);
    for j = 1:numOF
    %    h(j) = plot(lgdAx, nan, nan, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', colors(j,:));
    end
    %legend(lgdAx, h, OF_Plot, 'Orientation', 'horizontal', 'Location', 'southoutside');
end






%% PLOT CALIBRATION PERFORMANCE FOR EACH CATCHMENT (for a selected model)
% Choose one model to plot (adjust as needed)
selectedModel = model_list{1};

for catchmentIdx = 1:numCatchments
    catchment = catchments_aridity{catchmentIdx};
    
    % Create a new figure for the current catchment
    fig = figure;
    hold on;
    
    % First, plot the benchmark markers for each objective function.
    for funcIdx = 1:numFunctions
        benchmark_val = threshold(funcIdx, catchmentIdx);  % benchmark for this objective & catchment
        xPos = funcIdx;  % x-axis position for this objective function
        % Plot a black square marker for the benchmark
        plot(xPos, benchmark_val, 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
    end
    
    % Loop over each objective function (using OF_Plot order for labels)
    for funcIdx = 1:numFunctions
        obj_fun = objective_functions{funcIdx};  % assumed same order as OF_Plot
        xPos = funcIdx;  % x-axis position for this objective function
        
        % For each run, plot the performance value (with a small horizontal offset)
        for runIdx = 1:length(runFolders)
            currentRun = runFolders{runIdx};
            % Check if performance value exists for this combination
            if isfield(OF_value_cali.(catchment).(obj_fun).(selectedModel), currentRun)
                perf = OF_value_cali.(catchment).(obj_fun).(selectedModel).(currentRun);
                offset = (runIdx - (length(runFolders)+1)/2) * 0.05;  % small horizontal offset
                currentColor = colors(funcIdx, :);  % color based on the objective function
                plot(xPos + offset, perf, 'o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', currentColor);
            end
        end
    end
    
    % Format the plot
    set(gca, 'XTick', 1:numFunctions, 'XTickLabel', OF_Plot, 'XTickLabelRotation', 45);
    xlabel('Objective Function');
    ylabel('Calibration Performance Metric');
    title(sprintf('Calibration Performance for Catchment: %s (Model: %s)', catchment, selectedModel));
    
    % Optionally, create a legend using invisible markers for color coding.
    % Uncomment if needed.
    % h = gobjects(numFunctions,1);
    % for funcIdx = 1:numFunctions
    %     h(funcIdx) = plot(nan, nan, 'o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', colors(funcIdx, :));
    % end
    % legend(h, OF_Plot, 'Orientation', 'horizontal', 'Location', 'southoutside');
    
    hold off;
    exportgraphics(fig, sprintf('CalibrationPerformance_%s.jpeg', catchment), 'Resolution', 300);

    % Optionally, you can increase the quality (resolution) when saving:
    % exportgraphics(fig, sprintf('CalibrationPerformance_%s.jpeg', catchment), 'Resolution', 300);
end



%% Define colors and objective function order
colors = NaN(8,3);
colors(1,:) = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765];
colors(2,:) = [1.0, 0.4980392156862745, 0.054901960784313725];
colors(3,:) = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313];
colors(4,:) = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392];
colors(5,:) = [0.5803921568627451, 0.403921568627451, 0.7411764705882353];
colors(6,:) = [0.8901960784313725, 0.4666666666666667, 0.7607843137254902];
colors(7,:) = [0.7372549019607844, 0.7411764705882353, 0.13333333333333333];
colors(8,:) = [0.09019607843137255, 0.7450980392156863, 0.8117647058823529];

OF_Plot = {'KGE','NSE','log KGE','log NSE','DE','KGE-NP','KGE Split','SHE'};

%% Plot Parameter Variation per Catchment with Colored Objective Functions
% Assume optimal_parameters_all, runFolders, and model_list are already defined.
% Here, we choose one model for illustration.
currentModel = model_list{1};

% Loop through each catchment
catchmentNames = fieldnames(optimal_parameters_all);
for c = 1:length(catchmentNames)
    currentCatchment = catchmentNames{c};
    
    % Determine number of parameters for the current catchment and currentModel:
    nParams = [];
    for j = 1:length(OF_Plot)
        if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}), currentModel)
            runStruct = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(currentModel);
            runNames = fieldnames(runStruct);
            if ~isempty(runNames)
                temp = runStruct.(runNames{1});
                if iscell(temp)
                    temp = cell2mat(temp);
                end
                nParams = numel(temp);
                break;
            end
        end
    end
    if isempty(nParams)
        warning('No parameter data found for catchment %s', currentCatchment);
        continue;
    end
    
    % Create a figure with a tiled layout (one subplot per parameter)
    nRows = ceil(sqrt(nParams));
    nCols = ceil(nParams / nRows);
    fig = figure;
    t = tiledlayout(nRows, nCols, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    title(t, sprintf('Parameter Variation for Catchment: %s (Model: %s)', currentCatchment, currentModel), 'FontSize', 12);
    
    % Loop over each parameter (each subplot)
    for p = 1:nParams
        ax = nexttile;
        hold(ax, 'on');
        
        % Loop over each objective function (x-axis positions)
        for j = 1:length(OF_Plot)
            % Use the corresponding color from the colors array
            currentColor = colors(j,:);
            % For each objective function, loop over the equifinal runs (up to 6)
            for r = 1:length(runFolders)
                % Check if data exists for this catchment, objective function, model, and run.
                if isfield(optimal_parameters_all.(currentCatchment).(objective_functions{j}).(currentModel), runFolders{r})
                    temp = optimal_parameters_all.(currentCatchment).(objective_functions{j}).(currentModel).(runFolders{r});
                    if iscell(temp)
                        temp = cell2mat(temp);
                    end
                    if numel(temp) < p
                        warning('Parameter vector too short for catchment %s, objective %s, run %s', ...
                            currentCatchment, OF_Plot{j}, runFolders{r});
                        continue;
                    end
                    param_value = temp(p);
                    % Add a small offset to avoid overlapping markers based on the run index.
                    offset = (r - (length(runFolders)+1)/2)*0.1;
                    plot(ax, j+offset, param_value, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', currentColor);
                end
            end
        end
        
        % For GR4J, limit y-axis ranges for the 4 parameters using the specified bounds.
        if strcmp(currentModel, 'm_07_gr4j_4p_2s')
            parRanges = [1, 2000;    % x1 [mm]
                         -20, 20;    % x2 [mm/d]
                          1, 300;    % x3 [mm]
                         0.5, 15];   % x4 [d]
            if p <= size(parRanges,1)
                ylim(ax, parRanges(p, :));
            end
        end
        
        % Format the subplot:
        set(ax, 'XTick', 1:length(OF_Plot), 'XTickLabel', OF_Plot, 'XTickLabelRotation', 45);
        xlabel(ax, 'Objective Function');
        ylabel(ax, sprintf('Parameter %d Value', p));
        title(ax, sprintf('Parameter %d', p));
        hold(ax, 'off');
    end
    
    % Optionally, create a custom legend for the objective function colors.
    % Uncomment the block below to add a legend at the bottom of the figure.
    %{
    hold on;
    h = gobjects(length(OF_Plot),1);
    for j = 1:length(OF_Plot)
        h(j) = plot(nan, nan, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', colors(j,:));
    end
    legend(h, OF_Plot, 'Orientation', 'horizontal', 'Location', 'southoutside');
    hold off;
    %}
    
    % Optionally, save the figure:
    % saveas(fig, sprintf('Catchment_%s.png', currentCatchment));
end

%% Dotty Plot Definition

function dotty_plot(catchment, obj_fun, model_name, runFolders, ...
                    optimal_parameters_all, OF_value_cali, perf_threshold)
%DOTTY_PLOT  Parameter–performance scatter (“dotty”) for one catchment.
%
%  Inputs
%  ------
%  catchment               – e.g. 'camelsaus_143110A'
%  obj_fun                 – objective function used for the ordinate
%                            (must match the field name in OF_value_cali)
%  model_name              – e.g. 'm_07_gr4j_4p_2s'
%  runFolders              – cellstr of the six equifinal runs
%  optimal_parameters_all  – structure produced in your script
%  OF_value_cali           – structure of calibration performances
%  perf_threshold          – performance benchmark to highlight runs
%
%  Notes
%  -----
%  • One panel per parameter (GR4J ⇒ 4 panels).
%  • All runs plotted; points filled green when ≥ threshold,
%    open red circles otherwise.

    % ---------- collect data ----------
    runNames   = runFolders(:)';
    nRuns      = numel(runNames);

    % parameter vectors
    param_stack = nan(nRuns, 4);   % GR4J ⇒ 4 parameters
    % performance values
    perf_vec    = nan(nRuns, 1);

    for k = 1:nRuns
        r = runNames{k};

        % parameters
        pVec = optimal_parameters_all.(catchment).(obj_fun).(model_name).(r);
        if iscell(pVec);  pVec = cell2mat(pVec);  end
        param_stack(k, :) = pVec(:)';

        % performance
        perf_vec(k) = OF_value_cali.(catchment).(obj_fun).(model_name).(r);
    end

    % ---------- plotting ----------
    figure('Name', sprintf('%s – %s – %s', catchment, model_name, obj_fun),...
           'Color', 'w');
    tiledlayout(2,2,'TileSpacing','compact');

    parNames = {'x₁  [mm]','x₂  [mm d⁻¹]','x₃  [mm]','x₄  [d]'};

    for p = 1:4
        nexttile; hold on; grid on;
        good = perf_vec >= perf_threshold;
        bad  = ~good;

        % non-behavioural (below threshold) – open red circles
        scatter(param_stack(bad,p),  perf_vec(bad),  50, 'r', 'o', ...
                'filled', 'MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.6);
        % behavioural (above threshold) – filled green circles
        scatter(param_stack(good,p), perf_vec(good), 60, 'g', 'filled');

        xlabel(parNames{p});  ylabel(obj_fun);
        title(sprintf('Parameter %d', p));
    end

    sgtitle(sprintf('“Dotty” plots – %s  |  %s  |  %s', ...
                    catchment, model_name, obj_fun), 'FontWeight','bold');
end

%%
% Define catchments and models (adjust as needed)
catchments_aridity = {'camelsaus_143110A','camelsbr_60615000','camelsaus_607155',...
    'camels_12381400','camels_02017500','camelsgb_39037','camels_03460000',...
    'hysets_01AF007','camelsgb_27035','camelsgb_8013','lamah_200048'};

% choose the catchment and objective you want on the y–axis
myCatch    = 'lamah_200048';
myOF       = 'of_KGE';                 % or 'NSE', 'log NSE', …
myModel    = 'm_07_gr4j_4p_2s';
thr        = 0.5;                   % behavioural threshold

dotty_plot(myCatch, myOF, myModel, runFolders, ...
           optimal_parameters_all, OF_value_cali, thr);